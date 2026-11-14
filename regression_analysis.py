"""
节点级回归分析 — 基于风险感知的供应链脆弱性检验

论文 Section 5.5:
  因变量: 每个节点的 ARP(均值)、Variance、Gini
  自变量: RiskPerception (核心), size, degree_Gs, degree_Gp
  固定效应: 年份 + 参数组合(combo)

流程:
  1. 对每年 × 每个 combo 构建网络
  2. 对每个节点，逐个移除其供应商和客户，跑 ReactiveStrategy，记录该节点自身的删除概率
  3. 对每个节点的删除概率序列计算 Mean / Variance / Gini
  4. 汇总成面板数据，用 OLS + 年份固定效应 + combo 固定效应回归

多进程并行：按 (combo, year) 分配任务，8 进程并行。
  每个 worker 独立加载数据、构建网络、跑节点级模拟、保存中间 CSV。
  主进程收集所有 CSV 汇总面板后跑回归。
"""

import sys
import copy
import multiprocessing
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from tqdm import tqdm
import statsmodels.formula.api as smf

# ---- 路径设置 ----
ABM_DIR = Path(r"D:\c\科研代码\科研-供应链风险感知\大三上 2024.9\基于代理的模型\ABM")
sys.path.insert(0, str(ABM_DIR))

from config import get_config, get_data_files, PARAM_COMBOS, apply_combo
from data_loader import load_all_data, clean_firm_size_data
from network_builder import build_networks, add_node_attributes, remove_isolated_nodes_without_revenue
from strategy import ReactiveStrategy
from utils import set_seed

# ---- 配置 ----
YEARS = [str(y).zfill(2) for y in range(15, 23)]  # 15~22
OUTPUT_DIR = ABM_DIR / "output_alpha3" / "regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 工具函数 ----
def gini(array):
    """计算基尼系数"""
    arr = np.array(array, dtype=float)
    if len(arr) == 0 or np.sum(arr) == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    n = len(arr)
    cumulative_sum = np.cumsum(sorted_arr, dtype=float)
    cumulative_share = cumulative_sum / cumulative_sum[-1]
    cumulative_population = np.linspace(1 / n, 1, n)
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    B = _trapz(cumulative_share, cumulative_population)
    A = 0.5 - B
    return A / 0.5


def calculate_node_vulnerability(Gs, Gp, config, logger=None):

    # 创建一个 reactive 策略实例
    reactive = ReactiveStrategy(config, mode="full")

    results = {}

    nodes = list(Gs.nodes)
    for focal_node in tqdm(nodes, desc="  节点级模拟"):
        # 收集该节点的所有供应商和客户
        suppliers = list(Gs.predecessors(focal_node))
        customers = list(Gs.successors(focal_node))
        partners = suppliers + customers

        if not partners:
            # 没有供应商也没有客户，删除概率序列为空
            results[focal_node] = {
                'Mean': 0.0, 'Variance': 0.0, 'Gini': 0.0,
                'degree_Gs': Gs.degree(focal_node),
                'degree_Gp': Gp.degree(focal_node) if Gp.has_node(focal_node) else 0,
                'size': Gs.nodes[focal_node].get('size', 0),
                'RiskPerception': Gs.nodes[focal_node].get('RiskPerception', 0),
            }
            continue

        # 对每个伙伴，移除后跑 reactive，记录 focal_node 的删除概率
        dcp_values = []
        for partner in partners:
            Gs_copy = Gs.copy()
            probs = reactive.run(Gs_copy, Gp.copy(), [partner])
            dcp_values.append(probs.get(focal_node, 0))

        dcp_arr = np.array(dcp_values, dtype=float)

        # 计算统计量
        mu = float(np.mean(dcp_arr)) if len(dcp_arr) > 0 else 0.0
        var = float(np.var(dcp_arr)) if len(dcp_arr) > 1 else 0.0
        g = gini(dcp_arr) if len(dcp_arr) > 1 else 0.0

        results[focal_node] = {
            'Mean': mu,
            'Variance': var,
            'Gini': g,
            'degree_Gs': Gs.degree(focal_node),
            'degree_Gp': Gp.degree(focal_node) if Gp.has_node(focal_node) else 0,
            'size': Gs.nodes[focal_node].get('size', 0),
            'RiskPerception': Gs.nodes[focal_node].get('RiskPerception', 0),
        }

    return results


def run_regression_only():
    """从已保存的 panel_data.csv 读取数据，只跑回归（跳过模拟）"""
    panel_path = OUTPUT_DIR / "panel_data.csv"
    if not panel_path.exists():
        print(f"错误: 面板数据文件不存在: {panel_path}")
        print("请先运行完整模拟生成面板数据。")
        return

    panel_df = pd.read_csv(panel_path)
    print(f"已读取面板数据: {panel_path}")
    print(f"面板行数: {len(panel_df)}, 年份数: {panel_df['year'].nunique()}")
    print(f"年份分布: {panel_df.groupby('year').size().to_dict()}")

    _do_regression(panel_df)


def _do_regression(panel_df):
    """执行回归分析并保存结果"""
    # 标准化
    panel_df['log_RP'] = np.log1p(panel_df['RiskPerception'])
    panel_df['z_log_RP'] = (panel_df['log_RP'] - panel_df['log_RP'].mean()) / panel_df['log_RP'].std()
    panel_df['z_size'] = (panel_df['size'] - panel_df['size'].mean()) / panel_df['size'].std()

    # 多年 × 多combo面板：按 (year, combo) 双维聚类稳健标准误
    n_years = panel_df['year'].nunique()
    n_combos = panel_df['combo'].nunique()
    panel_df = panel_df.reset_index(drop=True)

    # 构建双维聚类变量
    panel_df['cluster_var'] = panel_df['year'].astype(str) + '_' + panel_df['combo'].astype(str)

    def fit_model(formula):
        m = smf.ols(formula, data=panel_df).fit()
        if n_years > 1 or n_combos > 1:
            m = _attach_cluster_se(m, panel_df['cluster_var'])
        return m

    # Model 1: ARP ~ RiskPerception + controls + year FE + combo FE
    arp_model = fit_model('Mean ~ z_log_RP + z_size + degree_Gs + degree_Gp + C(year) + C(combo)')

    # Model 2: Gini ~ RiskPerception + controls + year FE + combo FE
    gini_model = fit_model('Gini ~ z_log_RP + z_size + degree_Gs + degree_Gp + C(year) + C(combo)')

    # Model 3: Variance ~ RiskPerception + controls + year FE + combo FE
    var_model = fit_model('Variance ~ z_log_RP + z_size + degree_Gs + degree_Gp + C(year) + C(combo)')

    # 打印结果
    print("\n" + "=" * 70)
    print("Model 1: ARP (Mean) ~ RiskPerception + Controls + Year FE")
    print("=" * 70)
    print(arp_model.summary())

    print("\n" + "=" * 70)
    print("Model 2: Gini ~ RiskPerception + Controls + Year FE")
    print("=" * 70)
    print(gini_model.summary())

    print("\n" + "=" * 70)
    print("Model 3: Variance ~ RiskPerception + Controls + Year FE")
    print("=" * 70)
    print(var_model.summary())

    # 保存回归结果到 CSV
    for name, model in [('arp', arp_model), ('gini', gini_model), ('variance', var_model)]:
        ci = model.conf_int()
        ci_lower_col = ci.iloc[:, 0] if hasattr(ci, 'iloc') else ci[0]
        ci_upper_col = ci.iloc[:, 1] if hasattr(ci, 'iloc') else ci[1]
        summary_df = pd.DataFrame({
            'coef': model.params,
            'std_err': model.bse,
            't_stat': model.tvalues,
            'p_value': model.pvalues,
            'ci_lower': ci_lower_col,
            'ci_upper': ci_upper_col,
        })
        summary_df.to_csv(OUTPUT_DIR / f"regression_{name}.csv",
                          encoding='utf-8-sig')
        print(f"回归结果已保存: {OUTPUT_DIR / f'regression_{name}.csv'}")

    print(f"\n所有输出文件位于: {OUTPUT_DIR}")


def _attach_cluster_se(model, cluster_col):

    X = np.asarray(model.model.exog)          # (n, K) 设计矩阵
    resid = np.asarray(model.resid)            # (n,) 残差
    n, K = X.shape

    # 从 model 内部获取实际使用的行索引，对齐 cluster_var
    try:
        used_indices = model.model.data.row_labels
        if hasattr(used_indices, 'index'):
            # pandas Index，可能不是整数位置
            used_idx_array = np.asarray(used_indices)
            # 如果是整数索引，直接用
            if np.issubdtype(used_idx_array.dtype, np.integer):
                cluster_var = np.asarray(cluster_col)[used_idx_array]
            else:
                cluster_var = np.asarray(cluster_col)[:n]
        else:
            cluster_var = np.asarray(cluster_col)[:n]
    except Exception:
        cluster_var = np.asarray(cluster_col)[:n]

    # 确保长度一致
    if len(cluster_var) != n:
        # 回退：截取前 n 个
        cluster_var = cluster_var[:n]

    # (X'X)^{-1}
    XtX_inv = np.linalg.inv(X.T @ X)

    # 按 cluster 聚合: Σ_g X_g' e_g e_g' X_g = Σ_g (X_g * e_g)' (X_g * e_g)
    clusters = np.unique(cluster_var)
    G = len(clusters)

    bread = np.zeros((K, K))
    for c in clusters:
        idx = np.where(cluster_var == c)[0]
        Xg_e = X[idx] * resid[idx][:, np.newaxis]   # (n_g, K) * (n_g,1) -> (n_g, K)
        bread += Xg_e.T @ Xg_e                      # (K, K)

    # 自由度修正
    correction = (G / (G - 1)) * ((n - 1) / (n - K))

    # V_cluster = correction * (X'X)^{-1} * bread * (X'X)^{-1}
    V = correction * XtX_inv @ bread @ XtX_inv

    model.cov_params_default = V
    model.bse = pd.Series(np.sqrt(np.diag(V)), index=model.params.index)
    model.tvalues = model.params / model.bse
    from scipy import stats
    model.pvalues = pd.Series(2 * stats.t.sf(np.abs(model.tvalues), df=n - K),
                              index=model.params.index)
    # conf_int
    model.conf_int = lambda alpha=0.05: pd.DataFrame(
        {'lower': model.params - stats.t.ppf(1 - alpha/2, df=n-K) * model.bse,
         'upper': model.params + stats.t.ppf(1 - alpha/2, df=n-K) * model.bse},
        index=model.params.index
    )
    return model


# ---- 单 (combo, year) 任务 worker（多进程调用） ----
def _process_combo_year_worker(task):
    """
    多进程
    """
    combo_dict, year = task
    combo_name = combo_dict["name"]

    # 每个进程独立初始化
    config = get_config("default")
    apply_combo(config, combo_dict)
    config.strategy.reactive_mode = "reactive"
    set_seed(config.experiment.random_seed)

    # 加载数据
    data_files = get_data_files(year)
    data = load_all_data(data_files)
    data["firm_size"] = clean_firm_size_data(data["firm_size"])

    # 构建网络
    Gs, Gp = build_networks(data)
    Gs = add_node_attributes(Gs, data)
    Gs, Gp = remove_isolated_nodes_without_revenue(Gs, Gp)

    # 节点级模拟
    node_results = calculate_node_vulnerability(Gs, Gp, config)

    # 保存中间 CSV
    year_df = pd.DataFrame([{
        'CUSIP': node,
        'year': int('20' + year),
        'combo': combo_name,
        **res
    } for node, res in node_results.items()])
    csv_path = OUTPUT_DIR / f"node_vulnerability_{combo_name}_{year}.csv"
    year_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    return (combo_name, year, str(csv_path), len(node_results))


# ---- 主流程 ----
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--regression-only', action='store_true',
                        help='跳过模拟，直接从 panel_data.csv 读取数据跑回归')
    parser.add_argument('--processes', type=int, default=8,
                        help='并行进程数（默认8）')
    args = parser.parse_args()

    if args.regression_only:
        run_regression_only()
        return

    # 构建 (combo, year) 任务列表
    tasks = []
    for combo in PARAM_COMBOS:
        for year in YEARS:
            tasks.append((combo, year))

    total_tasks = len(tasks)
    num_processes = min(args.processes, total_tasks)

    print(f"\n{'='*60}")
    print(f"节点级回归分析 — 多进程并行")
    print(f"总任务数: {total_tasks} ({len(PARAM_COMBOS)} combos × {len(YEARS)} years)")
    print(f"并行进程数: {num_processes}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # 多进程并行运行
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(_process_combo_year_worker, tasks)

    # 打印完成情况
    print(f"\n{'='*60}")
    print("所有 (combo, year) 任务完成！")
    print(f"{'='*60}")
    for combo_name, year, csv_path, num_nodes in results:
        print(f"  ✅ {combo_name} | 20{year} | {num_nodes} nodes -> {csv_path}")

    # ---- 收集所有中间 CSV，汇总面板 ----
    print(f"\n{'='*60}")
    print("汇总面板数据...")
    print(f"{'='*60}")

    all_dfs = []
    for combo_name, year, csv_path, num_nodes in results:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        all_dfs.append(df)

    panel_df = pd.concat(all_dfs, ignore_index=True)
    panel_df.to_csv(OUTPUT_DIR / "panel_data.csv", index=False, encoding='utf-8-sig')
    print(f"面板数据已保存到: {OUTPUT_DIR / 'panel_data.csv'}")
    print(f"面板行数: {len(panel_df)}")
    print(f"年份数: {panel_df['year'].nunique()}, 参数组数: {panel_df['combo'].nunique()}")

    # ---- 跑回归 ----
    _do_regression(panel_df)


if __name__ == "__main__":
    main()
