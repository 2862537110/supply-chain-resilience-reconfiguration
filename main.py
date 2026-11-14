"""
供应链网络韧性分析 - 主入口
支持配置驱动的实验运行（年份 × 参数组合 × kcore）
多进程并行加速：8个进程并行跑不同alpha
"""
import logging
import sys
from pathlib import Path
import os
import multiprocessing

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    get_config, get_years_to_run, get_data_files, get_output_dir, get_log_file,
    LOG_CONFIG, RUN_ALL_COMBOS, get_combos_to_run,
    apply_combo, apply_kcore, get_kcore_configs_to_run,get_tau_values_to_run, apply_tau,
    get_alpha_values_to_run, apply_alpha,
)
from data_loader import load_all_data, clean_firm_size_data
from network_builder import build_networks, add_node_attributes, remove_isolated_nodes_without_revenue
from experiment import ExperimentRunner
from utils import set_seed, save_results_to_csv


def setup_logging(year: str, combo_name: str = None, kcore_k: int = None,
                  tau: float = None, alpha: float = None):
    # 1. 调用 get_log_file 时把 tau, alpha 传进去
    log_file = get_log_file(year, combo_name, kcore_k=kcore_k, tau=tau, alpha=alpha)
    
    # 2. 确保父目录存在（防止因为路径新建失败）
    os.makedirs(log_file.parent, exist_ok=True)

    # 3. 原本的 logger 配置逻辑
    logger = logging.getLogger(f"exp_{year}_{combo_name}_{kcore_k}_{tau}_{alpha}")
    logger.setLevel(logging.INFO)
    
    # 清除旧的 handler 防止重复输出
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件 handler
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

    return logger


def run_single_year(year: str, preset: str = "default",
                    combo: dict = None, kcore_k: int = None, 
                    tau: float = None,
                    alpha: float = None) -> tuple:
    """
    运行单一年份 + 单个参数组合 + 单个 kcore 设置 + 单个 tau + 单个 alpha 的实验

    Args:
        year: 年份 (如 "22")
        preset: 配置预设
        combo: 参数组合 dict
        kcore_k: kcore 的 k 值；None 表示不启用
        tau: 平滑参数；None 表示使用默认值
        alpha: 风险阈值百分位；None 表示使用默认值 (97)

    Returns:
        (strategy_stats, node_df)
    """
    combo_name = combo["name"] if combo else None
    kc_tag = "off" if kcore_k is None else f"k={kcore_k}"
    tau_tag = "default" if tau is None else f"tau={tau}"
    alpha_tag = "default" if alpha is None else f"alpha={alpha}"

    # 设置日志
    logger = setup_logging(year, combo_name, kcore_k, tau, alpha)
    logger.info("=" * 60)
    logger.info(
        f"供应链网络韧性分析启动 - {year}年"
        + (f" - 组合 {combo_name}" if combo_name else "")
        + f" - kcore={kc_tag} - {tau_tag} - {alpha_tag}"
    )
    logger.info(f"配置预设: {preset}")
    logger.info("=" * 60)

    # 1. 先取当年数据文件路径
    data_files = get_data_files(year)

    # 2. 加载配置 + 应用组合 + 应用 kcore + 应用 tau + 应用 alpha
    config = get_config(preset)
    if combo is not None:
        apply_combo(config, combo)
    apply_kcore(config, kcore_k)          
    apply_tau(config, tau)                # <-- 应用 tau
    apply_alpha(config, alpha)            # <-- 应用 alpha
    config.data_files = data_files        

    logger.info(f"实验次数: {config.experiment.num_experiments}")
    logger.info(f"删除概率计算模式: {config.removal_probability.mode}")
    logger.info(f"K值分布: {config.k_distribution.distribution}")
    logger.info(f"Uk分布: {config.uk_distribution.mode}")
    logger.info(f"删除概率用截断正态: {config.removal_probability.use_truncated_normal}")
    logger.info(f"平滑函数开启: {config.removal_probability.use_smooth_function}, tau={config.removal_probability.smooth_tau}")
    logger.info(f"风险阈值百分位 (alpha): {config.strategy.risk_threshold_percentile}")
    logger.info(
        f"kcore: enabled={config.kcore_constraint.enabled}, "
        f"k={config.kcore_constraint.k}, mode={config.kcore_constraint.check_mode}"
    )
    logger.info(f"响应式策略模式: {config.strategy.reactive_mode}")
    logger.info(f"前瞻性策略: {config.strategy.proactive_strategies}")

    set_seed(config.experiment.random_seed)

    # 3. 加载数据
    logger.info(f"加载 {year} 年数据...")
    data = load_all_data(data_files)

    # 4. 清理数据
    data["firm_size"] = clean_firm_size_data(data["firm_size"])

    # 5. 构建网络
    logger.info("构建网络...")
    Gs, Gp = build_networks(data, logger)
    logger.info(f"供应网络节点数: {len(Gs.nodes)}, 边数: {len(Gs.edges)}")
    logger.info(f"竞争网络节点数: {len(Gp.nodes)}, 边数: {len(Gp.edges)}")

    # 6. 添加节点属性
    logger.info("添加节点属性...")
    Gs = add_node_attributes(Gs, data, logger)

    # 7. 删除没有Revenue的孤立节点
    logger.info("删除没有Revenue的孤立节点...")
    Gs, Gp = remove_isolated_nodes_without_revenue(Gs, Gp, logger)
    logger.info(f"清理后供应网络节点数: {len(Gs.nodes)}")

    # 8. 运行实验
    logger.info("运行实验...")
    runner = ExperimentRunner(config, Gs, Gp, logger, combo_name=combo_name, year=year)
    strategy_results, node_results, experiment_df = runner.run()

    # 9. 格式化结果
    strategy_stats, node_df = runner.format_results(strategy_results, node_results)

    # 10. 打印
    logger.info("\n" + "=" * 60)
    logger.info(
        f"{year}年"
        + (f" [{combo_name}]" if combo_name else "")
        + f" [kcore={kc_tag}] [tau={tau_tag}] [alpha={alpha_tag}]"
        + " - 策略平均删除概率:"
    )
    for idx, row in strategy_stats.iterrows():
        logger.info(f"  {row['strategy']}: {row['mean']:.4f} ± {row['std']:.4f}")
    logger.info("=" * 60)

    # 11. 保存结果到 {combo}/[tau]/[alpha]/[kcore=k]/{year}/ 目录
    output_dir = get_output_dir(year, combo_name, kcore_k=kcore_k, tau=tau, alpha=alpha)
    
    output_file = output_dir / config.experiment.output_filename
    save_results_to_csv(node_df, output_file)

    strategy_file = output_dir / "strategy_summary.csv"
    strategy_stats.to_csv(strategy_file)
    logger.info(f"策略统计已保存到: {strategy_file}")

    experiment_df.to_csv(
        output_dir / "experiment_arp_details.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return strategy_stats, node_df


def _run_task_worker(task):
    """
    多进程 worker：接收一个任务元组，运行单次实验。
    每个 worker 在自己的进程里独立运行，互不干扰。

    Args:
        task: (year, preset, combo, kcore_k, tau, alpha) 元组

    Returns:
        (combo_name, tau, kcore_k, alpha, year, stats_dict)
        其中 stats_dict 转成 dict 传输，避免大对象在进程间传递
    """
    year, preset, combo, kcore_k, tau, alpha = task

    strategy_stats, node_df = run_single_year(
        year=year,
        preset=preset,
        combo=combo,
        kcore_k=kcore_k,
        tau=tau,
        alpha=alpha
    )

    combo_name = combo["name"] if combo else None

    stats_dict = {
        'strategy': strategy_stats['strategy'].tolist(),
        'mean': strategy_stats['mean'].tolist(),
        'std': strategy_stats['std'].tolist(),
    }

    return (combo_name, tau, kcore_k, alpha, year, stats_dict)


def main(preset: str = "default", combo_names: list = None):
    """
    主函数 - 运行 年份 × 参数组合 × kcore 的笛卡尔积
    使用多进程并行加速：8个进程并行处理 kcore × year 维度

    Args:
        preset: 配置预设 - "default" | "experiment" | "full" | "sensitivity"
        combo_names: 指定要跑的 combo 名称列表，如 ["0.1_N_N", "0.1_N_U"]
                     None 表示跑 get_combos_to_run() 返回的全部
    """
    years_to_run  = get_years_to_run()
    all_combos = get_combos_to_run()

    # 按 combo_names 筛选
    if combo_names:
        combos_to_run = [c for c in all_combos if c["name"] in combo_names]
        if not combos_to_run:
            raise ValueError(f"指定的 combo_names {combo_names} 不在 PARAM_COMBOS 中")
    else:
        combos_to_run = all_combos

    kcore_list    = get_kcore_configs_to_run()

    kc_desc = ["off" if k is None else f"k={k}" for k in kcore_list]

    print(f"\n{'='*60}")
    print(f"将要运行的年份: {years_to_run}")
    print(f"将要运行的参数组合: {[c['name'] for c in combos_to_run]}")
    print(f"将要运行的 kcore: {kc_desc}")
    print(f"配置预设: {preset}")

    # 获取需要跑的 tau 列表
    taus_to_run = get_tau_values_to_run()

    alpha = None

    # 总任务数 = combo × tau × kcore × year
    total_tasks = len(combos_to_run) * len(taus_to_run) * len(kcore_list) * len(years_to_run)
    print(f"总运行次数: {total_tasks}")
    print(f"并行进程数: 8")
    print(f"{'='*60}\n")

    all_results = {}   # key: (combo_name, tau, kcore_k, year)

    for combo in combos_to_run:
        combo_name = combo["name"]
        print(f"\n{'#'*60}")
        print(f"# 开始参数组合: {combo_name}")
        print(f"{'#'*60}\n")

        for tau in taus_to_run:
            tau_tag = "default" if tau is None else f"tau={tau}"
            print(f"\n{'='*60}")
            print(f"= 当前平滑参数: {tau_tag}")
            print(f"{'='*60}")

            # === 多进程并行 ===
            tasks = []
            for kcore_k in kcore_list:
                for year in years_to_run:
                    tasks.append((year, preset, combo, kcore_k, tau, alpha))

            num_processes = min(8, len(tasks))
            print(f"\n启动 {num_processes} 个进程并行运行 {len(tasks)} 个任务 ({len(kcore_list)} kcore × {len(years_to_run)} 年)...")
            print(f"S4/S5/S6 将自动使用各自最优 alpha（从 best_alpha_summary.csv 查表）")

            # 创建进程池，并行运行
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(_run_task_worker, tasks)

            # 收集结果
            for result in results:
                r_combo_name, r_tau, r_kcore_k, r_alpha, r_year, stats_dict = result
                kc_tag_r = "off" if r_kcore_k is None else f"k={r_kcore_k}"
                tau_tag_r = "default" if r_tau is None else f"tau={r_tau}"

                all_results[(r_combo_name, r_tau, r_kcore_k, r_year)] = {
                    'strategy_stats': stats_dict,
                }
                print(f"\n✅ {r_combo_name} | {tau_tag_r} | kcore={kc_tag_r} | {r_year} 年运行完成！")
                for s, m, st in zip(stats_dict['strategy'], stats_dict['mean'], stats_dict['std']):
                    print(f"  {s}: {m:.4f} ± {st:.4f}")

    # 打印总结
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("所有 组合/ tau / kcore / 年份 运行完成！总结:")
        print("=" * 60)
        for (combo_name, tau, kcore_k, year), results in all_results.items():
            kc_tag = "off" if kcore_k is None else f"k={kcore_k}"
            tau_tag = "default" if tau is None else f"tau={tau}"
            stats_dict = results['strategy_stats']
            print(f"\n[{combo_name}] [tau={tau_tag}] [kcore={kc_tag}] {year}年:")
            for s, m, st in zip(stats_dict['strategy'], stats_dict['mean'], stats_dict['std']):
                print(f"  {s}: {m:.4f} ± {st:.4f}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default="default",
                        choices=["default", "experiment", "full", "sensitivity"],
                        help="配置预设")
    parser.add_argument("--combos", type=str, default=None,
                        help="指定要跑的 combo 名称，逗号分隔，如 0.1_N_N,0.1_N_U")
    args = parser.parse_args()

    combo_names = None
    if args.combos:
        combo_names = [c.strip() for c in args.combos.split(",")]

    main(args.preset, combo_names)