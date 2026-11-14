"""
配置文件 - 集中管理所有可调参数
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np

# ============ 年份配置 ============
CURRENT_YEAR = "22"  # 可选: "15", "16", ..., "22"

# 是否一键运行所有年份 (15-22)
RUN_ALL_YEARS = True

# 所有年份列表（实际数据只有 15-22）
ALL_YEARS = [str(y).zfill(2) for y in range(15, 23)]  # ['15', '16', ..., '22']


# ============ 参数组合配置 ============
# 是否一键运行所有参数组合
RUN_ALL_COMBOS = True

# 当前单独运行的组合
CURRENT_COMBO = "0.1_N_N"

# 8 个参数组合定义
#   name          : 输出目录名，格式 {k_std}_{uk}_{removal}
#   k_dist        : "n1" -> K~N(1.5, 0.1) ; "n2" -> K~N(1.5, 0.2)
#   uk_dist       : "normal" | "uniform"  (对应 UkDistribution.mode)
#   removal_normal: True -> 删除概率用截断正态 ; False -> 用均匀分布
PARAM_COMBOS = [
    {"name": "0.1_N_N", "k_dist": "n1", "uk_dist": "normal",  "removal_normal": True},
    {"name": "0.1_N_U", "k_dist": "n1", "uk_dist": "normal",  "removal_normal": False},
    {"name": "0.1_U_N", "k_dist": "n1", "uk_dist": "uniform", "removal_normal": True},
    {"name": "0.1_U_U", "k_dist": "n1", "uk_dist": "uniform", "removal_normal": False},
    {"name": "0.2_N_N", "k_dist": "n2", "uk_dist": "normal",  "removal_normal": True},
    {"name": "0.2_N_U", "k_dist": "n2", "uk_dist": "normal",  "removal_normal": False},
    {"name": "0.2_U_N", "k_dist": "n2", "uk_dist": "uniform", "removal_normal": True},
    {"name": "0.2_U_U", "k_dist": "n2", "uk_dist": "uniform", "removal_normal": False},
]

# ============ k-core 运行范围配置 ============
# 🔥 是否遍历 kcore 的 k 值
RUN_KCORE_RANGE = True

# 🔥 k 值范围
# KCORE_RANGE = list(range(2, 10, 2))   # [2, 4, 6, 8]
KCORE_RANGE = [4]

# ============ alpha (risk_threshold_percentile) 敏感性分析配置 ============
# 🔥 是否遍历 alpha
RUN_ALPHA_RANGE = False

# 🔥 alpha 值列表
ALPHA_VALUES = [80.0 + 0.5 * i for i in range(41)]  # [80.0, 80.5, ..., 100.0]

# 当前单独运行的 alpha（RUN_ALPHA_RANGE=False 时生效）
CURRENT_ALPHA = 80.0


# ============ tau 平滑参数运行范围配置 ============
# 🔥 是否遍历 tau 的取值
RUN_TAU_RANGE = True

# 🔥 tau 值列表（你要跑的平滑参数）
TAU_VALUES = [0.1, 0.2]

# 当前单独运行的 tau（RUN_TAU_RANGE=False 时生效）
CURRENT_TAU = 0.05


# ============ 路径配置 ============
BASE_DIR = Path("D:/d/各学期资料/论文：双向供应链风险感知/数据处理/relationship及披露")
FIRM_SIZE_DIR = Path("D:/d/各学期资料/论文：双向供应链风险感知/结果和数据")
RISK_DIR = Path("D:/d/各学期资料/论文：双向供应链风险感知/数据处理/risk factor")

# 项目输出根目录
PROJECT_OUTPUT_DIR = Path("D:/c/科研代码/科研-供应链风险感知/大三上 2024.9/基于代理的模型/ABM")

# 本次实验的输出子目录（组合名 / 年份 会拼在其后）
EXPERIMENT_OUTPUT_SUBDIR = "output_alpha3"


def get_data_files(year: str) -> Dict[str, Any]:
    """根据年份获取对应的数据文件路径

    注意: firm_size 的值为 (路径, sheet名) 元组，读取时需解包。
    """
    return {
        "supply_chain": BASE_DIR / f"匹配后的供应链关系2/供应链关系{year}.xlsx",
        "competition": BASE_DIR / f"相似度/相似度{year}.xlsx",
        # 企业规模表按年份分 sheet（"15"~"22"），故返回 (路径, sheet名)
        "firm_size": (FIRM_SIZE_DIR / "企业规模.xlsx", f"{int(year) % 100:02d}"),
        "risk_perception": RISK_DIR / f"披露比例/披露比例{year}.xlsx",
    }


def get_output_dir(year: str, combo: str = CURRENT_COMBO,
                   kcore_k: Optional[int] = None,
                   tau: Optional[float] = None,
                   alpha: Optional[float] = None) -> Path:
    """
    根据年份 + 参数组合 (+ kcore) (+ tau) (+ alpha) 获取输出目录

    - 无 tau, 无 kcore, 无 alpha : {SUB}/{combo}/{year}
    - 有 tau, 无 kcore, 无 alpha  : {SUB}/tau/tau_{tau}/{combo}/{year}
    - 有 kcore, 有 tau            : {SUB}/tau/tau_{tau}/{combo}/kcore={kcore_k}/{year}
    - 有 alpha                    : {SUB}/alpha/{combo}/{year}/alpha_{alpha}
    """
    base = PROJECT_OUTPUT_DIR / EXPERIMENT_OUTPUT_SUBDIR
    if tau is not None:
        tau_str = f"{tau:g}"
        base = base / "tau" / f"tau_{tau_str}"

    if alpha is not None:
        alpha_str = f"{alpha:g}"
        base = base / "alpha"

    base = base / combo
    if kcore_k is not None:
        base = base / f"kcore={kcore_k}"
    base = base / year

    if alpha is not None:
        alpha_str = f"{alpha:g}"
        base = base / f"alpha_{alpha_str}"

    return base


def get_log_file(year: str, combo: str = CURRENT_COMBO,
                 kcore_k: Optional[int] = None,
                 tau: Optional[float] = None,
                 alpha: Optional[float] = None) -> Path:
    """根据年份 + 参数组合 (+ kcore) (+ tau) (+ alpha) 获取日志文件路径"""
    return get_output_dir(year, combo, kcore_k, tau, alpha) / "experiment.log"


# ============ 当前年份的数据文件 ============
DATA_FILES = get_data_files(CURRENT_YEAR)
OUTPUT_DIR = None

# ============ 日志配置 ============
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": get_log_file(CURRENT_YEAR, CURRENT_COMBO),
}


# ============ 实验配置 ============
@dataclass
class ExperimentConfig:
    """实验运行配置"""
    num_experiments: int = 1000         # 实验次数
    num_deleted_nodes: int = 1          # 每次删除的节点数
    random_seed: int = 42               # 随机种子
    max_iterations: int = 100           # 策略最大迭代次数
    max_trials: int = 10                # 最大尝试次数

    # 节点选择策略: "degree" | "random" | "betweenness" | "custom"
    node_selection: str = "degree"

    # 是否保存节点级别结果
    save_node_results: bool = True

    # 输出文件名（相对于年份输出目录）
    output_filename: str = "results.csv"


# ============ K-core 结构约束配置 ============
@dataclass
class KCoreConstraintConfig:
    """
    k-core 结构约束配置
    """
    # 是否启用（总开关）
    enabled: bool = False

    # 统一阈值：每个节点至少需要 k 个活跃连接
    k: int = 8

    # 检查哪些连接?
    check_mode: str = "both"  # "both" | "supplier_only" | "customer_only"

    # 是否只对原始度 >= k 的节点应用约束
    only_if_original_degree_ge_k: bool = True

    # 执行顺序
    execution_order: str = "structural_first"  # "structural_first" | "probability_first"


# ============ K值分布配置 ============
@dataclass
class ParameterDistribution:
    """
    节点级 K 值分布（与 Eq.1 的伙伴偏好 k 不同）
    """
    distribution: str = "n1"  # "n1"(std=0.1) 或 "n2"(std=0.2)

    def sample(self, size: Optional[int] = None):
        """采样K值"""
        import numpy as np
        if self.distribution == "n1":
            std = 0.1
        else:  # "n2"
            std = 0.2

        # 采样 N(1.5, std)
        if size is not None:
            samples = np.random.normal(1.5, std, size)
            return np.maximum(samples, 0.1)  # 确保K值 > 0
        else:
            sample = np.random.normal(1.5, std)
            return max(sample, 0.1)


@dataclass
class UkDistribution:
    """Uk（供应商/客户容量）分布配置

    论文 Table 5 两种设定：
      - tiered_normal : 小/中/大 按 size 的 33rd/66th 分位数分层，
                        分别 N(2,0.5) / N(4,1) / N(6,2)
      - size_uniform  : Unif(Z_i/2 - 2, Z_i/2 + 2)
    """
    mode: str = "tiered_normal"        # "tiered_normal" | "size_uniform"

    # tiered_normal 模式下的分层参数
    small: Dict[str, Any] = field(default_factory=lambda: {
        "mean": 2.0, "std": 0.5, "constant": 2
    })
    medium: Dict[str, Any] = field(default_factory=lambda: {
        "mean": 4.0, "std": 1.0, "constant": 4
    })
    large: Dict[str, Any] = field(default_factory=lambda: {
        "mean": 6.0, "std": 2.0, "constant": 6
    })

    # 规模分位数阈值（tiered_normal 用）
    small_threshold_percentile: float = 33.0
    medium_threshold_percentile: float = 66.0

    @property
    def distribution(self) -> str:
        """兼容旧字段：把 mode 映射回 'normal' / 'uniform'（供 main.py 日志读取）"""
        return "normal" if self.mode == "tiered_normal" else "uniform"


@dataclass
class RemovalProbabilityConfig:
    """删除概率计算配置"""
    # 选择计算方式: "full" | "supplier_only" | "customer_only" | "weighted"
    mode: str = "full"

    # 权重配置（当mode="weighted"时使用）
    supplier_weight: float = 0.5
    customer_weight: float = 0.5

    # 是否使用截断正态分布替代均匀分布
    use_truncated_normal: bool = True

    # 噪声标准差（用于随机扰动）
    noise_std: float = 0.1

    # 是否进行平滑
    use_smooth_function: bool = True
    smooth_tau: float = 0.05

    # 是否启用概率裁剪
    clip_probability: bool = True


# ============ 策略选择配置 ============
@dataclass
class StrategyConfig:
    """策略配置"""
    # 要运行的响应式策略: "reactive" | "reactive2" | "both"
    reactive_mode: str = "reactive"

    # 要运行的前瞻性策略: list of ["S1", "S4", "S5", "S6"]
    proactive_strategies: list = field(default_factory=lambda: ["S1", "S4", "S5", "S6"])

    # S4/S5/S6 的风险阈值百分位数
    risk_threshold_percentile: float = 85

    # 请求处理策略
    request_processing: str = "priority"  # "capacity" | "first_come" | "priority"


# ============ 节点属性默认值 ============
@dataclass
class NodeDefaults:
    """节点属性默认值"""
    size_default: float = 0.0
    risk_perception_default: float = 1.0
    uk_default: int = 3


# ============ 主配置 ============
@dataclass
class Config:
    """主配置类"""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    k_distribution: ParameterDistribution = field(default_factory=ParameterDistribution)
    uk_distribution: UkDistribution = field(default_factory=UkDistribution)
    removal_probability: RemovalProbabilityConfig = field(default_factory=RemovalProbabilityConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    node_defaults: NodeDefaults = field(default_factory=NodeDefaults)
    kcore_constraint: KCoreConstraintConfig = field(default_factory=KCoreConstraintConfig)

    # 数据文件路径（可覆盖）
    data_files: Dict[str, Any] = field(default_factory=lambda: DATA_FILES.copy())

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """从字典创建配置"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# ============ 预设配置 ============
def get_default_config() -> Config:
    """获取默认配置"""
    return Config()


def get_experiment_config() -> Config:
    """获取实验配置（快速运行）"""
    config = Config()
    config.experiment.num_experiments = 50
    config.experiment.max_iterations = 50
    return config


def get_full_analysis_config() -> Config:
    """获取完整分析配置（更精确）"""
    config = Config()
    config.experiment.num_experiments = 500
    config.experiment.max_iterations = 200
    config.experiment.max_trials = 20
    return config


def get_sensitivity_config() -> Config:
    """获取敏感性分析配置（测试不同参数）"""
    config = Config()
    config.experiment.num_experiments = 100
    config.k_distribution.distribution = "n2"
    config.removal_probability.noise_std = 0.2
    return config


# ============ 参数组合相关便捷函数 ============
def apply_combo(config: Config, combo: Dict[str, Any]) -> Config:
    """把一个参数组合应用到 Config 上，并返回该 Config（便于链式调用）

    combo["uk_dist"]:
      "normal"  -> tiered_normal（分层正态，按 size 分位数）
      "uniform" -> size_uniform（Unif(Z_i/2 - 2, Z_i/2 + 2)）
    """
    config.k_distribution.distribution = combo["k_dist"]
    config.uk_distribution.mode = (
        "tiered_normal" if combo["uk_dist"] == "normal" else "size_uniform"
    )
    config.removal_probability.use_truncated_normal = combo["removal_normal"]
    return config


def get_combos_to_run() -> List[Dict[str, Any]]:
    """获取需要运行的参数组合列表

    RUN_ALL_COMBOS=True  -> 返回全部 8 个
    RUN_ALL_COMBOS=False -> 只返回 CURRENT_COMBO 对应的那一个
    """
    if RUN_ALL_COMBOS:
        return PARAM_COMBOS
    matched = [c for c in PARAM_COMBOS if c["name"] == CURRENT_COMBO]
    if not matched:
        raise ValueError(
            f"CURRENT_COMBO={CURRENT_COMBO!r} 不在 PARAM_COMBOS 中，"
            f"可选: {[c['name'] for c in PARAM_COMBOS]}"
        )
    return matched


# ============ k-core 相关便捷函数 ============
def apply_kcore(config: Config, kcore_k: Optional[int]) -> Config:
    """
    应用 kcore 设置到 Config:
      - kcore_k is None : 关闭 kcore (enabled=False)
      - kcore_k = 10    : 启用 kcore，k=10
    """
    if kcore_k is None:
        config.kcore_constraint.enabled = False
    else:
        config.kcore_constraint.enabled = True
        config.kcore_constraint.k = int(kcore_k)
    return config


def get_kcore_configs_to_run() -> List[Optional[int]]:
    """
    返回本次要跑的 kcore k 值列表:
      - [None]          : 不启用 kcore（保存到 {combo}/{year}）
      - [2,3,...,10]    : 逐个 k 跑（保存到 {combo}/kcore={k}/{year}）
      - [10]            : 只跑 k=10

    规则:
      - 若 KCoreConstraintConfig.enabled = False（总开关关闭） -> 返回 [None]
      - 若 enabled=True 且 RUN_KCORE_RANGE=True            -> 返回 KCORE_RANGE
      - 若 enabled=True 且 RUN_KCORE_RANGE=False           -> 返回 [KCoreConstraintConfig.k]
    """
    base_cfg = get_default_config()

    # 总开关关闭：不跑任何 kcore 分支
    if not base_cfg.kcore_constraint.enabled:
        return [None]

    if RUN_KCORE_RANGE:
        return list(KCORE_RANGE)

    return [base_cfg.kcore_constraint.k]

# ============ tau 相关便捷函数 ============
def apply_tau(config: Config, tau_val: Optional[float]) -> Config:
    """应用 tau 设置到 Config:
      - tau_val is None : 使用默认 tau
      - tau_val = 0.05  : 覆盖平滑参数
    """
    if tau_val is not None:
        config.removal_probability.use_smooth_function = True
        config.removal_probability.smooth_tau = float(tau_val)
    return config

def get_tau_values_to_run() -> List[Optional[float]]:
    """返回本次要跑的 tau 值列表
      - RUN_TAU_RANGE=True -> 返回 TAU_VALUES 列表
      - RUN_TAU_RANGE=False -> 返回 [CURRENT_TAU]
    """
    if RUN_TAU_RANGE:
        return list(TAU_VALUES)
    return [CURRENT_TAU]

# ============ alpha 相关便捷函数 ============
def apply_alpha(config: Config, alpha_val: Optional[float]) -> Config:
    """应用 alpha (risk_threshold_percentile) 设置到 Config:
      - alpha_val is None : 使用默认值 (97)
      - alpha_val = 90.0  : 覆盖风险阈值百分位
    """
    if alpha_val is not None:
        config.strategy.risk_threshold_percentile = float(alpha_val)
    return config

def get_alpha_values_to_run() -> List[Optional[float]]:
    """返回本次要跑的 alpha 值列表
      - RUN_ALPHA_RANGE=True  -> 返回 ALPHA_VALUES 列表 [80.0, 80.5, ..., 100.0]
      - RUN_ALPHA_RANGE=False -> 返回 [None]（使用默认值）
    """
    if RUN_ALPHA_RANGE:
        return list(ALPHA_VALUES)
    return [None]

# ============ 便捷函数 ============
def get_config(preset: str = "default", combo_name: Optional[str] = None) -> Config:
    """
    获取预设配置

    Args:
        preset: "default" | "experiment" | "full" | "sensitivity"
        combo_name: 可选，若指定则应用对应参数组合（见 PARAM_COMBOS）
    """
    presets = {
        "default": get_default_config,
        "experiment": get_experiment_config,
        "full": get_full_analysis_config,
        "sensitivity": get_sensitivity_config,
    }
    config = presets.get(preset, get_default_config)()
    if combo_name is not None:
        combo = next((c for c in PARAM_COMBOS if c["name"] == combo_name), None)
        if combo is None:
            raise ValueError(f"未知 combo_name: {combo_name}")
        apply_combo(config, combo)
    return config


def get_years_to_run() -> List[str]:
    """获取需要运行的年份列表"""
    if RUN_ALL_YEARS:
        return ALL_YEARS
    return [CURRENT_YEAR]