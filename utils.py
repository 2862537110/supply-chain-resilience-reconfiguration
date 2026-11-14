import random
import numpy as np
import pandas as pd
from pathlib import Path


def set_seed(seed: int = 42):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)


def ensure_output_dir(path: Path) -> Path:
    """确保输出目录存在"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results_to_csv(results_df: pd.DataFrame, output_path: Path) -> None:
    """保存结果到CSV"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}")


def get_degree_sorted_nodes(G):
    """按度数从高到低排序节点"""
    return sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)