"""
实验运行器 - 使用配置驱动
"""
import gc
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Any

from config import Config
from strategy import ReactiveStrategy, ProactiveStrategies
from utils import get_degree_sorted_nodes


class ExperimentRunner:
    """实验运行器"""
    
    _best_alpha_cache = None

    @classmethod
    def _load_best_alpha(cls):
        """加载最优 alpha 对照表"""
        if cls._best_alpha_cache is not None:
            return cls._best_alpha_cache
        from pathlib import Path
        csv_path = Path(__file__).parent / "output_alpha3" / "best_alpha_summary.csv"
        if csv_path.exists():
            cls._best_alpha_cache = pd.read_csv(csv_path)
        return cls._best_alpha_cache

    def __init__(self, config: Config, Gs, Gp, logger=None, combo_name=None, year=None):
        self.config = config
        self.Gs = Gs
        self.Gp = Gp
        self.logger = logger
        self.combo_name = combo_name
        self.year = str(year) if year is not None else None
        
        # 创建策略实例
        self.reactive_strategies = {}
        self.proactive = ProactiveStrategies(config)
        
        # 运行前瞻性策略
        self._run_proactive_strategies()
    
    def _get_best_alpha(self, strategy: str):
        """从 best_alpha_summary.csv 查当前 combo×year 的最优 alpha"""
        df = self._load_best_alpha()
        if df is None or self.combo_name is None or self.year is None:
            return None
        try:
            year_val = int(self.year)
        except (ValueError, TypeError):
            year_val = self.year
        row = df[(df['Combo'] == self.combo_name) & (df['Year'] == year_val)]
        if len(row) == 0:
            return None
        return float(row[f'{strategy}_best_alpha'].values[0])

    def _run_proactive_strategies(self):
        """运行所有前瞻性策略"""
        self.Gs_s1 = None
        self.Gs_s4 = None
        self.Gs_s5 = None
        self.Gs_s6 = None
        self.added_edges = {}

        strategies = self.config.strategy.proactive_strategies

        if "S1" in strategies:
            if self.logger:
                self.logger.info("预计算 S1...")
            self.Gs_s1, self.added_edges["S1"] = self.proactive.s1(self.Gs, self.Gp)
            if self.logger:
                self.logger.info("S1 完成")

        if "S4" in strategies:
            alpha_s4 = self._get_best_alpha("S4")
            if self.logger:
                self.logger.info(f"预计算 S4 (alpha={alpha_s4})...")
            self.Gs_s4 = self.proactive.s4(self.Gs, self.Gp, alpha_override=alpha_s4)
            if self.logger:
                self.logger.info("S4 完成")

        if "S5" in strategies:
            alpha_s5 = self._get_best_alpha("S5")
            if self.logger:
                self.logger.info(f"预计算 S5 (alpha={alpha_s5})...")
            self.Gs_s5 = self.proactive.s5(self.Gs, self.Gp, alpha_override=alpha_s5)
            if self.logger:
                self.logger.info("S5 完成")

        if "S6" in strategies:
            alpha_s6 = self._get_best_alpha("S6")
            if self.logger:
                self.logger.info(f"预计算 S6 (alpha={alpha_s6})...")
            self.Gs_s6 = self.proactive.s6(self.Gs, self.Gp, alpha_override=alpha_s6)
            if self.logger:
                self.logger.info("S6 完成")

    def _get_reactive_strategy(self, mode: str) -> ReactiveStrategy:
        """获取响应式策略实例"""
        if mode not in self.reactive_strategies:
            self.reactive_strategies[mode] = ReactiveStrategy(self.config, mode)
        return self.reactive_strategies[mode]

    def run(self) -> tuple:
        """运行实验"""

        num_experiments = self.config.experiment.num_experiments
        num_deleted = self.config.experiment.num_deleted_nodes
        reactive_mode = self.config.strategy.reactive_mode

        # 选择节点
        sorted_nodes = get_degree_sorted_nodes(self.Gs)

        # 结果存储
        strategy_results = {
            "Reactive": [],
            "Reactive2": [],
            "S1": [], "S1_simplified": [],
            "S4": [], "S4_simplified": [],
            "S5": [], "S5_simplified": [],
            "S6": [], "S6_simplified": []
        }

        node_results = defaultdict(lambda: {
            "Reactive": [], "Reactive2": [],
            "S1": [], "S1_simplified": [],
            "S4": [], "S4_simplified": [],
            "S5": [], "S5_simplified": [],
            "S6": [], "S6_simplified": []
        })

        # 保存每次实验的 ARP 明细
        experiment_records = []

        for i in tqdm(range(num_experiments), desc="Running Experiments"):
            start_idx = i
            end_idx = start_idx + num_deleted
            deleted_nodes = sorted_nodes[start_idx:end_idx]

            # 记录本次实验的编号和初始删除节点
            experiment_record = {
                "experiment_id": i + 1,
                "deleted_nodes": ",".join(map(str, deleted_nodes))
            }

            # Reactive (full)
            if reactive_mode in ["reactive", "both"]:
                reactive = self._get_reactive_strategy("full")
                probs = reactive.run(self.Gs.copy(), self.Gp.copy(), deleted_nodes)
                arp = np.mean(list(probs.values()))
                strategy_results["Reactive"].append(arp)
                experiment_record["Reactive"] = arp

                for node, prob in probs.items():
                    node_results[node]["Reactive"].append(prob)
            
            # Reactive2 (simplified)
            if reactive_mode in ["reactive2", "both"]:
                reactive2 = self._get_reactive_strategy("simplified")
                probs = reactive2.run(self.Gs.copy(), self.Gp.copy(), deleted_nodes)
                arp = np.mean(list(probs.values()))
                strategy_results["Reactive2"].append(arp)
                experiment_record["Reactive2"] = arp

                for node, prob in probs.items():
                    node_results[node]["Reactive2"].append(prob)
            
            # S1
            if self.Gs_s1 is not None:
                if reactive_mode in ["reactive", "both"]:
                    reactive = self._get_reactive_strategy("full")
                    probs = reactive.run(self.Gs_s1.copy(), self.Gp.copy(), deleted_nodes)
                    arp = np.mean(list(probs.values()))
                    strategy_results["S1"].append(arp)
                    experiment_record["S1"] = arp

                    for node, prob in probs.items():
                        node_results[node]["S1"].append(prob)

                if reactive_mode in ["reactive2", "both"]:
                    reactive = self._get_reactive_strategy("simplified")
                    probs = reactive.run(self.Gs_s1.copy(), self.Gp.copy(), deleted_nodes)
                    arp = np.mean(list(probs.values()))
                    strategy_results["S1_simplified"].append(arp)
                    experiment_record["S1_simplified"] = arp

                    for node, prob in probs.items():
                        node_results[node]["S1_simplified"].append(prob)

            # S2
            if self.Gs_s4 is not None:
                if reactive_mode in ["reactive", "both"]:
                    reactive = self._get_reactive_strategy("full")
                    probs = reactive.run(self.Gs_s4.copy(), self.Gp.copy(), deleted_nodes)
                    arp = np.mean(list(probs.values()))
                    strategy_results["S4"].append(arp)
                    experiment_record["S4"] = arp

                    for node, prob in probs.items():
                        node_results[node]["S4"].append(prob)

                if reactive_mode in ["reactive2", "both"]:
                    reactive = self._get_reactive_strategy("simplified")
                    probs = reactive.run(self.Gs_s4.copy(), self.Gp.copy(), deleted_nodes)
                    arp = np.mean(list(probs.values()))
                    strategy_results["S4_simplified"].append(arp)
                    experiment_record["S4_simplified"] = arp

                    for node, prob in probs.items():
                        node_results[node]["S4_simplified"].append(prob)

            # S3
            if self.Gs_s5 is not None:
                if reactive_mode in ["reactive", "both"]:
                    reactive = self._get_reactive_strategy("full")
                    probs = reactive.run(self.Gs_s5.copy(), self.Gp.copy(), deleted_nodes)
                    arp = np.mean(list(probs.values()))
                    strategy_results["S5"].append(arp)
                    experiment_record["S5"] = arp

                    for node, prob in probs.items():
                        node_results[node]["S5"].append(prob)

                if reactive_mode in ["reactive2", "both"]:
                    reactive = self._get_reactive_strategy("simplified")
                    probs = reactive.run(self.Gs_s5.copy(), self.Gp.copy(), deleted_nodes)
                    arp = np.mean(list(probs.values()))
                    strategy_results["S5_simplified"].append(arp)
                    experiment_record["S5_simplified"] = arp

                    for node, prob in probs.items():
                        node_results[node]["S5_simplified"].append(prob)

            # S4
            if self.Gs_s6 is not None:
                if reactive_mode in ["reactive", "both"]:
                    reactive = self._get_reactive_strategy("full")
                    probs = reactive.run(self.Gs_s6.copy(), self.Gp.copy(), deleted_nodes)
                    arp = np.mean(list(probs.values()))
                    strategy_results["S6"].append(arp)
                    experiment_record["S6"] = arp

                    for node, prob in probs.items():
                        node_results[node]["S6"].append(prob)

                if reactive_mode in ["reactive2", "both"]:
                    reactive = self._get_reactive_strategy("simplified")
                    probs = reactive.run(self.Gs_s6.copy(), self.Gp.copy(), deleted_nodes)
                    arp = np.mean(list(probs.values()))
                    strategy_results["S6_simplified"].append(arp)
                    experiment_record["S6_simplified"] = arp

                    for node, prob in probs.items():
                        node_results[node]["S6_simplified"].append(prob)

            experiment_records.append(experiment_record)

            gc.collect()

        experiment_df = pd.DataFrame(experiment_records)

        return strategy_results, node_results, experiment_df

    def format_results(self, strategy_results: dict, node_results: dict) -> tuple:
        """格式化结果"""
        
        # 策略级别
        strategy_df = pd.DataFrame({
            strategy: results for strategy, results in strategy_results.items()
            if results
        })
        strategy_stats = strategy_df.agg(["mean", "std"]).T
        strategy_stats.columns = ["mean", "std"]
        strategy_stats["strategy"] = strategy_stats.index
        
        # 节点级别
        node_rows = []
        for node, strategies in node_results.items():
            row = {"CUSIP": node}
            for strategy, probs in strategies.items():
                row[strategy] = np.mean(probs) if probs else 0.0
            node_rows.append(row)
        
        node_df = pd.DataFrame(node_rows)
        column_order = ["CUSIP"] + [c for c in strategy_results.keys() if c in node_df.columns]
        node_df = node_df[column_order]
        
        return strategy_stats, node_df
