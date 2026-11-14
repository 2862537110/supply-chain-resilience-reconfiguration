"""
策略实现 - 支持配置注入
"""
import random
import numpy as np
import networkx as nx
import math
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

from config import Config, ParameterDistribution, UkDistribution


# ============ 参数采样器 ============
class ParameterSampler:
    """参数采样器，根据配置生成参数"""

    def __init__(self, config: Config):
        self.config = config

    def get_partner_preference_k(self) -> float:
        """Eq.(1) 的伙伴偏好系数 k
        """
        k_dist = self.config.k_distribution
        std = 0.1 if k_dist.distribution == "n1" else 0.2
        return max(1.0, float(np.random.normal(1.5, std)))

    def get_uk_value(self, size: float, config: UkDistribution,
                     all_sizes: Optional[np.ndarray] = None) -> int:
        """根据企业规模生成Uk值。
        """
        # ---- size_uniform 模式：直接按规模均匀采样 ----
        if config.mode == "size_uniform":
            low = size / 2.0 - 2.0
            high = size / 2.0 + 2.0
            value = np.random.uniform(low, high)
            return max(1, int(np.round(value)))

        # ---- tiered_normal 模式：按分位数分层正态采样 ----
        # 优先按配置的分位数分层
        if all_sizes is not None and len(all_sizes) > 0:
            q_small = float(np.percentile(all_sizes, config.small_threshold_percentile))
            q_medium = float(np.percentile(all_sizes, config.medium_threshold_percentile))
            if size <= q_small:  # 小
                params = config.small
            elif size <= q_medium:  # 中
                params = config.medium
            else:  # 大
                params = config.large
        else:
            # 兜底：原绝对阈值（仅兼容不传 all_sizes 的旧调用）
            if size < 2:  # 小
                params = config.small
            elif size < 4:  # 中
                params = config.medium
            else:  # 大
                params = config.large

        dist = params.get("distribution", "normal")
        if dist == "normal":
            value = np.random.normal(params.get("mean", 2), params.get("std", 0.5))
        elif dist == "uniform":
            value = np.random.uniform(params.get("low", 1), params.get("high", 3))
        else:
            value = params.get("constant", 2)

        return max(1, int(np.round(value)))

    def clear_cache(self):
        pass


# ============ 删除概率计算器 ============
@dataclass
class RemovalProbabilityConfig:
    mode: str
    use_smooth_function: bool
    smooth_tau: float
    supplier_weight: float
    customer_weight: float
    clip_probability: bool
    use_truncated_normal: bool
    noise_std: float


class RemovalProbabilityCalculator:
    """删除概率计算器"""

    def __init__(self, config: Config):
        self.config = config
        self.mode = config.removal_probability.mode

    def calculate(self, node: str, lost_suppliers: int, total_suppliers: int,
                  lost_customers: int, total_customers: int,
                  firm_size: float, min_size: float, max_size: float) -> float:
        if total_suppliers <= 0 and total_customers <= 0:
            return 0.0
        if lost_suppliers < 0 or lost_customers < 0:
            return 0.0

        rp = self.config.removal_probability

        loss_ratio_suppliers = lost_suppliers / total_suppliers if total_suppliers > 0 else 0.0
        loss_ratio_customers = lost_customers / total_customers if total_customers > 0 else 0.0

        if self.mode == "supplier_only":
            loss_ratio = loss_ratio_suppliers
        elif self.mode == "customer_only":
            loss_ratio = loss_ratio_customers
        elif self.mode == "weighted":
            loss_ratio = rp.supplier_weight * loss_ratio_suppliers + rp.customer_weight * loss_ratio_customers
        else:  # full
            loss_ratio = 0.5 * loss_ratio_suppliers + 0.5 * loss_ratio_customers

        if max_size > min_size:
            normalized_size = (firm_size - min_size) / (max_size - min_size + 1.0)
        else:
            normalized_size = 0.5

        base_prob = 1.0 - normalized_size * (1.0 - loss_ratio)

        # ---------------- A1 Logistic smooth‑step ----------------
        if rp.use_smooth_function:
            tau = rp.smooth_tau
            tau = max(tau, 1e-8)
            arg = (tau - loss_ratio) / tau
            h_tau = 1.0 / (1.0 + math.exp(arg))
            removal_prob = base_prob * h_tau
        else:
            if loss_ratio <= 0:
                removal_prob = 0.0
            else:
                removal_prob = base_prob

        if rp.clip_probability:
            removal_prob = max(0.0, min(1.0, removal_prob))
        return removal_prob

    def apply_noise(self, prob: float) -> float:
        """
        添加随机噪声
        """
        rp = self.config.removal_probability
        if rp.use_truncated_normal:
            noise = np.random.normal(0, rp.noise_std)
            result = prob + noise
        else:
            noise = np.random.uniform(-rp.noise_std, rp.noise_std)
            result = prob + noise
        return max(0.0, min(1.0, result))


# ============ 策略实现 ============
class ReactiveStrategy:
    """响应式策略"""

    def __init__(self, config: Config, mode: str = "full"):
        self.config = config
        self.mode = mode  # "full" or "simplified"
        self.sampler = ParameterSampler(config)
        self.prob_calculator = RemovalProbabilityCalculator(config)

    def run(self, Gs: nx.DiGraph, Gp: nx.Graph, initial_disruption: list,
            max_trials: Optional[int] = None, max_iterations: Optional[int] = None) -> Dict[str, float]:
        """运行响应式策略（内部在 Gs 副本上操作，不污染调用方传入的图）"""

        Gs = Gs.copy()

        if max_trials is None:
            max_trials = self.config.experiment.max_trials
        if max_iterations is None:
            max_iterations = self.config.experiment.max_iterations

        disrupted_nodes = set(initial_disruption)
        affected_nodes = set()
        removal_probabilities = {node: 0 for node in Gs.nodes}

        for node in initial_disruption:
            removal_probabilities[node] = 1

        # 获取 k-core 配置
        kcore_cfg = self.config.kcore_constraint
        k = kcore_cfg.k
        check_mode = kcore_cfg.check_mode

        valid_sizes = [Gs.nodes[node].get('size', 0) for node in Gs.nodes if 'size' in Gs.nodes[node]]
        average_size = np.mean(valid_sizes) if valid_sizes else self.config.node_defaults.size_default

        for node in Gs.nodes:
            if 'size' not in Gs.nodes[node]:
                Gs.nodes[node]['size'] = average_size

        min_size = min(Gs.nodes[node].get('size', average_size) for node in Gs.nodes)
        max_size = max(Gs.nodes[node].get('size', average_size) for node in Gs.nodes)

        all_sizes = np.array([Gs.nodes[n].get('size', average_size) for n in Gs.nodes])
        for node in Gs.nodes:
            if 'U_n' not in Gs.nodes[node]:
                u_n = self.sampler.get_uk_value(
                    Gs.nodes[node]['size'],
                    self.config.uk_distribution,
                    all_sizes=all_sizes,
                )
                Gs.nodes[node]['U_n'] = u_n
                Gs.nodes[node]['Uk'] = u_n
            elif 'Uk' not in Gs.nodes[node]:
                Gs.nodes[node]['Uk'] = Gs.nodes[node]['U_n']

        original_edges = set(Gs.edges())

        # 初始化
        supplier_request_queue = {node: [] for node in Gs.nodes}
        customer_request_queue = {node: [] for node in Gs.nodes}
        trial_count = {node: 0 for node in Gs.nodes}

        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            new_disrupted = set()
            new_affected = set()

            for node in Gs.nodes:
                if node in disrupted_nodes:
                    continue

                # ============================================================
                # k-core 结构性失效检查
                # ============================================================
                if kcore_cfg.enabled:
                    total_suppliers = Gs.in_degree(node)
                    total_customers = Gs.out_degree(node)

                    if check_mode == "supplier_only":
                        original_total = total_suppliers
                    elif check_mode == "customer_only":
                        original_total = total_customers
                    else:  # "both"
                        original_total = total_suppliers + total_customers

                    should_apply_constraint = True
                    if kcore_cfg.only_if_original_degree_ge_k:
                        should_apply_constraint = (original_total >= k)

                    if should_apply_constraint:
                        if check_mode == "supplier_only":
                            active_connections = sum(1 for _ in Gs.predecessors(node) if _ not in disrupted_nodes)
                        elif check_mode == "customer_only":
                            active_connections = sum(1 for _ in Gs.successors(node) if _ not in disrupted_nodes)
                        else:  # "both"
                            active_suppliers = sum(1 for _ in Gs.predecessors(node) if _ not in disrupted_nodes)
                            active_customers = sum(1 for _ in Gs.successors(node) if _ not in disrupted_nodes)
                            active_connections = active_suppliers + active_customers

                        if active_connections < k:
                            new_disrupted.add(node)
                            removal_probabilities[node] = 1
                            continue

                # ============================================================
                # 概率性失效检查
                # ============================================================
                lost_suppliers = sum(1 for _ in Gs.predecessors(node) if _ in disrupted_nodes)
                lost_customers = sum(1 for _ in Gs.successors(node) if _ in disrupted_nodes)
                total_suppliers = Gs.in_degree(node)
                total_customers = Gs.out_degree(node)
                firm_size = Gs.nodes[node]['size']

                if self.mode == "simplified":
                    removal_prob = self.prob_calculator.calculate(
                        node=node,
                        lost_suppliers=lost_suppliers,
                        total_suppliers=total_suppliers,
                        lost_customers=0,
                        total_customers=0,
                        firm_size=firm_size,
                        min_size=min_size,
                        max_size=max_size
                    )
                else:
                    removal_prob = self.prob_calculator.calculate(
                        node=node,
                        lost_suppliers=lost_suppliers,
                        total_suppliers=total_suppliers,
                        lost_customers=lost_customers,
                        total_customers=total_customers,
                        firm_size=firm_size,
                        min_size=min_size,
                        max_size=max_size
                    )

                random_value = self.prob_calculator.apply_noise(removal_prob)
                if random_value <= removal_prob:
                    new_disrupted.add(node)
                    continue

                if removal_prob != removal_probabilities[node]:
                    new_affected.add(node)
                    removal_probabilities[node] = removal_prob

                # ============================================================
                # 寻找替代供应商
                # ============================================================
                if (
                    lost_suppliers > 0
                    and trial_count[node] < max_trials
                    and Gp.has_node(node)
                ):
                    alternative_suppliers = [
                        n for n in Gp.neighbors(node)
                        if n not in disrupted_nodes
                    ]
                    if alternative_suppliers:
                        weighted = []
                        for s in alternative_suppliers:
                            if not Gp.has_edge(node, s):
                                continue
                            w = Gp.edges[node, s]['weight']
                            was_partner = (s, node) in original_edges or (node, s) in original_edges
                            k_pref = self.sampler.get_partner_preference_k() if was_partner else 1.0
                            weighted.append((s, k_pref * w))

                        total_weight = sum(w for _, w in weighted)
                        if total_weight > 0:
                            p_list = [w / total_weight for _, w in weighted]
                            chosen_idx = np.random.choice(len(weighted), p=p_list)
                            chosen_supplier = weighted[chosen_idx][0]
                            supplier_request_queue[chosen_supplier].append((node, 1.0))
                            trial_count[node] += 1

                # ============================================================
                # 寻找替代客户
                # ============================================================
                if (
                    self.mode != "simplified"
                    and lost_customers > 0
                    and trial_count[node] < max_trials
                ):
                    original_customers = set(Gs.successors(node))
                    competitors = set()
                    for customer in original_customers:
                        if not Gp.has_node(customer):
                            continue
                        competitors.update(Gp.neighbors(customer))
                    potential_new_customers = [
                        n for n in competitors
                        if n not in disrupted_nodes
                        and n in Gs.nodes
                    ]
                    if potential_new_customers:
                        weighted = []
                        for c in potential_new_customers:
                            if not Gp.has_edge(node, c):
                                continue
                            w = Gp.edges[node, c]['weight']
                            was_partner = (node, c) in original_edges or (c, node) in original_edges
                            k_pref = self.sampler.get_partner_preference_k() if was_partner else 1.0
                            weighted.append((c, k_pref * w))

                        total_weight = sum(w for _, w in weighted)
                        if total_weight > 0:
                            p_list = [w / total_weight for _, w in weighted]
                            chosen_idx = np.random.choice(len(weighted), p=p_list)
                            chosen_customer = weighted[chosen_idx][0]
                            customer_request_queue[chosen_customer].append((node, 1.0))
                            trial_count[node] += 1

            disrupted_nodes.update(new_disrupted)
            affected_nodes.update(new_affected)

            for removed in new_disrupted:
                if removed not in Gs:
                    continue
                for customer in Gs.successors(removed):
                    if customer in Gs.nodes:
                        Gs.nodes[customer]['Uk'] = min(
                            Gs.nodes[customer]['Uk'] + 1,
                            Gs.nodes[customer]['U_n'],
                        )
                for supplier in Gs.predecessors(removed):
                    if supplier in Gs.nodes:
                        Gs.nodes[supplier]['Uk'] = min(
                            Gs.nodes[supplier]['Uk'] + 1,
                            Gs.nodes[supplier]['U_n'],
                        )

            # 处理请求
            self._process_requests(Gs, Gp, supplier_request_queue,
                                   disrupted_nodes, "supplier", original_edges)
            if self.mode != "simplified":
                self._process_requests(Gs, Gp, customer_request_queue,
                                       disrupted_nodes, "customer", original_edges)

            supplier_request_queue = {node: [] for node in Gs.nodes}
            customer_request_queue = {node: [] for node in Gs.nodes}

            if not new_disrupted and not new_affected:
                break

        return removal_probabilities

    def _process_requests(self, Gs, Gp, request_queue, disrupted_nodes,
                          request_type, original_edges):
        """处理请求
        """
        for firm, requests in request_queue.items():
            if not requests or firm in disrupted_nodes:
                continue

            uk = Gs.nodes[firm].get('Uk', 0)
            if uk <= 0:
                continue

            partner_reqs = []   # [(requester, p_ir)]
            nonpartner_reqs = []
            for requester, p_ir in requests:
                if request_type == "supplier":
                    is_partner = (requester, firm) in original_edges or (firm, requester) in original_edges
                else:
                    is_partner = (firm, requester) in original_edges or (requester, firm) in original_edges

                if is_partner:
                    partner_reqs.append((requester, p_ir))
                else:
                    nonpartner_reqs.append((requester, p_ir))

            # ---- Step 2.1 ----
            accepted = 0
            while uk > 0 and partner_reqs:
                # 权重 = requester 的 firm size
                weights = np.array([
                    max(Gs.nodes[r].get('size', 0), 1e-8)
                    for r, _ in partner_reqs
                ])
                probs = weights / weights.sum()
                chosen_idx = np.random.choice(len(partner_reqs), p=probs)
                chosen_requester, _ = partner_reqs.pop(chosen_idx)

                if request_type == "supplier":
                    Gs.add_edge(firm, chosen_requester)
                else:
                    Gs.add_edge(chosen_requester, firm)
                Gs.nodes[firm]['Uk'] -= 1
                uk -= 1
                accepted += 1

            # ---- Step 2.2 ----
            while uk > 0 and nonpartner_reqs:
                # 权重 = requester size × sim(requester, firm) in Gp
                weights = []
                for r, _ in nonpartner_reqs:
                    r_size = max(Gs.nodes[r].get('size', 0), 1e-8)
                    if Gp.has_edge(r, firm):
                        sim = Gp.edges[r, firm].get('weight', 0)
                    else:
                        sim = 0
                    weights.append(r_size * sim)
                weights = np.array(weights)

                if weights.sum() <= 0:
                    probs = np.ones(len(nonpartner_reqs)) / len(nonpartner_reqs)
                else:
                    probs = weights / weights.sum()

                chosen_idx = np.random.choice(len(nonpartner_reqs), p=probs)
                chosen_requester, _ = nonpartner_reqs.pop(chosen_idx)

                if request_type == "supplier":
                    Gs.add_edge(firm, chosen_requester)
                else:
                    Gs.add_edge(chosen_requester, firm)
                Gs.nodes[firm]['Uk'] -= 1
                uk -= 1
                accepted += 1


# ============ 前瞻性策略 ============
class ProactiveStrategies:
    """前瞻性策略集合"""

    def __init__(self, config: Config):
        self.config = config
        self.reactive_strategy = ReactiveStrategy(config)

    def s1(self, Gs: nx.DiGraph, Gp: nx.Graph) -> tuple:
        """S1
        """
        Gs_copy = Gs.copy()
        added_edges = set()

        focal_limit = getattr(self.config.strategy, "s1_focal_limit", None)
        max_trials = self.config.experiment.max_trials
        reactive_uni = ReactiveStrategy(self.config, mode="full")

        focal_firms = list(Gs_copy.nodes)
        if focal_limit is not None and focal_limit > 0:
            degrees = {n: Gs.degree(n) for n in Gs.nodes}
            focal_firms = sorted(focal_firms, key=lambda n: degrees[n], reverse=True)[:focal_limit]

        for focal_firm in focal_firms:
            tier1_suppliers = set(Gs_copy.predecessors(focal_firm))
            if not tier1_suppliers:
                continue

            distant_suppliers = set()
            for supplier in tier1_suppliers:
                for distant in Gs_copy.predecessors(supplier):
                    if distant != focal_firm and distant in Gs_copy:
                        distant_suppliers.add(distant)

            if not distant_suppliers:
                continue

            impact = {}
            for distant in distant_suppliers:
                temp_Gs = Gs_copy.copy()
                temp_Gp = Gp.copy()
                temp_Gs.remove_node(distant)
                probs = reactive_uni.run(
                    temp_Gs, temp_Gp, [distant],
                    max_trials,
                    max_iterations=self.config.experiment.max_iterations,
                )
                impact[distant] = probs.get(focal_firm, 0.0)

            if not any(v > 0 for v in impact.values()):
                continue

            max_impact = max(impact.values())
            tied = [d for d, v in impact.items() if v == max_impact]
            risky_distant = np.random.choice(tied)

            if risky_distant not in Gp:
                continue
            candidates = [
                c for c in Gp.neighbors(risky_distant)
                if c not in tier1_suppliers and c != focal_firm
            ]
            if not candidates:
                continue
            chosen = np.random.choice(candidates)

            if not Gs_copy.has_edge(chosen, focal_firm):
                Gs_copy.add_edge(chosen, focal_firm)
                added_edges.add((chosen, focal_firm))
                # print(f"[S1] 新增边: {chosen} -> {focal_firm}")

        return Gs_copy, added_edges

    def s4(self, Gs: nx.DiGraph, Gp: nx.Graph, alpha_override: float = None) -> nx.DiGraph:
        """S2:"""
        Gs_new = Gs.copy()
        max_trials = self.config.experiment.max_trials
        threshold = alpha_override if alpha_override is not None else self.config.strategy.risk_threshold_percentile
        max_replacements = 1 

        all_risks = [Gs.nodes[s].get('RiskPerception', 1) for s in Gs.nodes]
        risk_threshold = np.percentile(all_risks, threshold)

        for node in Gs.nodes:
            suppliers = list(Gs_new.predecessors(node))
            if not suppliers:
                continue

            supplier_risks = {s: Gs.nodes[s].get('RiskPerception', 1) for s in suppliers}
            sorted_suppliers = sorted(supplier_risks.items(), key=lambda x: x[1], reverse=True)
            high_risk_suppliers = [
                (s, r) for s, r in sorted_suppliers if r > risk_threshold
            ][:max_replacements]

            if not high_risk_suppliers:
                continue

            replacements_done = 0
            for worst_supplier, worst_risk in high_risk_suppliers:
                if not Gs_new.has_edge(worst_supplier, node):
                    continue

                competitors = list(Gp.neighbors(worst_supplier)) if worst_supplier in Gp else []
                candidates = [c for c in competitors if c in Gs.nodes and c != node
                            and Gs.nodes[c].get('RiskPerception', 1) < worst_risk]

                if not candidates:
                    continue

                prob_before = self.reactive_strategy.run(
                    Gs_new.copy(), Gp.copy(), [worst_supplier], max_trials
                ).get(node, 0)
                candidates_sorted = sorted(candidates, key=lambda x: Gs.nodes[x]['RiskPerception'])

                for alt in candidates_sorted:
                    temp_Gs = Gs_new.copy()
                    if not temp_Gs.has_edge(alt, node):
                        temp_Gs.add_edge(alt, node)
                    if temp_Gs.has_edge(worst_supplier, node):
                        temp_Gs.remove_edge(worst_supplier, node)
                    prob_after = self.reactive_strategy.run(
                        temp_Gs.copy(), Gp.copy(), [worst_supplier], max_trials
                    ).get(node, 0)

                    if prob_after < prob_before:
                        if not Gs_new.has_edge(alt, node):
                            Gs_new.add_edge(alt, node)
                        if Gs_new.has_edge(worst_supplier, node):
                            Gs_new.remove_edge(worst_supplier, node)
                        replacements_done += 1
                        break

        return Gs_new

    def s5(self, Gs: nx.DiGraph, Gp: nx.Graph, alpha_override: float = None) -> nx.DiGraph:
        """S3"""
        Gs_new = Gs.copy()
        max_trials = self.config.experiment.max_trials
        threshold = alpha_override if alpha_override is not None else self.config.strategy.risk_threshold_percentile
        max_replacements = 1

        all_risks = [Gs.nodes[s].get('RiskPerception', 1) for s in Gs.nodes]
        risk_threshold = np.percentile(all_risks, threshold)

        for node in Gs.nodes:
            customers = list(Gs_new.successors(node))
            if not customers:
                continue

            customer_risks = {c: Gs.nodes[c].get('RiskPerception', 1) for c in customers}
            sorted_customers = sorted(customer_risks.items(), key=lambda x: x[1], reverse=True)
            high_risk_customers = [
                (c, r) for c, r in sorted_customers if r > risk_threshold
            ][:max_replacements]

            if not high_risk_customers:
                continue

            for worst_customer, worst_risk in high_risk_customers:
                if not Gs_new.has_edge(node, worst_customer):
                    continue

                competitors = list(Gp.neighbors(worst_customer)) if worst_customer in Gp else []
                candidates = [c for c in competitors if c in Gs.nodes and c != node
                            and Gs.nodes[c].get('RiskPerception', 1) < worst_risk]

                if not candidates:
                    continue

                prob_before = self.reactive_strategy.run(
                    Gs_new.copy(), Gp.copy(), [worst_customer], max_trials
                ).get(node, 0)
                candidates_sorted = sorted(candidates, key=lambda x: Gs.nodes[x]['RiskPerception'])

                for alt in candidates_sorted:
                    temp_Gs = Gs_new.copy()
                    if not temp_Gs.has_edge(node, alt):
                        temp_Gs.add_edge(node, alt)
                    if temp_Gs.has_edge(node, worst_customer):
                        temp_Gs.remove_edge(node, worst_customer)
                    prob_after = self.reactive_strategy.run(
                        temp_Gs.copy(), Gp.copy(), [worst_customer], max_trials
                    ).get(node, 0)

                    if prob_after < prob_before:
                        if not Gs_new.has_edge(node, alt):
                            Gs_new.add_edge(node, alt)
                        if Gs_new.has_edge(node, worst_customer):
                            Gs_new.remove_edge(node, worst_customer)
                        break

        return Gs_new

    def s6(self, Gs: nx.DiGraph, Gp: nx.Graph, alpha_override: float = None) -> nx.DiGraph:
        """S4"""
        Gs_new = Gs.copy()
        max_trials = self.config.experiment.max_trials
        threshold = alpha_override if alpha_override is not None else self.config.strategy.risk_threshold_percentile
        max_replacements = 1

        all_risks = [Gs.nodes[s].get('RiskPerception', 1) for s in Gs.nodes]
        risk_threshold = np.percentile(all_risks, threshold)

        for firm in Gs.nodes():
            suppliers = list(Gs_new.predecessors(firm))
            customers = list(Gs_new.successors(firm))

            partners = {}
            for s in suppliers:
                partners[s] = 'supplier'
            for c in customers:
                if c not in partners:
                    partners[c] = 'customer'

            if not partners:
                continue

            partner_risks = {
                p: Gs.nodes[p].get('RiskPerception', 1) for p in partners
            }
            sorted_partners = sorted(partner_risks.items(), key=lambda x: x[1], reverse=True)
            high_risk_partners = [
                (p, r) for p, r in sorted_partners if r > risk_threshold
            ][:max_replacements]

            if not high_risk_partners:
                continue

            for worst_partner, worst_risk in high_risk_partners:
                role = partners[worst_partner]

                if role == 'supplier':
                    if not Gs_new.has_edge(worst_partner, firm):
                        continue
                else:
                    if not Gs_new.has_edge(firm, worst_partner):
                        continue

                competitors = list(Gp.neighbors(worst_partner)) if worst_partner in Gp else []
                candidates = [
                    c for c in competitors
                    if c in Gs.nodes and c != firm
                    and Gs.nodes[c].get('RiskPerception', 1) < worst_risk
                ]
                if not candidates:
                    continue

                prob_before = self.reactive_strategy.run(
                    Gs_new.copy(), Gp.copy(), [worst_partner], max_trials
                ).get(firm, 0)

                candidates_sorted = sorted(
                    candidates, key=lambda x: Gs.nodes[x]['RiskPerception']
                )

                for alt in candidates_sorted:
                    temp_Gs = Gs_new.copy()

                    if role == 'supplier':
                        if not temp_Gs.has_edge(alt, firm):
                            temp_Gs.add_edge(alt, firm)
                        if temp_Gs.has_edge(worst_partner, firm):
                            temp_Gs.remove_edge(worst_partner, firm)
                    else:
                        if not temp_Gs.has_edge(firm, alt):
                            temp_Gs.add_edge(firm, alt)
                        if temp_Gs.has_edge(firm, worst_partner):
                            temp_Gs.remove_edge(firm, worst_partner)

                    prob_after = self.reactive_strategy.run(
                        temp_Gs.copy(), Gp.copy(), [worst_partner], max_trials
                    ).get(firm, 0)

                    if prob_after < prob_before:
                        if role == 'supplier':
                            if not Gs_new.has_edge(alt, firm):
                                Gs_new.add_edge(alt, firm)
                            if Gs_new.has_edge(worst_partner, firm):
                                Gs_new.remove_edge(worst_partner, firm)
                        else:
                            if not Gs_new.has_edge(firm, alt):
                                Gs_new.add_edge(firm, alt)
                            if Gs_new.has_edge(firm, worst_partner):
                                Gs_new.remove_edge(firm, worst_partner)
                        break

        return Gs_new


# ============ 工厂函数 ============
def create_reactive_strategy(config: Config, mode: str = "full") -> ReactiveStrategy:
    """创建响应式策略实例"""
    return ReactiveStrategy(config, mode)


def create_proactive_strategies(config: Config) -> ProactiveStrategies:
    """创建前瞻性策略实例"""
    return ProactiveStrategies(config)