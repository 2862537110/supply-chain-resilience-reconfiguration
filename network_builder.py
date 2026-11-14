import networkx as nx
import numpy as np
from data_loader import get_node_size_mapping, get_risk_perception_mapping


def build_networks(data, logger=None):
    """构建供应网络和竞争网络"""

    # 创建图
    Gs = nx.DiGraph()  # 供应网络（有向）
    Gp = nx.Graph()    # 竞争网络（无向）

    # 添加供应网络的边
    for _, row in data["supply_chain"].iterrows():
        Gs.add_edge(row['SOURCE_cusip'], row['TARGET_cusip'])

    # 添加竞争网络的边（带权重）
    for _, row in data["competition"].iterrows():
        Gp.add_edge(row['SOURCE_cusip'], row['TARGET_cusip'], weight=row['Similarity'])

    # 将竞争网络的节点添加到供应网络中
    Gs.add_nodes_from(Gp.nodes)

    return Gs, Gp


def add_node_attributes(Gs, data, logger=None):
    """为节点添加属性：规模、风险感知。
    注意：Uk / U_n 不在这里初始化，统一由 ReactiveStrategy.run() 按 config.uk_distribution 采样，
    避免两套规则不一致、重复消耗随机数。
    """

    # 1. 添加规模和Revenue
    size_mapping = get_node_size_mapping(data["firm_size"])

    for node, attrs in size_mapping.items():
        if node in Gs.nodes:
            Gs.nodes[node]['Revenue'] = attrs['Revenue']
            Gs.nodes[node]['size'] = attrs['size']

    # 为缺失size的节点填充平均值
    size_values = [data['size'] for data in Gs.nodes.values() if 'size' in data]
    average_size = np.mean(size_values) if size_values else 0

    for node in Gs.nodes:
        if 'Revenue' not in Gs.nodes[node]:
            Gs.nodes[node]['Revenue'] = None
        if 'size' not in Gs.nodes[node]:
            Gs.nodes[node]['size'] = average_size

    # 2. 添加风险感知
    risk_mapping = get_risk_perception_mapping(data["risk_perception"])

    for node, risk in risk_mapping.items():
        if node in Gs.nodes:
            Gs.nodes[node]['RiskPerception'] = risk

    average_risk = np.mean([
        data['RiskPerception'] for data in Gs.nodes.values()
        if 'RiskPerception' in data
    ])

    for node in Gs.nodes:
        if 'RiskPerception' not in Gs.nodes[node]:
            Gs.nodes[node]['RiskPerception'] = average_risk

    return Gs


def remove_isolated_nodes_without_revenue(Gs, Gp, logger=None):
    """删除没有Revenue的孤立节点"""

    # 找出Gs中的孤立节点（没有Revenue）
    isolated_nodes_Gs = [
        node for node in Gs.nodes
        if Gs.in_degree(node) == 0 and Gs.out_degree(node) == 0
        and Gs.nodes[node].get('Revenue') is None
    ]

    # 找出Gp中的孤立节点（没有Revenue）
    isolated_nodes_Gp = [
        node for node in Gp.nodes
        if node in nx.isolates(Gp)
        and Gs.nodes[node].get('Revenue') is None
    ]

    # 合并要删除的节点
    nodes_to_remove = set(isolated_nodes_Gs) | set(isolated_nodes_Gp)

    Gs.remove_nodes_from(nodes_to_remove)
    Gp.remove_nodes_from(nodes_to_remove)

    return Gs, Gp