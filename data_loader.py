import pandas as pd
from config import DATA_FILES
import numpy as np


def load_all_data(data_files=None):
    """
    加载所有数据文件
    
    Args:
        data_files: 数据文件路径字典，如果为 None 则使用默认的 DATA_FILES
    """
    if data_files is None:
        data_files = DATA_FILES
    
    data = {}
    
    data["supply_chain"] = pd.read_excel(data_files["supply_chain"])
    data["competition"] = pd.read_excel(data_files["competition"])
    path, sheet = data_files["firm_size"]
    data["firm_size"] = pd.read_excel(path, sheet_name=sheet)
    data["risk_perception"] = pd.read_excel(data_files["risk_perception"])
    
    return data


def clean_firm_size_data(df):
    """清理企业规模数据"""
    df = df.dropna(subset=['Revenue'])
    df = df.dropna(subset=['cusip'])
    return df


def get_node_size_mapping(df):
    """生成节点规模映射"""
    size_mapping = {}
    for _, row in df.iterrows():
        cusp = row['cusip']
        revenue = row['Revenue']
        size_mapping[cusp] = {
            'Revenue': revenue,
            'size': np.log(revenue + 30000)
        }
    return size_mapping


def get_risk_perception_mapping(df):
    """生成风险感知映射"""
    risk_mapping = {}
    for _, row in df.iterrows():
        cusp = row['CUSIP']
        risk_mapping[cusp] = row['特征单元格数量']
    return risk_mapping