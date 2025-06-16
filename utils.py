import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from utils import *
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
import random

def split_dataframe(df, column):
    # 使用groupby方法分割数据框
    grouped = df.groupby(column)
    
    # 创建一个字典来存储分割后的数据框
    df_dict = {}
    
    # 遍历groupby对象，将每个分组的数据框存储在字典中
    for name, group in grouped:
        df_dict[name] = group
    
    return df_dict

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


