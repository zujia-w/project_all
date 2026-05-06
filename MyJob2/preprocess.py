import numpy as np # 导入numpy库，用于数值计算
import torch # 导入torch库，用于张量操作和深度学习
import dgl # 导入dgl库，用于图数据处理
import os # 导入os库，用于文件路径操作
import sys # 导入sys库，用于系统相关操作
sys.path.append('../') # 将父目录添加到系统路径中，以便导入自定义模块
import utils # 导入自定义的utils模块
from scipy.spatial.distance import cosine # 从scipy库导入cosine函数，用于计算余弦距离
from sklearn.model_selection import train_test_split # 从sklearn库导入train_test_split函数，用于数据集划分
from dgl.data import FraudYelpDataset, FraudAmazonDataset # 从dgl.data导入FraudYelpDataset和FraudAmazonDataset数据集

# 定义build_DGraphFin函数，用于构建DGraphFin数据集的DGL图
def build_DGraphFin(data_path=None, save_path=None):
    # load data
    # 如果未指定数据路径，则使用默认路径
    if data_path is None:
        data_path = './dataset/DGraphFin/raw/dgraphfin.npz'
    ds = np.load(data_path) # 加载.npz格式的数据集
    
    x = ds['x'] # 节点特征
    # # node features normalization
    # # 节点特征归一化（可选，当前注释掉）
    # x = (x - x.mean(axis=0)) / x.std(axis=0)
    x = x.astype(dtype=np.float32) # 将节点特征转换为float32类型
    y = ds['y']  # label: 0, 1, 2, 3 for normal, fraud, background1 and background2 # 节点标签
    y = y.astype(dtype=np.int64) # 将节点标签转换为int64类型
    # 初始化训练、验证和测试掩码为全False的布尔数组
    trmask = np.zeros(x.shape[0]).astype(bool)
    trmask[ds['train_mask']] = True # 根据原始数据设置训练掩码
    valmask = np.zeros(x.shape[0]).astype(bool)
    valmask[ds['valid_mask']] = True # 根据原始数据设置验证掩码
    ttmask = np.zeros(x.shape[0]).astype(bool)
    ttmask[ds['test_mask']] = True # 根据原始数据设置测试掩码
    edge_type = ds['edge_type'] # 边类型
    ets = ds['edge_timestamp']  # edge time stamp # 边时间戳

    edge_idx = ds['edge_index'] # 边索引
    # 构建反向边索引，将(u, v)变为(v, u)
    rev_edge_idx = np.vstack((edge_idx.transpose()[1], edge_idx.transpose()[0])).transpose()
    # 构建自环边索引，每个节点到自身
    sl_idx = np.vstack((np.arange(x.shape[0]), np.arange(x.shape[0]))).transpose()  # self loop index
    # 自环边的类型，设置为最大边类型+1
    sl_type = (edge_type.max() * np.ones(x.shape[0]) + 1).astype(int)  # self loop type
    sl_ets = np.zeros(x.shape[0])  # self loop time stamp # 自环边的时间戳设置为0

    # build bi-directional homogeneous graph
    # 构建双向同构图
    edge_idx_set = set(map(tuple, edge_idx)) # 将原始边索引转换为集合，方便查找
    # 标记哪些反向边是原始边中不存在的（即非重复的反向边）
    rev_flg = np.array([False if tuple(e) in edge_idx_set else True for e in rev_edge_idx])  # non-repeated reverse edges' index
    # 合并原始边、非重复的反向边和自环边
    homo_edge_idx = np.vstack((edge_idx, rev_edge_idx[rev_flg], sl_idx))
    # 合并原始边类型、非重复反向边类型和自环边类型
    homo_edge_type = np.hstack((edge_type, edge_type[rev_flg], sl_type))
    # 合并原始边时间戳、非重复反向边时间戳和自环边时间戳
    homo_ets = np.hstack((ets, ets[rev_flg], sl_ets))
    # 创建一个标志，指示边是原始边（1）还是手动添加的反向边（-1）
    homo_rev = np.hstack((np.ones_like(edge_type), -1 * np.ones_like(edge_type)[rev_flg], -1 * np.ones_like(sl_type)))  # a flag to indicate whether it is added manually or not: 1 for original edges and -1 for manually-added reverse edges
    
    # 使用DGL构建图
    g = dgl.graph(tuple(homo_edge_idx.transpose()))
    g.ndata['feat'] = torch.from_numpy(x) # 将节点特征添加到图的ndata中
    g.ndata['label'] = torch.from_numpy(y) # 将节点标签添加到图的ndata中
    # 设置训练掩码
    mask = torch.zeros(x.shape[0]).bool()
    mask[trmask] = True
    g.ndata['train_mask'] = mask
    # 设置验证掩码
    mask = torch.zeros(x.shape[0]).bool()
    mask[valmask] = True
    g.ndata['val_mask'] = mask
    # 设置测试掩码
    mask = torch.zeros(x.shape[0]).bool()
    mask[ttmask] = True
    g.ndata['test_mask'] = mask

    # edge features
    # 将边时间戳添加到图的edata中
    g.edata['ts'] = torch.from_numpy(homo_ets)
    # 将边反向标志添加到图的edata中
    g.edata['rev'] = torch.from_numpy(homo_rev)
    # 将边类型进行one-hot编码并添加到图的edata中
    g.edata['type'] = torch.nn.functional.one_hot(torch.from_numpy(homo_edge_type))
        
    # 如果指定了保存路径，则保存图
    if save_path is not None:
        save_name = 'dgraphfin.bin'
        save_path = os.path.join(save_path, save_name)
        dgl.save_graphs(save_path, [g])
    
    return g

def split_DGraphFin(g, split):
    '''
    prepare data splits
    '''
    # 初始化参数字典，设置输出维度为2（通常对应二分类问题：正常/异常）
    params = {'out_dim': 2}
    # 获取图g的节点特征维度作为输入维度
    params['in_dim'] = g.ndata['feat'].shape[1]
    # 确保训练、验证和测试掩码是布尔类型
    g.ndata['train_mask'] = g.ndata['train_mask'].bool()
    g.ndata['val_mask'] = g.ndata['val_mask'].bool()
    g.ndata['test_mask'] = g.ndata['test_mask'].bool()

    # add edge type information
    # 为图中的每种规范边类型添加一个'typeid'边数据，用于标识边类型
    edgetypes = g.canonical_etypes # 获取所有规范边类型
    for i, etype in enumerate(edgetypes):
        # 为每种边类型设置一个唯一的typeid
        g.edges[etype].data['typeid'] = i * torch.ones(g.num_edges(etype)).long()

    # 如果split参数只有一个元素，表示不进行额外的划分，直接返回原始图和参数
    if len(split) == 1:
        return g, params
    
    # 获取节点标签
    labels = g.ndata['label']
    # 筛选出标签为0或1的节点索引（通常0代表正常，1代表欺诈）
    idx = np.arange(len(labels))[labels <= 1]  # only include normal and fraud nodes
    
    # 根据split参数的格式判断是按比例划分还是按数量划分
    if '.' in split[0]: # 如果split[0]包含小数点，表示按比例划分
        trsize = float(split[0]) # 训练集比例
        valsize, testsize = list(map(float, split[1].split('_'))) # 验证集和测试集比例
        if testsize == '': # 如果测试集比例为空，则计算剩余比例
            testsize = 1 - valsize
    else: # 否则，按数量划分
        trsize = int(split[0]) # 训练集数量
        valsize, testsize = list(map(int, split[1].split('_'))) # 验证集和测试集数量
        if testsize == -1: # 如果测试集数量为-1，则计算剩余数量
            testsize = len(idx) - trsize - valsize
    
    # 使用train_test_split进行分层抽样，划分训练集和剩余数据集
    idx_tr, idx_rest, _, y_rest = train_test_split(idx, labels[idx], train_size=trsize, stratify=labels[idx])
    # 再次使用train_test_split进行分层抽样，划分验证集和测试集
    idx_val, idx_test, _, _ = train_test_split(idx_rest, y_rest, train_size=valsize, stratify=y_rest)
    
    # 重置训练、验证和测试掩码
    g.ndata['train_mask'] = torch.zeros(len(labels)).bool()
    g.ndata['train_mask'][idx_tr] = True # 根据划分结果设置训练掩码
    g.ndata['val_mask'] = torch.zeros(len(labels)).bool()
    g.ndata['val_mask'][idx_val] = True # 根据划分结果设置验证掩码
    g.ndata['test_mask'] = torch.zeros(len(labels)).bool()
    g.ndata['test_mask'][idx_test] = True # 根据划分结果设置测试掩码

    return g, params

def split_tfintsoc(g, split):
    '''
    prepare data splits for t-finance and t-social
    split: "X" or "X,Y_Z", X stands for the ratio of training set over the whole set, Y and Z stands for the ratio over data for evaluations (val + test)
    '''
    # 初始化参数字典，设置输出维度为2（通常对应二分类问题：正常/异常）
    params = {'out_dim': 2}
    # 获取图g的节点特征维度作为输入维度
    params['in_dim'] = g.ndata['feature'].shape[1]
    # 设置关系数量为2
    params['n_rel'] = 2
    # 为边数据添加'type'特征，初始化为全零，并将第一列设置为1（表示一种默认边类型）
    g.edata['type'] = torch.zeros(g.num_edges(), 2).float()
    g.edata['type'][:, 0] = 1
    # 重命名节点特征属性，并将数据类型转换为float
    g.ndata['feat'] = g.ndata.pop('feature').float()  # rename the node feature attributes
    # normalize input features
    # 对输入节点特征进行标准化处理
    g.ndata['feat'] = (g.ndata['feat'] - g.ndata['feat'].mean(dim=0)) / g.ndata['feat'].std(dim=0)
    
    # 如果split参数只有一个元素或更少，表示不进行额外的划分，直接返回原始图和参数
    if len(split) <= 1:
        return g, params
    
    # 获取节点标签
    labels = g.ndata['label']
    # 获取所有节点索引
    idx = list(range(len(labels)))
    
    # 根据split参数的格式判断是按比例划分还是按数量划分
    if '.' in split[0]: # 如果split[0]包含小数点，表示按比例划分
        trsize = float(split[0]) # 训练集比例
        valsize, testsize = list(map(float, split[1].split('_'))) # 验证集和测试集比例
    else: # 否则，按数量划分
        trsize = int(split[0]) # 训练集数量
        valsize, testsize = list(map(int, split[1].split('_'))) # 验证集和测试集数量
        if testsize == -1: # 如果测试集数量为-1，则计算剩余数量
            testsize = len(labels) - trsize - valsize
    
    # 使用train_test_split进行分层抽样，划分训练集和剩余数据集
    idx_tr, idx_rest, _, y_rest = train_test_split(idx, labels, train_size=trsize, stratify=labels)
    # 再次使用train_test_split进行分层抽样，划分验证集和测试集
    idx_val, idx_test, _, _ = train_test_split(idx_rest, y_rest, train_size=valsize, stratify=y_rest)
    
    # 重置训练、验证和测试掩码
    g.ndata['train_mask'] = torch.zeros(len(labels)).bool()
    g.ndata['train_mask'][idx_tr] = True # 根据划分结果设置训练掩码
    g.ndata['val_mask'] = torch.zeros(len(labels)).bool()
    g.ndata['val_mask'][idx_val] = True # 根据划分结果设置验证掩码
    g.ndata['test_mask'] = torch.zeros(len(labels)).bool()
    g.ndata['test_mask'][idx_test] = True # 根据划分结果设置测试掩码

    return g, params  

def split_yelpamz(g, split, normfeat=True):
    '''
    prepare data splits for yelp and amazon
    split: "X" or "X,Y_Z", X stands for the ratio of training set over the whole set, Y and Z stands for the ratio over data for evaluations (val + test)
    '''
    # 初始化参数字典，设置输出维度为2（通常对应二分类问题：正常/异常）
    params = {'out_dim': 2}
    # 获取图g的节点特征维度作为输入维度
    params['in_dim'] = g.ndata['feature'].shape[1]
    # 重命名节点特征属性，并将数据类型转换为float
    g.ndata['feat'] = g.ndata.pop('feature').float()  # rename the node feature attributes

    # 如果normfeat为True，则对节点特征进行归一化
    if normfeat:
        # normalize input features
        # 将节点特征归一化到[0, 1]范围
        g.ndata['feat'] = (g.ndata['feat'] - g.ndata['feat'].min(dim=0)[0]) / (g.ndata['feat'].max(dim=0)[0] - g.ndata['feat'].min(dim=0)[0])

    # add edge type information
    # 为图中的每种规范边类型添加一个'typeid'边数据，用于标识边类型
    edgetypes = g.canonical_etypes # 获取所有规范边类型
    for i, etype in enumerate(edgetypes):
        # 为每种边类型设置一个唯一的typeid
        g.edges[etype].data['typeid'] = i * torch.ones(g.num_edges(etype)).long()

    # 如果split参数只有一个元素或更少，表示不进行额外的划分，直接返回原始图和参数
    if len(split) <= 1:
        return g, params
    
    # 获取节点标签
    labels = g.ndata['label']
    # 获取所有节点索引
    idx = list(range(len(labels)))
    
    # 根据split参数的格式判断是按比例划分还是按数量划分
    if '.' in split[0]: # 如果split[0]包含小数点，表示按比例划分
        trsize = float(split[0]) # 训练集比例
        valsize, testsize = list(map(float, split[1].split('_'))) # 验证集和测试集比例
    else: # 否则，按数量划分
        trsize = int(split[0]) # 训练集数量
        valsize, testsize = list(map(int, split[1].split('_'))) # 验证集和测试集数量
        if testsize == -1: # 如果测试集数量为-1，则计算剩余数量
            testsize = len(labels) - trsize - valsize
    
    # 使用train_test_split进行分层抽样，划分训练集和剩余数据集
    idx_tr, idx_rest, _, y_rest = train_test_split(idx, labels, train_size=trsize, stratify=labels)
    # 再次使用train_test_split进行分层抽样，划分验证集和测试集
    idx_val, idx_test, _, _ = train_test_split(idx_rest, y_rest, train_size=valsize, stratify=y_rest)
    
    # 重置训练、验证和测试掩码
    g.ndata['train_mask'] = torch.zeros(len(labels)).bool()
    g.ndata['train_mask'][idx_tr] = True # 根据划分结果设置训练掩码
    g.ndata['val_mask'] = torch.zeros(len(labels)).bool()
    g.ndata['val_mask'][idx_val] = True # 根据划分结果设置验证掩码
    g.ndata['test_mask'] = torch.zeros(len(labels)).bool()
    g.ndata['test_mask'][idx_test] = True # 根据划分结果设置测试掩码

    return g, params

def load_dataset(dataset_name, split, seed=0, het=False, save_path=None):
    # 设置随机种子，保证实验的可复现性
    utils.set_random_seed(1000 + seed)
    # 将split字符串按逗号分割，并去除首尾空格
    split = split.strip().split(',')
    
    # 根据数据集名称加载和处理数据
    if dataset_name =='dgraphfin':
        gfilename = 'dgraphfin.bin'
        gpath = os.path.join('data_processing', gfilename)
        # 如果图文件已存在，则直接加载
        if os.path.exists(gpath):
            g, _ = dgl.load_graphs(gpath)
            g = g[0] # DGL load_graphs返回一个图列表，取第一个
        else:
            # 否则，构建DGraphFin图
            g = build_DGraphFin(het=het, save_path='data_processing' if save_path is None else save_path)
        # 划分DGraphFin数据集
        g, params = split_DGraphFin(g, split)
        # 设置关系数量为边类型特征的维度
        params['n_rel'] = g.edata['type'].shape[1]
        # 确保边类型特征为浮点型
        g.edata['type'] = g.edata['type'].float()
        
    elif dataset_name == 'tfin':
        gpath = './dataset/T-Finance/tfinance'
        g, _ = dgl.load_graphs(gpath)
        g = g[0]
        # 将标签转换为one-hot编码的argmax形式
        g.ndata['label'] = g.ndata['label'].argmax(1)
        # 划分T-Finance数据集
        g, params = split_tfintsoc(g, split)
    elif dataset_name == 'tsoc':
        gpath = './dataset/T-Social/tsocial'
        g, _ = dgl.load_graphs(gpath)
        g = g[0]
        # 划分T-Social数据集
        g, params = split_tfintsoc(g, split)
    elif dataset_name == 'yelp':
        gpath = './dataset/FraudYelp'
        g = FraudYelpDataset(gpath) # 加载Yelp欺诈数据集
        g = g[0]
        
        # 转换为同构图，并保留节点数据
        g = dgl.to_homogeneous(g, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        # 将原始的边类型编码为one-hot向量作为边特征
        g.edata['type'] = torch.nn.functional.one_hot(g.edata['_TYPE']).float()
        # 确保节点掩码为布尔类型
        g.ndata['train_mask'] = g.ndata['train_mask'].bool()
        g.ndata['val_mask'] = g.ndata['val_mask'].bool()
        g.ndata['test_mask'] = g.ndata['test_mask'].bool()
        # 划分Yelp数据集
        g, params = split_yelpamz(g, split)
        # 设置关系数量为边特征的维度
        params['n_rel'] = g.edata['type'].shape[1]
    elif dataset_name == 'amazon':
        gpath = './dataset/FraudAmazon'
        g = FraudAmazonDataset(gpath)  # 1. 加载原始的 FraudAmazonDataset
        g = g[0]
        
        # 2. 转换为同构图，并保留节点数据
        g = dgl.to_homogeneous(g, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        # 3. 将原始的边类型编码为 one-hot 向量作为边特征
        g.edata['type'] = torch.nn.functional.one_hot(g.edata['_TYPE']).float()
        # 4. 确保节点掩码为布尔类型
        g.ndata['train_mask'] = g.ndata['train_mask'].bool()
        g.ndata['val_mask'] = g.ndata['val_mask'].bool()
        g.ndata['test_mask'] = g.ndata['test_mask'].bool()
        # 5. 划分数据集并获取参数
        g, params = split_yelpamz(g, split)
        # 6. 设置关系数量为边特征的维度
        params['n_rel'] = g.edata['type'].shape[1]
    
    else:
        # 如果数据集名称未实现，则抛出错误
        raise NotImplementedError('dataset {} not implemented'.format(dataset_name))

    return g, params

