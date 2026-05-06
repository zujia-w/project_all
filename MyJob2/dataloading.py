# 导入必要的库
import numpy as np
import torch
from torch.utils.data import Dataset # 导入PyTorch的Dataset基类
import dgl # 导入DGL库，用于图数据处理
import copy # 导入copy模块，用于对象的深拷贝

# 定义DSDataset类，继承自PyTorch的Dataset
class DSDataset(Dataset):
    '''
    down-sampling dataset
    sample equal-sized positive and negative samples for each batch
    下采样数据集 (down-sampling dataset)
    为每个批次采样等量的正样本和负样本
    '''
    def __init__(self, graph) -> None:
        super().__init__() # 调用父类构造函数
        self.num_nodes = graph.num_nodes() # 图中的节点总数
        # 获取训练集、验证集和测试集的节点掩码，并转换为布尔类型的numpy数组
        self.train_mask = graph.ndata['train_mask'].bool().numpy()
        self.val_mask = graph.ndata['val_mask'].bool().numpy()
        self.test_mask = graph.ndata['test_mask'].bool().numpy()
        self.labels = graph.ndata['label'].numpy() # 获取节点标签

        # 找出训练集中所有正样本节点ID
        self.pos_nids = np.arange(self.num_nodes)[(self.labels == 1) * self.train_mask]
        # 找出训练集中所有负样本节点ID
        self.neg_nids = np.arange(self.num_nodes)[(self.labels == 0) * self.train_mask]
        self.pos_list = self.pos_nids # 正样本列表初始化为所有正样本节点ID
        self.resample() # 调用resample方法，初始化负样本列表

    def resample(self):
        '''
        resample a list of negative nodes, with the same size of positive nodes
        重新采样负样本节点列表，使其数量与正样本节点数量相同
        '''
        # 随机打乱所有负样本节点ID，并选取与正样本数量相同的负样本
        self.neg_list = np.random.permutation(self.neg_nids)[:len(self.pos_nids)]
    
    def __getitem__(self, index):
        '''
        return format: [neg_x, neg_y, pos_x, pos_y], where pos_x and neg_x are node ids
        返回格式: [neg_x, neg_y, pos_x, pos_y]，其中pos_x和neg_x是节点ID
        '''
        pos_x = self.pos_list[index] # 获取指定索引的正样本节点ID
        neg_x = self.neg_list[index] # 获取指定索引的负样本节点ID
        pos_x = torch.LongTensor([pos_x]) # 将正样本节点ID转换为LongTensor
        neg_x = torch.LongTensor([neg_x]) # 将负样本节点ID转换为LongTensor
        # 返回负样本节点ID、负样本标签（0）、正样本节点ID、正样本标签（1）
        return neg_x, torch.zeros(len(neg_x)).long(), pos_x, torch.ones(len(pos_x)).long()
    
    def __len__(self):
        # 返回数据集中正样本节点的总数，这决定了每个epoch的迭代次数
        return len(self.pos_nids)


