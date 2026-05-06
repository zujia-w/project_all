import torch
import torch.nn as nn
import numpy as np
import os

class MLEDEnhancer(nn.Module):
    """
    原论文Algorithm 1的正确实现
    - type维度: λ=8
    - relation维度: γ=16
    - 融合方式: 加权求和 (公式7)
    """
    def __init__(self, dataset_name, embedding_dir, in_dim=32, llm_dim=4096):
        super().__init__()
        self.dataset_name = dataset_name
        self.embedding_dir = embedding_dir
        self.in_dim = in_dim
        self.llm_dim = llm_dim
        
        # 论文Section 4.1: λ=8, γ=16
        self.type_dim = 8
        self.rel_dim = 16
        
        # 加载预生成的embedding
        self.type_embeddings, self.rel_embeddings = self.load_embeddings()
        
        # 1. MLP降维到论文指定的维度
        self.type_mlp = nn.Sequential(
            nn.Linear(llm_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.type_dim)  # 输出8维
        )
        
        self.rel_mlp = nn.Sequential(
            nn.Linear(llm_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.rel_dim)   # 输出16维
        )
        
        # 2. 投影层：将8/16维投影到in_dim(32)以便加权求和
        self.type_proj = nn.Linear(self.type_dim, in_dim)
        self.rel_proj = nn.Linear(self.rel_dim, in_dim)
        
        # 3. 可学习的融合权重
        self.type_weight = nn.Parameter(torch.tensor(0.1))
        self.rel_weight = nn.Parameter(torch.tensor(0.1))
        
    def load_embeddings(self):
        """加载之前生成的embedding文件"""
        name_map = {
            'yelp': 'yelpchi',
            'yelpchi': 'yelpchi',
            'amazon': 'amazon',
            'tfinance': 'tfinance',
        }
        file_prefix = name_map.get(self.dataset_name, self.dataset_name)
        
        type_path = os.path.join(self.embedding_dir, f"{file_prefix}_type_embeddings.npz")
        rel_path = os.path.join(self.embedding_dir, f"{file_prefix}_relation_embeddings.npz")
        
        print(f"加载类型embedding: {type_path}")
        print(f"加载关系embedding: {rel_path}")
        
        type_data = np.load(type_path)
        type_embeddings = {}
        for key in type_data.keys():
            type_embeddings[key] = torch.tensor(type_data[key], dtype=torch.float32)
        
        rel_data = np.load(rel_path)
        rel_embeddings = {}
        for key in rel_data.keys():
            rel_embeddings[key] = torch.tensor(rel_data[key], dtype=torch.float32)
        
        print(f"加载成功: {len(type_embeddings)}个类型, {len(rel_embeddings)}个关系")
        return type_embeddings, rel_embeddings
    
    def get_dataset_config(self):
        """获取每个数据集的节点类型和关系列表"""
        configs = {
            'yelp': {
                'node_type': 'review',
                'rel_names': ['R-U-R', 'R-T-R', 'R-S-R']
            },
            'yelpchi': {
                'node_type': 'review',
                'rel_names': ['R-U-R', 'R-T-R', 'R-S-R']
            },
            'amazon': {
                'node_type': 'user',
                'rel_names': ['U-P-U', 'U-S-U']
            },
            'tfinance': {
                'node_type': 'account',
                'rel_names': ['transaction']
            }
        }
        return configs.get(self.dataset_name, configs['yelp'])
    
    def forward(self, g, original_features):
        """
        论文公式(7)的实现: F_vt = h_vt + w_tp·z_vt + w_re·m_vt
        """
        device = original_features.device
        num_nodes = original_features.shape[0]
        
        config = self.get_dataset_config()
        node_type = config['node_type']
        rel_names = config['rel_names']
        
        # 1. 类型增强
        if node_type in self.type_embeddings:
            type_emb = self.type_embeddings[node_type].to(device)  # [4096]
            type_8d = self.type_mlp(type_emb)        # [8]
            type_32d = self.type_proj(type_8d)       # [32]
            type_feat = type_32d.unsqueeze(0).expand(num_nodes, -1)  # [num_nodes, 32]
        else:
            type_feat = torch.zeros(num_nodes, self.in_dim).to(device)
        
        # 2. 关系增强（多个关系取平均）
        rel_feat = torch.zeros(num_nodes, self.in_dim).to(device)
        for rel_name in rel_names:
            if rel_name in self.rel_embeddings:
                rel_emb = self.rel_embeddings[rel_name].to(device)  # [4096]
                rel_16d = self.rel_mlp(rel_emb)      # [16]
                rel_32d = self.rel_proj(rel_16d)     # [32]
                rel_feat += rel_32d.unsqueeze(0).expand(num_nodes, -1)
        
        if rel_names:
            rel_feat = rel_feat / len(rel_names)
        
        # 3. 加权求和 (论文公式7)
        enhanced_features = original_features + \
                           self.type_weight * type_feat + \
                           self.rel_weight * rel_feat
        
        return enhanced_features