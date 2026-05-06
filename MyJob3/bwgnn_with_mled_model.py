import torch.nn as nn
from BWGNN import BWGNN
from mled_enhancer import MLEDEnhancer

class BWGNNWithMLED(nn.Module):
    """包装BWGNN，添加MLED增强"""
    def __init__(self, in_feats, h_feats, num_classes, graph, d, 
                 dataset_name, embedding_dir):
        super().__init__()
        
        self.g = graph
        
        # MLED增强器 - 输出维度保持in_feats不变
        self.mled = MLEDEnhancer(
            dataset_name=dataset_name,
            embedding_dir=embedding_dir,
            in_dim=in_feats  # 原始特征维度 (Yelp是32)
        )
        
        # BWGNN - 输入维度不变！
        self.bwgnn = BWGNN(
            in_feats=in_feats,  # 保持32，不是160！
            h_feats=h_feats,
            num_classes=num_classes,
            graph=graph,
            d=d
        )
        
    def forward(self, in_feat):
        # 增强后的特征维度不变 [num_nodes, in_feats]
        enhanced_feat = self.mled(self.g, in_feat)
        return self.bwgnn(enhanced_feat)
    
    def testlarge(self, g, in_feat):
        enhanced_feat = self.mled(g, in_feat)
        return self.bwgnn.testlarge(g, enhanced_feat)