import numpy as np
import torch
import torch.nn as nn
import dgl
from dgl.nn import SAGEConv, GINConv


# 定义SAGE模型类，继承自nn.Module
class SAGE(nn.Module):
    '''
    SAGE with MLP
    the forward/ inference function will store the learned node embeddings before the final mlp layer
    '''
    def __init__(self, params):
        super(SAGE, self).__init__()
        self.params = params
        
        # 定义图神经网络层列表
        self.layers = nn.ModuleList()
        # 定义批量归一化层列表
        self.bns = nn.ModuleList()
        # 定义Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(p=self.params['dropout'])
        # self.activation = nn.LeakyReLU()
        self.activation = nn.ReLU()

        # 根据参数配置构建多层SAGEConv层
        for i in range(self.params['n_layer']):
            # 计算当前层的输入维度
            in_dim = self.params['in_dim'] if i == 0 else self.params['hid_dim']
            # 计算当前层的输出维度
            out_dim = self.params['hid_dim']
            # 添加SAGEConv层，聚合类型为'mean'
            self.layers.append(SAGEConv(in_dim, out_dim, aggregator_type='mean'))
            # 判断是否需要批量归一化，最后一层通常不需要
            bn = False if i == self.params['n_layer'] - 1 else self.params['bn']
            # 添加批量归一化层或恒等映射层
            self.bns.append(nn.BatchNorm1d(out_dim) if bn else nn.Identity())
        # 定义分类器，将隐藏层输出映射到最终输出维度
        self.classifier = nn.Linear(self.params['hid_dim'], self.params['out_dim'])
        
    def forward(self, g, x):
        # 初始特征
        h = x
        # 遍历每一层SAGEConv和批量归一化层
        for i, (layer, bn) in enumerate(zip(self.layers, self.bns)):
            # 获取当前层的图块（如果g是列表，表示是多层图）
            block = g[i] if type(g) == list else g
            # 应用SAGEConv层
            h = layer(block, h)
            # 应用批量归一化
            h = bn(h)
            # 应用激活函数
            h = self.activation(h)
            # 应用Dropout
            h = self.dropout(h)
        
        # 经过分类器得到最终的logits
        logits = self.classifier(h)
        
        # 返回logits和最终的隐藏层表示
        return logits, h
    
    def inference(self, x, dataloader):
        '''
        inference part: full graph inference layer by layer.
        return all node embeddings.
        dataloader: a dgl dataloader with one-layer fullsampler.
        '''

        # 在推理阶段不计算梯度
        with torch.no_grad():
            h = x
            # 定义MLP的批处理大小
            mlp_bs = 10 * dataloader.dataset.batch_size
            
            # gnn layers
            # GNN层推理
            for _, (layer, bn) in enumerate(zip(self.layers, self.bns)):
                # 初始化一个新的张量来存储当前层的输出
                new_h = torch.empty(h.shape[0], self.params['hid_dim'], device=h.device)
                
                # 遍历数据加载器中的每个批次
                for input_nodes, output_nodes, blocks in dataloader:
                    # 获取输入节点的特征并移动到指定设备
                    batch_h = h[input_nodes].to(self.params['device'])
                    # 获取图块并移动到指定设备
                    block = blocks[0].to(self.params['device'])
                    # 应用SAGEConv层
                    batch_h = layer(block, batch_h)
                    # 应用批量归一化
                    batch_h = bn(batch_h)
                    # 应用激活函数
                    batch_h = self.activation(batch_h)
                    # 应用Dropout
                    batch_h = self.dropout(batch_h)
                    # 将结果存储到new_h中对应的输出节点位置
                    new_h[output_nodes] = batch_h
                h = new_h
            
            # new_h = []
            # 初始化一个张量来存储最终的logits
            logits = torch.empty(x.shape[0], self.params['out_dim'], dtype=x.dtype, device=h.device)
            # predictions
            # 预测部分
            # 遍历所有节点，分批进行预测
            for i in range(np.ceil(len(h) / mlp_bs).astype(int)):
                # 获取当前批次的隐藏层表示
                batch_h = h[i*mlp_bs: (i+1)*mlp_bs]
                # 应用分类器
                batch_h = self.classifier(batch_h)
                # 将结果存储到logits中对应的位置
                logits[i*mlp_bs: (i+1)*mlp_bs] = batch_h
            
            # 返回logits和最终的隐藏层表示
        return logits, h


# 定义SAGE_Large模型类，继承自nn.Module
class SAGE_Large(nn.Module):
    '''
    SAGE with MLP;
    The input will first go through a 2-layer MLP;
    Use a 2-layer MLP as the predictor;
    '''
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        # input transformation layers
        # 输入转换层（MLP）
        self.translayers = nn.ModuleList()
        # 第一层线性变换
        self.translayers.append(nn.Linear(self.params['in_dim'], self.params['hid_dim']))
        # 第二层线性变换
        self.translayers.append(nn.Linear(self.params['hid_dim'], self.params['hid_dim']))
        
        # SAGE layers
        # SAGE GNN层
        self.gnnlayers = nn.ModuleList()
        # 批量归一化层
        self.bns = nn.ModuleList()
        # Dropout层
        self.dropout = nn.Dropout(p=self.params['dropout'])
        # 激活函数
        self.activation = nn.ReLU()
        # self.activation = nn.LeakyReLU()
        for i in range(self.params['n_layer']):
            in_dim = out_dim = self.params['hid_dim']
            # 添加SAGEConv层
            self.gnnlayers.append(SAGEConv(in_dim, out_dim, aggregator_type='mean'))
            # 批量归一化设置
            bn = False if i == self.params['n_layer'] - 1 else self.params['bn']
            self.bns.append(nn.BatchNorm1d(out_dim) if bn else nn.Identity())
        
        # predictor
        # 预测器（MLP）
        self.classifier = nn.ModuleList()
        # 预测器第一层
        self.classifier.append(nn.Linear(self.params['hid_dim'], self.params['hid_dim']))
        # 预测器第二层
        self.classifier.append(nn.Linear(self.params['hid_dim'], self.params['out_dim']))
        
    def forward(self, g, x):
        h = x
        # 经过输入转换层
        for layer in self.translayers:
            h = layer(h)
            h = self.activation(h)
            h = self.dropout(h)
        
        # 经过GNN层
        for i, (layer, bn) in enumerate(zip(self.gnnlayers, self.bns)):
            block = g[i] if type(g) == list else g
            h = layer(block, h)
            h = bn(h)
            h = self.activation(h)
            h = self.dropout(h)
        
        # store the node embeddings
        emb = h  # 存储节点嵌入

        # prediction
        # 经过预测器
        for i, layer in enumerate(self.classifier):
            h = layer(h)
            if i < len(self.classifier) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        
        return h, emb
    
    def inference(self, x, dataloader):
        '''
        inference part: full graph inference layer by layer.
        return all node embeddings.
        dataloader: a dgl dataloader with one-layer fullsampler.
        '''

        # 在推理阶段不计算梯度
        with torch.no_grad():
            h = x
            # new_h = []
            # 初始化一个新的张量来存储转换后的特征
            new_h = torch.empty(x.shape[0], self.params['hid_dim'], dtype=x.dtype, device=h.device)
            # transformation layers
            # 定义MLP的批处理大小
            mlp_bs = 10 * dataloader.dataset.batch_size
            # 转换层推理
            for i in range(np.ceil(len(h) / mlp_bs).astype(int)):
                batch_h = h[i*mlp_bs: (i+1)*mlp_bs]
                for layer in self.translayers:
                    batch_h = layer(batch_h)
                    batch_h = self.activation(batch_h)
                    batch_h = self.dropout(batch_h)
                # new_h.append(batch_h)
                new_h[i*mlp_bs: (i+1)*mlp_bs] = batch_h
            # h = torch.cat(new_h, dim=0)
            h = new_h

            # gnn layers
            # GNN层推理
            for _, (layer, bn) in enumerate(zip(self.gnnlayers, self.bns)):
                new_h = torch.empty_like(h, device=h.device)
                
                for input_nodes, output_nodes, blocks in dataloader:
                    batch_h = h[input_nodes].to(self.params['device'])
                    block = blocks[0].to(self.params['device'])
                    batch_h = layer(block, batch_h)
                    batch_h = bn(batch_h)
                    batch_h = self.activation(batch_h)
                    batch_h = self.dropout(batch_h)
                    new_h[output_nodes] = batch_h
                h = new_h
            
            # 存储节点嵌入
            emb = h
            # new_h = []
            # 初始化一个张量来存储最终的logits
            new_h = torch.empty(x.shape[0], self.params['out_dim'], dtype=x.dtype, device=h.device)
            # predictions
            for i in range(np.ceil(len(h) / mlp_bs).astype(int)):
                batch_h = h[i*mlp_bs: (i+1)*mlp_bs]
                for k, layer in enumerate(self.classifier):
                    batch_h = layer(batch_h)
                    if k < len(self.classifier) - 1:
                        batch_h = self.activation(batch_h)
                        batch_h = self.dropout(batch_h)
                # new_h.append(batch_h)
                new_h[i*mlp_bs: (i+1)*mlp_bs] = batch_h
            
            h = new_h
        return h, emb


# 定义RelSAGEConv类，这是一个关系感知的SAGEConv层
class RelSAGEConv(nn.Module):
    '''
    SAGEConv with Relation-aware Joint Aggregation;
    aggregation type: mean;
    '''
    def __init__(self, in_dim, edge_dim, out_dim):
        super().__init__()
        # 边特征映射到输出维度
        self.edge_map = nn.Linear(edge_dim, out_dim, bias=False)
        # 合并邻居节点特征和边特征的线性层
        self.fc_neigh_edge = nn.Linear(in_dim + out_dim, out_dim)  # a layer to merge edge type and neighbor node features

        # 自身节点特征的线性变换层
        self.fc_self = nn.Linear(in_dim, out_dim)
        # 激活函数
        self.activation = nn.ReLU()
    
    def msgfunc(self, edges):
        # 获取源节点的特征
        src_h = edges.src['h']
        # 获取边的类型特征
        efeat = edges.data['type']
        # 将边特征通过线性映射
        eh = self.edge_map(efeat)
        # ne_feat = torch.cat((src_h, efeat), dim=1)
        # 合并源节点特征和映射后的边特征
        ne_feat = torch.cat((src_h, eh), dim=1)
        # 返回经过激活函数和线性层处理后的消息
        return {'m': self.activation(self.fc_neigh_edge(ne_feat))}
    
    def forward(self, graph, nfeat):
        '''
        graph: a dgl graph;
        nfeat: node features;
        '''
        # 在图的局部作用域内执行操作
        with graph.local_scope():
            # 将节点特征存储为源节点数据
            graph.srcdata['h'] = nfeat
            # 更新所有节点，通过msgfunc发送消息，并使用mean聚合函数聚合消息到'neigh'
            graph.update_all(self.msgfunc, dgl.function.mean('m', 'neigh'))
            # 获取聚合后的邻居特征
            neigh = graph.dstdata['neigh']
            # 对目标节点自身的特征进行线性变换
            self_feat = self.fc_self(nfeat[:neigh.shape[0]])  # destination features transformation
            # 将邻居特征和自身特征相加
            out = neigh + self_feat
        return out


# 定义CRoCSAGE模型类，继承自SAGE
class CRoCSAGE(SAGE):
    '''
    GraphSAGE with CRoC;
    '''
    def __init__(self, params):
        super().__init__(params)

        # replace SAGEConv with RelSAGEConv
        # 替换SAGEConv为RelSAGEConv，使其具有关系感知能力
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.params['n_layer']):
            in_dim = self.params['in_dim'] if i == 0 else self.params['hid_dim']
            out_dim = self.params['hid_dim']
            self.layers.append(RelSAGEConv(in_dim, params['n_rel'], out_dim))
            bn = False if i == self.params['n_layer'] - 1 else self.params['bn']
            self.bns.append(nn.BatchNorm1d(out_dim) if bn else nn.Identity())


# 定义CRoCSAGE_Large模型类，继承自SAGE_Large
class CRoCSAGE_Large(SAGE_Large):
    '''
    SAGEPlus with edge relation type as input;
    '''
    def __init__(self, params):
        super().__init__(params)

        # replace SAGEConv with RelSAGEConv
        # 替换SAGEConv为RelSAGEConv，使其具有关系感知能力
        self.gnnlayers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.params['n_layer']):
            in_dim = self.params['hid_dim']
            out_dim = self.params['hid_dim']
            self.gnnlayers.append(RelSAGEConv(in_dim, params['n_rel'], out_dim))
            bn = False if i == self.params['n_layer'] - 1 else self.params['bn']
            self.bns.append(nn.BatchNorm1d(out_dim) if bn else nn.Identity())
        

# 定义GIN模型类，继承自SAGE
class GIN(SAGE):
    '''
    GIN with a linear layer as predictor;
    use max aggregator;
    '''
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(self.params['n_layer']):
            in_dim = self.params['in_dim'] if i == 0 else self.params['hid_dim']
            out_dim = self.params['hid_dim']
            # GINConv需要一个内部的线性层
            lin = nn.Linear(in_dim, out_dim)
            self.layers.append(GINConv(lin, 'max'))   
            bn = False if i == self.params['n_layer'] - 1 else self.params['bn']
            self.bns.append(nn.BatchNorm1d(out_dim) if bn else nn.Identity())


# 定义GIN_Large模型类，继承自SAGE_Large
class GIN_Large(SAGE_Large):
    '''
    GIN with a MLP as input transformer and a MLP as predictor;
    use max aggregator;
    '''
    def __init__(self, params):
        super().__init__(params)
        
        self.gnnlayers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = nn.Dropout(p=self.params['dropout'])
        self.activation = nn.ReLU()
        for i in range(self.params['n_layer']):
            in_dim = out_dim = self.params['hid_dim']
            # GINConv需要一个内部的线性层
            lin = nn.Linear(in_dim, out_dim)
            self.gnnlayers.append(GINConv(lin, 'max'))
            bn = False if i == self.params['n_layer'] - 1 else self.params['bn']
            self.bns.append(nn.BatchNorm1d(out_dim) if bn else nn.Identity())


# 定义RelGINConv类，这是一个关系感知的GINConv层
class RelGINConv(nn.Module):
    '''
    GINConv with Relation-aware Joint Aggregation;
    aggregation type: mean;
    '''
    def __init__(self, in_dim, edge_dim, out_dim, apply_func, aggregator_type='max', init_eps=0, learn_eps=False):
        super().__init__()
        # 边特征映射到输出维度
        self.edge_map = nn.Linear(edge_dim, out_dim, bias=False)
        # 合并邻居节点特征和边特征的线性层
        self.fc_neigh_edge = nn.Linear(in_dim + out_dim, out_dim)  # a layer to merge edge type and neighbor node features

        # 应用函数，通常是MLP
        self.apply_func = apply_func
        # 自身节点特征的线性变换层
        self.fc_self = nn.Linear(in_dim, out_dim)
        # 检查聚合器类型是否有效
        if aggregator_type not in ("sum", "max", "mean"):
            raise KeyError(
                "Aggregator type {} not recognized.".format(aggregator_type)
            )
        self._aggregator_type = aggregator_type
        # to specify whether eps is trainable or not.
        # 判断eps是否可训练
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))
        # 激活函数
        self.activation = nn.ReLU()
    
    def msgfunc(self, edges):
        # 获取源节点的特征
        src_h = edges.src['h']
        # 获取边的类型特征
        efeat = edges.data['type']
        # 将边特征通过线性映射
        eh = self.edge_map(efeat)
        # 合并源节点特征和映射后的边特征
        ne_feat = torch.cat((src_h, eh), dim=1)
        # 返回经过激活函数和线性层处理后的消息
        return {'m': self.activation(self.fc_neigh_edge(ne_feat))}
    
    def forward(self, graph, nfeat):
        '''
        graph: a dgl graph;
        nfeat: node features;
        '''
        # 获取聚合器函数
        _reducer = getattr(dgl.function, self._aggregator_type)
        # 在图的局部作用域内执行操作
        with graph.local_scope():
            # 将节点特征存储为源节点数据
            graph.srcdata['h'] = nfeat
            # 更新所有节点，通过msgfunc发送消息，并使用指定的聚合函数聚合消息到'neigh'
            graph.update_all(self.msgfunc, _reducer('m', 'neigh'))
            # 获取聚合后的邻居特征
            neigh = graph.dstdata['neigh']
            # 对目标节点自身的特征进行线性变换
            self_feat = self.fc_self(nfeat[:neigh.shape[0]])  # destination features transformation
            # GIN的聚合方式：(1 + epsilon) * 邻居特征 + 自身特征
            out = (1 + self.eps) * neigh + self_feat
            # 如果存在应用函数，则应用它
            if self.apply_func is not None:
                out = self.apply_func(out)
        return out


# 定义CRoCGIN模型类，继承自GIN
class CRoCGIN(GIN):
    '''
    GIN with CRoC;
    '''
    def __init__(self, params):
        super().__init__(params)

        # replace GINConv with RelGINConv
        # 替换GINConv为RelGINConv，使其具有关系感知能力
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.params['n_layer']):
            in_dim = self.params['in_dim'] if i == 0 else self.params['hid_dim']
            out_dim = self.params['hid_dim']
            # GINConv需要一个内部的线性层作为apply_func
            lin = nn.Linear(out_dim, out_dim)
            self.layers.append(RelGINConv(in_dim, params['n_rel'], out_dim, lin, 'max'))
            bn = False if i == self.params['n_layer'] - 1 else self.params['bn']
            self.bns.append(nn.BatchNorm1d(out_dim) if bn else nn.Identity())


# 定义CRoCGIN_Large模型类，继承自GIN_Large
class CRoCGIN_Large(GIN_Large):
    '''
    GIN_Large with edge relation type as input;
    '''
    def __init__(self, params):
        super().__init__(params)

        # replace GINConv with RelSAGEConv
        # 替换GINConv为RelSAGEConv，使其具有关系感知能力
        self.gnnlayers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.params['n_layer']):
            in_dim = self.params['hid_dim']
            out_dim = self.params['hid_dim']
            # GINConv需要一个内部的线性层作为apply_func
            lin = nn.Linear(out_dim, out_dim)
            self.gnnlayers.append(RelGINConv(in_dim, params['n_rel'], out_dim, lin, 'max'))
            bn = False if i == self.params['n_layer'] - 1 else self.params['bn']
            self.bns.append(nn.BatchNorm1d(out_dim) if bn else nn.Identity())
      