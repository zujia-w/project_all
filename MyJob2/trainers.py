import torch
import os
import numpy as np
import time
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
)
import utils
from utils import mixed_negative_sampling, compute_negative_logits
import dataloading
import copy


# 定义GNNTrainer类，用于训练和评估GNN模型
class GNNTrainer():
    '''
    Basic trainer for GNNs
    '''
    def __init__(self, g, model, params) -> None:
        self.g = g  # DGL图对象
        self.model = model  # GNN模型
        self.params = params  # 参数字典
        self.model.to(self.params['device'])  # 将模型移动到指定设备
        # 获取训练集、验证集和测试集的节点ID
        self.tr_nid = self.g.ndata['train_mask'].nonzero().squeeze()
        self.val_nid = self.g.ndata['val_mask'].nonzero().squeeze()
        self.tt_nid = self.g.ndata['test_mask'].nonzero().squeeze()
        self.ulb_mask = ~self.g.ndata['train_mask']  # 未标记节点的掩码
        self.labels = self.g.ndata['label']  # 节点标签

        self.pred_labels = torch.zeros_like(self.labels)  # 预测标签初始化
        self.pred_labels[self.tr_nid] = self.labels[self.tr_nid]  # 训练集标签已知
        # 未标记节点的标签随机初始化，用于CRoC的伪标签
        self.pred_labels[self.ulb_mask] = (torch.rand(len(self.pred_labels[self.ulb_mask])) > 0.5).long()
        self.embs = None  # 节点嵌入初始化

        # build datalaoders
        # 构建数据加载器
        self.build_loaders()

        # 添加这行，记录当前epoch（用于调试日志）
        self.current_epoch = 0

        self.loss_fn = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
        # self.loss_fn = torch.nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'], weight_decay=self.params['wd'])  # Adam优化器

        # 最佳指标初始化
        self.best_val_loss = float('inf')
        self.best_tt_loss = float('inf')
        self.best_val_auc = 0
        self.best_tt_auc = 0
        self.best_val_metrics = None
        self.best_tt_metrics = None
        self.best_epoch = 0

        # set model save path
        self.model_path = os.path.join('exp', self.params['split'], self.params['model'], self.params['dataset'], 'checkpoints', 'ckpt_{}.pth'.format(self.params['run']))

        # 打印异常节点的索引列表
        print('The index list of abnormal nodes:', torch.arange(self.g.num_nodes())[self.g.ndata['train_mask'] & (self.g.ndata['label'] == 1)].numpy().tolist())
    
    def build_loaders(self):
        # 创建训练数据集
        train_dataset = dataloading.DSDataset(self.g)
        # 创建训练数据加载器
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.params['bs'], shuffle=True, drop_last=False, num_workers=4)

        # 创建评估数据采样器，使用MultiLayerFullNeighborSampler进行全邻居采样
        sampler = MultiLayerFullNeighborSampler(1)
        # 创建评估数据加载器，用于全图推理
        self.eval_dataloader = DataLoader(self.g, torch.arange(self.g.num_nodes()), sampler, batch_size=10 * self.params['bs'], shuffle=False, drop_last=False, num_workers=4)
    
    def train(self):
        
        # 遍历训练轮次
        for e in range(1, self.params['n_epoch'] + 1):
            self.current_epoch = e  # 添加这行，记录当前epoch
            print('Epoch {:d}'.format(e))
            # 训练函数方法train_epoch()
            # 训练一个epoch
            ep_tr_metrics, ep_tr_tc = self.train_epoch()
            # 每隔一定轮次或达到一定轮次后进行评估
            if e % 1 == 0:
            # if e >= 50:
                # 验证函数方法eval_epoch()
                # 评估一个epoch
                ep_val_metrics, ep_val_tc = self.eval_epoch()
                # 如果当前验证集的AUC优于历史最佳，则更新最佳指标并保存模型
                if ep_val_metrics['auc'] > self.best_val_auc:
                    self.best_val_auc = ep_val_metrics['auc']
                    self.best_val_loss = ep_val_metrics['loss']
                    self.best_val_metrics = ep_val_metrics
                    self.best_epoch = e
                    self.save_model()

                
        
        # testing
        ep_tt_metrics, ep_tt_tc = self.eval_epoch(mode='test')
        self.best_tt_loss = ep_tt_metrics['loss']
        self.best_tt_metrics = ep_tt_metrics
        print('Testing results:\n', ep_tt_metrics)
        

        # 返回最佳验证集和测试集指标
        return self.best_val_metrics, self.best_tt_metrics
    
    def sample_ulb(self, n):
        '''
        sample a set of unlabeled nodes
        '''
        # 获取所有未标记节点的ID
        ulb_nids = torch.arange(self.g.num_nodes())[self.ulb_mask]
        # 随机打乱索引
        premute_idx = torch.randperm(len(ulb_nids))
        # 返回前n个未标记节点的ID
        return ulb_nids[premute_idx[:n]]

    def sample_nids(self, n):
        '''
        random sample a set of nodes for feature shuffling
        '''
        # 从所有节点中随机采样n个节点，用于特征混洗（feature shuffling）
        # torch.randperm(self.g.num_nodes()) 生成一个从0到图节点总数-1的随机排列
        # [:n] 选取前n个节点ID
        return torch.randperm(self.g.num_nodes())[:n]
  
    def train_epoch(self):
        # 记录当前epoch的开始时间
        tic = time.time()
        # 将模型设置为训练模式
        self.model.train()
        # 初始化列表，用于存储每个batch的损失和预测结果
        ep_loss = []  # 存储总损失
        ep_raw_ce = []  # 存储原始交叉熵损失
        ep_shuf_ce = []  # 存储混洗特征的交叉熵损失
        ep_ctr = []  # 存储对比损失
        ep_preds = []  # 存储预测分数
        ep_labels = []  # 存储真实标签
        
        # 对训练数据集进行重采样，这对于DSDataset（下采样数据集）是必要的，以确保每个epoch使用不同的负样本
        self.train_dataloader.dataset.resample()  # resample nodes for training
        # 遍历训练数据加载器中的每个batch
        for i, (batch_neg_nids, batch_neg_y, batch_pos_nids, batch_pos_y) in enumerate(self.train_dataloader):
            
            batch_neg_nids = batch_neg_nids.squeeze()  # 负样本节点ID
            batch_neg_y = batch_neg_y.squeeze()  # 负样本标签
            batch_pos_nids = batch_pos_nids.squeeze()  # 正样本节点ID
            batch_pos_y = batch_pos_y.squeeze()  # 正样本标签
            # 将正负样本节点ID和标签拼接起来，形成当前批次的有标签数据
            batch_lb_nids = torch.cat((batch_neg_nids, batch_pos_nids))
            batch_lb_y = torch.cat((batch_neg_y, batch_pos_y)).to(self.params['device'])
            # 从未标记节点中采样q个节点，用于对比学习
            batch_ulb_nids = self.sample_ulb(self.params['q'])
            
            # 将有标签节点和采样的未标记节点合并，形成当前批次需要处理的所有节点
            batch_nids = torch.cat((batch_neg_nids, batch_pos_nids, batch_ulb_nids))
            # generate blocks
            # 根据当前批次的节点生成多层图块（blocks），用于GNN的邻居采样
            input_nids, _, blocks = utils.sample_blocks(self.g, batch_nids, self.params['n_neb'])
            # 将生成的图块移动到指定设备
            blocks = [b.to(self.params['device']) for b in blocks]
            
            # raw node features
            # 获取原始节点特征
            batch_raw_feat = self.g.ndata['feat'][input_nids]
            # feature shuffling
            # 特征混洗（feature shuffling）
            alpha = self.params['alpha']  # 混洗比例参数
            # 通过线性插值将原始特征与随机采样的节点特征进行混合
            batch_shuf_feat = alpha * batch_raw_feat + (1 - alpha) * self.g.ndata['feat'][self.sample_nids(len(batch_raw_feat))] # mix shuffle
            
            # 将原始特征和混洗特征移动到指定设备
            batch_raw_x = batch_raw_feat.to(self.params['device'])
            batch_shuf_x = batch_shuf_feat.to(self.params['device'])
            # 使用模型对原始特征和混洗特征进行前向传播，得到预测分数和节点嵌入
            batch_raw_scores, batch_raw_embs = self.model(blocks, batch_raw_x)
            batch_shuf_scores, batch_shuf_embs = self.model(blocks, batch_shuf_x)

            # cross-entropy loss
            # 计算原始特征的交叉熵损失，只针对有标签数据
            batch_raw_ce = self.loss_fn(batch_raw_scores[:len(batch_lb_y)], batch_lb_y)
            # 计算混洗特征的交叉熵损失，只针对有标签数据
            batch_shuf_ce = self.loss_fn(batch_shuf_scores[:len(batch_lb_y)], batch_lb_y)
            

            # contrastive loss: apply contrastive learning within a set of nodes.
            # 对比损失：在一组节点内应用对比学习。
            # positive pairs logits for constrastive learning
            # 用于对比学习的正样本对的logits
            # 提取未标记节点（用于对比学习）的原始特征嵌入和混洗特征嵌入
            batch_pos_raw_embs = batch_raw_embs[len(batch_lb_y):]
            batch_pos_shuf_embs = batch_shuf_embs[len(batch_lb_y):]
            n_ctr_pos, ndim = batch_pos_raw_embs.shape  # 获取对比学习正样本的数量和嵌入维度
            # 计算正样本对的logits，通过原始嵌入和混洗嵌入的点积
            batch_pos_ctr_logits = torch.bmm(batch_pos_raw_embs.view(n_ctr_pos, 1, ndim), batch_pos_shuf_embs.view(n_ctr_pos, ndim, 1))

            # negative logits for contrastive learning
            # 负样本部分 - 修复后的代码
            if self.params.get('hard_ratio', 0.3) > 0:
                # 使用混合负采样
                neg_indices, sample_types = mixed_negative_sampling(
                    batch_pos_raw_embs,
                    batch_pos_raw_embs,  # 候选池
                    self.params['k'],
                    hard_ratio=self.params.get('hard_ratio', 0.3),
                    temperature=self.params.get('hard_temp', 0.5)
                )
                
                # 确保没有采到自身
                anchor_indices = torch.arange(n_ctr_pos, device=self.params['device']).view(-1, 1)
                self_mask = (neg_indices == anchor_indices)
                if self_mask.any():
                    # 用随机采样替换采到自身的位置
                    random_fix = torch.randint(0, n_ctr_pos, (n_ctr_pos, 1), device=self.params['device'])
                    neg_indices = torch.where(self_mask, random_fix, neg_indices)
                
                # 计算负样本logits [n_ctr_pos, k]
                batch_neg_ctr_logits = compute_negative_logits(
                    batch_pos_raw_embs,
                    batch_pos_raw_embs,
                    neg_indices
                ).squeeze(-1)  # [n_ctr_pos, k]
                
                # 记录难例比例
                if n_ctr_pos > 0 and self.current_epoch % 10 == 0 and i == 0:
                    hard_count = sample_types.sum().item()
                    total_count = n_ctr_pos * self.params['k']
                    hard_ratio_actual = hard_count / total_count
                    print(f"  Actual hard ratio: {hard_ratio_actual:.3f}")
            else:
                # 随机采样
                batch_ctr_neg_idx = torch.cat([torch.randperm(n_ctr_pos) for _ in range(self.params['k'])], dim=0).view(self.params['k'], n_ctr_pos).transpose(0, 1).to(self.params['device'])
                batch_neg_ctr_embs = batch_pos_raw_embs[batch_ctr_neg_idx].transpose(1, 2)
                batch_neg_ctr_logits = torch.bmm(
                    batch_pos_raw_embs.view(n_ctr_pos, 1, ndim), 
                    batch_neg_ctr_embs
                ).squeeze(-1)  # [n_ctr_pos, k]

            # 正样本logits已经是 [n_ctr_pos, 1]
            batch_pos_ctr_logits = batch_pos_ctr_logits.squeeze(-1)  # [n_ctr_pos, 1]

            # 拼接正负样本logits [n_ctr_pos, 1 + k]
            batch_ctr_logits = torch.cat([batch_pos_ctr_logits, batch_neg_ctr_logits], dim=1)
            
            # 标签：正样本在第0个位置
            ctr_labels = torch.zeros(n_ctr_pos, dtype=torch.long, device=self.params['device'])
            
            # 计算InfoNCE损失
            temp = self.params.get('ctr_temp', 2)
            batch_ctr_loss = self.loss_fn(batch_ctr_logits / temp, ctr_labels)

            # backward
            # 清零优化器梯度
            self.optimizer.zero_grad()
            # 计算总损失，包括原始交叉熵损失、混洗交叉熵损失和对比损失，并乘以相应的权重
            batch_loss = batch_raw_ce + self.params['gamma'] * batch_shuf_ce + self.params['eta'] * batch_ctr_loss
            # 反向传播，计算梯度
            batch_loss.backward()
            # 更新模型参数
            self.optimizer.step()

            # 记录当前batch的各种损失和预测结果
            ep_raw_ce.append(batch_raw_ce.detach().item())
            ep_shuf_ce.append(batch_shuf_ce.detach().item())
            ep_ctr.append(batch_ctr_loss.detach().item())
            ep_loss.append(batch_loss.detach().item())
            ep_preds.append(batch_raw_scores[:len(batch_lb_y)].detach().cpu())
            ep_labels.append(batch_lb_y.detach().cpu())
        
        # 计算整个epoch的平均损失
        ep_raw_ce = np.mean(ep_raw_ce).item()
        ep_shuf_ce = np.mean(ep_shuf_ce).item()
        ep_ctr = np.mean(ep_ctr).item()
        ep_loss = np.mean(ep_loss).item()
        # 将所有batch的预测结果和标签拼接起来
        ep_preds = torch.cat(ep_preds, dim=0)
        ep_labels = torch.cat(ep_labels, dim=0)

        # compute metrics
        # 计算评估指标（如AUC, F1等）
        metrics = utils.cal_metrics(ep_preds, ep_labels)  
        metrics.update({'loss': ep_loss})  # 将总损失添加到指标字典中
        toc = time.time()
        tc = toc - tic  # time cost  计算时间成本

        # 打印训练日志
        print('Training loss: {:.4f}, time cost: {:.4f}s'.format(metrics['loss'], tc))
        print('Raw CE: {:.4f}, Shuf CE: {:.4f}, Ctr: {:.4f}'.format(ep_raw_ce, ep_shuf_ce, ep_ctr))
        return metrics, tc

    def eval_epoch(self, mode='val'):
        # 记录当前epoch的开始时间
        tic = time.time()
        
        # 根据模式（验证或测试）选择模型
        if mode == 'val':
            eval_model = self.model  # 验证模式下使用当前训练的模型
        else:
            # test
            # 测试模式下，复制当前模型并加载最佳模型参数
            eval_model = copy.deepcopy(self.model)
            eval_model.load_state_dict(torch.load(self.model_path))
        # 将评估模型移动到指定设备
        eval_model.to(self.params['device'])
        # 将模型设置为评估模式（关闭dropout等）
        eval_model.eval()
        # 根据图的节点数量选择推理方式：如果节点数量大，使用inference方法进行分批推理；否则直接对整个图进行推理
        if self.g.num_nodes() > 10000:
            pred_scores, embs = eval_model.inference(self.g.ndata['feat'].to(self.params['device']), self.eval_dataloader)
        else:
            pred_scores, embs = eval_model(self.g.to(self.params['device']), self.g.ndata['feat'].to(self.params['device']))
        # 根据模式选择评估节点ID（验证集或测试集）
        eval_nids = self.val_nid if mode == 'val' else self.tt_nid
        # 提取评估节点的预测分数
        eval_preds = pred_scores[eval_nids]
        # 提取评估节点的真实标签，并移动到指定设备
        eval_labels = self.labels[eval_nids].to(self.params['device'])
        # 计算评估损失
        eval_loss = self.loss_fn(eval_preds, eval_labels).cpu().item()
        
        # compute metrics
        # 计算评估指标
        metrics = utils.cal_metrics(eval_preds, eval_labels)
        metrics.update({'loss': eval_loss})  # 将损失添加到指标字典中
        self.embs = embs.detach().cpu()  # 存储节点嵌入

        toc = time.time()
        tc = toc - tic  # time cost  计算时间成本
  
        # log metrics 
        # 打印评估日志
        if mode == 'val':
            print('Validation loss: {:.4f}, time cost: {:.4f}s'.format(metrics['loss'], tc))
            print('Validation results:\n', metrics)
        else:
            print('Testing loss: {:.4f}, time cost: {:.4f}s'.format(metrics['loss'], tc))
            print('Testing results:\n', metrics)

        return metrics, tc
    
    def save_model(self):
        # 构建模型保存的文件夹路径
        save_folder = os.path.join('exp', self.params['split'], self.params['model'], self.params['dataset'], 'checkpoints')
        # 如果文件夹不存在，则创建
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # 构建检查点文件名
        ckpname = 'ckpt_{}.pth'.format(self.params['run'])
        # format: exp/{split}/{model}/{dataset}/checkpoints/{ts}_{seed}.pth
        # 构建完整的保存路径
        save_path = os.path.join(save_folder, ckpname)
        # 保存模型的参数字典
        torch.save(self.model.state_dict(), save_path)
