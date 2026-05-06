
# 导入必要的库
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, precision_score, precision_recall_curve
import torch
from collections import OrderedDict
import os
import dgl


def find_best_f1(probs, labels):
    """
    从一系列阈值中找到最佳的F1分数及其对应的阈值。
    该函数通常用于二分类问题中，当模型输出概率时，需要将概率转换为具体的类别预测。
    通过遍历不同的阈值，计算每个阈值下的F1分数，并选择F1分数最高的阈值。
    
    Args:
        probs (np.ndarray): 模型的预测概率，通常是正类的概率。
        labels (np.ndarray): 真实的标签。
        
    Returns:
        tuple: 包含最佳F1分数和最佳阈值的元组。
    """
    '''
    copy from ConsisGAD
    '''
    best_f1, best_thre = -1., -1.
    thres_arr = np.linspace(0.05, 0.95, 19)
    for thres in thres_arr:
        preds = np.zeros_like(labels)
        preds[probs > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


def cal_metrics(logits, labels):
    """
    计算并返回模型性能的各项指标，包括准确率、AUC、F1分数、召回率、精确率和平均精确率。
    该函数首先将模型的原始输出（logits）通过softmax转换为概率，然后使用find_best_f1函数确定最佳阈值，
    最后根据最佳阈值进行二值化预测，并计算各种评估指标。

    Args:
        logits (torch.Tensor): 模型的原始输出，形状为 (n_nodes, n_classes)。
        labels (torch.Tensor): 真实的标签，形状为 (n_nodes,)。

    Returns:
        collections.OrderedDict: 包含各项评估指标的有序字典。
                                 键包括 'acc' (准确率), 'auc' (ROC AUC),
                                 'micro-f1' (微平均F1), 'macro-f1' (宏平均F1),
                                 'recall' (召回率), 'prec' (精确率), 'ap' (平均精确率)。
    """
    '''
    calculate metrics
    logits: a tensor of shape (n_nodes, n_classes)
    labels: a tensor of shape (n_nodes,)
    threshold: a float, use it to binarize the logits
    '''
    logits = logits.softmax(dim=-1)

    logits = logits.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    macf1, best_thre = find_best_f1(logits[:, 1], labels)
    pred = (logits[:, 1] > best_thre).astype(int)
    acc = accuracy_score(labels, pred)
    auc = roc_auc_score(labels, logits[:, 1])
    recall = recall_score(labels, pred)
    prec = precision_score(labels, pred)
    micf1 = 2 * (prec * recall) / (prec + recall + 1e-20)
    ap = average_precision_score(labels, logits[:, 1])
    
    return OrderedDict({'acc': acc, 'auc': auc, 'micro-f1': micf1, 'macro-f1': macf1, 'recall': recall, 'prec': prec, 'ap': ap})


def update_info_dict(args, info_dict):
    """
    更新信息字典，将命令行参数和设备信息添加到其中。
    这个函数通常用于在实验开始前，将所有重要的配置参数和运行环境信息整合到一个字典中，
    方便后续的记录和管理。

    Args:
        args (argparse.Namespace): 包含命令行参数的对象。
        info_dict (dict): 待更新的信息字典。

    Returns:
        dict: 更新后的信息字典，包含了命令行参数和设备信息。
    """
    info_dict.update(args.__dict__)
    info_dict.update({'device': torch.device('cpu') if args.gpu == -1 else torch.device('cuda:{}'.format(args.gpu)),})
    
    return info_dict


def set_random_seed(seed):
    """
    设置所有必要的随机种子，以确保实验的可复现性。
    这包括NumPy、PyTorch的CPU和CUDA随机种子，以及PyTorch的CUDNN确定性设置。

    Args:
        seed (int): 用于设置随机数的种子值。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-gpu
    torch.backends.cudnn.deterministic = True


def record_results(best_val, best_tt, params):
    """
    将实验结果记录到CSV文件中。
    该函数会创建一个以实验参数命名的文件夹结构，并在其中保存一个results.csv文件。
    如果文件不存在，会先写入标题行；否则，直接追加结果数据。
    这有助于组织和管理多次实验的结果。

    Args:
        best_val (dict): 验证集上的最佳性能指标字典。
        best_tt (dict): 测试集上的最佳性能指标字典。
        params (dict): 实验参数字典。
    """
    '''
    record results: add a new row to the results.csv file for each run
    '''
    save_folder = os.path.join('exp', params['split'], params['model'], params['dataset'], 'results')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    fdir = os.path.join(save_folder, 'results.csv')
    if not os.path.exists(fdir):
        need_title = True
    else:
        need_title = False
    
    with open(fdir, 'a', newline='') as f:
        if need_title:
            f.write(','.join(list(best_tt.keys()) + list(best_val.keys()) + list(params.keys())) + '\n')
        f.write(','.join(['{:.4f}'.format(best_tt[k]) for k in best_tt.keys()] + ['{:.4f}'.format(best_val[k]) for k in best_val.keys()] + [str(params[k]) for k in params.keys()]) + '\n')


def sample_blocks(g, seed_nodes, fanouts):
    """
    从图中采样邻居节点，并返回DGL的块（blocks）。
    这个函数实现了多层邻居采样，用于构建GNN的计算图。
    它从给定的种子节点开始，根据fanouts参数逐层采样邻居，生成一系列的图块。

    Args:
        g (dgl.DGLGraph): 原始的DGL图。
        seed_nodes (torch.Tensor): 用于采样的起始种子节点ID列表。
        fanouts (list): 一个整数列表，表示每一层采样的邻居数量。
                        例如，[10, 5] 表示第一层采样10个邻居，第二层采样5个邻居。

    Returns:
        tuple: 包含以下元素的元组：
               - input_nodes (torch.Tensor): 最终采样得到的输入节点ID列表。
               - output_nodes (torch.Tensor): 原始的种子节点ID列表。
               - blocks (list): 包含DGLGraph对象的列表，每个对象代表一层采样得到的图块。
    """
    '''
    sample neighbors from graph, return dgl blocks
    g: a dgl graph
    seed_nodes: a list of nodes to start the sampling
    fanouts: a list of fanouts for each layer
    '''
    blocks = []
    output_nodes = seed_nodes
    for fanout in reversed(fanouts):
        # extract the marginal graph
        sg = g.sample_neighbors(seed_nodes, fanout)
        # sg = dgl.remove_self_loop(sg)
        # sg = dgl.add_self_loop(sg)  # make sure that each node have a self-loop link (for GCN and GAT)
        sgb = dgl.to_block(sg, seed_nodes)
        seed_nodes = sgb.srcdata[dgl.NID]
        blocks.insert(0, sgb)
        input_nodes = seed_nodes
    return input_nodes, output_nodes, blocks

# =====================================================

def cosine_similarity_matrix(x, y=None):
    """
    计算两个矩阵之间的余弦相似度
    Args:
        x: [n, d] 矩阵
        y: [m, d] 矩阵，如果为None则计算x与自身的相似度
    Returns:
        sim: [n, m] 相似度矩阵
    """
    if y is None:
        y = x
    
    x_norm = torch.nn.functional.normalize(x, dim=1)
    y_norm = torch.nn.functional.normalize(y, dim=1)
    
    return torch.mm(x_norm, y_norm.t())


def hard_negative_mining(anchor_embs, candidate_embs, k, exclude_self=True, temperature=0.5):
    """
    难例负样本挖掘
    Args:
        anchor_embs: [n, d] 锚点嵌入
        candidate_embs: [m, d] 候选负样本嵌入
        k: 每个锚点需要的难例数量
        exclude_self: 是否排除自身（当候选池包含锚点时）
        temperature: 温度参数，用于调整相似度的softmax分布
    Returns:
        hard_indices: [n, k] 难例负样本的索引
        hard_similarities: [n, k] 对应的相似度分数
    """
    # 计算相似度
    sim_matrix = cosine_similarity_matrix(anchor_embs, candidate_embs)
    
    # 如果需要排除自身（当候选池包含锚点时）
    if exclude_self and anchor_embs.shape[0] == candidate_embs.shape[0]:
        # 将对角线设为负无穷，避免选择自身
        sim_matrix = sim_matrix - torch.eye(sim_matrix.shape[0]).to(sim_matrix.device) * 1e10
    
    # 应用温度
    sim_matrix = sim_matrix / temperature
    
    # 选择相似度最高的k个作为难例
    hard_scores, hard_indices = sim_matrix.topk(k, dim=1)
    
    return hard_indices, hard_scores


def mixed_negative_sampling(anchor_embs, candidate_embs, k, hard_ratio=0.3, temperature=0.5):
    """
    混合负采样：随机采样 + 难例采样（确保不采到自身）
    Args:
        anchor_embs: [n, d] 锚点嵌入
        candidate_embs: [m, d] 候选负样本嵌入
        k: 总负样本数量
        hard_ratio: 难例比例
        temperature: 难例采样的温度
    Returns:
        mixed_indices: [n, k] 混合后的负样本索引
        sample_types: [n, k] 每个样本的类型（0:随机, 1:难例）
    """
    n = len(anchor_embs)
    m = len(candidate_embs)
    k_hard = int(k * hard_ratio)
    k_random = k - k_hard
    
    device = anchor_embs.device
    
    # 初始化返回矩阵
    mixed_indices = torch.zeros(n, k, dtype=torch.long, device=device)
    sample_types = torch.zeros(n, k, dtype=torch.long, device=device)
    
    # 创建自身索引（用于排除）
    self_indices = torch.arange(n, device=device).view(-1, 1)
    
    # 1. 随机采样（确保不采到自身）
    if k_random > 0:
        random_indices = torch.randint(0, m, (n, k_random), device=device)
        
        # 如果候选池就是锚点池，需要排除自身
        if m == n:
            # 循环直到没有采到自身
            self_mask = (random_indices == self_indices)
            max_iter = 10
            iter_count = 0
            while self_mask.any() and iter_count < max_iter:
                # 重新采样被污染的位置
                fix_indices = torch.randint(0, m, (n, k_random), device=device)
                random_indices = torch.where(self_mask, fix_indices, random_indices)
                self_mask = (random_indices == self_indices)
                iter_count += 1
            
            # 如果还有采到自身的，用次优值替换
            if self_mask.any():
                # 用 (index + 1) % n 替换
                fallback = (self_indices + 1) % n
                random_indices = torch.where(self_mask, fallback, random_indices)
        
        mixed_indices[:, :k_random] = random_indices
        sample_types[:, :k_random] = 0
    
    # 2. 难例采样（hard_negative_mining已经处理了exclude_self）
    if k_hard > 0:
        hard_indices, hard_scores = hard_negative_mining(
            anchor_embs, candidate_embs, k_hard,
            exclude_self=True, temperature=temperature
        )
        mixed_indices[:, k_random:] = hard_indices
        sample_types[:, k_random:] = 1
    
    return mixed_indices, sample_types


def compute_negative_logits(anchor_embs, neg_embs, indices):
    """
    高效计算负样本logits
    Args:
        anchor_embs: [n, d] 锚点嵌入
        neg_embs: [m, d] 负样本候选嵌入
        indices: [n, k] 负样本索引
    Returns:
        neg_logits: [n, k, 1] 负样本logits
    """
    n, k = indices.shape
    d = anchor_embs.shape[1]
    
    # 收集负样本嵌入 [n, k, d]
    selected_neg = neg_embs[indices.view(-1)].view(n, k, d)
    
    # 计算点积 [n, 1, k]
    neg_logits = torch.bmm(
        anchor_embs.view(n, 1, d),
        selected_neg.transpose(1, 2)
    )
    
    # 转置为 [n, k, 1]
    neg_logits = neg_logits.transpose(1, 2)
    
    return neg_logits
