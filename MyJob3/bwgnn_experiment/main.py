import dgl
import torch
import torch.nn.functional as F
import numpy
import argparse
import time
from dataset import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from BWGNN import *
from sklearn.model_selection import train_test_split


def train(model, g, args, dataset_name=None):
    features = g.ndata['feature']
    labels = g.ndata['label']
    index = list(range(len(labels)))
    
    current_dataset = dataset_name if dataset_name is not None else args.dataset
    
    if current_dataset == 'amazon':
        index = list(range(3305, len(labels)))

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=args.train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()

    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_f1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0.

    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    
    for e in range(args.epoch):
        # ========== 训练阶段 ==========
        model.train()
        logits_train = model(features)
        loss = F.cross_entropy(logits_train[train_mask], labels[train_mask], 
                              weight=torch.tensor([1., weight]).to(features.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # ========== 验证阶段 ==========
        model.eval()
        with torch.no_grad():
            logits_eval = model(features)
            probs = logits_eval.softmax(1)
            
            # 验证集上找最佳阈值
            val_labels = labels[val_mask].cpu().numpy()
            val_probs = probs[val_mask][:, 1].cpu().numpy()
            
            best_val_f1 = 0
            best_thre = 0.5
            
            # 只搜索有正样本的阈值范围
            unique_probs = np.unique(val_probs)
            if len(unique_probs) > 1:
                min_prob = max(0.05, np.percentile(val_probs[val_labels==1], 5) if (val_labels==1).sum()>0 else 0.05)
                max_prob = min(0.95, np.percentile(val_probs[val_labels==1], 95) if (val_labels==1).sum()>0 else 0.95)
                thresholds = np.linspace(min_prob, max_prob, 19)
            else:
                thresholds = np.linspace(0.05, 0.95, 19)
            
            for thres in thresholds:
                preds = (val_probs > thres).astype(int)
                # 如果有预测样本才计算F1
                if preds.sum() > 0:
                    val_f1 = f1_score(val_labels, preds, average='macro')
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_thre = thres
            
            # 如果所有阈值都没有预测样本，使用默认阈值
            if best_val_f1 == 0:
                best_thre = 0.5
                preds = (val_probs > best_thre).astype(int)
                best_val_f1 = f1_score(val_labels, preds, average='macro')
            
            # 测试集上评估（用验证集找到的最佳阈值）
            test_labels = labels[test_mask].cpu().numpy()
            test_probs = probs[test_mask][:, 1].cpu().numpy()
            test_preds = (test_probs > best_thre).astype(int)
            
            # 使用zero_division=0避免警告
            trec = recall_score(test_labels, test_preds, zero_division=0)
            tpre = precision_score(test_labels, test_preds, zero_division=0)
            tmf1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
            tauc = roc_auc_score(test_labels, test_probs)

            if best_val_f1 > best_f1:
                best_f1 = best_val_f1
                final_trec = trec
                final_tpre = tpre
                final_tmf1 = tmf1
                final_tauc = tauc
        
        if (e+1) % 20 == 0:
            print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(
                e+1, loss.item(), best_val_f1, best_f1))

    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(
        final_trec*100, final_tpre*100, final_tmf1*100, final_tauc*100))
    return final_tmf1, final_tauc


# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="amazon",
                        help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim
    graph = Dataset(dataset_name, homo).graph
    in_feats = graph.ndata['feature'].shape[1]
    num_classes = 2

    if args.run == 1:
        if homo:
            model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
        else:
            model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
        train(model, graph, args)

    else:
        final_mf1s, final_aucs = [], []
        for tt in range(args.run):
            if homo:
                model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
            else:
                model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
            mf1, auc = train(model, graph, args)
            final_mf1s.append(mf1)
            final_aucs.append(auc)
        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        print('MF1-mean: {:.2f}, MF1-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(100 * np.mean(final_mf1s),
                                                                                            100 * np.std(final_mf1s),
                                                               100 * np.mean(final_aucs), 100 * np.std(final_aucs)))
