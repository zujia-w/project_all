import sys
import argparse
import numpy as np
from dataset import Dataset
from main import train
from bwgnn_with_mled_model import BWGNNWithMLED

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="yelp")
    parser.add_argument("--train_ratio", type=float, default=0.01)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--homo", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--run", type=int, default=10)
    parser.add_argument("--embedding_dir", type=str, 
                       default="/home/usr01/yk/mled/models/embeddings")
    
    args = parser.parse_args()
    
    print("="*60)
    print("BWGNN+MLED 原论文正确实现")
    print(f"Dataset: {args.dataset}")
    print("="*60)
    
    # 加载数据
    graph = Dataset(args.dataset, args.homo).graph
    in_feats = graph.ndata['feature'].shape[1]
    num_classes = 2
    
    print(f"原始特征维度: {in_feats}")
    print(f"论文设置: type_dim=8, rel_dim=16, 融合方式=加权求和")
    
    results = {'f1': [], 'auc': []}
    
    for run_idx in range(args.run):
        print(f"\n--- Run {run_idx + 1}/{args.run} ---")
        
        model = BWGNNWithMLED(
            in_feats=in_feats,
            h_feats=args.hid_dim,
            num_classes=num_classes,
            graph=graph,
            d=args.order,
            dataset_name=args.dataset,
            embedding_dir=args.embedding_dir
        )
        
        mf1, auc = train(model, graph, args, args.dataset)
        
        results['f1'].append(mf1)
        results['auc'].append(auc)
    
    print("\n" + "="*60)
    print(f"BWGNN+MLED 结果 ({args.run}次运行)")
    print("="*60)
    print(f"MF1:     {100*np.mean(results['f1']):.2f} ± {100*np.std(results['f1']):.2f}")
    print(f"AUCROC:  {100*np.mean(results['auc']):.2f} ± {100*np.std(results['auc']):.2f}")
    print("="*60)

if __name__ == '__main__':
    main()