import argparse
import os
import sys
# 导入自定义的模型定义模块
import models
import torch
# 导入自定义的工具函数模块
import utils
# 导入自定义的数据预处理模块，并重命名为 preprocess
import preprocess as preprocess
import datetime
# 导入自定义的训练器模块
import trainers
# 导入 NumPy 库，用于数值计算
import numpy as np

# 定义主函数，接收命令行参数 args
def main(args):
    
    # 获取当前时间戳，格式为 YYYYMMDDHHMM，用于文件命名或记录
    cur_ts = datetime.datetime.now().strftime('%Y%m%d%H%M')

    # go through multiple runs
    # 用于存储每次运行的验证结果
    val_results = []
    # 用于存储每次运行的测试结果
    tt_results = []
    # 根据命令行参数 n_run 进行多次实验运行，以获取更稳定的结果
    for i in range(args.n_run):
        
        # 加载数据集并进行预处理
        # preprocess.load_dataset 返回图对象 g 和包含数据集信息的参数字典 params
        g, params = preprocess.load_dataset(args.dataset, args.split, seed=i,)
        # update params
        params = utils.update_info_dict(args, params)
        # 将当前时间戳添加到参数中
        params['ts']= cur_ts
        # 处理标签参数，将其从逗号分隔的字符串转换为列表
        params['tags'] = params['tags'].strip().split(',')
        # 处理邻居采样数量参数 n_neb，将其从字符串转换为整数列表
        # 如果 n_neb 为空，则默认为每层采样 -1（表示全连接）
        params['n_neb'] = list(map(int, params['n_neb'].strip().split(','))) if params['n_neb'] else [-1] * params['n_layer']
        # 设置当前运行的编号
        params['run'] = i

        # 初始化模型
        # 根据数据集类型选择不同的模型构造函数
        if args.dataset in ['yelp', 'amazon']:
            # 对于 yelp 和 amazon 数据集，直接使用 args.model 指定的模型
            model = getattr(models, args.model)(params)
        else:
            # 对于其他数据集，使用 args.model 加上 '_Large' 后缀的模型
            model = getattr(models, args.model + '_Large')(params)
        # 打印模型结构
        print(model)
        # 打印模型参数的总数量
        print('Number of parameters: {}'.format(sum(p.numel() for p in model.parameters())))
        # 将模型移动到指定的设备（CPU 或 GPU）
        model = model.to(params['device'])

        # initialize a trainer
        trainername = 'GNNTrainer'
        # 使用 getattr 动态获取 trainers 模块中的 GNNTrainer 类，并实例化
        trainer = getattr(trainers, trainername)(g, model, params=params)
        
        # 打印最终的参数配置
        print(params)
        # 调用训练器的 train 方法开始训练和评估，并获取最佳验证和测试指标
        best_val_metrics, best_tt_metrics = trainer.train()
        # 将当前运行的最佳验证结果添加到列表中
        val_results.append(best_val_metrics)
        # 将当前运行的最佳测试结果添加到列表中
        tt_results.append(best_tt_metrics)
        # 记录当前运行的验证和测试结果到文件或日志中
        utils.record_results(best_val_metrics, best_tt_metrics, params)
    
    # print averaged result
    print('Average validation results:')
    # 计算所有运行的验证结果的平均值
    val_avg = {'val {}'.format(k): np.mean([r[k] for r in val_results]) for k in val_results[0].keys()}
    # 打印平均验证结果
    [print('{}: {:.4f}'.format(k, v)) for k, v in val_avg.items()]
    print('Average testing results:')
    # 计算所有运行的测试结果的平均值
    tt_avg = {'tt {}'.format(k): np.mean([r[k] for r in tt_results]) for k in tt_results[0].keys()}
    # 打印平均测试结果
    [print('{}: {:.4f}'.format(k, v)) for k, v in tt_avg.items()]

        

if __name__ == '__main__':
    # 创建 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser(description='main script for GAD')
    # 添加数据集名称参数，默认值为 'yelp'
    parser.add_argument('--dataset', type=str, default='yelp', help='dataset name')
    # 添加运行次数参数，默认值为 10
    parser.add_argument('--n_run', type=int, default=1)
    # 添加模型名称参数，默认值为 'CRoCGIN'
    parser.add_argument('--model', type=str, default='CRoCGIN', help='model name')
    # 添加 GNN 层数参数，默认值为 2
    parser.add_argument('--n_layer', type=int, default=2)
    # 添加邻居采样数量参数，默认值为 '10,5'
    parser.add_argument('--n_neb', type=str, default='10,5')
    # 添加隐藏层维度参数，默认值为 64
    parser.add_argument('--hid_dim', type=int, default=64)
    # 添加是否使用 BatchNorm 的参数，默认不使用
    parser.add_argument('--bn', action='store_true', default=False)
    # 添加 Dropout 比例参数，默认值为 0
    parser.add_argument('--dropout', type=float, default=0)
    # 添加训练 epoch 数量参数，默认值为 300
    parser.add_argument('--n_epoch', type=int, default=300)
    # 添加学习率参数，默认值为 0.003
    parser.add_argument('--lr', type=float, default=0.003)
    # 添加权重衰减（L2 正则化）参数，默认值为 0
    parser.add_argument('--wd', type=float, default=0)
    # 添加批次大小参数，默认值为 1024
    parser.add_argument('--bs', type=int, default=1024)
    # 添加 CRoC 中特征混洗的混合比例参数 alpha，默认值为 0
    parser.add_argument('--alpha', type=float, default=0)
    # 添加 CRoC 中混洗分类损失的权重参数 gamma，默认值为 1
    parser.add_argument('--gamma', type=float, default=1)
    # 添加 CRoC 中对比损失的权重参数 eta，默认值为 0.5
    parser.add_argument('--eta', type=float, default=0.5)
    # 添加用于对比学习的未标记节点数量参数 q，默认值为 10240
    parser.add_argument('--q', type=int, default=10240, help='Number of unlabeled nodes used for each batch of training.')
    # 添加对比学习中负样本对的数量参数 k，默认值为 10
    parser.add_argument('--k', type=int, default=10, help='Number of negative pairs for contrastive learning.')
    # 添加 GPU ID 参数，默认值为 0
    parser.add_argument('--gpu', type=int, default=0)
    # 添加数据集划分比例参数，默认值为 '0.01,0.33_0.67'  这里可以看到训练集的比例非常的低
    parser.add_argument('--split', type=str, default='0.01,0.33_0.67')
    # 添加用于 wandb 运行的标签参数，多个标签用逗号分隔，默认值为 'debug'
    parser.add_argument('--tags', type=str, default='debug', help='tags for a wandb run; separated by comma')
    # 添加难例负采样相关参数
    parser.add_argument('--hard_ratio', type=float, default=0.3,
                        help='ratio of hard negatives in contrastive learning (0 for random only)')
    parser.add_argument('--hard_temp', type=float, default=0.5,
                        help='temperature for hard negative mining (lower = harder negatives)')
    parser.add_argument('--ctr_temp', type=float, default=2,
                        help='temperature for InfoNCE loss')
    # 解析命令行参数
    args = parser.parse_args()
    # 调用主函数 main，传入解析后的参数
    main(args)