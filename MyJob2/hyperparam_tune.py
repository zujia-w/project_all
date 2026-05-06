import subprocess
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import itertools
import os
import time

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# =====================================================
# 根据您的main.py地址设置保存路径
# main.py 地址: /home/usr01/yk/scrc0/main.py
# 项目根目录: /home/usr01/yk/scrc0/
# =====================================================
PROJECT_ROOT = '/home/usr01/yk/scrc0'  # 项目根目录
SAVE_DIR = os.path.join(PROJECT_ROOT, '调优结果')  # 在项目根目录下创建"调优结果"文件夹

# 创建保存图片的文件夹
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"创建文件夹: {SAVE_DIR}")

def run_experiment(params, n_epoch=50, n_run=1, dataset='yelp'):
    """
    运行单次实验
    params: 参数字典，如 {'hard_ratio': 0.5, 'hard_temp': 0.3}
    """
    # main.py的完整路径
    main_py_path = os.path.join(PROJECT_ROOT, 'main.py')
    
    # 构建基础命令
    cmd = [
        "python", main_py_path,  # 使用完整的main.py路径
        "--n_run", str(n_run),
        "--n_epoch", str(n_epoch),
        "--dataset", dataset,
        "--model", "CRoCGIN",
        "--gpu", "0",
    ]
    
    # 添加所有参数
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"\n运行命令: {' '.join(cmd)}")
    print("-" * 50)
    
    # 运行命令并捕获输出
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=PROJECT_ROOT)  # 在项目根目录运行
        end_time = time.time()
        
        # 保存完整输出到日志文件（便于调试）
        log_file = os.path.join(SAVE_DIR, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"命令: {' '.join(cmd)}\n\n")
            f.write(f"标准输出:\n{result.stdout}\n\n")
            f.write(f"错误输出:\n{result.stderr}\n")
        
        print(f"运行完成，耗时: {end_time - start_time:.1f}秒")
        print(f"日志保存到: {log_file}")
        
        # 提取AUC
        auc = extract_auc(result.stdout)
        
        if auc is None:
            # 尝试从stderr提取
            auc = extract_auc(result.stderr)
        
        return auc
        
    except subprocess.TimeoutExpired:
        print(f"实验超时（>1小时）")
        return None
    except Exception as e:
        print(f"运行出错: {e}")
        return None

def extract_auc(output):
    """从输出中提取AUC - 支持多种格式"""
    if output is None:
        return None
    
    # 尝试不同的模式
    patterns = [
        r"tt auc: (\d+\.\d+)",           # "tt auc: 0.8263"
        r"auc[:\s]+(\d+\.\d+)",          # "auc: 0.8263"
        r"Average testing results:.*?tt auc: (\d+\.\d+)",  # 多行匹配
        r"testing results.*?auc[:\s]*(\d+\.\d+)",  # 不区分大小写
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
        if match:
            auc = float(match.group(1))
            print(f"提取到AUC: {auc:.4f}")
            return auc
    
    # 如果还是没找到，打印部分输出以便调试
    print("无法提取AUC，输出预览:")
    print(output[-500:] if len(output) > 500 else output)
    return None

def tune_hard_ratio_and_temp():
    """调优 hard_ratio 和 hard_temp"""
    
    print("\n" + "="*60)
    print("【步骤1】调优 hard_ratio 和 hard_temp")
    print("="*60)
    
    # 参数范围
    hard_ratios = [0.3, 0.4, 0.5, 0.6]
    hard_temps = [0.2, 0.3, 0.4, 0.5]
    
    # 固定其他参数
    fixed_params = {
        'eta': 0.5,
        'k': 10,
        'ctr_temp': 2,
        'alpha': 0,
        'gamma': 1
    }
    
    results = []
    total = len(hard_ratios) * len(hard_temps)
    current = 0
    
    for hard_ratio in hard_ratios:
        for hard_temp in hard_temps:
            current += 1
            print(f"\n[{current}/{total}] 测试: hard_ratio={hard_ratio}, hard_temp={hard_temp}")
            
            params = {
                'hard_ratio': hard_ratio,
                'hard_temp': hard_temp,
                **fixed_params
            }
            
            # 每个组合跑2次取平均（增加可靠性）
            aucs = []
            for trial in range(2):
                print(f"  第{trial+1}次运行...")
                auc = run_experiment(params, n_epoch=30)  # 用30轮快速验证
                if auc is not None:
                    aucs.append(auc)
                time.sleep(2)  # 避免GPU过热
            
            if len(aucs) > 0:
                avg_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                results.append({
                    'hard_ratio': hard_ratio,
                    'hard_temp': hard_temp,
                    'auc': avg_auc,
                    'std': std_auc
                })
                print(f"  平均AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    
    # 保存结果
    csv_path = os.path.join(SAVE_DIR, 'hard_ratio_temp_调优结果.csv')
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到 {csv_path}")
    
    return df

def tune_eta():
    """调优 eta（对比损失权重）"""
    
    print("\n" + "="*60)
    print("【步骤2】调优 eta")
    print("="*60)
    
    # 先用找到的最佳 hard_ratio 和 hard_temp
    best_params = find_best_hard_params()
    
    if best_params is None:
        print("未找到之前的调优结果，使用默认值")
        best_params = {'hard_ratio': 0.5, 'hard_temp': 0.3}
    
    etas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []
    
    for eta in etas:
        print(f"\n测试: eta={eta}")
        
        params = {
            'hard_ratio': best_params['hard_ratio'],
            'hard_temp': best_params['hard_temp'],
            'eta': eta,
            'k': 10,
            'ctr_temp': 2,
            'alpha': 0,
            'gamma': 1
        }
        
        # 每个值跑2次取平均
        aucs = []
        for trial in range(2):
            print(f"  第{trial+1}次运行...")
            auc = run_experiment(params, n_epoch=30)
            if auc is not None:
                aucs.append(auc)
            time.sleep(2)
        
        if len(aucs) > 0:
            avg_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            results.append({
                'eta': eta,
                'auc': avg_auc,
                'std': std_auc
            })
            print(f"  平均AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    
    # 保存结果
    csv_path = os.path.join(SAVE_DIR, 'eta_调优结果.csv')
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到 {csv_path}")
    
    return df

def find_best_hard_params():
    """从之前的结果中找到最佳的 hard_ratio 和 hard_temp"""
    try:
        csv_path = os.path.join(SAVE_DIR, 'hard_ratio_temp_调优结果.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            best = df.loc[df['auc'].idxmax()]
            return {
                'hard_ratio': best['hard_ratio'],
                'hard_temp': best['hard_temp']
            }
        else:
            return None
    except Exception as e:
        print(f"读取结果文件出错: {e}")
        return None

def plot_heatmap(df, save_path='hard_ratio_temp_热力图.png'):
    """绘制热力图"""
    
    if df.empty:
        print("没有数据可绘制")
        return
    
    # 创建透视表
    pivot = df.pivot_table(
        values='auc',
        index='hard_ratio',
        columns='hard_temp',
        aggfunc='mean'
    )
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    
    # 使用更好的配色
    sns.heatmap(pivot, 
                annot=True, 
                fmt='.4f',
                cmap='RdYlGn',
                center=0.8,
                vmin=0.75, 
                vmax=0.85,
                cbar_kws={'label': 'AUC'})
    
    plt.title('难例比例和温度对AUC的影响', fontsize=14, fontweight='bold')
    plt.xlabel('hard_temp (难例采样温度)', fontsize=12)
    plt.ylabel('hard_ratio (难例比例)', fontsize=12)
    
    # 标记最佳点
    best_idx = df['auc'].idxmax()
    best_ratio = df.loc[best_idx, 'hard_ratio']
    best_temp = df.loc[best_idx, 'hard_temp']
    best_auc = df.loc[best_idx, 'auc']
    
    # 在热力图上标记
    temp_idx = list(pivot.columns).index(best_temp)
    ratio_idx = list(pivot.index).index(best_ratio)
    plt.plot(temp_idx + 0.5, ratio_idx + 0.5, 'r*', markersize=15, 
             label=f'最佳: AUC={best_auc:.4f}')
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图片
    full_save_path = os.path.join(SAVE_DIR, save_path)
    plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"\n热力图已保存到: {full_save_path}")
    print(f"最佳参数: hard_ratio={best_ratio}, hard_temp={best_temp}, AUC={best_auc:.4f}")

def plot_line_chart(df, x_col, y_col, save_path='eta_敏感度分析.png'):
    """绘制折线图"""
    
    if df.empty:
        print("没有数据可绘制")
        return
    
    plt.figure(figsize=(10, 6))
    
    # 按x排序
    df = df.sort_values(x_col)
    
    plt.errorbar(df[x_col], df[y_col], yerr=df['std'], 
                fmt='bo-', linewidth=2, markersize=8, capsize=5)
    
    # 标记最佳点
    best_idx = df[y_col].idxmax()
    best_x = df.loc[best_idx, x_col]
    best_y = df.loc[best_idx, y_col]
    
    plt.plot(best_x, best_y, 'r*', markersize=15, 
             label=f'最佳: {x_col}={best_x}, AUC={best_y:.4f}')
    
    # 设置中文标签
    xlabel_map = {
        'eta': '对比损失权重 η',
        'hard_ratio': '难例比例',
        'hard_temp': '难例温度'
    }
    
    plt.xlabel(xlabel_map.get(x_col, x_col), fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title(f'{xlabel_map.get(x_col, x_col)} 敏感度分析', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图片
    full_save_path = os.path.join(SAVE_DIR, save_path)
    plt.savefig(full_save_path, dpi=300)
    plt.show()
    plt.close()
    
    print(f"折线图已保存到: {full_save_path}")

def plot_parameter_importance():
    """绘制参数重要性图"""
    
    # 读取所有结果
    df1_path = os.path.join(SAVE_DIR, 'hard_ratio_temp_调优结果.csv')
    df2_path = os.path.join(SAVE_DIR, 'eta_调优结果.csv')
    
    df1_found = os.path.exists(df1_path)
    df2_found = os.path.exists(df2_path)
    
    if not df1_found or not df2_found:
        print("缺少结果文件，无法绘制参数重要性图")
        return
    
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    
    # 计算每个参数与AUC的相关性
    correlations = []
    
    # hard_ratio的相关性
    corr = df1['hard_ratio'].corr(df1['auc'])
    correlations.append({'parameter': '难例比例', 'correlation': abs(corr) if not pd.isna(corr) else 0})
    
    # hard_temp的相关性
    corr = df1['hard_temp'].corr(df1['auc'])
    correlations.append({'parameter': '难例温度', 'correlation': abs(corr) if not pd.isna(corr) else 0})
    
    # eta的相关性
    corr = df2['eta'].corr(df2['auc'])
    correlations.append({'parameter': '对比损失权重', 'correlation': abs(corr) if not pd.isna(corr) else 0})
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('correlation', ascending=True)
    
    # 绘制
    plt.figure(figsize=(10, 6))
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    bars = plt.barh(corr_df['parameter'], corr_df['correlation'], 
                    color=colors, alpha=0.8)
    
    # 在条形上添加数值
    for i, (bar, val) in enumerate(zip(bars, corr_df['correlation'])):
        plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center')
    
    plt.xlabel('与AUC的绝对相关性', fontsize=12)
    plt.title('参数重要性分析', fontsize=14, fontweight='bold')
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # 保存图片
    full_save_path = os.path.join(SAVE_DIR, '参数重要性分析.png')
    plt.savefig(full_save_path, dpi=300)
    plt.show()
    plt.close()
    
    print(f"参数重要性图已保存到: {full_save_path}")

def main():
    """主函数"""
    
    print("="*60)
    print("CRoC 超参数自动调优")
    print("="*60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"main.py路径: {os.path.join(PROJECT_ROOT, 'main.py')}")
    print(f"结果保存目录: {SAVE_DIR}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 步骤1: 调优 hard_ratio 和 hard_temp
    df_ht = tune_hard_ratio_and_temp()
    if not df_ht.empty:
        plot_heatmap(df_ht, 'hard_ratio_temp_热力图.png')
    
    # 步骤2: 调优 eta
    df_eta = tune_eta()
    if not df_eta.empty:
        plot_line_chart(df_eta, 'eta', 'auc', 'eta_敏感度分析.png')
    
    # 步骤3: 参数重要性分析
    plot_parameter_importance()
    
    # 步骤4: 输出最终建议
    print("\n" + "="*60)
    print("调优完成！最终建议：")
    
    best_ht = find_best_hard_params()
    
    eta_csv_path = os.path.join(SAVE_DIR, 'eta_调优结果.csv')
    if os.path.exists(eta_csv_path):
        best_eta_df = pd.read_csv(eta_csv_path)
        best_eta = best_eta_df.loc[best_eta_df['auc'].idxmax()]
        
        print(f"\n最佳参数组合:")
        print(f"  hard_ratio = {best_ht['hard_ratio'] if best_ht else 0.5}")
        print(f"  hard_temp  = {best_ht['hard_temp'] if best_ht else 0.3}")
        print(f"  eta        = {best_eta['eta']}")
        print(f"  预期AUC    = {best_eta['auc']:.4f}")
        
        print("\n运行完整实验的命令:")
        print(f"cd {PROJECT_ROOT}")
        print(f"python main.py --n_run 10 --n_epoch 300 \\")
        print(f"  --hard_ratio {best_ht['hard_ratio'] if best_ht else 0.5} \\")
        print(f"  --hard_temp {best_ht['hard_temp'] if best_ht else 0.3} \\")
        print(f"  --eta {best_eta['eta']}")
    
    print("\n结束时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"所有结果已保存到 {SAVE_DIR} 文件夹")

if __name__ == "__main__":
    # 检查必要的库
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn']
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"请先安装缺失的库: pip install {' '.join(missing)}")
    else:
        main()