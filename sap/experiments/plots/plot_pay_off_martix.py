import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 设置绘图风格与中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

def plot_payoff_matrix(file_path="sap/data/payoff/payoff_matrix.csv"):
    if not os.path.exists(file_path):
        print(f"错误：未找到文件 {file_path}")
        return

    # 1. 读取数据（第一列作为索引）
    df = pd.read_csv(file_path, index_col=0)
    
    # 2. 创建画布
    fig, ax = plt.subplots(figsize=(14, 11), dpi=300)
    
    # 3. 绘制热力图
    # 3. 绘制热力图
    # cmap 选择 'RdBu_r' (红-白-蓝翻转)，蓝色代表正收益，红色代表负收益
    sns.heatmap(
        df, 
        annot=True,           
        fmt=".1f",            
        cmap='RdBu_r',        # 修改此处：蓝色(正) - 红色(负)
        center=0,             
        linewidths=0.2,       
        linecolor='#eeeeee',  
        cbar_kws={'shrink': 0.8},
        square=True,
        xticklabels=False,  # <--- 隐藏 x 轴刻度标签
        yticklabels=False   # <--- 隐藏 y 轴刻度标签
    )

    # 4. 细节美化
    # ax.set_title('策略库对抗收益矩阵', fontsize=18, pad=20)
    ax.set_xlabel('对手策略 $\psi^{-1}$', fontsize=14, labelpad=10)
    ax.set_ylabel('受控智能体策略 $\psi^{1}$', fontsize=14, labelpad=10)
    
    # 调整坐标轴刻度标签方向
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # 5. 保存为高清 PDF 供论文使用
    plt.tight_layout()
    output_filename = 'sap/experiments/plots/payoff_matrix.pdf'
    plt.savefig(output_filename, bbox_inches='tight')

if __name__ == "__main__":
    plot_payoff_matrix()