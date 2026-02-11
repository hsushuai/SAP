from sap.strategy import Strategy
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def get_all_strategy(base_dir="sap/data/strategies") -> list[Strategy]:
    if not os.path.exists(base_dir): return []
    strategy_filenames = sorted([f for f in os.listdir(base_dir) if f.endswith(".json")])
    return [Strategy.load_from_json(f"{base_dir}/{f}") for f in strategy_filenames]

def plot():
    strategies = get_all_strategy()
    if not strategies: return
    
    # 1. 准备数据
    strategies_vector = np.array([s.encode() for s in strategies])
    vectors_scaled = StandardScaler().fit_transform(strategies_vector)

    # 2. t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=15, init='pca', random_state=42)
    reduced_data = tsne.fit_transform(vectors_scaled)
    
    # 3. 开始绘图：使用艺术化的风格
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    
    # 设置极简背景
    ax.set_facecolor('#fdfdfd')
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    # 绘制点：增加边缘发光感和阴影效果
    # c 使用 y 轴坐标进行映射，制造一种空间纵深感的色彩过渡
    colors = reduced_data[:, 1] 
    
    # 绘制背景散点（模拟光晕）
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, 
               cmap='magma', s=250, alpha=0.1, edgecolors='none')
    
    # 绘制主体散点
    scatter = ax.scatter(
        reduced_data[:, 0], 
        reduced_data[:, 1], 
        c=colors, 
        cmap='magma', 
        s=120, 
        alpha=0.9, 
        edgecolors='white', 
        linewidth=1.5,
        zorder=3
    )

    # 4. 隐藏坐标轴标签和数值
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # plt.title('策略空间特征分布图谱', fontsize=18, fontproperties=my_font, pad=20, color='#333333')

    # 6. 保存
    plt.tight_layout()
    plt.savefig('strategy_tsne_art.pdf', bbox_inches='tight')

if __name__ == "__main__":
    plot()