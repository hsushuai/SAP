import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格与中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

def plot_confusion_mat():
    confusion_mat = [
        [0.84, 0.16],
        [0.26, 0.74]
    ]
    plt.rcParams.update({"font.size": 20})

    labels = ["0", "1"]
    tick_labels = ["0", "1"]
    sns.heatmap(confusion_mat, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=tick_labels, annot_kws={"size": 20})
    plt.xlabel("预测")
    plt.ylabel("真实")
    plt.tight_layout()
    plt.savefig("sap/experiments/plots/confusion_matrix_cn.pdf")


def plot_win_loss():
    # Data
    runs_dir = "runs/eval_sen"
    win_loss = []
    
    # 假设路径和文件存在，这里保留你的逻辑
    for filename in os.listdir(runs_dir):
        with open(f"{runs_dir}/{filename}/metric.json") as f:
            metric = json.load(f)
        win_loss.append(metric["win_loss"][0])
        
    num_win, num_draw, num_loss = 0, 0, 0
    for wl in win_loss:
        if wl > 0:
            num_win += 1
        elif wl == 0:
            num_draw += 1
        else:
            num_loss += 1
            
    total = len(win_loss)
    data = [num_win / total, num_draw / total, num_loss / total]
    data = [d * 100 for d in data]
    labels = ["获胜", "平局", "失败"]
    
    # 注意：如果 colors 只有一个值，建议定义为列表以匹配 labels 数量，或者直接使用单色
    colors = ["#d86c50", "#0ac9bf", "#a39aef"] 
    bar_width = 0.5
    plt.rcParams.update({"font.size": 20})

    # Plot
    plt.clf()
    # 捕获 barh 返回的对象以便后续添加标签
    bars = plt.barh(labels[::-1], data[::-1], color=colors[::-1], hatch="/", edgecolor="black", height=bar_width)
    
    # --- 新增：在条形图上显示具体数值 ---
    # fmt='%.1f%%' 表示保留一位小数并带上百分号
    # padding=3 表示标签距离条形末端 3 个单位
    plt.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=16)
    
    plt.xlabel("胜率 (%)")

    # 适当增加横轴范围，防止数值标签超出边界
    plt.xlim(0, max(data) * 1.15) 

    # Save plot
    plt.tight_layout()
    plt.savefig("sap/experiments/plots/eval_sen_cn.pdf")


if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    plot_confusion_mat()
    plot_win_loss()
