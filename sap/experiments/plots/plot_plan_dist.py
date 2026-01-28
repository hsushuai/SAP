import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


STRATEGY1 = "## Strategy\n- Economic Feature: 2\n- Barracks Feature: resource >= 8\n- Military Feature: Range and Worker\n- Aggression Feature: True\n- Attack Feature: Unit\n- Defense Feature: None\n\n"
STRATEGY2 = "## Strategy\n- Economic Feature: 1\n- Barracks Feature: False\n- Military Feature: Worker\n- Aggression Feature: False\n- Attack Feature: None\n- Defense Feature: Medium\n\n"
SKILLS = [
    "[Harvest Mineral]",
    "[Produce Unit]",
    "[Build Building]",
    "[Deploy Unit]",
    "[Attack Enemy]"
]


def find_all_runs(run_dir):
    result_dirs = []
    for root, dirs, files in os.walk(run_dir):
        if 'plans.json' in files:
            result_dirs.append(root)
    return sorted(result_dirs)


def stat_all_plan_dist(run_dir, output_path):
    result_dirs = find_all_runs(run_dir)
    print(f"发现 {len(result_dirs)} 个结果目录")
    res = defaultdict(list)
    num_plans = 0
    for result_dir in result_dirs:
        with open(f"{result_dir}/plans.json") as f:
            data = json.load(f)
        for item in data:
            strategy_id = None
            if item["players"][0]["strategy"] == STRATEGY1:
                strategy_id = 1
            elif item["players"][0]["strategy"] == STRATEGY2:
                strategy_id = 2
            else:
                continue
            
            num_plans += 1
            res[f"strategy_{strategy_id}"].append(stat_plan_dist(item["players"][0]["plan"]))
    
    print(f"统计 {num_plans} 条计划")
    with open(output_path, "w") as f:
        json.dump(res, f, indent=4)
    print(f"结果保存到 {output_path}")


def stat_plan_dist(plan: str):
    res = {}
    total = 0
    for skill in SKILLS:
        res[skill] = plan.count(skill)
        total += res[skill]
    for skill in SKILLS:
        res[skill] = res[skill] / total
    return res


def plot_strategy_boxplot(json_path, output_path):
    # 1. 加载数据
    with open(json_path) as f:
        data = json.load(f)
    
    # 2. 数据清洗：将嵌套 JSON 转换为长表格式 (Long Format)
    # 目标格式：| Strategy | Action | Value |
    all_records = []
    for strategy_name, time_steps in data.items():
        for step in time_steps:
            for action_name, value in step.items():
                all_records.append({
                    "Strategy": strategy_name,
                    "Action": action_name.replace('[', '').replace(']', ''), # 去掉括号美化标签
                    "Percentage": value
                })
    
    df = pd.DataFrame(all_records)
    
    # 3. 绘图
    # 3. 绘图
    colors = ["#d86c50", "#0ac9bf", "#a39aef", "#f4cc71"]
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white") # 使用 whitegrid 辅助观察数值
    
    ax = sns.boxplot(
        data=df, 
        x="Action", 
        y="Percentage", 
        hue="Strategy",
        palette=colors,
        width=0.5,
        fliersize=2,
        linewidth=1.2
    )
    
    # 4. 图表细节美化
    plt.legend(loc='best', fontsize=16)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.tick_params(axis='x', labelsize=16) 
    ax.tick_params(axis='y', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


def merge_result(filenames, output_path):
    data = {}
    for filename in filenames:
        with open(filename, 'r') as f:
            d = json.load(f)
            data.update(d) 
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    # run_dir = "runs/eval_expert2_sap"
    # output_path = f"{run_dir}/plan_dist.json"
    # stat_all_plan_dist(run_dir, output_path)

    # result_filenames = [
    #     "runs/eval_expert_sap/plan_dist.json",
    #     "runs/eval_expert_sap_no_tips/plan_dist.json",
    #     "runs/eval_expert2_sap/plan_dist.json",
    #     "runs/eval_expert2_sap_no_tips/plan_dist.json"
    # ]
    # output_path = "sap/experiments/plots/plan_dist.json"
    # merge_result(result_filenames, output_path)

    result_path = "sap/experiments/plots/plan_dist.json"
    output_path = "sap/experiments/plots/plan_dist.pdf"
    plot_strategy_boxplot(result_path, output_path)