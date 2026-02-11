import argparse
from omegaconf import OmegaConf
import json
from skill_rts.envs import MicroRTSLLMEnv
from sap.agent import Planner, SAPVanilla, SAPAgent, SAPAgentWithoutSEN, SAPCoT
from skill_rts.agents import bot_ais
from skill_rts import logger
import traceback
import time


def run(opponent):
    # Initialize
    cfg = OmegaConf.load("sap/configs/sap.yaml")
    
    runs_dir = f"runs/eval-SAP-Distill/SAP-Distill-V2-Qwen2.5-32B/{opponent['name']}"
    logger.set_level(logger.DEBUG)

    model_cfg0 = {"model": "SAP-Distill-V2-Qwen2.5-32B", "temperature": 0, "max_tokens": 2048}
    # agent0 = SAPAgent(player_id=0, prompt="zero-shot-w-strategy", map_name="basesWorkers8x8", strategy_interval=200, **model_cfg0)
    agent0 = Planner(player_id=0, prompt="zero-shot", map_name="basesWorkers8x8", **model_cfg0)
    # agent0 = SAPCoT(player_id=0, map_name="basesWorkers8x8", **model_cfg0)
    # env = MicroRTSLLMEnv([agent, opponent["agent"]], **cfg.env)
    env = MicroRTSLLMEnv([agent0, opponent["agent"]], **cfg.env)
    
    # Run the episodes
    for episode in range(3):
        run_dir = f"{runs_dir}/run_{episode}"
        env.set_dir(run_dir)
        start_time = time.time()
        try:
            payoffs, trajectory = env.run()
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            env.close()
            traceback.print_exc()
            continue

        # Save the results
        OmegaConf.save(cfg, f"{run_dir}/config.yaml")
        trajectory.to_json(f"{run_dir}/traj.json")
        env.metric.to_json(f"{run_dir}/metric.json")
        with open(f"{run_dir}/plans.json", "w") as f:
            json.dump(env.plans, f, indent=4)
        print(f"Match {episode} | {runs_dir} | Payoffs: {payoffs} | Runtime: {(time.time() - start_time) / 60:.2f}min, {env.time}steps")


if __name__ == "__main__":
    import os

    baseline_model_cfg = {"model": "Qwen2.5-72B-Instruct", "temperature": 0, "max_tokens": 4096}
    # opponents = [
    #     # {"name": "Vanilla", "agent": VanillaAgent(player_id=1, **baseline_model_cfg)},
    #     # {"name": "CoT", "agent": CoTAgent(player_id=1, **baseline_model_cfg)},
    #     # {"name": "PLAP", "agent": PLAPAgent(player_id=1, **baseline_model_cfg)},
    #     {"name": "SAP-Gen", "agent": SAPVanilla(prompt="zero-shot-w-strategy", player_id=1, map_name="basesWorkers16x16", strategy_interval=200, **baseline_model_cfg)},
    #     {"name": "SAP-OM-wo_SEN", "agent": SAPAgentWithoutSEN(prompt="zero-shot-w-strategy", player_id=1, map_name="basesWorkers16x16", strategy_interval=200, **baseline_model_cfg)},
    #     # {"name": "ExpertStrategy", "agent": Planner(prompt="zero-shot-w-strategy", player_id=1, map_name="basesWorkers8x8", strategy="sap/data/expert_strategy.json", **baseline_model_cfg)}
    # ]
    opponents = []
    for name, bot in bot_ais.items():
        opponents.append({"name": name, "agent": bot})
    for opponent in opponents:
        run(opponent)
