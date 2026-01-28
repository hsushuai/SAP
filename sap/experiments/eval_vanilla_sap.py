import argparse
from omegaconf import OmegaConf
import json
from skill_rts.envs import MicroRTSLLMEnv
from sap.agent import Planner, SAPVanilla
from skill_rts.agents import bot_ais, VanillaAgent, CoTAgent, PLAPAgent
from skill_rts import logger
import traceback
import time


llm_based_baselines = {
    "Vanilla": VanillaAgent,
    "CoT": CoTAgent,
    "PLAP": PLAPAgent
}


def parse_args():

    cfg = OmegaConf.load("sap/configs/sap.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", type=str, help="Path to the map file")
    parser.add_argument("--max_steps", type=int, help="Maximum steps for the environment")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--temperature", type=float, help="Temperature for LLM")
    parser.add_argument("--max_tokens", type=int, help="Maximum tokens for LLM")
    parser.add_argument("--num_generations", type=int, help="Number of generations for LLM")
    parser.add_argument("--opponent", type=str, help="Strategy for opponent")
    parser.add_argument("--interval", type=int, help="Interval for update plan")

    args = parser.parse_args()

    if args.map_path is not None:
        cfg.env.map_path = args.map_path
    if args.model is not None:
        cfg.agents[0].model = args.model
        cfg.agents[1].model = args.model
    if args.temperature is not None:
        cfg.agents[0].temperature = args.temperature
        cfg.agents[1].temperature = args.temperature
    if args.max_tokens is not None:
        cfg.agents[0].max_tokens = args.max_tokens
        cfg.agents[1].max_tokens = args.max_tokens
    if args.opponent is not None:
        cfg.agents[1].strategy = args.opponent
    if args.interval is not None:
        cfg.env.interval = args.interval
    
    return cfg


def run():
    # Initialize
    cfg = parse_args()
    map_name = cfg.env.map_path.split("/")[-1].split(".")[0]
    
    runs_dir = "runs/eval_scling_law/7b_vs_3b"
    logger.set_level(logger.DEBUG)

    model_cfg0 = {"model": "Qwen2.5-7B-Instruct", "temperature": 0, "max_tokens": 4096}
    model_cfg1 = {"model": "Qwen2.5-3B-Instruct", "temperature": 0, "max_tokens": 4096}
    agent0 = SAPVanilla(player_id=0, map_name=map_name, prompt="zero-shot-w-strategy", strategy_interval=200, **model_cfg0)
    agent1 = SAPVanilla(player_id=1, map_name=map_name, prompt="zero-shot-w-strategy", strategy_interval=200, **model_cfg1)
    # agent = Planner(prompt="zero-shot-w-strategy", player_id=0, map_name="basesWorkers8x8", strategy="sap/data/expert_strategy2.json", **model_cfg)
    # env = MicroRTSLLMEnv([agent, opponent["agent"]], **cfg.env)
    env = MicroRTSLLMEnv([agent0, agent1], **cfg.env)
    
    # Run the episodes
    for episode in range(cfg.episodes):
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
    # baseline_model_cfg = {"model": "Qwen2.5-72B-Instruct", "temperature": 0, "max_tokens": 4096}
    # opponents = [
    #     {"name": "Vanilla", "agent": VanillaAgent(player_id=1, **baseline_model_cfg)},
    #     {"name": "CoT", "agent": CoTAgent(player_id=1, **baseline_model_cfg)},
    #     {"name": "PLAP", "agent": PLAPAgent(player_id=1, **baseline_model_cfg)},
    #     {"name": "ExpertStrategy", "agent": Planner(prompt="zero-shot-w-strategy", player_id=1, map_name="basesWorkers8x8", strategy="sap/data/expert_strategy.json", **baseline_model_cfg)}
    # ]
    run()
