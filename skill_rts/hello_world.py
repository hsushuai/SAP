from skill_rts.envs import MicroRTSLLMEnv
from skill_rts.agents import bot_ais


def main():
    agents = [bot_ais.get("coacAI"), bot_ais.get("randomAI")]
    env = MicroRTSLLMEnv(agents, record_video=True, display=False, auto_build=False)
    payoffs, _ = env.run()
    print("Payoffs:", payoffs)


if __name__ == "__main__":
    main()
