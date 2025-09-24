import argparse
import yaml
import sys

sys.path.append('darer')
from ATRPO import ATRPO
from CustomSAC import CustomSAC
from ASAC import ASAC
from arDDPG import arDDPG
from utils import safe_open


env_to_steps = {
    'Humanoid-v5': 10_000_000,
    'HalfCheetah-v5': 3_000_000,
    'Ant-v5': 3_000_000,
    'Swimmer-v5': 3_000_000,
    'Walker2d-v5': 3_000_000,
    'Hopper-v5': 1_000_000,
    'Pusher-v5': 1_000_000,
    'Reacher-v5': 100_000,
    'InvertedPendulum-v5': 50_000,
    'InvertedDoublePendulum-v5': 50_000,
}

env_to_logfreq = {
    'Humanoid-v5': 10000,
    'HalfCheetah-v5': 5000,
    'Ant-v5': 5000,
    'Swimmer-v5': 5000,
    'Walker2d-v5': 5000,
    'Hopper-v5': 2500,
    'Pusher-v5': 2500,
    'Reacher-v5': 1000,
    'InvertedPendulum-v5': 1000,
    'InvertedDoublePendulum-v5': 1000,
}
algo_to_agent_class = {
    'sac': CustomSAC,
    'asac': ASAC,
    'arddpg': arDDPG,
    'atrpo': ATRPO
}


if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--count', type=int, default=1)
    args.add_argument('--env_id', type=str, default='HalfCheetah-v5')
    args.add_argument('--algo', type=str, default='asac')
    args.add_argument('--device', type=str, default='auto')
    args.add_argument('--exp-name', type=str, default='asac-vs-sac')
    args.add_argument('--name', type=str, default='')
    args.add_argument('--eval_steps', type=int, default=None)
    args.add_argument('--save-best', type=bool, default=True)

    args = args.parse_args()
    env_id = args.env_id
    experiment_name = args.exp_name
    device = args.device
    name_suffix = args.name

    print("Running finetuned hyperparameters...")
    algo = args.algo
    algo = algo.lower()
    print(algo)

    hparams = safe_open(f'hparams/{env_id}/{algo}.yaml')
    AgentClass = algo_to_agent_class[algo]
    if args.save_best:
        hparams['save_best'] = True

    for i in range(args.count):
        full_config = {}
        from stable_baselines3.sac import SAC
        # agent = SAC('MlpPolicy', env_id, **hparams, device=device)
        agent = AgentClass(env_id, **hparams, policy="MlpPolicy",
                            device=device, log_interval=env_to_logfreq.get(env_id, 1000),
                            tensorboard_log=f'ft_logs/{experiment_name}/{env_id}',
                            max_eval_steps=args.eval_steps,
                            name_suffix=f'{name_suffix}',
                            render_mode='human',
                            )
        # Measure the time it takes to learn:
        agent.learn(total_timesteps=env_to_steps.get(env_id, 100_000))
        del agent
