"""
this is a modified version of https://github.com/xtma/apo/blob/main/main.py
"""
import multiprocessing
import os
import random

import numpy as np

import sys

import psutil

sys.path.append('rlpyt')
sys.path.append('apo')
sys.path.append('darer')
from rlpyt.agents.pg.mujoco import MujocoFfAgent
from rlpytGymnasiumWrappers import gym_make
#from rlpyt.envs.gym import make as gym_make
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.utils.logging.context import logger_context
from apo.envs.traj_info import AverageTrajInfo

from finetuned_runs import env_to_steps, env_to_logfreq


def import_config(algo_name):
    if algo_name == "appo":
        from apo.algos.apg.appo import APPO as Algo
        from apo.experiments.configs.mujoco.apg.mujoco_appo import config
    elif algo_name == "appo2":
        from apo.algos.apg.appo2 import APPO2 as Algo
        from apo.experiments.configs.mujoco.apg.mujoco_appo import config
    elif algo_name == "aac":
        from apo.algos.apg.aac import AAC as Algo
        from apo.experiments.configs.mujoco.apg.mujoco_aac import config
    elif algo_name == "atrpo":
        from apo.algos.apg.atrpo import ATRPO as Algo
        from apo.experiments.configs.mujoco.apg.mujoco_atrpo import config
    elif algo_name == "ppo":
        from apo.algos.apg.appo import APPO as Algo
        from apo.experiments.configs.mujoco.apg.mujoco_appo import config
        config["algo"]["longrun"] = False
    elif algo_name == "ppo_norm":
        from apo.algos.apg.appo import APPO as Algo
        from apo.experiments.configs.mujoco.apg.mujoco_appo import config
        config["algo"]["longrun"] = False
        config["algo"]["normalize_advantage"] = True
    elif algo_name == "trpo":
        from apo.algos.pg.trpo import TRPO as Algo
        from apo.experiments.configs.mujoco.pg.mujoco_trpo import config
    else:
        assert NotImplementedError
    return config, Algo

def build_and_train(
    algo_name='appo',
    env_id="Swimmer-v4",
    cuda_idx=None,
    gamma=0.9,
    lamda=0.8,
    lr_eta=0.1,
    rm_vb_coef=0.1,
    exp_name='tests',
):
    config, Algo = import_config(algo_name)
    config["env"]["id"] = env_id
    config["algo"]["discount"] = gamma
    config["algo"]["gae_lambda"] = lamda
    config["algo"]["lr_eta"] = lr_eta
    config["algo"]["rm_vbias_coeff"] = rm_vb_coef

    config["runner"]["n_steps"] = env_to_steps[env_id]
    config["runner"]["log_interval_steps"] = env_to_logfreq[env_id]

    sampler = GpuSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id),
        eval_env_kwargs=dict(id=env_id),
        TrajInfoCls=AverageTrajInfo,
        **config["sampler"],
    )
    algo = Algo(optim_kwargs=config["optim"], **config["algo"])
    agent = MujocoFfAgent(model_kwargs=config["model"], **config["agent"])
    n_cpus = config["sampler"]["batch_B"]
    # use current process's affinity to avoid oversubscription
    availalbe_cpus = psutil.Process(os.getpid()).cpu_affinity()
    workers_cpus = np.random.choice(availalbe_cpus, n_cpus).tolist()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=dict(cuda_idx=cuda_idx, workers_cpus=workers_cpus),
        **config["runner"],
    )
    if algo_name in ['ppo', 'trpo']:
        name = f"{algo_name}_g-{gamma}_l-{lamda}_{env_id}"
    else:
        name = f"{algo_name}_g-{gamma}_l-{lamda}_e-{lr_eta}_v-{rm_vb_coef}_{env_id}"
    log_dir = f"ft_logs/{exp_name}/{env_id}/{name}"
    with logger_context(
            log_dir,
            random.randint(0,1048576),
            name,
            config,
            override_prefix=True,
            use_summary_writer=True,
    ):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--algo', help='algorithm', default='ppo')
    parser.add_argument('--env_id', help='environment ID', default='HalfCheetah-v5')
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--gamma', help='discount', type=float, default=0.99)
    parser.add_argument('--lamda', help='gae lambda', type=float, default=0.95)
    parser.add_argument('--lr_eta', help='lr_eta', type=float, default=0.1)
    parser.add_argument('--rm_vb_coef', type=float, default=0.1)
    parser.add_argument('--exp_name', type=str, default="tests")
    args = parser.parse_args()
    build_and_train(
        algo_name=args.algo,
        env_id=args.env_id,
        cuda_idx=args.cuda_idx,
        gamma=args.gamma,
        lamda=args.lamda,
        lr_eta=args.lr_eta,
        rm_vb_coef=args.rm_vb_coef,
    )
