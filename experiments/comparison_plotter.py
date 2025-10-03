from operator import indexOf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from tbparse import SummaryReader

from apo_runs import import_config

def clean_algorithm_name(algo_name, env):
    prefix = f'{env}-'
    if algo_name.startswith(prefix):
        return algo_name[len(prefix):]
    return algo_name

def rlpyt_step_to_global_step(step, batch_B, batch_T):
    return step * batch_B * batch_T

metrics_to_ylabel = {
    'eval/avg_reward': 'Average Return',
    'rollout/ep_reward': 'Average Rollout Reward',
    'train/theta': r'Reward-rate, $\theta$',
    'train/avg logu': r'Average of $\log u(s,a)$',
    'rollout/avg_entropy': r'Policy Entropy',
}
metric_to_rlpyt_metric = {
    'eval/avg_reward': 'Return/Average'
}
all_metrics = [
    'eval/avg_reward',
]
sns.set_theme(style="whitegrid")
sns.set_context("poster")
plt.rcParams['text.color'] = 'black'
# increase font size
plt.rcParams.update({'font.size': 24})

algo_to_color = {
    'SAC0.990.2': 'orange',
    'ASAC': 'blue',
    'ASACmax': 'blue',
    'arDDPG': 'darkred',
    'SQL': 'orange',
    'SQL1net': 'red',
    'ASQL': 'blue',
    'ASQL1net': 'purple',
    'DQN': 'green',
    "SAC":  'red',
    'SAC0.990.05': 'red',
    'SAC0.9990.05': 'orange',
    'SAC0.99990.05': 'green',
    'appo': 'darkgreen',
    'atrpo': 'darkblue',
}
algos = ['ASAC', 'arDDPG', 'appo', 'atrpo']


def plotter(env, folder, x_axis='step', metric='eval/avg_reward',
            exclude_algos=[], include_algos=[],
            xlim=None, ylim=None, ax=None):
    algo_data = pd.DataFrame()
    subfolders = glob(os.path.join(folder, '*'))
    print("Found subfolders:", subfolders)
    subfolders = sorted(subfolders)
    for subfolder in subfolders:
        if not os.path.isdir(subfolder) or subfolder.endswith('.png'):
            continue

        algo_name = os.path.basename(subfolder).split('_')[0]
        if algo_name in include_algos or len(include_algos) == 0:
            if algo_name in exclude_algos:
                print(f"Skipping {algo_name}, in exclude_algos.")
                continue
            log_files = glob(os.path.join(subfolder, '*.tfevents.*'))
            if not log_files:
                print(f"No log files found in {subfolder}")
                continue
            
            print("Processing", os.path.basename(subfolder))
            if algo_name.startswith("appo") or algo_name.startswith("atrpo"):
                _metric = metric_to_rlpyt_metric[metric]
            else:
                _metric = metric
            try:
                for log_file in log_files:
                    reader = SummaryReader(log_file)
                    df = reader.scalars
                    df = df[df['tag'].isin([_metric, x_axis])]
                    # handle rlpyt format
                    if algo_name.startswith("appo") or algo_name.startswith("atrpo"):
                        rlpyt_config = import_config(algo_name)
                        batch_B = rlpyt_config[0]['sampler']['batch_B']
                        batch_T = rlpyt_config[0]['sampler']['batch_T']
                        # convert step to globel step
                        df['step'] = df['step'].apply(lambda x: rlpyt_step_to_global_step(x, batch_B, batch_T))
                        # convert the metric name
                        df['tag'] = df['tag'].replace(to_replace=_metric, value=metric)

                    clean_algo_name = clean_algorithm_name(algo_name, env)
                    print(env)
                    clean_algo_name = clean_algorithm_name(clean_algo_name, env.split(':')[-1])
                    clean_algo_name = clean_algo_name.replace('-', '')
                    clean_algo_name = clean_algo_name.replace('g99', '')

                    print(clean_algo_name)
                    df['algo'] = clean_algo_name
                    df['run'] = os.path.basename(subfolder).split('_')[1]
                    algo_data = pd.concat([algo_data, df])
            except Exception as e:
                print(f"Error processing: {e}", log_file)
                continue

    metric_data = algo_data[algo_data['tag'] == metric]
    if not metric_data.empty:
        print(f"Plotting {metric}...")
        algo_runs = metric_data.groupby('algo')['run'].nunique()


        for algo, runs in algo_runs.items():
            i = indexOf(algos, "ASAC" if "ASACmax" else algo)
            sns.lineplot(data=metric_data[metric_data['algo']==algo], x='step', y='value', ax=ax,
                         label=algo, color=algo_to_color.get(algo, None),
                         lw=8 if 'ASAC' in algo else 5)
            print(f"Plotted {algo}.")
        if metric == 'rollout/avg_entropy':
            ax.set_yscale('log')

        ax.set_title(env)
        if xlim is None:
            xlim = (0, metric_data['step'].max())
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_ylabel(metrics_to_ylabel.get(metric, metric), fontsize=48)
        # set xtick fontsize:
        ax.tick_params(axis='x', labelsize=36)
        # set ytick fontsize:
        ax.tick_params(axis='y', labelsize=36)
        ax.legend(fontsize=36)

cc = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'LunarLander-v2']
ps = ['DiscA:Pendulum-v1', 'Disc2000A:Pendulum-v1']
sw = ['Swimmer-v4/gamma']
mujoco_envs = ['Humanoid-v5','HalfCheetah-v5','Ant-v5','Hopper-v5','Walker2d-v5','Swimmer-v5']

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--envs', type=str, nargs='+', default=["HalfCheetah-v5",])
    parser.add_argument('-n', '--experiment_name', type=str, default='paper')
    args = parser.parse_args()
    envs = args.envs
    n_rows = 1 if len(envs) == 1 else 2
    n_cols = len(envs) // n_rows
    metrics = ['eval/avg_reward']
    for metric in metrics:
        fig, axis = plt.subplots(n_rows, n_cols, figsize=(15*n_cols, 12*n_rows))
        if 'DiscA:' in envs[0]:
            fig, axis = plt.subplots(1, len(envs), figsize=(15*len(envs), 10))
        if 'Swimmer' in envs[0]:
            fig, axis = plt.subplots(1, len(envs), figsize=(15*len(envs), 10))

        if len(envs) == 1:
            axis = [axis]
            env_name = envs[0]
        else:
            env_name = ''
        for i, env in enumerate(envs):
            ax = axis[i] if n_rows==1 else axis[i//n_cols, i%n_cols]
            env_title = env
            if 'Disc' in env:
                if '2000' in env:
                    n_steps = 2000
                else:
                    n_steps = 200
                env_title = f"Discretized {env.split(':')[-1]} ({n_steps} steps)"

            if env == 'Swimmer-v4/gamma':
                env_title = r'SAC for Different Discount Factors, $\gamma$'
            elif env == 'Swimmer-v4':
                env_title = 'Average-Reward Algorithms'
            ax.set_title(env_title, fontsize=48)
            print(f"Plotting for {env} env.")
            # folder = f'/hpcstor6/scratch01/j/jacob.adamczyk001/logs'
            folder = f'ft_logs/{args.experiment_name}/{env}'
            env_to_settings = {
                "Acrobot-v1": {
                    "xlim": (0, 10000),
                },
                "BreakoutNoFrameskip-v4": {
                    # "xlim": (0, 10e6),
                    # "ylim": (-1,80)
                },
                "DiscA:Pendulum-v1": {
                    "xlim": (200, 2000),
                    "include_algos": {'DQN-g99', 'Pendulum-v1-SQLg99', 'Pendulum-v1-ASQL'}
                },
                "Disc2000A:Pendulum-v1": {
                    "xlim": (200, 2000),
                },
                "Swimmer-v4/gamma": {
                    "ylim": (0, 350),},
                "Swimmer-v5": {
                    "ylim": (0, 350),
                    "include_algos": {'arDDPG', 'Swimmer-v5-ASAC', 'appo', 'atrpo'}
                },
                "HalfCheetah-v4": {
                    "include_algos": {"arDDPG", "HalfCheetah-v4-ASAC", "SAC0.990.2"},
                    # "exclude_algos": {"HalfCheetah-v4-ASAC"}
                }

            }
            try:
                plotter(env=env,
                        folder=folder,
                        metric=metric,
                        exclude_algos=[],
                        **env_to_settings.get(env, {}),
                        # include_algos=['SAC0.990.05', 'SAC0.9990.05', 'SAC0.99990.05'],
                        ax=ax
                        )
                # set y-label on the leftmost bottom plot
                # targ_i = len(envs) - (n_rows - 1) * n_cols
                if i == 0:
                    ax.set_ylabel(metrics_to_ylabel[metric])
                else:
                    ax.set_ylabel('')
            except KeyError:
                print("No data to plot.")
        
        unique_handles, unique_labels = [], []
        for i, env in enumerate(envs):
            ax = axis[i] if n_rows == 1 else axis[i // n_cols, i % n_cols]
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    # Change 'SAC0.990.05' to '$\gamma=0.99$':
                    if 'SAC0.990.05' in label:
                        label = r'$\gamma=0.99$'
                    elif 'SAC0.9990.05' in label:
                        label = r'$\gamma=0.999$'
                    elif 'SAC0.99990.05' in label:
                        label = r'$\gamma=0.9999$'
                    elif 'SAC0.990.2' in label:
                        label = 'SAC'
                    elif 'ASACmax' in label:
                        label = 'ASAC'
                    elif 'appo' in label or 'atrpo' in label:
                        label = label.upper()
                    unique_labels.append(label)
                    unique_handles.append(handle)
            ax.set_xlabel('')
            ax.legend().remove()

        final_labels = sorted(list(set(unique_labels)))
        # use color dict to get handles:
        unique_handles = [unique_handles[unique_labels.index(label)] for label in final_labels]
        # add legend to the first axis:
        first_ax = axis[0] if n_rows == 1 else axis[0, 0]
        first_ax.legend(loc='upper left', ncol=1, borderaxespad=0., labels=final_labels, handles=unique_handles, fontsize=36)
        # add X-axis annotation to the last column of the last row
        ax = axis[-1] if n_rows == 1 else axis[-1, -1]
        ax.set_xlabel('Environment Steps', fontsize=48)
        fig.tight_layout()
        save_path = os.path.join(f'ft_logs/{args.experiment_name}', env_name, f"{metric.split('/')[-1]}.png")
        print(f"Saving plot in {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

