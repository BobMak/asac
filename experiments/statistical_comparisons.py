from matplotlib import pyplot as plt
import seaborn as sns
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import numpy as np
import os
from glob import glob
import pandas as pd
from tbparse import SummaryReader
import argparse

env_to_steps = {
    'CartPole-v1': 50_000,
    'Acrobot-v1': 100_000,
    'LunarLander-v2': 200_000,
    'MountainCar-v0': 200_000,
    'HalfCheetah-v4': 1_000_000,
    'Ant-v4': 1_000_000,
    'Swimmer-v4': 1_000_000,
    'Humanoid-v4': 5_000_000,
    'Pusher-v4': 1_000_000,
    'Pendulum-v1': 10_000,
}
plot_metrics = ['eval/avg_reward']

if __name__ == "__main__":
    # parse env arg:
    args = argparse.ArgumentParser()
    args.add_argument('--env', type=str, default='MountainCar-v0')
    args = args.parse_args()

    env = args.env
    # env='Ant-v4'
    total_steps = env_to_steps[env]
    algorithms = [f'{env}-ASAC', 'SAC0.990.2', 'SAC', f'{env}-arDDPG']
    algorithms += [f'{env}-arSAC-newh', 'SAC0.990.2', 'SAC0.9990.05', 'SAC', f'{env}-arDDPG']
    algorithms += ['DQN', f'{env}-ASQL', 'ASAC', 'arDDPG']
    algorithms = list(set(algorithms))

    folder = f'ft_logs/EVAL/{env}/'

    score_dict = dict([(algo, []) for algo in algorithms])
    # temporal_score_dict = dict([(algo, []) for algo in algorithms])
    # Load the scores from the files
    for subfolder in os.listdir(folder):
        subfolder = os.path.join(folder, subfolder)
        if not os.path.isdir(subfolder) or subfolder.endswith('.png'):
            continue

        algo_name = os.path.basename(subfolder).split('_')[0]
        if algo_name in algorithms:
            # Replace algo_name with the matching value in the lsit:
            # algo_name = [algo for algo in algorithms if algo in algo_name][0]

            log_files = glob(os.path.join(subfolder, '*.tfevents.*'))
            if not log_files:
                print(f"No log files found in {subfolder}")
                continue
            
            # Require only one log file per folder:
            # assert len(log_files) == 1
            print("Processing", os.path.basename(subfolder))

            try:
                for log_file in log_files:
                    reader = SummaryReader(log_file)
                    df = reader.scalars
                    df = df[df['tag'].isin(plot_metrics)]
                    # Add a new column with the algo name:
                    df['algo'] = algo_name
                    # Add run number:
                    df['run'] = os.path.basename(subfolder).split('_')[1]
                    # algo_data = pd.concat([algo_data, df])
                    last_steps = df['step'].values[-1]
                    log_freq = df['step'].values[1] - df['step'].values[0]
                    # This (roughly) ensures the run is complete:
                    if not last_steps in [total_steps, total_steps-log_freq+1]:
                        print(f"Warning: {algo_name} has {last_steps} steps. Skipping...")
                        continue
                    score_dict[algo_name].append(sum(df['value'].tolist()) * log_freq / total_steps)
                    # Get rewards at intervals of 5000:
                    # reward_sequence = np.array(df['value'][df['step'] % 5000 == 0].tolist())
                    # if len(reward_sequence) != 200:
                    #     print(f"Warning: {algo_name} has {len(reward_sequence)} frames. Skipping...")
                    #     continue

                    # temporal_score_dict[algo_name].append(reward_sequence)
            except Exception as e:
                print(f"Error processing: {e}", log_file)
                continue

    # Check if no runs were found:
    for algo in algorithms:
        if len(score_dict[algo]) == 0:
            print(f"No runs found for {algo} in {env}.")

            # Remove it
            del score_dict[algo]

    # Properly shape the score_dict as n_runs by n_tasks (1):
    algorithms = list(score_dict.keys())
    for algo in algorithms:
        score_dict[algo] = np.array(score_dict[algo]).reshape(-1, 1)


    # Sort the score_dict by alphabetical name of algo to ensure plotting color consistency:
    # First get names:
    algo_names = list(score_dict.keys())
    # Then sort them:
    algo_names.sort()
    # Then create a new score_dict with the sorted names:
    score_dict = {name: score_dict[name] for name in algo_names}
    # score_dict = dict(sorted(score_dict.items(), key=lambda x: x[0][0], reverse=True))


    aggregate_func = lambda x: np.array([
      metrics.aggregate_median(x),
      metrics.aggregate_iqm(x),
      metrics.aggregate_mean(x),
      # metrics.aggregate_optimality_gap(x)
      ])
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
      score_dict, aggregate_func, reps=2000)
    fig, axes = plot_utils.plot_interval_estimates(
      aggregate_scores, aggregate_score_cis,
      metric_names=['Median', 'IQM', 'Mean'],
      algorithms=algorithms, xlabel='Average Evaluation Reward',
      xlabel_y_coordinate=-1,
      legend=True)

    # change the legend labels:
    # labels, handles = axes[0].get_legend_handles_labels()
    # labels = [label.split('-')[-1] for label in labels]

    fig.savefig(f'ft_logs/EVAL/{env}/rliabl.png', bbox_inches='tight', dpi=300)
    min_t = np.min([np.min(scores) for scores in aggregate_scores.values()])
    max_t = np.max([np.max(scores) for scores in aggregate_scores.values()])
    if min_t > 0:
        min_t *= 0.75
    else:
        min_t *= 1.25

    if max_t > 0:
        max_t *= 1.25
    else:
        max_t *= 0.75
    thresholds = np.linspace(min_t, max_t, 81)
    score_distributions, score_distributions_cis = rly.create_performance_profile(
        score_dict, thresholds, reps=2000)
    # Plot score distributions
    fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    plot_utils.plot_performance_profiles(
      score_distributions, thresholds,
      performance_profile_cis=score_distributions_cis,
      colors=dict(zip(algorithms, sns.color_palette('colorblind'))),
      xlabel=r'Average Evaluation Reward $(\tau)$',
      ax=ax,
      legend=True,
    )
    fig.suptitle(f'Performance Profile for {env}', fontsize=24)
    fig.savefig(f'ft_logs/EVAL/{env}/perf_profile.png', bbox_inches='tight', dpi=300)
