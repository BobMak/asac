
# Average-reward Soft Actor Critic
Experiments used in [Average-Reward Soft Actor-Critic](https://arxiv.org/pdf/2501.09080v2) by Jacob Adamczyk, Volodymyr Makarenko,
Stas Tiomkin, and Rahul V. Kulkarni.

Environments: Gridworlds, Gymnasium's classic control and Mujoco.

# Installation:
tested with python 3.12

1. Clone the repo
`git clone --recurse-submodules https://github.com/BobMak/asac.git`

2. Install the dependencies

2.a using pdm
```
pdm use 3.12
pdm install
```

2.b using venv with python3.12
```
pip venv .
source .venv/bin/activate
pip install -e .
```

3. activate the environment   
pdm: `$(pdm venv activate)`   
venv: `source .venv/bin/activate`

# Experiments
Run a single ASAC or arDDPG training run for a specified environment:   
```
python experiments/finetuned_runs.py --algo [asac|arddpg|sac] --env_id HalfCheetah-v5 [--exp-name experimentname]
```   

Run a single ATRPO and APO training run for a specified environment:
```
python experiments/apo_runs.py --algo [atrpo|appo] --env_id HalfCheetah-v5 [--exp_name experimentname]
```

We run each experimental configuration 30 times.   

Put all of the ATRPO and APO results into the common experiment directory:
```
./process_rlpyt_logs.sh [experimentname]
```

Plotting the results:
```
python experiments/comparison_plotter.py -e HalfCheetah-v5 [-n experimentname]
```
The default experiment name is "paper." You will find the average reward plot from Figure 2 in the output directory
`ft_logs/<experiment_name>/<env_name>/avg_reward.png`   
Note: plotting will fail if not all of the environments specified after the `-e` are present in the `ft_logs/<experiment_name>` directory
