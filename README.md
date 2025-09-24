
# Average-reward Soft Actor Critic
Experiments used in [Average-Reward Soft Actor-Critic](https://arxiv.org/pdf/2501.09080v2) by Jacob Adamczyk, Volodymyr Makarenko,
Stas Tiomkin, and Rahul V. Kulkarni.

Environments: Gridworlds, Gymnasium's classic control and Mujoco.

# Installation:
tested with python 3.12

1. clone the repo
`git clone --recurse-submodules https://github.com/BobMak/asac.git`

2. install the dependencies using pdm

get pdm 3.12   
`pdm use 3.12`   

install packages
`pdm install`

2.b using venv with python3.12
```
pip venv .
source .venv/bin/activate
pip install -e .
```

3. activate the environmnet
pdm: `$(pdm venv activate)`
venv: `source .venv/bin/activate`

# Experiments
We run each experimental configuration 30 times.   

Reproducing the ASAC and arDDPG results:   
```
python experiments/finetuned_runs.py --algo [asac|arddpg|sac] --env_id HalfCheetah-v5 [--exp-name experimentname]
```   
   
Reproducing ATRPO and APO results:
```
python experiments/apo_runs.py --algo [atrpo|appo] --env_id HalfCheetah-v5 [--exp_name experimentname]
```

Put the ATRPO and APO results into the common experiment directory:
```
./process_rlpyt_logs.sh [experimentname]
```

Plotting the results:
```
python experiments/comparison_plotter.py -e HalfCheetah-v5 [-n experimentname]
```
The default experiment name is "paper." You will find the average reward plot from the Figure 2 in an output directory
`ft_logs/<experiment_name>/<env_name>/avg_reward.png`
note: plotting will fail if not all of the environments specified after the `-e` are present in the `ft_logs/<experiment_name>` directory
