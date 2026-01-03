### Prerequisites

We recommend to use our code under the following environment:

- Ubuntu 20.04/22.04 Operating System
- IsaacGym Preview 4.0
  - NVIDIA GPU (RTX 2070 or higher)
  - NVIDIA GPU Driver (recommended version 535.183)
- Conda
  - Python 3.8

### Installation
A. Create a virtual environment and install Isaac Gym:
```
1. conda create -n homierl python=3.8
2. conda activate homierl
3. cd path_to_downloaded_isaac_gym/python
4. pip install -e .
```
B. Install this repository:
```
1. git clone https://github.com/OpenRobotLab/HomieRL.git
2. cd HomieRL && pip install -r requirements.txt
3. cd rsl_rl && pip install -e .
4. cd ../legged_gym && pip install -e .
```
### Train lower body policy
You can train your own policy with our code by running the command below.
```
python legged_gym/legged_gym/scripts/train.py --task g1 --num_envs 4096 --headless --run_name my_policy --rl_device cuda:0 --sim_device cuda:0
```
The meanings of the parameters in this command are listed below:
* `--task`: the training task
* `--num_envs`: the number of parallel environments used for training
* `--headless`: don't use the visualization window; you cannot use it to visualize the training process
* `--run_name`: name of this training
* `--rl_device` & `--sim_device`: which device is used for training

The default logging method is [wandb](https://wandb.ai/), and you have to set the values of ***run_name***, ***experiment_name***, ***wandb_project***, and ***wandb_user*** to yours in `legged_gym/legged_gym/envs/g1/g1_29dof_config.py`. You can also change the ***logger*** to **tensorboard**. The training results will be saved in `legged_gym/logs/`.

If you encounter the error: ***"ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory"***, you can run this command to solve it, where path_to_miniconda3 is the absolute path of your miniconda directory:
```
export LD_LIBRARY_PATH=path_to_miniconda3/envs/homierl/lib:$LD_LIBRARY_PATH
```
### Play lower body policy
Once you train a policy, you can first set the [resume_path](https://github.com/OpenRobotLab/HomieRL/blob/main/legged_gym/legged_gym/utils/task_registry.py#L6) to the path of your checkpoint, and run the command below:
```
python legged_gym/legged_gym/scripts/play.py --num_envs 32 --task g1 --resume
```
Then you can view the performance of your trained policy.

### Play whole body manipulation
If you train a better policy of lower body policy, you can change ***HomieRL/legged_gym/logs/exported/policies/policy.pt*** to ***HomieRL/legged_gym/logs/exported/policies/default_policy.pt***
```
python legged_gym/legged_gym/scripts/test_env.py 
```
