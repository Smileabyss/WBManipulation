import os
import argparse
from time import time
import numpy as np
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, change_to_lowpolicy_obs
import torch
import pickle

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10) 
    env_cfg.terrain.mesh_type = 'plane' 
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False 
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.commands.use_random = False 
    env_cfg.commands.heading_command = False 
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)


    total_steps = 10000
    print(f"开始仿真，总步数：{total_steps}")
    start_time = time()
    
    commands_a = torch.zeros((env.num_envs,6), dtype=torch.float32, device=env.device)
    for i in range(total_steps):
        env.apply_torque_yes = True
        env.add_force = False
        obs,_,_,_,_,_,_,commands_a = env.step(commands_a)
        if i < 10:
            commands_a[:,3] = 0.8
        if i==10:
            commands_a[:,2] = 0.5
        if i==75:
            commands_a[:,2] = 0
            commands_a[:,0] = 0.5

        if i==354:
            commands_a[:,0] = 0
        
        if i==400:
            commands_a[:,5] = 1
            commands_a[:,4] = -0.02
        
        if i== 450:
            commands_a[:,3] = 0.2
        
        if i== 500:
            commands_a[:,4] = 0
        if i== 520:
            commands_a[:,4] = -0.01
        if i== 530:
            commands_a[:,4] = 0.01

        if i==570:
            commands_a[:,3] = 0.5
        if i==580:
            commands_a[:,3] = 0.8

        if i==600:
            commands_a[:,2] = -0.5

        if i==820:
            commands_a[:,2] = 0
        
        if i==900:
            commands_a[:,0] = 0.5
        
        if i==930:
            commands_a[:,0] = 0

        if i==1020:
            commands_a[:,4] = -0.02
        
        if i==1040:
            commands_a[:,0] = -0.5

        if i==1100:
            commands_a[:,0] = 0.



        if (i + 1) % 1000 == 0:
            elapsed_time = time() - start_time
            print(f"步数：{i+1}/{total_steps}，耗时：{elapsed_time:.2f}s，FPS：{(i+1)/elapsed_time:.2f}")

    print("\n仿真结束。")

if __name__ == '__main__':
    args = get_args()
    args.task = 'g1inspire'  # 指定任务
    args.num_envs = 1
    args.headless = False
    play(args)
