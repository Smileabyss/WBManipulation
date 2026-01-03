import torch
def process_single_step_obs(obs_step):
    """
    将新版205维obs转换成旧版76维obs：
    - 跳过新增42维commands扩展
    - 从扩展后的dof_pos/dof_vel（51维）中提取原来的前27维
    - 从扩展后的actions历史（51维）中提取前12维
    """

    commands_part1 = obs_step[:3]                 # 0–2
    commands_part2 = obs_step[3].unsqueeze(0)    # 3

    imu_ang_vel = obs_step[32:35]                # 32–34
    imu_gravity = obs_step[35:38]                # 35–37

    dof_pos_part = obs_step[38:65]               # 38–64，27维
    dof_vel_part = obs_step[89:116]             # 取前27维（旧版）

    actions_history = obs_step[-51:-51+12]       # 动作前12维

    # ================================
    # 4. 拼接旧版结构：共76维
    # ================================
    processed_step = torch.cat([
        commands_part1,      # 3
        commands_part2,      # 1
        imu_ang_vel,         # 3
        imu_gravity,         # 3
        dof_pos_part,        # 27
        dof_vel_part,        # 27
        actions_history         # 12
    ])

    assert processed_step.shape[0] == 76, f"维度错误！实际{processed_step.shape[0]}"
    return processed_step


def change_to_lowpolicy_obs(obs):
    """
    修改序列形式的观测值（适配新版obs，输出与旧版一致的序列）。
    obs shape: (num_envs, sequence_length) where sequence_length = 6 * 新版onestepdim
    新版onestepdim = 旧版76 + 42（力/力矩） + (51-12)（动作新增维度）= 157（自动推断，无需硬编码）
    """
    num_envs = obs.shape[0]
    sequence_length = obs.shape[1]
    
    # 自动推断新版单步维度（包含42维力/力矩+51维动作）
    onestepdim = sequence_length // 6  # 新版单步维度（无需修改，自动适配）
    
    # 处理后单步维度仍为76（与旧版一致）
    new_onestepdim = 3+1+3+3 + 27 + 27 + 12  # 固定76维
    
    # 初始化处理后的观测张量
    processed_obs = torch.empty(
        num_envs, 6 * new_onestepdim,
        device=obs.device, dtype=obs.dtype
    )
    
    for env_idx in range(num_envs):
        # 提取当前环境的整个观测序列（含新版特征）
        env_obs_sequence = obs[env_idx, :]
        processed_sequence = []
        
        for t in range(6):
            # 1. 按新版单步维度拆分序列（关键：用推断出的onestepdim）
            start = t * onestepdim
            end = start + onestepdim
            obs_step = env_obs_sequence[start:end]  # 新版单步观测（含42维力/力矩+51维动作）
            # print("obs_step",obs_step.shape)
            
            # 2. 处理单步观测（提取旧版76维特征）
            processed_step = process_single_step_obs(obs_step)
            
            # 3. 收集处理后的时间步
            processed_sequence.append(processed_step)
        
        # 4. 拼接6个时间步，恢复序列形式
        full_processed_sequence = torch.cat(processed_sequence)
        processed_obs[env_idx, :] = full_processed_sequence
            
    return processed_obs