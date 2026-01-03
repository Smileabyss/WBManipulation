# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import copy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math_self import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.base.legged_robot import euler_from_quaternion
import threading
import time
from legged_gym.utils import Cv2Display,  change_to_lowpolicy_obs
import torchvision
from legged_gym.utils import RobotArmKinematics
import trimesh

class G1Inspire(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        self.num_one_step_privileged_obs = self.cfg.env.num_one_step_privileged_obs
        self.actor_history_length = self.cfg.env.num_actor_history
        self.critic_history_length = self.cfg.env.num_critic_history
        self.actor_proprioceptive_obs_length = self.num_one_step_obs * self.actor_history_length
        self.critic_proprioceptive_obs_length = self.num_one_step_privileged_obs * self.critic_history_length
        self.actor_use_height = True if self.num_obs > self.actor_proprioceptive_obs_length else False
        self.num_lower_dof = self.cfg.env.num_lower_dof
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        # ------------creat images data and visualiize ---------##
        if self.cfg.env.use_camera:
            self.create_cameras()
        # ------------resample elbow and wrist's position and rpy angels --##
        # self.kinematics = RobotArmKinematics(
        #         body_x_block_range=[-0.1, 0.1],
        #         device=self.device,
        #         dtype=torch.float32
        #     )
        ## -------------low policy env add ------------##
        self.legged_policy = torch.jit.load(self.cfg.env.low_policy_model_path, map_location=self.device)
        self.legged_policy.eval()  
        ### -------------------------------------------##
        self.reset_catch = False
    
    def step(self, commands_a):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        ##---update commands actions---------------
        self.commands[:,0:3] = commands_a[:,0:3]
        self.commands[:,4] = commands_a[:,3]
        self.commands[:,5] = commands_a[:,4]
        mask = torch.isclose(commands_a[:,5], torch.tensor(1.0, device=commands_a.device), atol=1e-6)
        ##----------init upper zeros actions------
        actions = torch.zeros((self.num_envs,39), dtype=torch.float32, device=self.device)
        ##
        actions[mask,1] = -0.02
        actions[mask,4] = 0.02
        actions[mask,7+12+1] = -0.02
        actions[mask,7+12+4] = 0.02
        ##
        actions[mask,3] = -self.commands[:,5]
        actions[mask,7+5] = 100*self.commands[:,5]
        actions[mask,7+5+2] = 100*self.commands[:,5]
        actions[mask,7+12+3] = self.commands[:,5]
        ### ------- add legged policy to env step: 1, justyfy obs to acurate low policy needed obs  2,computer actions 12  ----###
        processed_obs = change_to_lowpolicy_obs(self.obs_buf)
        with torch.no_grad():
            lower_actions = self.legged_policy(processed_obs)
        actions =  torch.cat([lower_actions,actions], dim=-1)
        ### -------------------------------------------##

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.origin_actions[:] = self.actions[:]
        self.delayed_actions = self.actions.clone().view(1, self.num_envs, self.num_actions).repeat(self.cfg.control.decimation, 1, 1)
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
                
        # Randomize Joint Injections
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        # step physics and render each frame
        self.render()
        if self.cfg.env.use_camera:
            if self.image_display  is not None:
                rgbs = self.cam_tensors.permute(0, 3, 1, 2)  # (n, 3, H, W)
                N = rgbs.shape[0]
                rgb_to_display = torchvision.utils.make_grid(rgbs, nrow=N // 2)
                self.image_display(rgb_to_display)

            if self.cam_tensors is not None:
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)

            if self.cam_tensors is not None:
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)


        for _ in range(self.cfg.control.decimation):
            
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # upper-body with position control; lower-body with force control;
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            # # -------------------------- 施加刚体力/力矩 --------------------------
            # self.apply_forces.zero_()
            # self.apply_torque.zero_()

            # force_torque_cmds = self.commands[:, -10:].clone() 
            # self.apply_forces[:, self.tip_indice, 0] = -force_torque_cmds
            
            
            # self.gym.apply_rigid_body_force_tensors(
            #     self.sim,
            #     gymtorch.unwrap_tensor(-self.apply_forces),
            #     gymtorch.unwrap_tensor(-self.apply_torque),
            #     gymapi.LOCAL_SPACE,
            # )
            
            # # ------------------------------------------------------------------
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        ##
        self.commands_a[:,0:3] = self.commands[:,0:3] 
        self.commands_a[:,3] = self.commands[:,4] 
        self.commands_a[:,4] = self.commands[:,5]
        self.commands_a[:,5] = commands_a[:,5]
    
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs, self.commands_a

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[self._global_indices[:,0], 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[self._global_indices[:,0], 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[self._global_indices[:,0], 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[self._global_indices[:,0], 7:10] - self.last_root_vel[:, :3]) / self.dt
        
        self.feet_pos[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 3:7]
        self.feet_vel[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self.tip_vel[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.tip_indice, 7:10]
        
        
        # compute joint power
        joint_power = torch.abs(self.torques * self.dof_vel).unsqueeze(1)
        self.joint_powers = torch.cat((self.joint_powers[:, 1:], joint_power), dim=1)
        
        if self.cfg.env.is_train:
            self._post_physics_step_callback()


        if self.cfg.env.use_camera:
            for i, (env, handle) in enumerate(zip(self.envs, self.cameras)):
                if self.cfg.camera.type == "d":
                    depth_image_gpu_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, handle, gymapi.IMAGE_DEPTH)
                    self.cam_tensors[i] = gymtorch.wrap_tensor(depth_image_gpu_tensor).unsqueeze(-1)
                elif self.cfg.camera.type == "rgb":
                    color_image_gpu_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, handle, gymapi.IMAGE_COLOR)
                    self.cam_tensors[i] = gymtorch.wrap_tensor(color_image_gpu_tensor)[..., :-1] 
                elif self.cfg.camera.type == "rgbd":
                    color_image_gpu_tensor = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, env, handle, gymapi.IMAGE_COLOR))
                    depth_image_gpu_tensor = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, env, handle, gymapi.IMAGE_DEPTH))
                    image_gpu_tensor = torch.cat([color_image_gpu_tensor[..., :-1] , depth_image_gpu_tensor], dim=-1)
                    self.cam_tensors[i] = image_gpu_tensor


        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[self._global_indices[:,0], 7:13]
        self.last_tip_vel[:] = self.tip_vel[:]
        

        return env_ids, termination_privileged_obs

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 10., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.gravity_termination_buf = torch.any(torch.norm(self.projected_gravity[:, 0:2], dim=-1, keepdim=True) > 0.8, dim=1)
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.gravity_termination_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        # update action curriculum for specific dofs
        if self.cfg.env.action_curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_action_curriculum(env_ids)
            
        self.refresh_actor_rigid_shape_props(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # resample commands
        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_tip_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.joint_powers[env_ids] = 0.
        self.random_upper_actions[env_ids] = 0. 
        self.current_upper_actions[env_ids] = 0.
        self.delta_upper_actions[env_ids] = 0.
        reset_roll, reset_pitch, reset_yaw = euler_from_quaternion(self.base_quat[env_ids])
        self.roll[env_ids] = reset_roll
        self.pitch[env_ids] = reset_pitch
        self.yaw[env_ids] = reset_yaw
        self.reset_buf[env_ids] = 1
        
         #reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (len(env_ids), self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            # self.extras["episode"]["height_curriculum_ratio"] = self.height_curriculum_ratio
        if self.cfg.env.action_curriculum:
            self.extras["episode"]["action_curriculum_ratio"] = self.action_curriculum_ratio
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if torch.isnan(rew).any():
                import ipdb; ipdb.set_trace()
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """
        imu_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.rigid_body_states[:, self.imu_index,10:13])
        imu_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.gravity_vec)

        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.commands[:, 4].unsqueeze(1),
                                    self.commands[:, 5:],
                                    imu_ang_vel  * self.obs_scales.ang_vel,
                                    imu_projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    ),dim=-1)

        current_actor_obs = torch.clone(current_obs)
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec[0:(10 + 2 * self.num_actions + self.num_lower_dof)]           
        self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_obs:self.actor_proprioceptive_obs_length], current_actor_obs[:, :self.num_one_step_obs]), dim=-1)
        current_critic_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf[:, self.num_one_step_privileged_obs:self.critic_proprioceptive_obs_length], current_critic_obs), dim=-1)
        
    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        imu_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.rigid_body_states[:, self.imu_index,10:13])
        imu_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.gravity_vec)
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.commands[:, 4].unsqueeze(1),
                                    self.commands[:, 5:],
                                    imu_ang_vel  * self.obs_scales.ang_vel,
                                    imu_projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    ),dim=-1)

        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(10 + 2 * self.num_actions + self.num_lower_dof)]
        current_critic_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        return torch.cat((self.privileged_obs_buf[:, self.num_one_step_privileged_obs:self.critic_proprioceptive_obs_length], current_critic_obs), dim=-1)[env_ids]
            
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()
        
    def create_cameras(self):
        """ Creates camera for each robot
        """
        self.camera_params = gymapi.CameraProperties()
        self.camera_params.width = self.cfg.camera.width
        self.camera_params.height = self.cfg.camera.height
        self.camera_params.horizontal_fov = self.cfg.camera.horizontal_fov
        self.camera_params.enable_tensors = True
        self.cameras = []
        for env_handle in self.envs:
            camera_handle = self.gym.create_camera_sensor(env_handle, self.camera_params)
            head_handle = self.gym.get_actor_rigid_body_handle(env_handle, 0, self.head_indice)
            camera_offset = gymapi.Vec3(self.cfg.camera.offset[0], self.cfg.camera.offset[1], self.cfg.camera.offset[2])
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(self.cfg.camera.angle_randomization * (2 * np.random.random() - 1) + self.cfg.camera.angle))
            self.gym.attach_camera_to_body(camera_handle, env_handle, head_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
            self.cameras.append(camera_handle)

        self.image_display = Cv2Display("IsaacGym") if self.cfg.camera.display else None
     
    def post_process_camera_tensor(self):
        """
        First, post process the raw image and then stack along the time axis
        """
        new_images = torch.stack(self.cam_tensors)
        new_images = torch.nan_to_num(new_images, neginf=0)
        new_images = torch.clamp(new_images, min=-self.cfg.camera.far, max=-self.cfg.camera.near)
        # new_images = new_images[:, 4:-4, :-2] # crop the image
        self.last_visual_obs_buf = torch.clone(self.visual_obs_buf)
        self.visual_obs_buf = new_images.view(self.num_envs, -1)
  
    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            env_handle = self.envs[env_id]
            actor_handle = self.actor_handles[env_id]
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)

            for i in range(len(rigid_shape_props)):
                if self.cfg.domain_rand.randomize_friction:
                    rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                if self.cfg.domain_rand.randomize_restitution:
                    rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id==0:
            sum = 0
            for i, p in enumerate(props):
                sum += p.mass
                print(f"Mass of body {i}: {p.mass} (before randomization)")
            print(f"Total mass {sum} (before randomization)")
        # randomize base masss
        if self.cfg.domain_rand.randomize_payload_mass:
            props[self.torso_body_index].mass = self.default_rigid_body_mass[self.torso_body_index] + self.payload[env_id, 0]

        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = self.default_com + gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])
        if self.cfg.domain_rand.randomize_body_displacement:
            props[self.torso_body_index].com = self.default_body_com + gymapi.Vec3(self.body_displacement[env_id, 0], self.body_displacement[env_id, 1], self.body_displacement[env_id, 2])

        
        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
                
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        set_x = torch.rand(len(env_ids), 1).to(self.device)
        is_height = set_x < 1/3
        is_vel = set_x > 1/2
        # self.commands[env_ids, 0] = (torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device) * is_vel).squeeze(1) 
        # self.commands[env_ids, 1] = (torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device) * is_vel).squeeze(1) 
        self.commands[env_ids, 0] = 0
        self.commands[env_ids, 1] = 0
        self.commands[env_ids, 4] = 0.8
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = (torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device) * is_vel).squeeze(1)
            self.commands[env_ids, 4] = (torch_rand_float(self.command_ranges["height"][0], self.command_ranges["height"][1], (len(env_ids), 1), device=self.device) * is_height).squeeze(1) + self.cfg.rewards.base_height_target # height
        else:
            self.commands[env_ids, 2] = (torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device) * is_vel).squeeze(1)
            self.commands[env_ids, 4] = (torch_rand_float(self.command_ranges["height"][0], self.command_ranges["height"][1], (len(env_ids), 1), device=self.device) * is_height).squeeze(1) + self.cfg.rewards.base_height_target # height
        
        # self.commands[env_ids, 5:] = self.random_force_pos(env_ids)

    
    
    @staticmethod
    def quat_inverse(q):
        x, y, z, w = q.unbind(-1)
        return torch.stack([-x, -y, -z,  w], dim=-1)

    @staticmethod
    def quat_multiply(q1, q2):
        x1, y1, z1, w1 = q1.unbind(-1)
        x2, y2, z2, w2 = q2.unbind(-1)

        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        return torch.stack([x, y, z, w], dim=-1)

    @staticmethod
    def quat_rotate_vector(q, v):
        """旋转 v: (N,3)  by quaternion q: (N,4)"""
        # v → 纯四元数
        zeros = torch.zeros_like(v[..., :1])
        qv = torch.cat([v, zeros], dim=-1)
        return G1Inspire.quat_multiply(
                    G1Inspire.quat_multiply(q, qv),
                    G1Inspire.quat_inverse(q)
                )[..., :3]


    @staticmethod
    def global_to_waist(target_pos_global,
                        target_quat_global,
                        waist_pos_global,
                        waist_quat_global):

        # waist 四元数 inverse
        waist_inv = G1Inspire.quat_inverse(waist_quat_global)

        # 位置转换
        translated = target_pos_global - waist_pos_global
        target_pos_waist = G1Inspire.quat_rotate_vector(waist_inv, translated)

        # 姿态转换
        target_quat_waist = G1Inspire.quat_multiply(waist_inv, target_quat_global)

        return target_pos_waist, target_quat_waist
    

    @staticmethod
    def quat_to_rpy(q, degrees=False):
        """
        将四元数转换为RPY角（滚转/俯仰/偏航）
        适配任意维度输入（最后一维为四元数分量），严格遵循PyTorch张量操作逻辑

        参数说明：
            q: torch.Tensor - 输入四元数，形状为 (..., 4)，最后一维为 [x, y, z, w]（与你的代码格式一致）
            degrees: bool - 若为True返回角度制，否则返回弧度制（默认）

        旋转规则：
            RPY顺序：Z-Y-X（偏航yaw绕Z轴 → 俯仰pitch绕Y轴 → 滚转roll绕X轴）
            旋转方向：右手定则（拇指指向轴正方向，四指为旋转正方向）

        返回值：
            torch.Tensor - RPY角，形状为 (..., 3)，顺序为 [roll, pitch, yaw]
        """
        # 解绑定四元数分量（最后一维拆分，保持维度兼容）
        x, y, z, w = q.unbind(dim=-1)

        # 1. 计算滚转角 roll (绕X轴)
        roll_numerator = 2 * (w * x + y * z)
        roll_denominator = 1 - 2 * (x**2 + y**2)
        roll = torch.atan2(roll_numerator, roll_denominator)

        # 2. 计算俯仰角 pitch (绕Y轴) - 钳位防止数值误差导致nan
        pitch_input = 2 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(pitch_input, min=-1.0, max=1.0))

        # 3. 计算偏航角 yaw (绕Z轴)
        yaw_numerator = 2 * (w * z + x * y)
        yaw_denominator = 1 - 2 * (y**2 + z**2)
        yaw = torch.atan2(yaw_numerator, yaw_denominator)

        # 组合RPY张量（恢复最后一维为3）
        rpy = torch.stack([roll, pitch, yaw], dim=-1)

        # 转换为角度制（若需要）
        if degrees:
            rpy = torch.rad2deg(rpy)

        return rpy


    def random_force_pos(self, env_ids):
        if len(env_ids) == 0:
            return self.commands[env_ids, 5:]
        
        # ================== 取当前 batch ==================
        wp = self.rigid_body_states[env_ids, self.waist_indice, :3].unsqueeze(1)    # (K,1,3)
        wq = self.rigid_body_states[env_ids, self.waist_indice, 3:7].unsqueeze(1)      # (K,1,4)

        body_pos = self.rigid_body_states[env_ids, :, :3] 
        body_quat = self.rigid_body_states[env_ids, :, 3:7]

        gep = body_pos[:,self.elbow_wrist_indice,:] # (K,4,3)
        geq = body_quat[:,self.elbow_wrist_indice,:]# (K,4,4)

        # ================== global → waist ==================
        pos_w, _ = G1Inspire.global_to_waist(
            gep,     # (K,4,3)
            geq,     # (K,4,4)
            wp,  # (K,1,3)
            wq   # (K,1,4)
        )  # 输出: (K,4,3), (K,4,4)

        # ================== 分开左右 arm ==================
        prev_l_elbow_pos  = pos_w[:, 0]
        prev_l_base_pos   = pos_w[:, 1]
        prev_r_elbow_pos  = pos_w[:, 2]
        prev_r_base_pos   = pos_w[:, 3]


        # ================== 调用运动学（逐环境） ==================
        K = len(env_ids)

        new_l_elbow_pos  = torch.zeros_like(prev_l_elbow_pos)
        new_l_base_pos   = torch.zeros_like(prev_l_base_pos)
        new_l_base_quat  = torch.zeros_like(prev_l_base_pos)

        new_r_elbow_pos  = torch.zeros_like(prev_r_elbow_pos)
        new_r_base_pos   = torch.zeros_like(prev_r_base_pos)
        new_r_base_quat   = torch.zeros_like(prev_r_base_pos)

        for i in range(K):

            le, lb, lbq = self.kinematics.generate_next_position(
                prev_l_elbow_pos[i], prev_l_base_pos[i], step_count= self.common_step_counter,arm_side='left'
            )
            new_l_elbow_pos[i] = le
            new_l_base_pos[i]  = lb
            new_l_base_quat[i] = lbq

            re, rb, rbq = self.kinematics.generate_next_position(
                prev_r_elbow_pos[i], prev_r_base_pos[i], step_count= self.common_step_counter,arm_side='right'
            )
            new_r_elbow_pos[i] = re
            new_r_base_pos[i]  = rb
            new_r_base_quat[i] = rbq

        pos_quat = torch.stack([
            new_l_elbow_pos,
            new_l_base_pos,
            new_r_elbow_pos,
            new_r_base_pos,
            new_l_base_quat,
            new_r_base_quat
        ], dim=1)   # (K,6,3)

        # ================== 梯形力 ==================
        pre_force = self.commands[env_ids, -10:]
        new_force = pre_force.clone()

        for i in range(K):
            state = self.force_state[env_ids[i]]
            current = pre_force[i]
            hold = self.current_hold_steps[env_ids[i]]

            if state == 0:
                # 核心：先维持力=0，直到hold步数达标
                if hold < self.cfg.commands.force_hold_steps:
                    nf = 0.0  # 力固定为0，维持该状态
                    hold += 1  # 维持步数计数+1
                else:
                    # 力=0维持时间达标后，执行原有力递增逻辑
                    nf = current + self.cfg.commands.force_step * self.dt
                    # 力达到上限后切换状态，重置计数器
                    if torch.all(nf >= self.cfg.commands.ranges.force[1] - 1e-3):
                        nf = self.cfg.commands.ranges.force[1]
                        self.force_state[env_ids[i]] = 1
                        hold = 0  # 切换状态后重置计数器

            elif state == 1:
                # 力上限维持（原有逻辑）
                nf = self.cfg.commands.ranges.force[1]
                hold += 1
                if hold >= self.cfg.commands.force_hold_steps:
                    self.force_state[env_ids[i]] = 2
                    hold = 0

            elif state == 2:
                # 力递减（原有逻辑）
                nf = current - self.cfg.commands.force_step * self.dt
                if torch.all(nf <= self.cfg.commands.ranges.force[0] + 1e-3):
                    nf = self.cfg.commands.ranges.force[0]
                    self.force_state[env_ids[i]] = 3
                    hold = 0

            else:  # state == 3
                # 力下限维持（原有逻辑，若需要此处也可加力=0维持，可按需调整）
                nf = self.cfg.commands.ranges.force[0]
                hold += 1
                if hold >= self.cfg.commands.force_hold_steps:
                    self.force_state[env_ids[i]] = 0
                    hold = 0  # 切回0阶段后，重新开始力=0的维持

            new_force[i] = nf
            self.current_hold_steps[env_ids[i]] = hold


        # ================== 拼接输出 ==================
        joint_force_data = torch.cat([
            pos_quat.reshape(K, -1),
            new_force
        ], dim=-1)  # (K,(3+3+3+5)*2=14*2=28)

        return joint_force_data



    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        self.joint_pos_target = self.default_dof_pos + actions_scaled
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
            torques = torques + self.actuation_offset + self.joint_injection
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
            torques = torques + self.actuation_offset + self.joint_injection
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        elif control_type=="M":
            torques = self.p_gains * self.Kp_factors * (
                    self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
            
            torques = torques + self.actuation_offset + self.joint_injection
            torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
            actions = actions*self.torque_limits
            actions = actions + self.actuation_offset + self.joint_injection
            actions = torch.clip(actions, -self.torque_limits, self.torque_limits)
            return torch.cat((torques[..., :self.num_lower_dof], actions[..., self.num_lower_dof:]), dim=-1)
        elif control_type=="X":
            torques = self.p_gains * self.Kp_factors * (
                    self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
            
            torques = torques + self.actuation_offset + self.joint_injection
            torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
            actions = actions*self.torque_limits
            actions = actions + self.actuation_offset + self.joint_injection
            actions = torch.clip(actions, -self.torque_limits, self.torque_limits)
            return torch.cat((torques[..., :12 +1+7],  actions[...,12+1+7:12+1+7+12], 
                              torques[..., 12+1+7+12:12+1+7+12+7], actions[...,-12:],   ), dim=-1)
        
        else:
            raise NameError(f"Unknown controller type: {control_type}")

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
        if self.cfg.domain_rand.randomize_initial_joint_pos:
            init_dos_pos = self.default_dof_pos * torch_rand_float(self.cfg.domain_rand.initial_joint_pos_scale[0], self.cfg.domain_rand.initial_joint_pos_scale[1], (len(env_ids), self.num_dof), device=self.device)
            init_dos_pos += torch_rand_float(self.cfg.domain_rand.initial_joint_pos_offset[0], self.cfg.domain_rand.initial_joint_pos_offset[1], (len(env_ids), self.num_dof), device=self.device)
            self.dof_pos[env_ids] = torch.clip(init_dos_pos, dof_lower, dof_upper)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch.ones((len(env_ids), self.num_dof), device=self.device)

        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[self._global_indices[env_ids,0]] = self.base_init_state
            self.root_states[self._global_indices[env_ids,0], :3] += self.env_origins[env_ids]
            self.root_states[self._global_indices[env_ids,0], :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            self.root_states[self._global_indices[env_ids,0], 2:3] += torch_rand_float(0.0, 0.1, (len(env_ids), 1), device=self.device) # z position within 0.1m of the ground
        else:
            self.root_states[self._global_indices[env_ids,0]] = self.base_init_state
            self.root_states[self._global_indices[env_ids,0], :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[self._global_indices[:,0], 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        multi_env_ids_int32 = multi_env_ids_int32.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 75% of the maximum, increase the range of commands
        if (torch.mean(self.episode_sums["tracking_x_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_x_vel"]) and (torch.mean(self.episode_sums["tracking_y_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_y_vel"]):
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum)

        
    def update_action_curriculum(self, env_ids):
        """ Implements a curriculum of increasing action range

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if (torch.mean(self.episode_sums["tracking_x_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_x_vel"]):
            self.action_curriculum_ratio += 0.05
            self.action_curriculum_ratio = min(self.action_curriculum_ratio, 1.0)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(10 + 2*self.num_actions + self.num_lower_dof, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:4] = 0. # commands
        noise_vec[4:7] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[7:10] = noise_scales.gravity * noise_level
        noise_vec[10:(10 + self.num_actions)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(10 + self.num_actions):(10 + 2 * self.num_actions)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(10 + 2 * self.num_actions):(10 + 2 * self.num_actions + self.num_lower_dof)] = 0. # previous actions
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        ##global indice
        self._global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[self._global_indices[:,0], 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

 
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.origin_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[self._global_indices[:,0], 7:13])
        self.tip_vel = torch.zeros(self.num_envs, len(self.tip_names),3, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_tip_vel = torch.zeros(self.num_envs,len(self.tip_names),3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.commands_a = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False) 
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_max_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.first_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[self._global_indices[:,0], 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[self._global_indices[:,0], 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.cam_tensors = torch.zeros(self.num_envs, self.cfg.camera.height,self.cfg.camera.width,self.cfg.camera.channels, dtype=torch.uint8, device=self.device, requires_grad=False)
        self.force_state = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.current_hold_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            print(f"Joint {self.gym.find_actor_dof_index(self.envs[0], self.actor_handles[0], name, gymapi.IndexDomain.DOMAIN_ACTOR)}: {name}")
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.action_max = (self.hard_dof_pos_limits[:, 1].unsqueeze(0) - self.default_dof_pos) / self.cfg.control.action_scale
        self.action_min = (self.hard_dof_pos_limits[:, 0].unsqueeze(0) - self.default_dof_pos) / self.cfg.control.action_scale
        self.action_curriculum_ratio = self.cfg.domain_rand.init_upper_ratio
        self.target_heights = torch.ones((self.num_envs), device=self.device) * self.cfg.rewards.base_height_target
        print(f"Action min: {self.action_min}")
        print(f"Action max: {self.action_max}")
        
        self.random_upper_actions = torch.zeros((self.num_envs, self.num_actions - self.num_lower_dof), device=self.device)
        self.current_upper_actions = torch.zeros((self.num_envs, self.num_actions - self.num_lower_dof), device=self.device)
        self.delta_upper_actions = torch.zeros((self.num_envs, 1), device=self.device)
        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_injection = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuation_offset = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)

        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
        if self.cfg.domain_rand.randomize_body_displacement:
            self.body_displacement = torch_rand_float(self.cfg.domain_rand.body_displacement_range[0], self.cfg.domain_rand.body_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        
        #joint powers
        self.joint_powers = torch.zeros(self.num_envs, 100, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        

        self.apply_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float, requires_grad=False)
        self.apply_torque = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float, requires_grad=False)
        
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity


        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_bodies += 2
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)


        # save body names from the asset
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])
            
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
        if self.cfg.domain_rand.randomize_body_displacement:
            self.body_displacement = torch_rand_float(self.cfg.domain_rand.body_displacement_range[0], self.cfg.domain_rand.body_displacement_range[1], (self.num_envs, 3), device=self.device)
        
        self.torso_body_index = self.body_names.index("torso_link")  

        ###########load tablet ################
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True

        table_width_offset = 0.2
        table_width = 0.5 + table_width_offset
        table_length = 1.6 
        table_asset = self.gym.create_box(self.sim, 0.5 + table_width_offset, 1.6, 0.05, table_asset_options)

        table_pos = gymapi.Vec3(-table_width_offset / 2 + 2, 0, 0.4)
        table_half_height = 0.05
        self._table_surface_z = table_pos.z + table_half_height
        self.obj_x_lower = - table_width/6 
        self.obj_x_upper =   table_width/6 - 0.2
        self.obj_y_lower =   table_length/6 + 0.1
        self.obj_y_upper =   table_length/6 - 0.1
        self.obj_z_lower =   self._table_surface_z + 0.05  
        self.obj_z_upper =   self._table_surface_z + 0.3    
        ########### load obj urdf pool ###########
        selected_urdf_paths = ["/home/cyrus/HomieRL/legged_gym/resources/objs/O02@0018@00002/scan.urdf"]
        n = 1
        self.object_asset_pool = [] 
        self.object_asset_options = gymapi.AssetOptions()
        self.object_asset_options.fix_base_link = False
        self.object_asset_options.angular_damping = 5.0
        self.object_asset_options.linear_damping = 5.0
        self.object_asset_options.max_linear_velocity = 10.0
        self.object_asset_options.max_angular_velocity = 20.0
        self.object_asset_options.use_mesh_materials = True

        for idx, urdf_path in enumerate(selected_urdf_paths):
            obj_urdf_dir = os.path.dirname(urdf_path)
            obj_urdf_file = os.path.basename(urdf_path)
            asset = self.gym.load_asset(
                self.sim, obj_urdf_dir, obj_urdf_file, self.object_asset_options
            )
            
            # 随机设置物理属性（从配置范围采样）
            obj_rigid_props = self.gym.get_asset_rigid_shape_properties(asset)
            rand_friction = torch_rand_float(
                self.cfg.asset.friction_range[0], self.cfg.asset.friction_range[1], (1,1), self.device
            ).item()
            rand_contact_offset = torch_rand_float(
                self.cfg.asset.friction_range[0], self.cfg.asset.friction_range[1], (1,1), self.device
            ).item()
            for prop in obj_rigid_props:
                prop.friction = rand_friction
                prop.contact_offset = rand_contact_offset
            self.gym.set_asset_rigid_shape_properties(asset, obj_rigid_props)
            
            # 存入Asset池
            self.object_asset_pool.append(asset)
            print(f"预加载Asset {idx+1}/{n} → {obj_urdf_file}（摩擦系数：{rand_friction:.2f}）")

            # rigid_body_props = self.gym.get_asset_rigid_shape_properties(asset)
            # rigid_body_props[0].mass = 0.5  
            # self.gym.set_asset_rigid_shape_properties(asset, rigid_body_props)


        # --------------------------  构建“环境-Asset”映射（按顺序循环分配） --------------------------
        # 例：n=3，num_envs=10 → 映射为 [0,1,2,0,1,2,0,1,2,0]
        self.env_to_asset_idx = torch.tensor(
            [i % n for i in range(self.num_envs)],  
            device=self.device, dtype=torch.long
        )
        print(f"\n环境-Asset映射示例（前10个环境）：{self.env_to_asset_idx[:10].tolist()}")

        # -------------------------- creat obj points ------------------------------
                
        self.object_shape_points = torch.zeros((self.num_envs, self.cfg.asset.obj_surface_points, 3), dtype=torch.float32, 
                                                device=self.device)
        self.object_shape_points_waist = torch.zeros((self.num_envs, self.cfg.asset.obj_surface_points, 3), dtype=torch.float32, 
                                                device=self.device)
        self.object_shape_points_world = torch.zeros((self.num_envs, self.cfg.asset.obj_surface_points, 3), dtype=torch.float32, 
                                                device=self.device)                                      
        

        for i in range(len(self.env_to_asset_idx)):
            asset_idx = self.env_to_asset_idx[i].item()
            urdf_path = selected_urdf_paths[asset_idx]
            base_path, _ = os.path.splitext(urdf_path)
            ply_file = base_path + ".ply"  # 构造 PLY 文件的路径
            mesh = trimesh.load_mesh(ply_file)
            points, _ = trimesh.sample.sample_surface(mesh, count=self.cfg.asset.obj_surface_points)
            points_tensor = torch.tensor(points, dtype=torch.float32, device=self.device)
            self.object_shape_points[i, ...] = points_tensor



        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            ###
            max_agg_bodies = self.num_bodies + 2  # 机器人body数 + 桌子 + 物体
            max_agg_shapes = self.gym.get_asset_rigid_shape_count(robot_asset) + 1 + 1  # 机器人形状+桌子+物体
            #  开启聚合模式
            self.gym.begin_aggregate(env_handle, max_agg_bodies, max_agg_shapes, True)
            ##
            pos = self.env_origins[i].clone()
            pos[0]  -= 0.55
            pos[2]  += 0.8
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name,
                                                  i, self.cfg.asset.self_collisions, 0)
            
            dof_props = self._process_dof_props(dof_props_asset, i)
            if self.cfg.control.control_type == 'S':
                dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
                dof_props["stiffness"][:] = [300.,  
                                            200., 200., 200., 100.,  20.,  20.,  20.,
                                            20,20,20,20,20,20, 20,20,20,20,20,20,
                                            200., 200., 200., 100.,  20.,  20.,  20.,
                                            20,20,20,20,20,20, 20,20,20,20,20,20]
                dof_props["damping"][:] = [5.0000, 
                                            4.0000, 4.0000, 4.0000, 1.0000, 0.5000, 0.5000,0.5000, 
                                            2,2,2,2,2,2, 2,2,2,2,2,2,
                                            4.0000, 4.0000, 4.0000, 1.0000, 0.5000, 0.5000, 0.5000,
                                            2,2,2,2,2,2, 2,2,2,2,2,2]
            elif self.cfg.control.control_type == 'P':
                dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
                dof_props["stiffness"].fill(0.0)
                dof_props["damping"].fill(0.0)
            
                       
            elif self.cfg.control.control_type == 'M':
                dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
                dof_props["stiffness"].fill(0.0)
                dof_props["damping"].fill(0.0)
        
                dof_props["driveMode"][12:].fill(gymapi.DOF_MODE_POS)
                dof_props["stiffness"][12:] = [300.,  
                                            200., 200., 200., 100.,  20.,  20.,  20.,
                                            20,20,20,20,20,20, 20,20,20,20,20,20,
                                            200., 200., 200., 100.,  20.,  20.,  20.,
                                            20,20,20,20,20,20, 20,20,20,20,20,20]
                dof_props["damping"][12:] = [5.0000, 
                                            4.0000, 4.0000, 4.0000, 1.0000, 0.5000, 0.5000,0.5000, 
                                            2,2,2,2,2,2, 2,2,2,2,2,2,
                                            4.0000, 4.0000, 4.0000, 1.0000, 0.5000, 0.5000, 0.5000,
                                            2,2,2,2,2,2, 2,2,2,2,2,2]

            
            elif self.cfg.control.control_type == 'X':

                dof_props["driveMode"][8:20].fill(gymapi.DOF_MODE_POS)
                dof_props["stiffness"][8:20].fill(20.0)
                dof_props["damping"][8:20].fill(2.0)

                dof_props["driveMode"][27:39].fill(gymapi.DOF_MODE_POS)
                dof_props["stiffness"][27:39].fill(20.0)
                dof_props["damping"][27:39].fill(2.0)
        
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            if i == 0:
                self.default_com = copy.deepcopy(body_props[0].com)
                self.default_body_com = copy.deepcopy(body_props[self.torso_body_index].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            ######## record body mass
            self.body_masses = torch.zeros(97, dtype=torch.float32, device=self.device)
            for body_idx  in range(97):
                self.body_masses[body_idx ] = float(body_props[body_idx ].mass)
      
             # Create table 
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(self.env_origins[i][0] + table_pos.x, self.env_origins[i][1] + table_pos.y, table_pos.z)
            table_actor = self.gym.create_actor(
                env_handle, table_asset, table_pose, "table", i , 1, 0
            )  
            
            table_props = self.gym.get_actor_rigid_shape_properties(env_handle, table_actor)
            table_props[0].friction = 0.5  # ? only one table shape in each env
            self.gym.set_actor_rigid_shape_properties(env_handle, table_actor, table_props)
            # set table's color to be dark gray
            self.gym.set_rigid_body_color(env_handle, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))

            ##Create obstacles
            table_center_x = self.env_origins[i][0] + table_pos.x
            table_center_y = self.env_origins[i][1] + table_pos.y
            obj_x_lower = table_center_x + self.obj_x_lower
            obj_x_upper = table_center_x + self.obj_x_upper
            obj_y_lower = table_center_y + self.obj_y_lower  + 2
            obj_y_upper = table_center_y + self.obj_y_upper  + 2

            obj_x = torch_rand_float(obj_x_lower, obj_x_upper, (1, 1), self.device).item()
            obj_y = torch_rand_float(obj_y_lower, obj_y_upper, (1, 1), self.device).item()
            obj_z = self._table_surface_z + 0.2


            # 设置物体姿态
            obj_pose = gymapi.Transform()
            obj_pose.p = gymapi.Vec3(obj_x, obj_y, obj_z)
            # 随机偏航角（-π/4 ~ π/4），shape=(1,1) 
            obj_yaw = torch_rand_float(-np.pi/4, np.pi/4, (1, 1), self.device).item()
            obj_pose.r = gymapi.Quat.from_euler_zyx(obj_yaw, 0, 0)

            asset_idx = self.env_to_asset_idx[i].item()  
            current_asset = self.object_asset_pool[asset_idx] 
            current_urdf_file = os.path.basename(selected_urdf_paths[asset_idx])  # 用于日志

            obj_actor = self.gym.create_actor(
                env_handle,
                current_asset,  
                obj_pose,
                f"object_{current_urdf_file}",
                i,  
                2, 0
            )

            # 结束聚合模式
            self.gym.end_aggregate(env_handle)
            # Store the created env pointers
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

    
 
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
       
        self.imu_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.asset.imu_link)
        
        head_name = self.cfg.asset.head_name
        self.head_indice = torch.zeros(len(head_name), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(head_name)):
            self.head_indice = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], head_name[i])
        
        self.tip_names = self.cfg.asset.tip_names
        self.tip_indice =  torch.zeros(len(self.tip_names), dtype=torch.int, device=self.device, requires_grad=False)
        for i in range(len(self.tip_names)):
            self.tip_indice[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.tip_names[i])
       
        self.elbow_wrist_names = self.cfg.asset.elbow_wrist_names
        self.elbow_wrist_indice =  torch.zeros(len(self.elbow_wrist_names), dtype=torch.int, device=self.device, requires_grad=False)
        for i in range(len(self.elbow_wrist_names)):
            self.elbow_wrist_indice[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.elbow_wrist_names[i])
    
        self.waist_indice = self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], self.cfg.asset.waist_link_name 
                )
        
        self.hand_base_names = self.cfg.asset.hand_base_names
        self.hand_base_indice =  torch.zeros(len(self.hand_base_names), dtype=torch.int, device=self.device, requires_grad=False)
        for i in range(len(self.hand_base_names)):
            self.hand_base_indice[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.hand_base_names[i])
  
        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
            

    def _get_env_origins(self):
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.upper_interval = np.ceil(self.cfg.domain_rand.upper_interval_s / self.dt)
    #------------  draw target pos and force------

    def _draw_commands_and_tips(self):
        """
        绘制:
        1) 四个 command 点（waist 坐标系 → world）
        2) 两个手腕姿态（local rpy → quaternion → world）
        3) 10 个 tip x-force（local tip frame → world → 线段）
        最终完整修复：
        - 修复姿态位置错误：调整pos_local索引（从0:2→2:4，避开左手肘/腕）
        - 移除 AxesGeometry 不兼容参数（tube_radius/color_x/y/z）
        - 修正 Transform 构造参数名（pos→p，rot→r）
        - 四元数顺序修正（x,y,z,w → w,x,y,z）+ 归一化
        - 增大坐标轴缩放（scale=0.4）提升可视性
        - 数据合法性校验 + 异常处理
        - 添加坐标验证打印，方便确认映射关系
        """
        self.gym.clear_lines(self.viewer)
        # ----------------------------------------------
        # 1. 基础：腰部刚体的世界姿态
        # ----------------------------------------------
        waist_state = self.rigid_body_states[:, [self.waist_indice], 0:7]
        waist_pos = waist_state[:,: ,0:3].expand(self.num_envs,4,3)     # (num_envs,1,3)
        waist_quat = waist_state[:,:, 3:7].expand(self.num_envs,4,4)   # (num_envs,1,4)

        # ----------------------------------------------
        # 2. commands 中提取 4 个点 + 2 个 rpy
        # ----------------------------------------------
        pos_local = self.commands[:, 5:5+12].reshape(self.num_envs, 4, 3)
        rpy_local = self.commands[:, 17:17+6].reshape(self.num_envs, 2, 3)

        # # 【新增】验证pos_local映射关系（运行一次后可注释）
        # if self.num_envs > 0:
        #     pos_local_sample = pos_local[0].cpu().numpy()
        #     print("\n=== pos_local 4个点坐标（第一个环境）===")
        #     for idx, point in enumerate(pos_local_sample):
        #         print(f"pos_local[{idx}]: {point}")
        #     # 若有肘/腕索引，可添加真实坐标对比（根据你的代码补充indice）
        #     if hasattr(self, 'left_elbow_indice') and hasattr(self, 'left_wrist_indice'):
        #         left_elbow = self.rigid_body_states[0, self.left_elbow_indice, 0:3].cpu().numpy()
        #         left_wrist = self.rigid_body_states[0, self.left_wrist_indice, 0:3].cpu().numpy()
        #         print("左手肘真实坐标:", left_elbow)
        #         print("左手腕真实坐标:", left_wrist)

        # ----------------------------------------------
        # 3. 四个点 local → world
        # ----------------------------------------------
        pos_world = quat_apply(
            waist_quat,
            pos_local
        ) + waist_pos

        # 4 个点用不同颜色（保留原逻辑）
        point_colors = [(1,0,0), (0,1,0), (0,0,1), (1,0,1)]
        point_geoms = [
            gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=c)
            for c in point_colors
        ]

        # ----------------------------------------------
        # 4. 手腕 rpy → local quat → world quat
        # ----------------------------------------------
        wrist_local_quat = quat_from_euler_xyz(
            rpy_local[:, :, 0],
            rpy_local[:, :, 1],
            rpy_local[:, :, 2]
        )  # (num_envs,2,4)
        waist_quat_crat = waist_quat[:,[0,1],:]
        wrist_world_quat = quat_mul(
            waist_quat_crat,
            wrist_local_quat
        )  # (num_envs,2,4)


        reflect_induce =[1,3] 
        wrist_pos_world = quat_apply(
            waist_quat_crat,
            pos_local[:, reflect_induce, :]  
        ) + waist_pos[:,[0,1],:]

        axes_geom = gymutil.AxesGeometry(scale=0.4)


        # ----------------------------------------------
        # 6. 绘制所有对象（最终兼容版）
        # ----------------------------------------------
        for i in range(self.num_envs):

            # ---- 1. 四个 command 点（修正Transform参数名）----
            for j in range(4):
                pos = pos_world[i, j]
                pose = gymapi.Transform(p=gymapi.Vec3(*pos.cpu()), r=None)
                gymutil.draw_lines(point_geoms[j], self.gym, self.viewer, self.envs[i], pose)

            # ---- 2. 两个姿态坐标轴（核心兼容修复）----
            for j in range(2):
                # 获取当前手腕的位置和姿态
                pos = wrist_pos_world[i, j]
                quat = wrist_world_quat[i, j]

                # 校验1：位置是否合法（NaN/超出场景范围）
                if torch.isnan(pos).any() or torch.abs(pos).max() > 10.0:
                    print(f"[绘制警告] Env{i} Wrist{j} 位置异常: {pos.cpu().numpy()}, 跳过绘制")
                    continue

                # 校验2：四元数是否合法（NaN/模长过小）
                quat_norm = torch.norm(quat)
                if torch.isnan(quat).any() or quat_norm < 0.1:
                    print(f"[绘制警告] Env{i} Wrist{j} 姿态异常（模长={quat_norm:.2f}）, 跳过绘制")
                    continue

                # 四元数归一化（避免姿态畸变）
                quat_normalized = quat / quat_norm

                # 转换为Isaac Gym兼容的四元数（w,x,y,z顺序）
                gym_quat = gymapi.Quat(
                    w=quat_normalized[3].item(),  # w分量（原第4位）
                    x=quat_normalized[0].item(),  # x分量（原第1位）
                    y=quat_normalized[1].item(),  # y分量（原第2位）
                    z=quat_normalized[2].item()   # z分量（原第3位）
                )

                # 最终兼容：Transform仅用p/r参数
                pose = gymapi.Transform(
                    p=gymapi.Vec3(pos[0].item(), pos[1].item(), pos[2].item()),
                    r=gym_quat
                )

                # 绘制坐标轴（捕获所有异常）
                try:
                    gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)
                except Exception as e:
                    print(f"[绘制错误] Env{i} Wrist{j} 绘制失败: {str(e)}")
                    continue




    #------------ reward functions----------------
    def _reward_tracking_x_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :1] - self.base_lin_vel[:, :1]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_y_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, 1:2] - self.base_lin_vel[:, 1:2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    # def _reward_tracking_positions(self):
    #     wp = self.rigid_body_states[:, self.waist_indice, :3]    # (K,3)
    #     wq = self.rigid_body_states[:, self.waist_indice, 3:7]   # (K,4)

    #     gep = self.rigid_body_states[:, self.elbow_wrist_indice, :3]  # (K,4,3)
    #     # geq = self.rigid_body_states[:, self.elbow_wrist_indice, 3:7] # (K,4,4)
    #     geq = self.rigid_body_states[:, [self.elbow_wrist_indice[1],self.elbow_wrist_indice[3]], 3:7] 

    #     # ================== global → waist ==================
    #     pos_w, quat_w = G1Inspire.global_to_waist(
    #         gep,     # (K,4,3)
    #         geq,     # (K,4,4)
    #         wp.unsqueeze(1),  # (K,1,3)
    #         wq.unsqueeze(1)   # (K,1,4)
    #     )  # 输出: (K,4,3), (K,4,4)
    #     pos_error = torch.sum(torch.square(self.commands[:, 5:5+12] - pos_w.reshape(self.num_envs,-1)), dim=1)
    #     reward_pos = torch.exp(-pos_error/self.cfg.rewards.tracking_sigma)

    #     rpy_w = G1Inspire.quat_to_rpy(quat_w)
    #     rpy_error = torch.sum(torch.square(self.commands[:, 17:17+6] - rpy_w.reshape(self.num_envs,-1)), dim=1)
    #     reward_rpy  = torch.exp(-rpy_error/self.cfg.rewards.tracking_sigma)
    #     return reward_pos  +  reward_rpy
    
    # def  _reward_balance_to_force(self):
    #     tip_vel_change = self.tip_vel - self.last_tip_vel
    #     gap = torch.sum(torch.square(tip_vel_change.reshape(self.num_envs,-1)), dim=1)
    #     reward_pos = torch.exp(-gap/self.cfg.rewards.tracking_sigma)
    #     return reward_pos


    
    
