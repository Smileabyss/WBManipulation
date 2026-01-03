import torch
import numpy as np


class RobotArmKinematics:
    def __init__(self, body_x_block_range=[-0.1, 0.1], device='cpu', dtype=torch.float32):

        self.device = device
        self.dtype = dtype

        # ============================================================
        # 标准 Waist 坐标系原始数据
        # ============================================================
        self.shoulder_pos = {
            'left': torch.tensor([0.0, 0.1, 0.292], device=device, dtype=dtype),
            'right': torch.tensor([0.0, -0.1, 0.292], device=device, dtype=dtype)
        }

        # 肩 -> 肘 长度
        left_SE = torch.norm(torch.tensor([0, 0.147, 0.105]) - torch.tensor([0, 0.1, 0.292]))
        right_SE = torch.norm(torch.tensor([0, -0.147, 0.105]) - torch.tensor([0, -0.1, 0.292]))

        # 肘 -> 基座长度
        left_EB = torch.norm(torch.tensor([0.233, 0.150, 0.072]) - torch.tensor([0, 0.147, 0.105]))
        right_EB = torch.norm(torch.tensor([0.233, -0.151, 0.072]) - torch.tensor([0, -0.147, 0.105]))

        self.joint_lengths = {
            'left_shoulder_elbow': left_SE.item(),
            'left_elbow_base': left_EB.item(),
            'right_shoulder_elbow': right_SE.item(),
            'right_elbow_base': right_EB.item(),
        }

        # 身体阻挡范围
        self.body_x_min = torch.tensor(body_x_block_range[0], device=device, dtype=dtype)
        self.body_x_max = torch.tensor(body_x_block_range[1], device=device, dtype=dtype)
        self.min_body_y = torch.tensor(-0.1, device=device, dtype=dtype)
        self.max_body_y = torch.tensor(0.1, device=device, dtype=dtype)

    # ======================================================================
    # 随机旋转方向的 Euler
    # ======================================================================
    def _random_euler_along_direction(self, direction):
        direction = direction / (torch.norm(direction) + 1e-8)
        angle = torch.rand(1, device=self.device) * 2 * np.pi

        k = direction
        K = torch.tensor([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ], device=self.device, dtype=self.dtype)

        R = torch.eye(3, device=self.device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

        yaw = torch.atan2(R[1, 0], R[0, 0])
        pitch = torch.atan2(-R[2, 0], torch.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        roll = torch.atan2(R[2, 1], R[2, 2])
        return torch.stack([roll, pitch, yaw])

    # ======================================================================
    # 这里是新增的 ———— 完整的全局重置
    # ======================================================================
    def global_random_reset(self, arm_side='left'):
        shoulder = self.shoulder_pos[arm_side]
        SE_len = self.joint_lengths[f"{arm_side}_shoulder_elbow"]
        EB_len = self.joint_lengths[f"{arm_side}_elbow_base"]

        # -------- 肘部：直接在肩部球面均匀采样 --------
        dir_e = torch.randn(3, device=self.device)
        dir_e = dir_e / (torch.norm(dir_e) + 1e-8)
        elbow = shoulder + dir_e * SE_len

        # y 侧边避障
        if arm_side == 'left':
            if elbow[1] <= self.max_body_y:
                elbow[1] = self.max_body_y + 0.001
        else:
            if elbow[1] >= self.min_body_y:
                elbow[1] = self.min_body_y - 0.001

        # -------- 基座：在肘部球面随机采样 --------
        dir_b = torch.randn(3, device=self.device)
        dir_b = dir_b / (torch.norm(dir_b) + 1e-8)
        base = elbow + dir_b * EB_len

        # -------- X 避障 --------
        if self.body_x_min <= base[0] <= self.body_x_max:
            # 推到左右侧
            center = (self.body_x_min + self.body_x_max) / 2
            if base[0] < center:
                base[0] = self.body_x_min - 0.002
            else:
                base[0] = self.body_x_max + 0.002

        # -------- 姿态 --------
        euler = self._random_euler_along_direction(base - elbow)
        return elbow, base, euler

    # ======================================================================
    # 主函数（加入周期性全局重置）
    # ======================================================================
    def generate_next_position(self, prev_elbow, prev_base,
                               arm_side='left', max_step=0.1,
                               step_count=0, reset_interval=30):

        # ---------- 周期性全局随机重置 ----------
        if step_count % reset_interval == 0:
            return self.global_random_reset(arm_side)

        prev_elbow = prev_elbow.to(self.device)
        prev_base = prev_base.to(self.device)

        shoulder = self.shoulder_pos[arm_side]
        SE_len = self.joint_lengths[f"{arm_side}_shoulder_elbow"]
        EB_len = self.joint_lengths[f"{arm_side}_elbow_base"]

        # ===========================================================
        # 1. 肘部随机步进
        # ===========================================================
        rnd = torch.randn(3, device=self.device)
        rnd = rnd / (torch.norm(rnd) + 1e-8)
        step = torch.rand(1, device=self.device) * max_step

        new_elbow = prev_elbow + rnd * step

        v = new_elbow - shoulder
        v = v / (torch.norm(v) + 1e-8) * SE_len
        new_elbow = shoulder + v

        # y 避障
        if arm_side == 'left':
            if new_elbow[1] <= self.max_body_y:
                new_elbow[1] = self.max_body_y + 0.001
        else:
            if new_elbow[1] >= self.min_body_y:
                new_elbow[1] = self.min_body_y - 0.001

        # ===========================================================
        # 2. 基座随机步进
        # ===========================================================
        rnd2 = torch.randn(3, device=self.device)
        rnd2 = rnd2 / (torch.norm(rnd2) + 1e-8)
        step2 = torch.rand(1, device=self.device) * max_step

        new_base = prev_base + rnd2 * step2

        v2 = new_base - new_elbow
        v2 = v2 / (torch.norm(v2) + 1e-8) * EB_len
        new_base = new_elbow + v2

        # ===========================================================
        # 3. X 避障
        # ===========================================================
        if self.body_x_min <= new_base[0] <= self.body_x_max:
            center = (self.body_x_min + self.body_x_max) / 2
            if new_base[0] < center:
                new_base[0] = self.body_x_min - 0.001
            else:
                new_base[0] = self.body_x_max + 0.001

        # ===========================================================
        # 4. 姿态
        # ===========================================================
        direction = new_base - new_elbow
        new_euler = self._random_euler_along_direction(direction)

        return new_elbow, new_base, new_euler
