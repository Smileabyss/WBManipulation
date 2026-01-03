import os
from typing import List, Optional
import numpy as np
import pickle


def get_all_urdf_paths(main_folder: str, print_log: bool = True) -> List[str]:
    """
    遍历主文件夹下的所有子文件夹，收集每个子文件夹中的URDF文件绝对路径（假设每个子文件夹仅1个URDF）

    Args:
        main_folder: 主文件夹路径（如 "/home/cyrus/ManipTrans/data/OakInk-v2/coacd_object_preview/align_ds"）
        print_log: 是否打印日志（默认打印，方便调试）

    Returns:
        List[str]: 所有有效URDF文件的绝对路径列表
    """
    # 存储最终URDF路径
    urdf_paths = []

    # 验证主文件夹是否存在
    if not os.path.exists(main_folder):
        if print_log:
            print(f"❌ 主文件夹不存在：{main_folder}")
        return urdf_paths

    # 遍历主文件夹下的所有子文件夹（递归遍历所有层级）
    for root, dirs, files in os.walk(main_folder):
        # 筛选当前文件夹下的URDF文件（忽略大小写，支持 .urdf 和 .URDF）
        urdf_files = [f for f in files if f.lower().endswith(".urdf")]
        
        # 处理不同情况
        if len(urdf_files) == 0:
            # 无URDF文件，跳过并提示
            if print_log:
                print(f"⚠️  子文件夹无URDF文件，跳过：{root}")
            continue
        
        elif len(urdf_files) > 1:
            # 多个URDF文件，提示并取第一个（或可修改为取所有）
            if print_log:
                print(f"⚠️  子文件夹存在多个URDF文件，仅取第一个：{root}")
                for idx, f in enumerate(urdf_files, 1):
                    print(f"    {idx}. {f}")
            selected_urdf = urdf_files[0]
        
        else:
            # 仅1个URDF文件，直接选中
            selected_urdf = urdf_files[0]
        
        # 拼接URDF文件的绝对路径
        urdf_abs_path = os.path.abspath(os.path.join(root, selected_urdf))
        urdf_paths.append(urdf_abs_path)
        
        # 打印日志（可选）
        if print_log:
            print(f"✅ 找到URDF：{urdf_abs_path}")

    # 最终统计
    if print_log:
        total_valid = len(urdf_paths)
        print(f"\n📊 遍历完成：共找到 {total_valid} 个有效URDF文件")

    return urdf_paths

def get_all_pkl_paths(main_folder: str, print_log: bool = True) -> List[str]:
    """
    遍历主文件夹下的所有子文件夹，收集每个子文件夹中的PKL文件绝对路径（假设每个子文件夹仅1个PKL）
    
    Args:
        main_folder: 主文件夹路径（如 "/home/cyrus/data/pkl_files"）
        print_log: 是否打印日志（默认打印，方便调试）
    
    Returns:
        List[str]: 所有有效PKL文件的绝对路径列表
    """
    # 存储最终PKL路径
    pkl_paths = []
    
    # 验证主文件夹是否存在
    if not os.path.exists(main_folder):
        if print_log:
            print(f"❌ 主文件夹不存在：{main_folder}")
        return pkl_paths
    
    # 遍历主文件夹下的所有子文件夹（递归遍历所有层级）
    for root, dirs, files in os.walk(main_folder):
        # 筛选当前文件夹下的PKL文件（忽略大小写，支持 .pkl 和 .PKL）
        pkl_files = [f for f in files if f.lower().endswith(".pkl")]
        
        # 处理不同情况
        if len(pkl_files) == 0:
            # 无PKL文件，跳过并提示
            if print_log:
                print(f"⚠️  子文件夹无PKL文件，跳过：{root}")
            continue
        
        elif len(pkl_files) > 1:
            # 多个PKL文件，提示并取第一个（可修改为取所有，见扩展说明）
            if print_log:
                print(f"⚠️  子文件夹存在多个PKL文件，仅取第一个：{root}")
                for idx, f in enumerate(pkl_files, 1):
                    print(f"    {idx}. {f}")
            selected_pkl = pkl_files[0]
        
        else:
            # 仅1个PKL文件，直接选中
            selected_pkl = pkl_files[0]
        
        # 拼接PKL文件的绝对路径
        pkl_abs_path = os.path.abspath(os.path.join(root, selected_pkl))
        pkl_paths.append(pkl_abs_path)
        
        # 打印日志（可选）
        if print_log:
            print(f"✅ 找到PKL：{pkl_abs_path}")
    
    # 最终统计
    if print_log:
        total_valid = len(pkl_paths)
        print(f"\n📊 遍历完成：共找到 {total_valid} 个有效PKL文件")
    
    return pkl_paths




def generate_finger_grasp_trajectory(
    selected_finger_ids: list = [0],  # 选中的手指索引列表（0-3，支持1-4个手指，如[0]、[0,1]、[0,1,2,3]）
    target_angle_ratio: float = 0.8,  # 弯曲程度（0=不弯，1=最大弯曲，不超过upper_bound）
    num_steps: int = 100,  # 插值步数（length=num_steps）
    hand_angle_bounds: np.ndarray = None,  
    save_path: str = "grasp_trajectory.npy"  # 轨迹保存路径
) -> np.ndarray:
    """
    生成大拇指+任意1-4个手指弯曲的100步插值轨迹（NumPy实现）
    输出形状：[length, actdim=39]（步数×39维动作）
    
    参数：
        selected_finger_ids: 选中的手指索引列表（0-3对应4个手指，支持1-4个元素，如[0]、[0,2]、[1,2,3]）
        target_angle_ratio: 弯曲程度（0~1，对应[lower_bound, upper_bound]的比例）
        num_steps: 轨迹总步数（length=num_steps，默认100）
        hand_angle_bounds: 关节边界数组（12,2），格式：[关节数, [下界, 上界]]
        save_path: 生成轨迹的保存路径（.npy格式）
    
    返回：
        action_trajectory: 轨迹数组，形状[num_steps, 39]
    """
    # -------------------------- 输入验证 --------------------------
    assert hand_angle_bounds is not None, "必须传入hand_angle_bounds（12,2）"
    assert hand_angle_bounds.shape == (12, 2), f"hand_angle_bounds形状需为(12,2)，当前为{hand_angle_bounds.shape}"
    assert isinstance(selected_finger_ids, list), "selected_finger_ids必须是列表（如[0]、[0,1]）"
    assert 1 <= len(selected_finger_ids) <= 4, "选中的手指数量必须在1-4之间"
    for finger_id in selected_finger_ids:
        assert 0 <= finger_id <= 3, f"手指索引必须在0-3之间，当前存在无效索引：{finger_id}"
    
    # 1. 计算有效关节边界（上界=原上界×0.6，与原代码一致）
    lower_bound = hand_angle_bounds[:, 0].copy()  # (12,)：12个关节的下界
    upper_bound = hand_angle_bounds[:, 1] * 0.6  # (12,)：12个关节的有效上界（×0.6）
    
    # 2. 定义运动关节索引（选中的所有手指+大拇指）
    moving_joints = []
    # 遍历选中的每个手指，收集其2个关节索引（前8个关节：4手指×2关节/手指）
    for finger_id in selected_finger_ids:
        finger_joints = [finger_id * 2, finger_id * 2 + 1]
        moving_joints.extend(finger_joints)
    # 加入大拇指的4个关节索引（后4个关节：8-11）
    thumb_joints = list(range(8, 12))
    moving_joints.extend(thumb_joints)
    # 去重（防止极端情况重复输入同一手指）
    moving_joints = list(set(moving_joints))
    
    # 3. 定义初始状态和目标状态
    # 初始状态：所有关节处于下界（未弯曲）
    init_joints = lower_bound.copy()
    # 目标状态：运动关节弯曲到目标角度，其他关节保持下界
    target_joints = lower_bound.copy()
    for joint_idx in moving_joints:
        # 目标角度 = 下界 + 比例×(有效上界-下界)（确保在安全范围内）
        target_joints[joint_idx] = lower_bound[joint_idx] + target_angle_ratio * (
            upper_bound[joint_idx] - lower_bound[joint_idx]
        )
    
    # 4. 生成100步线性插值轨迹（关节角度轨迹）
    # 形状：[num_steps, 12] → 100步，每步12个关节角度
    joint_trajectory = np.linspace(
        init_joints, target_joints, num_steps, axis=0, dtype=np.float32
    )
    
    # 5. 构建完整动作轨迹（形状[num_steps, 39]，符合要求）
    action_trajectory = np.zeros((num_steps, 24 + 15), dtype=np.float32)  # [length, 39]
    
    # 6. 填充动作的目标列（8:20和27:39，各12列，对应12个关节）
    action_trajectory[:, 8:8+12] = joint_trajectory  # 第一个关节区间（8~19列）
    action_trajectory[:, 27:27+12] = joint_trajectory  # 第二个关节区间（27~38列）
    
    # 7. 保存轨迹
    np.save(save_path, action_trajectory)
    print(f"轨迹已保存到：{save_path}")
    print(f"轨迹形状：{action_trajectory.shape}（length={num_steps}, actdim=39）")
    print(f"选中的手指：{selected_finger_ids}（对应关节索引：{moving_joints[:-4]}）")
    print(f"运动关节总数：{len(moving_joints)}（{len(selected_finger_ids)}个手指×2关节 + 大拇指4关节）")
    
    return action_trajectory

def data_process(left_adr, right_adr, adr_save):
    with open(left_adr, "rb") as f:
        data_left = pickle.load(f)
    with open(right_adr, "rb") as f:
        data_right = pickle.load(f)

    data_left = data_left["opt_dof_pos"]
    data_right = data_right["opt_dof_pos"]
    print("length:",len(data_right))


    upper_actions = np.zeros((len(data_left), 24+15),dtype=np.float32)

    upper_actions[:,1:1+7] = data_left[:,0:7]

    idx = [8,10,12,14] 
    upper_actions[:,idx] = data_left[:,7:7+4]
    idx = [9,11,13,15] 
    upper_actions[:,idx] = data_left[:,7:7+4]*1.0843


    idx = [16,17] 
    upper_actions[:,idx] = data_left[:,11:11+2]
    upper_actions[:,18] = data_left[:,12]*0.8024
    upper_actions[:,19] = data_left[:,12]*0.8024*0.9487

    upper_actions[:,20:20+7] = data_right[:,0:7]

    idx = [27,29,31,33] 
    upper_actions[:,idx] = data_right[:,7:7+4]
    idx = [28,30,32,34] 
    upper_actions[:,idx] = data_right[:,7:7+4]*1.0843

    idx = [35,36] 
    upper_actions[:,idx] = data_right[:,11:11+2]
    upper_actions[:,18] = data_right[:,12]*0.8024
    upper_actions[:,19] = data_right[:,12]*0.8024*0.9487

    np.save(adr_save,upper_actions)

# # -------------------------- 用法示例 --------------------------
# if __name__ == "__main__":
#     # 你的关节角度边界（12,2）
#     action_bounds = np.array([
#         [ 0.0180,  1.4201],
#         [ 0.0393,  3.1007],
#         [ 0.0180,  1.4201],
#         [ 0.0393,  3.1007],
#         [ 0.0180,  1.4201],
#         [ 0.0393,  3.1007],
#         [ 0.0180,  1.4201],
#         [ 0.0393,  3.1007],
#         [ 0.0146,  1.1495],
#         [ 0.0073,  0.5791],
#         [ 0.0062,  0.4938],
#         [ 0.0393,  3.1007]
#     ])
    
#     trajectory1 = generate_finger_grasp_trajectory(
#         selected_finger_ids=[1,2,3],  # 选中第0个手指
#         target_angle_ratio=0.8,
#         num_steps=100,
#         hand_angle_bounds=action_bounds,
#         save_path="/home/cyrus/OpenHomie/Action_Trajs/base/grasp_trajectory_0111.npy"
#     )
    
