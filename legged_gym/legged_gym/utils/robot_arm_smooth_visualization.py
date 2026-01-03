import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from robot_arm_kinematics import RobotArmKinematics

def main():
    # 配置PyTorch设备（自动检测GPU）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== Using Device: {device} ===")
    
    # 1. Initialize kinematics library (PyTorch version)
    kinematics = RobotArmKinematics(body_x_block_range=[-0.1, 0.1], device=device, dtype=torch.float32)

    # 2. Initial state (in Waist coordinate system, Torch Tensor格式)
    waist_positions = torch.tensor([0,0,0],dtype=float)

    left_elbow_waist_t0 = torch.tensor([0,0.147,0.105],dtype=float)
    left_base_waist_t0 =  torch.tensor([0.233,0.150,0.072],dtype=float)
    
    right_elbow_waist_t0 =  torch.tensor([0,-0.147,0.105],dtype=float)
    right_base_waist_t0 =  torch.tensor([0.233,-0.151,0.072],dtype=float)
    
    left_shoulder_waist = kinematics.shoulder_pos['left']
    right_shoulder_waist = kinematics.shoulder_pos['right']
    
    # 3. Initialize visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    
    # Set axis limits（转换为numpy计算）
    padding = 0.3
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    
    # Axis labels (English only)
    ax.set_xlabel('X-axis (Waist Coordinate System)')
    ax.set_ylabel('Y-axis (Waist Coordinate System)')
    ax.set_zlabel('Z-axis (Waist Coordinate System)')
    ax.set_title(f'Robot Arm Motion Simulation (PyTorch Version)\nDevice: {device}')
    
    # 4. Plot elements (English labels)
    # Left arm
    left_upper, = ax.plot([], [], [], 'b-', linewidth=3, label='Left Upper Arm')
    left_lower, = ax.plot([], [], [], 'b--', linewidth=3, label='Left Lower Arm')
    left_shoulder_scatter = ax.scatter([], [], [], c='blue', s=150, marker='o', label='Left Shoulder (Fixed)')
    left_elbow_scatter = ax.scatter([], [], [], c='cyan', s=150, marker='^', label='Left Elbow')
    left_base_scatter = ax.scatter([], [], [], c='darkblue', s=150, marker='s', label='Left Base')
    
    # Right arm
    right_upper, = ax.plot([], [], [], 'r-', linewidth=3, label='Right Upper Arm')
    right_lower, = ax.plot([], [], [], 'r--', linewidth=3, label='Right Lower Arm')
    right_shoulder_scatter = ax.scatter([], [], [], c='red', s=150, marker='o', label='Right Shoulder (Fixed)')
    right_elbow_scatter = ax.scatter([], [], [], c='magenta', s=150, marker='^', label='Right Elbow')
    right_base_scatter = ax.scatter([], [], [], c='darkred', s=150, marker='s', label='Right Base')
    
    # Waist origin
    ax.scatter(0, 0, 0, c='orange', s=300, marker='*', label='Waist Origin (0,0,0)')
    
    # Body range (Y ∈ [-0.1, 0.1] in Waist coordinate system)
    # 绘图部分仍用numpy（matplotlib不支持torch tensor）
    x_body = torch.linspace(torch.tensor(ax.get_xlim()[0]), torch.tensor(ax.get_xlim()[1]), 5).cpu().numpy()
    z_body = torch.linspace(torch.tensor(ax.get_zlim()[0]), torch.tensor(ax.get_zlim()[1]), 5).cpu().numpy()
    X_body, Z_body = plt.np.meshgrid(x_body, z_body)  # 用matplotlib的numpy兼容
    Y_min = plt.np.full_like(X_body, kinematics.min_body_y.item())
    Y_max = plt.np.full_like(X_body, kinematics.max_body_y.item())
    ax.plot_wireframe(X_body, Y_min, Z_body, color='gray', alpha=0.2, linestyle=':')
    ax.plot_wireframe(X_body, Y_max, Z_body, color='gray', alpha=0.2, linestyle=':')
    
    ax.legend()
    plt.tight_layout()
    
    # 5. Core loop
    print("\n=== Motion Simulation Started (Press Ctrl+C to stop) ===")
    print(f"Constraints: Fixed Joint Lengths | Elbow Outside Body (Y ∈ [-0.1, 0.1]) | Base Outside X Range [{kinematics.body_x_min.item():.3f}, {kinematics.body_x_max.item():.3f}] | Max Step: 0.05")
    
    # Initialize previous state (Torch Tensor)
    prev_left_elbow = left_elbow_waist_t0
    prev_left_base = left_base_waist_t0
    prev_left_quat = None
    
    prev_right_elbow = right_elbow_waist_t0
    prev_right_base = right_base_waist_t0
    prev_right_quat = None
    
    step = 0
    try:
        while True:
            step += 1
            # Generate next position (Torch Tensor运算，全程在GPU/CPU上)
            left_elbow, left_base, left_quat = kinematics.generate_next_position(
                prev_left_elbow, prev_left_base, step_count=step,arm_side='left'
            )
            right_elbow, right_base, right_quat = kinematics.generate_next_position(
                prev_right_elbow, prev_right_base, step_count=step,arm_side='right'
            )
            
            # Update plot（转换为numpy进行绘图）
            # Left arm
            left_shoulder_np = left_shoulder_waist.cpu().numpy()
            left_elbow_np = left_elbow.cpu().numpy()
            left_base_np = left_base.cpu().numpy()
            
            left_upper.set_data([left_shoulder_np[0], left_elbow_np[0]], [left_shoulder_np[1], left_elbow_np[1]])
            left_upper.set_3d_properties([left_shoulder_np[2], left_elbow_np[2]])
            left_lower.set_data([left_elbow_np[0], left_base_np[0]], [left_elbow_np[1], left_base_np[1]])
            left_lower.set_3d_properties([left_elbow_np[2], left_base_np[2]])
            
            # Right arm
            right_shoulder_np = right_shoulder_waist.cpu().numpy()
            right_elbow_np = right_elbow.cpu().numpy()
            right_base_np = right_base.cpu().numpy()
            
            right_upper.set_data([right_shoulder_np[0], right_elbow_np[0]], [right_shoulder_np[1], right_elbow_np[1]])
            right_upper.set_3d_properties([right_shoulder_np[2], right_elbow_np[2]])
            right_lower.set_data([right_elbow_np[0], right_base_np[0]], [right_elbow_np[1], right_base_np[1]])
            right_lower.set_3d_properties([right_elbow_np[2], right_base_np[2]])
            
            # Update scatter points
            left_shoulder_scatter._offsets3d = ([left_shoulder_np[0]], [left_shoulder_np[1]], [left_shoulder_np[2]])
            left_elbow_scatter._offsets3d = ([left_elbow_np[0]], [left_elbow_np[1]], [left_elbow_np[2]])
            left_base_scatter._offsets3d = ([left_base_np[0]], [left_base_np[1]], [left_base_np[2]])
            right_shoulder_scatter._offsets3d = ([right_shoulder_np[0]], [right_shoulder_np[1]], [right_shoulder_np[2]])
            right_elbow_scatter._offsets3d = ([right_elbow_np[0]], [right_elbow_np[1]], [right_elbow_np[2]])
            right_base_scatter._offsets3d = ([right_base_np[0]], [right_base_np[1]], [right_base_np[2]])
            
            plt.draw()
            plt.pause(0.01)
            
            # Print state every 5 steps（转换为numpy打印）
            if step % 5 == 0:
                print(f"\nStep {step} | Waist Coordinate System (Torch Tensor):")
                print(f"Left Elbow: {left_elbow_np.round(3)} | Left Base Quaternion: {left_quat.cpu().numpy().round(3)}")
                print("left_base:",left_base)
                print(f"Right Elbow: {right_elbow_np.round(3)} | Right Base Quaternion: {right_quat.cpu().numpy().round(3)}")
                print("right_base:",right_base)
            
            # Update previous state for next iteration
            prev_left_elbow, prev_left_base, prev_left_quat = left_elbow, left_base, left_quat
            prev_right_elbow, prev_right_base, prev_right_quat = right_elbow, right_base, right_quat
            
            time.sleep(0.3)
            
    except KeyboardInterrupt:
        print("\nSimulation Stopped")
    finally:
        plt.ioff()
        plt.close()

if __name__ == "__main__":
    main()
