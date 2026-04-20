import argparse
import os
import csv
import glob
import yaml
from datetime import datetime

# 1. 엔진 시동 (절대 위치 고정)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--load_run", type=str, required=True)
parser.add_argument("--num_steps", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =================================================================
# 2. 엔진 가동 후 필수 모듈 수입 (pxr 에러 방지)
import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# 환경 생성
task_name = "Isaac-Velocity-Rough-Anymal-C-Play-v0"
env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make(task_name, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

log_root = os.path.join(os.path.dirname(__file__), "../../logs/rsl_rl/anymal_c_rough")
run_dir = os.path.join(log_root, args_cli.load_run)

# [Fix 1] 학습할 때 썼던 알고리즘 설정(agent.yaml) 완벽 로드
config_path = os.path.join(run_dir, "params", "agent.yaml")
with open(config_path, "r") as f:
    agent_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

# [Fix 2] 폴더가 아닌 진짜 신경망 파일(.pt) 자동 탐색
model_files = glob.glob(os.path.join(run_dir, "model_*.pt"))
model_files.sort(key=os.path.getmtime)
resume_path = model_files[-1]

# 러닝머신 가동
runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=args_cli.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=args_cli.device)

# CSV 저장 경로
save_dir = os.path.expanduser(f"~/IsaacLab/datasets/anymal_c_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(save_dir, exist_ok=True)
csv_path = os.path.join(save_dir, "control_data.csv")

obs, _ = env.reset()

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([f"joint_pos_{i}" for i in range(12)] + 
                    [f"joint_vel_{i}" for i in range(12)] + 
                    [f"joint_torque_{i}" for i in range(12)] + 
                    ["base_pos_x", "base_pos_y", "base_pos_z"] + 
                    ["base_quat_w", "base_quat_x", "base_quat_y", "base_quat_z"] + 
                    ["step"])

    for step in range(args_cli.num_steps):
        with torch.no_grad():
            action = policy(obs)
        obs, _, _, _ = env.step(action)

        robot_data = env.unwrapped.scene["robot"].data
        joint_pos = robot_data.joint_pos[0].cpu().numpy()
        joint_vel = robot_data.joint_vel[0].cpu().numpy()
        joint_torque = robot_data.applied_torque[0].cpu().numpy()
        base_pos = robot_data.root_pos_w[0].cpu().numpy()
        base_quat = robot_data.root_quat_w[0].cpu().numpy()

        writer.writerow(list(joint_pos) + list(joint_vel) + list(joint_torque) + 
                        list(base_pos) + list(base_quat) + [step])

print(f"\n[INFO] 🚀 드디어 데이터 저장 완료!: {csv_path}\n")
env.close()
simulation_app.close()
