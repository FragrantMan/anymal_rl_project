import argparse
from isaaclab.app import AppLauncher

# ==========================================
# [파트 1] 터미널 주문서(argparse)와 물리 엔진 시동
# ==========================================
parser = argparse.ArgumentParser() # 터미널 주문을 받을 종업원 고용
parser.add_argument("--num_envs", type=int, default=1)        # 놀이터 개수 (기본 1개)
parser.add_argument("--load_run", type=str, required=True)    # [필수] 불러올 뇌(가중치) 폴더명
parser.add_argument("--num_steps", type=int, default=1000)    # 몇 스텝(프레임) 뽑을지

AppLauncher.add_app_launcher_args(parser) # 시뮬레이터 기본 설정 메뉴 추가
args_cli = parser.parse_args()            # 터미널 입력을 모아 '최종 주문서' 완성!

app_launcher = AppLauncher(args_cli)      # 주문서를 시동 도우미에게 전달
simulation_app = app_launcher.app         # 무거운 3D 물리 엔진(Omniverse) 전원 ON!


import torch
import csv
import os
from datetime import datetime
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

# ==========================================
# [파트 2] 로봇의 '놀이터(환경)'와 '뇌(가중치)' 세팅
# ==========================================
# 1. 험지 놀이터(Environment) 규격 설정 및 생성
env_cfg = parse_env_cfg(
    "Isaac-Velocity-Rough-Anymal-C-Play-v0",
    device=args_cli.device,
    num_envs=args_cli.num_envs,
)
env = gym.make("Isaac-Velocity-Rough-Anymal-C-Play-v0", cfg=env_cfg)

# 2. 로봇의 뇌(체크포인트)를 찾아 러닝머신에 올리기
log_root = os.path.join(os.path.dirname(__file__), "../../logs/rsl_rl/anymal_c_rough")
runner = OnPolicyRunner(env, {}, log_dir=None, device=args_cli.device) # 러닝머신
runner.load(os.path.join(log_root, args_cli.load_run)) # 똑똑해진 뇌 이식 완료

# 3. [중요] 탐험(딴짓)을 끄고, 배운 대로만 완벽하게 걷는 모드(Inference) 추출
policy = runner.get_inference_policy(device=args_cli.device)


# ==========================================
# 데이터 저장 폴더 및 CSV 파일 준비
# ==========================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.expanduser(f"~/IsaacLab/datasets/anymal_c_{timestamp}")
os.makedirs(save_dir, exist_ok=True)
csv_path = os.path.join(save_dir, "control_data.csv")

obs, _ = env.reset() # 시뮬레이션 초기화 (로봇 출발선 정렬)

# 엑셀(CSV) 파일 쓰기 모드로 열기
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    
    # 엑셀 첫 줄(헤더)에 센서 이름들 적기
    writer.writerow(
        [f"joint_pos_{i}" for i in range(12)] +
        [f"joint_vel_{i}" for i in range(12)] +
        [f"joint_torque_{i}" for i in range(12)] +
        ["base_pos_x", "base_pos_y", "base_pos_z"] +
        ["base_quat_w", "base_quat_x", "base_quat_y", "base_quat_z"] +
        ["step"]
    )
    
    # ==========================================
    # [파트 3] 핵심 데이터 추출 루프 (지정한 스텝만큼 반복)
    # ==========================================
    for step in range(args_cli.num_steps):
        
        # 뇌(policy)가 현재 상태(obs)를 보고 다음 행동(action) 결정
        with torch.no_grad():
            action = policy(obs)
            
        # 시뮬레이터 1스텝(프레임) 재생!
        obs, _, _, _, _ = env.step(action)
        
        # 껍질(unwrapped)을 까서 무대(scene) 위 'robot'의 데이터(센서 캐비닛) 열기
        # [0] = 0번 로봇 타겟팅 / .cpu().numpy() = GPU 텐서를 일반 숫자로 번역!
        joint_pos = env.unwrapped.scene["robot"].data.joint_pos[0].cpu().numpy()
        joint_vel = env.unwrapped.scene["robot"].data.joint_vel[0].cpu().numpy()
        joint_torque = env.unwrapped.scene["robot"].data.applied_torque[0].cpu().numpy()
        base_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
        base_quat = env.unwrapped.scene["robot"].data.root_quat_w[0].cpu().numpy()
        
        # 번역된 숫자들을 엑셀 파일 한 줄로 예쁘게 써넣기
        writer.writerow(
            list(joint_pos) +
            list(joint_vel) +
            list(joint_torque) +
            list(base_pos) +
            list(base_quat) +
            [step]
        )
        
        # 100스텝마다 터미널에 생존 신고
        if step % 100 == 0:
            print(f"[INFO] Step {step}/{args_cli.num_steps} 완료")

print(f"[INFO] 데이터 저장 완료: {csv_path}")
env.close()             # 놀이터 폐쇄
simulation_app.close()  # 물리 엔진 전원 OFF
