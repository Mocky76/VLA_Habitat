#这是Gr00t专用的server文件
'''端口逻辑：
eval_main.py
  |
  | POST http://127.0.0.1:9000/act
  v
server_Gr00t.py   (uvicorn 监听 9000)
  |
  | Gr00tHTTPClient.post → http://127.0.0.1:8000/act
  v
Gr00t 原生模型服务 (监听 8000)
'''
'''启动方式：
uvicorn scripts.eval.server_Gr00t:app \
    --host 127.0.0.1 \
    --port 9000
'''

from fastapi import FastAPI
import numpy as np
from PIL import Image
from internnav.evaluator.gr00t_http_client import Gr00tHTTPClient   # 🔴 Gr00t 模型调用
import torch
    
app = FastAPI()

# ===== 🔴 Gr00t 模型加载（模型特有）=====
#注：这里正常启动模型即可，如果在本地有直接加载即可，这里是gr00t需要一个新端口运行推理服务
#注意eval_main.py中的url要指向本服务启动时的端口("http://127.0.0.1:9000/act")，而不是这个端口，这个端口只是用于原本的gr00t推理服务的，更换模型不需要这个端口
gr00t_client = Gr00tHTTPClient(
    url="http://127.0.0.1:8000/act"  # 真正的 Gr00t 服务
)

# ===== 构建Gr00t 模型输入 =====
def build_gr00t_obs(
    obs: dict,
):
    """
    obs: build_traj_request() 生成的 dict
    """
    rgb = obs["rgb"]
    gps = obs["gps"]
    yaw = obs["yaw"]
    camera_height = obs["camera_height"]
    instruction = obs["instruction"]

    image = Image.fromarray(rgb).convert("RGB")
    rgb_256 = np.array(image.resize((256, 256)))

    drone_state = np.array(
        [gps[0], gps[1], camera_height, yaw],
        dtype=np.float32
    ).reshape(1, 4)

    return {
        "video.ego_view": rgb_256.reshape(1, 256, 256, 3),
        "state.drone": drone_state,
        "annotation.human.action.task_description": [instruction],
    }

#接受请求并完成实际推理的主要逻辑
@app.post("/act")
def act(req: dict):
    """
    req = {
        "observation": build_traj_request(...) 的输出
    }
    """
    obs = req["observation"]

    # ===== 🔴 Gr00t 私有 preprocessing =====
    #构造输入
    gr00t_obs = build_gr00t_obs(obs)

    # 🔴 调用“真正的 Gr00t 服务”
    gr00t_output = gr00t_client.get_action(gr00t_obs)
    if isinstance(gr00t_output, dict) and "action.delta_pose" in gr00t_output:
        delta_poses = gr00t_output["action.delta_pose"]
    else:
        delta_poses = gr00t_output
    #到这里为止拿到了gr00t模型的输出

    # ===== 🔴 Gr00t 私有 postprocess =====
    #将输出转换成正确的格式，如删除一列(1,T,4)变为(1,T,3)或者相关数据比如角度的归一化
    dp_actions = gr00t_output_to_dp_actions(delta_poses)
    # print("dp_actions:",dp_actions)

    #将转换为正确格式的输出转换为离散动作action如[1，2，0]
    actions = traj_to_actions_Gr00t(dp_actions)

    # 返回 Habitat action id list
    return {
        "actions": actions
    }

#下面是用到的函数，按情况更改使用或新增即可
def gr00t_output_to_dp_actions(gr00t_out):
        """
        把 Gr00t 输出转换为 traj_to_actions_Gr00t 需要的格式。

        支持以下 gr00t_out 形式：
        - numpy array shape (T, 4)  # 单序列
        - numpy array shape (1, T, 4)  # batch=1
        - torch tensor 同上

        Gr00t 输出列 assumed: [dx, dy, dz, dyaw_degrees]
        返回: torch.Tensor shape (1, T, 3) dtype=float32, last dim = [dx, dy, dyaw_rad*12]
        """
        # 转 numpy / torch 兼容
        if isinstance(gr00t_out, torch.Tensor):
            arr = gr00t_out.detach().cpu().numpy()
        else:
            arr = np.asarray(gr00t_out)

        # 支持 (T,4) 或 (1,T,4) 或 (B,T,4)
        if arr.ndim == 2 and arr.shape[1] == 4:
            arr = arr[None, :, :]  # -> (1, T, 4)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            pass
        else:
            raise ValueError(f"Unsupported gr00t_out shape: {arr.shape}, expected (T,4) or (1,T,4) or (B,T,4)")

        # 取 (dx, dy, dyaw)
        # 列索引假设： 0=dx, 1=dy, 2=dz (unused), 3=dyaw (单位：度)
        dx = arr[:, :, 0].astype(np.float32)
        dy = arr[:, :, 1].astype(np.float32)
        dyaw_deg = arr[:, :, 3].astype(np.float32)

        # deg -> rad
        dyaw_rad = np.deg2rad(dyaw_deg)

        # 根据之前讨论，把 yaw 放大（保持和 traj_to_actions_Gr00t 里相同的放大逻辑）
        dyaw_rad = dyaw_rad * 1.0  # base conversion
        # 注意：traj_to_actions_Gr00t 会再做 *=12 的处理（如果你在函数里保留那一行）
        # 此处不再重复乘 12，除非你在 traj_to_actions_Gr00t 中没有加那一行。

        dp = np.stack([dx, dy, dyaw_rad], axis=-1)  # (B, T, 3)

        return torch.from_numpy(dp).float()  # 返回 torch Tensor (B, T, 3)

def traj_to_actions_Gr00t(dp_actions,use_discrate_action=True):
    def reconstruct_xy_from_delta(delta_xyt):
        """
        Input:
            delta_xyt: [B, T, 3], dx, dy are position increments in global coordinates, dθ is heading difference (not used for position)
            start_xy: [B, 2] starting point
        Output:
            xy: [B, T+1, 2] reconstructed global trajectory
        """
        start_xy = np.zeros((len(delta_xyt), 2))
        delta_xy = delta_xyt[:, :, :2]  # Take dx, dy parts
        cumsum_xy = np.cumsum(delta_xy, axis=1)  # [B, T, 2]

        B = delta_xyt.shape[0]
        T = delta_xyt.shape[1]
        xy = np.zeros((B, T + 1, 2))
        xy[:, 0] = start_xy
        xy[:, 1:] = start_xy[:, None, :] + cumsum_xy

        return xy

    def trajectory_to_discrete_actions_close_to_goal(trajectory, step_size=0.25, turn_angle_deg=15, lookahead=4):
        actions = []
        yaw = 0.0
        pos = trajectory[0]
        turn_angle_rad = np.deg2rad(turn_angle_deg)
        traj = trajectory
        goal = trajectory[-1]

        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        while np.linalg.norm(pos - goal) > 0.2:
            # Find the nearest trajectory point index to current position
            dists = np.linalg.norm(traj - pos, axis=1)
            nearest_idx = np.argmin(dists)
            # Look ahead a bit (not exceeding trajectory end)
            target_idx = min(nearest_idx + lookahead, len(traj) - 1)
            target = traj[target_idx]
            # Target direction
            target_dir = target - pos
            if np.linalg.norm(target_dir) < 1e-6:
                break
            target_yaw = np.arctan2(target_dir[1], target_dir[0])
            # Difference between current yaw and target yaw
            delta_yaw = normalize_angle(target_yaw - yaw)
            n_turns = int(round(delta_yaw / turn_angle_rad))
            if n_turns > 0:
                actions += [2] * n_turns
            elif n_turns < 0:
                actions += [3] * (-n_turns)
            yaw = normalize_angle(yaw + n_turns * turn_angle_rad)

            # Move forward one step
            next_pos = pos + step_size * np.array([np.cos(yaw), np.sin(yaw)])

            # If moving forward one step makes us farther from goal, stop
            if np.linalg.norm(next_pos - goal) > np.linalg.norm(pos - goal):
                break

            actions.append(1)
            pos = next_pos

        return actions

    # unnormalize
    dp_actions[:, :, :2] /= 4.0
    all_trajectory = reconstruct_xy_from_delta(dp_actions.float().cpu().numpy())
    trajectory = np.mean(all_trajectory, axis=0)
    if use_discrate_action:
        actions = trajectory_to_discrete_actions_close_to_goal(trajectory)
        return actions
    else:
        return trajectory