import argparse
import copy
import itertools
import json
import os
import random
import re
from collections import OrderedDict
from typing import Any

import habitat
import numpy as np
import quaternion
import torch
import tqdm
from depth_camera_filtering import filter_depth
from habitat import Env
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat_baselines.config.default import get_config as get_habitat_config
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from transformers.image_utils import to_numpy_array
from .gr00t_http_client import Gr00tHTTPClient


from internnav.model.utils.vln_utils import (
    chunk_token,
    image_resize,
    open_image,
    rho_theta,
    split_and_clean,
    traj_to_actions,
    traj_to_actions_Gr00t,
)
from internnav.utils.dist import *  # noqa: F403

DEFAULT_IMAGE_TOKEN = "<image>"

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

def local_dp_to_global_dp(dp_actions, init_xy, init_yaw, yaw_in_degree=False):
    if dp_actions.ndim == 3:
        dp = dp_actions[0]
    else:
        dp = dp_actions

    if yaw_in_degree:
        init_yaw = np.deg2rad(init_yaw)

    T = dp.shape[0]
    global_dp = np.zeros((T, 3), dtype=np.float32)

    yaw = init_yaw

    for i in range(T):
        dx, dy, dyaw = dp[i]

        # 如果 Gr00t dyaw 是 degree，这里要转
        # dyaw = np.deg2rad(dyaw)

        c = np.cos(yaw)
        s = np.sin(yaw)
        R = np.array([[c, -s],
                      [s,  c]])

        dxy_world = R @ np.array([dx, dy])

        global_dp[i, 0] = dxy_world[0]
        global_dp[i, 1] = dxy_world[1]
        global_dp[i, 2] = 0.0   # ⚠️ 关键：不再把 dyaw 传下去

        yaw += dyaw             # 用 dyaw 更新朝向

        # print("=== yaw evolution ===")
        # yaw_dbg = init_yaw
        # for i in range(min(10, T)):
        #     _, _, dyaw = dp[i]
        #     print(f"step {i}: yaw={yaw_dbg:.4f}, dyaw={dyaw:.4f}")
        #     yaw_dbg += dyaw

    return global_dp[None, ...]


def should_stop(dp_actions, dist_thresh=0.15):
    """
    判断 Gr00t 是否认为已经到终点
    """
    dp = dp_actions[0]          # (T, 3)
    disp = dp[:, :2].sum(axis=0)
    dist = np.linalg.norm(disp)
    return dist < dist_thresh

def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 1,
        output_path: str = None,
        model: Any = None,
        processor: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device('cuda')
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        with habitat.config.read_write(self.config):
            # self.config.habitat.task.measurements.success.success_distance=3.0
            self.config.habitat.dataset.split = self.split
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        print(f"config = {type(self.config)}")
        print(OmegaConf.to_yaml(self.config))

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))

        self.model = self.model = Gr00tHTTPClient(url="http://127.0.0.1:8000/act")
        self.processor = processor

        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint\'s coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

        self.objectnav_instructions = ["Search for the {target_object}."]

        self.num_frames = args.num_frames
        self.num_future_steps = args.num_future_steps
        self.num_history = args.num_history

    def preprocess_depth_image_v2(
        self, depth_image, do_depth_scale=True, depth_scale=1000, target_height=None, target_width=None
    ):
        if target_height is None:
            target_height = self.image_processor.crop_size['height']  # 384
            target_width = self.image_processor.crop_size['width']  # 384

        resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)

        img = to_numpy_array(resized_depth_image)
        if do_depth_scale:
            img = img / depth_scale

        return img, (target_width, target_height)

    def get_intrinsic_matrix(self, sensor_cfg) -> np.ndarray:
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx  # Assuming square pixels (fx = fy)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        intrinsic_matrix = np.array(
            [[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        return intrinsic_matrix

    def preprocess_instrinsic(self, intrinsic, ori_size, target_size):  # (V, 4, 4) (resize_shape) (h, w)
        intrinsic = copy.deepcopy(intrinsic)
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[None, :, :]  # (1, 4, 4) or (B, 4, 4)

        intrinsic[:, 0] /= ori_size[0] / target_size[0]  # width
        intrinsic[:, 1] /= ori_size[1] / target_size[1]  # height

        # for crop transform
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2

        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)

        return intrinsic

    def get_axis_align_matrix(self):
        ma = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        return ma

    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],
                [np.sin(yaw), np.cos(yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )
        return transformation_matrix

    def xyz_pitch_to_tf_matrix(self, xyz: np.ndarray, pitch: float) -> np.ndarray:
        """Converts a given position and pitch angle to a 4x4 transformation matrix.

        Args:
            xyz (np.ndarray): A 3D vector representing the position.
            pitch (float): The pitch angle in radians for y axis.
        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """

        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch), x],
                [0, 1, 0, y],
                [-np.sin(pitch), 0, np.cos(pitch), z],
                [0, 0, 0, 1],
            ]
        )
        return transformation_matrix

    def xyz_yaw_pitch_to_tf_matrix(self, xyz: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
        """Converts a given position and yaw, pitch angles to a 4x4 transformation matrix.

        Args:
            xyz (np.ndarray): A 3D vector representing the position.
            yaw (float): The yaw angle in radians.
            pitch (float): The pitch angle in radians for y axis.
        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """
        x, y, z = xyz
        rot1 = self.xyz_yaw_to_tf_matrix(xyz, yaw)[:3, :3]
        rot2 = self.xyz_pitch_to_tf_matrix(xyz, pitch)[:3, :3]
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rot1 @ rot2
        transformation_matrix[:3, 3] = xyz
        return transformation_matrix

    def pixel_to_gps(self, pixel, depth, intrinsic, tf_camera_to_episodic):
        '''
        Args:
            pixel: (2,) - [u, v] pixel coordinates
            depth: (H, W) - depth image where depth[v, u] gives depth in meters
            intrinsic: (4, 4) - camera intrinsic matrix
            tf_camera_to_episodic: (4, 4) - transformation from camera to episodic frame
        Returns:
            (x, y): (x, y) coordinates in the episodic frame
        '''
        v, u = pixel
        z = depth[v, u]
        print("depthhhhhhhhhhhhhh", z)

        x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
        y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
        point_camera = np.array([x, y, z, 1.0])

        # Transform to episodic frame
        point_episodic = tf_camera_to_episodic @ point_camera
        point_episodic = point_episodic[:3] / point_episodic[3]

        x = point_episodic[0]
        y = point_episodic[1]

        return (x, y)  # same as habitat gps

    def config_env(self) -> Env:
        env = Env(config=self.config)
        # env.episodes = env.episodes[0:1]
        return env
    


    def dot_matrix_two_dimensional(
        self,
        image_or_image_path,
        save_path=None,
        dots_size_w=8,
        dots_size_h=8,
        save_img=False,
        font_path='fonts/arial.ttf',
        pixel_goal=None,
    ):
        """
        takes an original image as input, save the processed image to save_path. Each dot is labeled with two-dimensional Cartesian coordinates (x,y). Suitable for single-image tasks.
        control args:
        1. dots_size_w: the number of columns of the dots matrix
        2. dots_size_h: the number of rows of the dots matrix
        """
        with open_image(image_or_image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            draw = ImageDraw.Draw(img, 'RGB')

            width, height = img.size
            grid_size_w = dots_size_w + 1
            grid_size_h = dots_size_h + 1
            cell_width = width / grid_size_w
            cell_height = height / grid_size_h

            font = ImageFont.truetype(font_path, width // 40)  # Adjust font size if needed; default == width // 40

            target_i = target_j = None
            if pixel_goal is not None:
                y_pixel, x_pixel = pixel_goal[0], pixel_goal[1]
                # Validate pixel coordinates
                if not (0 <= x_pixel < width and 0 <= y_pixel < height):
                    raise ValueError(f"pixel_goal {pixel_goal} exceeds image dimensions ({width}x{height})")

                # Convert to grid coordinates
                target_i = round(x_pixel / cell_width)
                target_j = round(y_pixel / cell_height)

                # Validate grid bounds
                if not (1 <= target_i <= dots_size_w and 1 <= target_j <= dots_size_h):
                    raise ValueError(
                        f"pixel_goal {pixel_goal} maps to grid ({target_j},{target_i}), "
                        f"valid range is (1,1)-({dots_size_h},{dots_size_w})"
                    )

            count = 0

            for j in range(1, grid_size_h):
                for i in range(1, grid_size_w):
                    x = int(i * cell_width)
                    y = int(j * cell_height)

                    pixel_color = img.getpixel((x, y))
                    # choose a more contrasting color from black and white
                    if pixel_color[0] + pixel_color[1] + pixel_color[2] >= 255 * 3 / 2:
                        opposite_color = (0, 0, 0)
                    else:
                        opposite_color = (255, 255, 255)

                    if pixel_goal is not None and i == target_i and j == target_j:
                        opposite_color = (255, 0, 0)  # Red for target

                    circle_radius = width // 240  # Adjust dot size if needed; default == width // 240
                    draw.ellipse(
                        [(x - circle_radius, y - circle_radius), (x + circle_radius, y + circle_radius)],
                        fill=opposite_color,
                    )

                    text_x, text_y = x + 3, y
                    count_w = count // dots_size_w
                    count_h = count % dots_size_w
                    label_str = f"({count_w+1},{count_h+1})"
                    draw.text((text_x, text_y), label_str, fill=opposite_color, font=font)
                    count += 1
            if save_img:
                print(">>> dots overlaid image processed, stored in", save_path)
                img.save(save_path)
            return img

    def _pointnav(
        self,
        goal: np.ndarray,
        depth: np.ndarray,
        step_id: int,
        robot_xy: np.ndarray,
        robot_heading: float,
        stop: bool = False,
    ) -> Tensor:
        '''
        Args:
            goal (np.ndarray): goal position
            stop (bool): whether to stop
        Returns:
            action: action tensor
        '''

        masks = torch.tensor([step_id != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                print('Pointnav policy reset!')
                masks = torch.zeros_like(masks)
            self._last_goal = goal
        rho, theta = rho_theta(robot_xy, robot_heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                depth,
                (self._pointnav_depth_image_shape[0], self._pointnav_depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }

        if rho < self._pointnav_stop_radius and stop:
            return 0
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        return action

    def eval_action(self, idx) -> None:  # noqa: C901
        # self.model.eval()
        if hasattr(self.model, "eval"):
            self.model.eval()
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        intrinsic_matrix = self.get_intrinsic_matrix(
            self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
        )
        sucs, spls, oss, nes = [], [], [], []
        done_res = []

        if os.path.exists(os.path.join(self.output_path, 'result.json')):
            with open(os.path.join(self.output_path, 'result.json'), 'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:  # noqa: F405
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        nes.append(res['ne'])

        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            process_bar = tqdm.tqdm(range(len(episodes[idx :: self.env_num])), desc=f"scene {scene_id}")
            for episode in episodes[idx :: self.env_num]:
                episode_instruction = (
                    episode.instruction.instruction_text
                    if 'objectnav' not in self.config_path
                    else episode.object_category
                )
                print("episode start", episode_instruction)
                episode_id = int(episode.episode_id)
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue

                env.current_episode = episode
                observations = env.reset()

                agent_state = env.sim.get_agent_state()
                rotation = agent_state.rotation
                translation = agent_state.position
                rotation_matrix = quaternion.as_rotation_matrix(rotation)
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, :3] = rotation_matrix
                transformation_matrix[:3, 3] = translation

                agent = ShortestPathFollower(env.sim, 0.25, False)

                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
                Image.fromarray(observations['rgb']).save(
                    os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{idx}.jpg')
                )

                vis_frames = []
                step_id = 0

                if self.save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
                initial_height = env.sim.get_agent_state().position[1]

                rgb_list = []
                action_seq = []
                output_ids = None

                goal = None
                action = None
                messages = []
                local_actions = []
                pending_actions = []
                forward_action = 0          # 等价 InternVA 的 forward_action
                MAX_FORWARD = 8             # 建议和 InternVA 一致
                gr00t_plan_step = 0  

                observations = env.reset()
                step_id = 0
                camera_yaw = observations["compass"][0]   # rad
                adapter_state = {
                    "yaw": camera_yaw,
                    "mode": "forward",
                    "forward_left": 0,
                }


                while not env.episode_over and step_id <= 500:
                    rgb = observations["rgb"]
                    depth = observations["depth"]
                    x, y = observations["gps"]
                    camera_yaw = observations["compass"][0]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000

                    agent_state = env.sim.get_agent_state()
                    height = agent_state.position[1] - initial_height
                    camera_position = np.array([x, -y, self._camera_height + height])
                    tf_camera_to_episodic = (
                        self.xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30))
                        @ self.get_axis_align_matrix()
                    )

                    image = Image.fromarray(rgb).convert('RGB')
                    save_raw_image = image.copy()

                    info = env.get_metrics()

                    # ================= STOP by env metric =================
                    if info["distance_to_goal"] < 0.25:
                        print("[STOP] reached goal by env metric")
                        break
                    # ======================================================

                    # 输入给 Gr00t 的 RGB（必须 256 × 256）
                    rgb_256 = np.array(image.resize((256, 256)))
                    
                    height = env.sim.get_agent_state().position[1] - initial_height

                    # 当前位姿（Habitat → Gr00t）
                    # Gr00t 坐标系：右 +x，前 +y，上 +z
                    # Habitat 坐标：x, y, z, yaw（z 是高度）
                    drone_state = np.array([x, y, height, camera_yaw], dtype=np.float32).reshape(1, 4)
                    # print("drone_state",drone_state)
                    # print("episode_instruction",episode_instruction)

                    # 构建 Gr00t 输入
                    gr00t_obs = {
                        "video.ego_view": rgb_256.reshape(1, 256, 256, 3),
                        "state.drone": drone_state,          # (1,4)
                        "annotation.human.action.task_description":[episode_instruction],
                    }

                    # ============================================================
                    #           Gr00t trajectory -> Habitat discrete action
                    # ============================================================

                    # 如果当前没有缓存动作，则调用一次 Gr00t 规划一整段轨迹
                    if len(pending_actions) == 0 and adapter_state.get("forward_left", 0) == 0:

                        gr00t_out = self.model.get_action(gr00t_obs)

                        # 解析 Gr00t 输出
                        if isinstance(gr00t_out, dict) and "action.delta_pose" in gr00t_out:
                            delta_poses = gr00t_out["action.delta_pose"]
                        else:
                            delta_poses = gr00t_out

                        # print("[DEBUG] delta_poses type:", type(delta_poses))
                        # print("[DEBUG] delta_poses shape:", np.array(delta_poses).shape)
                        # print("[DEBUG] delta_poses first 3:", np.array(delta_poses)[:3])

                        # delta_poses: (T, 4) or (1, T, 4)
                        dp_actions = gr00t_output_to_dp_actions(delta_poses)  # (B=1, T, 3)

                        # dp_np = dp_actions.detach().cpu().numpy()[0]
                        # print("=== Gr00t raw dp_actions (first 10) ===")
                        # for i in range(min(10, len(dp_np))):
                        #     dx, dy, dyaw = dp_np[i]
                        #     print(f"{i}: dx={dx:.4f}, dy={dy:.4f}, dyaw={dyaw:.4f}")


                        global_dp = local_dp_to_global_dp(
                            dp_actions.detach().cpu().numpy(),
                            init_xy=np.array([x, y]),
                            init_yaw=camera_yaw
                        )

                        global_dp = torch.from_numpy(global_dp).to(dp_actions.device)
                        print("global_dp:",global_dp)
                        # print("global_dp.shape:",global_dp.shape)

                        # dp_np = dp_actions.detach().cpu().numpy()
                        # print("[DEBUG] dp_actions shape:", dp_np.shape)
                        # print("[DEBUG] dp_actions min/max:", dp_np.min(), dp_np.max())
                        # print("[DEBUG] dp_actions first step:", dp_np[0, 0])

                        # ========== ⭐ STOP 判据在这里 ==========
                        if info["distance_to_goal"] < 0.25:
                            break
                       
                                         # ⬅️⬅️⬅️ 关键：直接结束 episode loop
                        # ======================================
                        
                        # 连续轨迹 -> 离散动作序列
                        REPLAN_INTERVAL = 4
                        if len(pending_actions) == 0 or step_id % REPLAN_INTERVAL == 0:
                            pending_actions = traj_to_actions_Gr00t(
                                global_dp
                            )
                        print("[DEBUG] pending_actions:", pending_actions)
                        print("[DEBUG] pending_actions len:", len(pending_actions))

                        gr00t_plan_step = 0

                    # 从缓存中取一个动作执行
                    if len(pending_actions) > 0:
                        action = pending_actions.pop(0)
                        gr00t_plan_step += 1
                    else:
                        action = 0  # fallback: STOP

                    print(f"[Gr00t] step {step_id}, planned_step {gr00t_plan_step}, action {action}")

                    # ============================================================
                    #          原 evaluator 的可视化和日志完全保留
                    # ============================================================
                    if info['top_down_map'] is not None:
                        frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                        vis_frames.append(frame)

                    print("step_id", step_id, "action", action)

                    # === 执行动作 ===
                    observations = env.step(action)
                    step_id += 1

                    if action == 2:
                        adapter_state["yaw"] += np.deg2rad(15)
                    elif action == 3:
                        adapter_state["yaw"] -= np.deg2rad(15)

                    adapter_state["yaw"] = normalize_angle(adapter_state["yaw"])

                    # 清空（但 Gr00t 不需要 messages）
                    messages = []



                process_bar.update(1)

                metrics = env.get_metrics()
                if self.save_video:
                    images_to_video(
                        vis_frames,
                        os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                        f'{episode_id:04d}',
                        fps=6,
                        quality=9,
                    )
                vis_frames.clear()
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                nes.append(metrics["distance_to_goal"])
                print(
                    f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}"
                )

                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction,
                }

                with open(os.path.join(self.output_path, 'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")
        env.close()
        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(len(sucs)).to(self.device),
        )

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def preprocess_qwenvl(self, source):
        prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
        if len(source[0]["value"]) != 0:
            source[0]["value"] += f" {prompt}."
        else:
            source[0]["value"] = f"{prompt}."  # Please output the next waypoint\'s coordinates in the image."
        return source
