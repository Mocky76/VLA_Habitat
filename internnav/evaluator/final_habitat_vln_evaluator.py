#本文件定义了基本类和函数，分为Evaluator（在Habitat中执行）和LLMAgent（调用模型推理）两部分
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
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass

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

def build_traj_request(obs, instruction: str, rel_height: float):
    return {
        "rgb": obs.rgb,
        "depth": obs.depth,
        "gps": obs.gps,
        "yaw": obs.compass,
        "camera_height": rel_height,
        "instruction": instruction,
        "step_id": obs.step_id,
    }

def preprocess_depth_image(
    depth_image,
    target_height: int = 384,
    target_width: int = 384,
    do_depth_scale: bool = True,
    depth_scale: float = 1000.0,
):
    if isinstance(depth_image, np.ndarray):
        depth_image = Image.fromarray(depth_image)
    resized_depth_image = depth_image.resize(
        (target_width, target_height),
        Image.NEAREST
    )

    img = to_numpy_array(resized_depth_image)
    if do_depth_scale:
        img = img / depth_scale

    return img

def get_intrinsic_matrix(width, height, hfov) -> np.ndarray:
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    fy = fx
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0

    return np.array(
        [
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

def preprocess_intrinsic(intrinsic, ori_size, target_size):
    intrinsic = copy.deepcopy(intrinsic)

    if len(intrinsic.shape) == 2:
        intrinsic = intrinsic[None, :, :]

    intrinsic[:, 0] /= ori_size[0] / target_size[0]
    intrinsic[:, 1] /= ori_size[1] / target_size[1]

    intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2

    if intrinsic.shape[0] == 1:
        intrinsic = intrinsic.squeeze(0)

    return intrinsic

def get_axis_align_matrix():
    return np.array(
        [
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

def xyz_yaw_to_tf_matrix(xyz: np.ndarray, yaw: float) -> np.ndarray:
    x, y, z = xyz
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0, x],
            [np.sin(yaw),  np.cos(yaw), 0, y],
            [0,            0,           1, z],
            [0,            0,           0, 1],
        ],
        dtype=np.float32,
    )

def xyz_pitch_to_tf_matrix(xyz: np.ndarray, pitch: float) -> np.ndarray:
    """Converts a given position and pitch angle to a 4x4 transformation matrix.

    Args:
        xyz (np.ndarray): A 3D vector representing the position.
        pitch (float): The pitch angle in radians for y axis.
    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    x, y, z = xyz
    return np.array(
        [
            [ np.cos(pitch), 0, np.sin(pitch), x],
            [ 0,             1, 0,             y],
            [-np.sin(pitch), 0, np.cos(pitch), z],
            [ 0,             0, 0,             1],
        ],
        dtype=np.float32,
    )

def xyz_yaw_pitch_to_tf_matrix(xyz, yaw, pitch):
    T = np.eye(4, dtype=np.float32)
    R = (
        xyz_yaw_to_tf_matrix(xyz, yaw)[:3, :3]
        @ xyz_pitch_to_tf_matrix(xyz, pitch)[:3, :3]
    )
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T

def pixel_to_gps(pixel, depth, intrinsic, tf_camera_to_episodic):
    """
    Args:
        pixel: (2,) [v, u]
        depth: (H, W)
        intrinsic: (4, 4)
        tf_camera_to_episodic: (4, 4)
    Returns:
        (x, y) in episodic frame
    """
    v, u = pixel
    # depth is assumed to be in meters
    z = depth[v, u]

    x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]

    point_camera = np.array([x, y, z, 1.0], dtype=np.float32)

    point_episodic = tf_camera_to_episodic @ point_camera
    point_episodic = point_episodic[:3] / point_episodic[3]

    x = point_episodic[0]
    y = point_episodic[1]

    return (x, y)

def dot_matrix_two_dimensional(
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

@dataclass
class Observation:
    rgb: np.ndarray            # (H, W, 3)
    depth: np.ndarray          # (H, W)
    gps: np.ndarray            # (2,)
    compass: float
    step_id: int
    height: float
    # info: dict                 # top_down_map / collisions

class Action(Enum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    LOOK_UP = 4
    LOOK_DOWN = 5

class BaseAgent(ABC):
    @abstractmethod
    def reset(self, instruction: str, **kwargs):
        pass

    @abstractmethod
    def act(self, obs: Observation) -> Action:
        pass

    def set_camera_params(self, params: dict):
        pass

    def on_episode_end(self, metrics: dict):
        pass

class HabitatSensorAPI:
    def __init__(self, observations, step_id):
        self.obs = observations
        self.step_id = step_id

    def get_rgb(self):
        return self.obs["rgb"]

    def get_depth(self):
        return self.obs["depth"]

    def get_pose(self):
        return self.obs["gps"], self.obs["compass"][0]

    def get_step(self):
        return self.step_id

class HabitatMotionAPI:

    def __init__(self, env):
        self.env = env

    def step_action(self, action: Action):
        self.env.step(action.value)

    def step_trajectory(self, traj, local=True):
        actions = traj_to_actions_Gr00t(traj)
        for act in actions:
            self.env.step(act)

class HabitatEpisodeAPI:

    def __init__(self, env):
        self.env = env

    def reset(self) -> None:
        self.env.reset()

    def is_done(self) -> bool:
        return self.env.episode_over

    def get_metrics(self) -> dict:
        return self.env.get_metrics()
    
class BaseTrajectoryClient:
    def reset(self, instruction: str, **kwargs):
        pass

    def query(self, obs: dict) -> list[int]:
        """
        返回 Habitat action id list
        """
        raise NotImplementedError

class Evaluator:
    def __init__(
        self,
        config_path: str,
        split: str,
        output_path: str,
        args: argparse.Namespace,
        agent: BaseAgent,
        max_steps: int = 500,
        idx: int = 0,
        env_num: int = 1,
    ):
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent = agent

        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = split
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

        self.env = Env(config=self.config)
        if hasattr(self.agent, "set_env"):
            self.agent.set_env(self.env)
        self.idx = idx
        self.env_num = env_num

        self.agent = agent
        self.max_steps = max_steps
        self.output_path = output_path
        self.save_video = args.save_video
        self.vis_frames = []

        self.sucs = []
        self.spls = []
        self.oss = []
        self.nes = []
        self.steps = []

        sensor_cfg = self.config.habitat.simulator.agents.main_agent.sim_sensors

        camera_params = {
            "camera_height": sensor_cfg.rgb_sensor.position[1],
            "min_depth": sensor_cfg.depth_sensor.min_depth,
            "max_depth": sensor_cfg.depth_sensor.max_depth,
            "hfov": sensor_cfg.depth_sensor.hfov,
            "width": sensor_cfg.depth_sensor.width,
            "height": sensor_cfg.depth_sensor.height,
        }

        self.agent.set_camera_params(camera_params)

    def iter_episodes(self):
        """
        Iterate over all episodes, grouped by scene.
        Skip episodes that already exist in result.json.

        Yields:
            episode: habitat episode
            scene_id: str
            episode_instruction: str
        """
        env = self.env
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)
        done_res = set()
        result_path = os.path.join(self.output_path, "result.json")

        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                for line in f:
                    res = json.loads(line)
                    done_res.add((
                        res["scene_id"],
                        int(res["episode_id"]),
                        res["episode_instruction"],
                    ))

        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split("/")[-2]

            for episode in episodes[self.idx :: self.env_num]:
                episode_instruction = (
                    episode.instruction.instruction_text
                    if "objectnav" not in self.config_path
                    else episode.object_category
                )

                episode_key = (
                    scene_id,
                    int(episode.episode_id),
                    episode_instruction,
                )

                if episode_key in done_res:
                    continue

                yield episode, scene_id, episode_instruction

    def _init_episode(self, episode):
        """
        Episode 初始化逻辑（无任何模型相关内容）
        - reset
        - 相机视角对齐（俯视 30°）
        """
        self.env.current_episode = episode
        observations = self.env.reset()

        # === 初始高度（给 agent 用）===
        initial_height = self.env.sim.get_agent_state().position[1]

        # === 固定初始化动作：LOOK_DOWN × 2 ===
        # observations = self.env.step(Action.LOOK_DOWN.value)
        # observations = self.env.step(Action.LOOK_DOWN.value)

        self.initial_yaw = observations["compass"][0]

        self.vis_frames = []
        self.initial_height = initial_height

        return observations, initial_height

    def run_episode(self, episode):
        # ===== Episode init =====
        observations, initial_height = self._init_episode(episode)
        self.agent.reset(
            episode.instruction.instruction_text
            if hasattr(episode, "instruction")
            else "",
            init_yaw=self.initial_yaw,
            initial_height=self.initial_height
        )

        step = 0
        done = False

        self.vis_frames = []

        while not done and step < self.max_steps:
            if self.env.episode_over:
                break
            # === 模块 3：Habitat → Observation ===
            obs = self._build_observation(observations, step, agent_height=self.initial_height)

            # === 模块 5（之后）：Observation → Action ===
            action = self.agent.act(obs)

            # === 模块 4：STOP by env metric（关键）===
            # info = self.env.get_metrics()
            # if info.get("distance_to_goal", float("inf")) < 0.25:
            #     # 这是 evaluator 的 stop，不是 agent 的 stop
            #     break

            # === 执行动作 ===
            observations = self.env.step(action.value)
            done = self.env.episode_over
            step += 1

            if self.save_video:
                frame = observations_to_image(
                    {"rgb":  observations["rgb"]},
                    self.env.get_metrics(),
                )
                self.vis_frames.append(frame)

        # ===== episode end =====
        metrics = self.env.get_metrics()
        self.agent.on_episode_end(metrics)
        
        # ===== evaluator-level metric（和之前完全一致）=====
        success = metrics["success"]
        spl = metrics["spl"]
        ne = metrics["distance_to_goal"]
        # print("self.config.habitat.task",self.config.habitat.task)
        # oracle_success：自己算（等价于你之前的）
        oracle_success = float(
            ne < self.config.habitat.task.measurements.success.success_distance
        )

        self.sucs.append(success)
        self.spls.append(spl)
        self.oss.append(oracle_success)
        self.nes.append(ne)
        self.steps.append(step)

        result = {
            "scene_id": episode.scene_id.split("/")[-2],
            "episode_id": int(episode.episode_id),
            "success": success,
            "spl": spl,
            "os": oracle_success,   
            "ne": ne,
            "steps": step,
            "episode_instruction": (
                episode.instruction.instruction_text
                if hasattr(episode, "instruction")
                else ""
            ),
        }

        with open(os.path.join(self.output_path, "result.json"), "a") as f:
            f.write(json.dumps(result) + "\n")

        print(
            f"[Eval] Episode {int(episode.episode_id)} finished | "
            f"success={success}, spl={spl:.3f}, ne={ne:.3f} | "
            f"result.json updated"
        )

        if self.save_video and len(self.vis_frames) > 0:
            scene_id = episode.scene_id.split("/")[-2]
            save_dir = os.path.join(
                self.output_path,
                "vis",
                scene_id,
            )
            os.makedirs(save_dir, exist_ok=True)

            images_to_video(
                self.vis_frames,
                save_dir,
                f"{int(episode.episode_id):04d}",
                fps=6,
                quality=9,
            )
            video_name = "video"

            # print(
            #     f"[Eval] 🎬 Video saved: "
            #     f"{os.path.join(save_dir, video_name)}.mp4"
            # )

        self.vis_frames.clear()

        return metrics


    def _build_observation(self, observations, step_id, agent_height=None):
        return Observation(
            rgb=observations["rgb"],
            depth=observations["depth"],
            gps=observations["gps"],
            compass=observations["compass"][0],
            step_id=step_id,
            height=agent_height,
            # info=self.env.get_metrics(),
        )
    
    def run(self):
        current_scene = None
        process_bar = None

        for episode, scene_id, episode_instruction in self.iter_episodes():

            # === new scene ===
            if scene_id != current_scene:
                if process_bar is not None:
                    process_bar.close()

                current_scene = scene_id
                process_bar = tqdm.tqdm(
                    desc=f"scene {scene_id}",
                    unit="ep",
                )

            # === run one episode ===
            self.run_episode(episode)

            # === update bar ===
            process_bar.update(1)

        if process_bar is not None:
            process_bar.close()

        if dist.is_initialized():
            dist.barrier()

        # 只让 rank 0 负责 summary
        if not dist.is_initialized() or dist.get_rank() == 0:
            self._summarize_results()

    def _summarize_results(self):
        import json, os

        sucs, spls, oss, nes = [], [], [], []

        result_path = os.path.join(self.output_path, "result.json")

        with open(result_path) as f:
            for line in f:
                r = json.loads(line)
                sucs.append(r["success"])
                spls.append(r["spl"])
                oss.append(r["os"])
                nes.append(r["ne"])

        summary = {
            "sucs_all": sum(sucs) / len(sucs),
            "spls_all": sum(spls) / len(spls),
            "oss_all": sum(oss) / len(oss),
            "nes_all": sum(nes) / len(nes),
            "length": len(sucs),
        }

        print("===== EVAL SUMMARY =====")
        print(summary)

        with open(os.path.join(self.output_path, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

class LLMAgent(BaseAgent):
    def __init__(
        self,
        traj_client: BaseTrajectoryClient,
        processor,
        args: argparse.Namespace,
        device: str = "cuda",
    ):
        # ===== 模型相关 =====
        # self.model = Gr00tHTTPClient(url="http://127.0.0.1:8000/act")
        self.traj_client = traj_client
        self.processor = processor
        self.device = torch.device(device)
        self.env = None

        self.last_pixel_goal = None

        # ===== 超参数 =====
        self.num_frames = args.num_frames
        self.num_future_steps = args.num_future_steps
        self.num_history = args.num_history

        # ===== prompt / language =====
        self.base_prompt = (
            "You are an autonomous navigation assistant. "
            "Your task is to <instruction>. "
            "Where should you go next to stay on track? "
            "Please output the next waypoint's coordinates in the image. "
            "Please output STOP when you have successfully completed the task."
        )

        self.conjunctions = [
            "you can see ",
            "in front of you is ",
            "there is ",
            "you can spot ",
            "you are toward the ",
            "ahead of you is ",
            "in your sight is ",
        ]

        self.actions2idx = OrderedDict(
            {
                "STOP": 0,
                "↑": 1,
                "←": 2,
                "→": 3,
                "↓": 5,
            }
        )

        # ===== episode state =====
        self.objectnav_instructions = [
            "Search for the {target_object}."
        ]

        self._last_goal = None
        self._pointnav_policy = None
        self._pointnav_depth_image_shape = (256, 256)
        self._pointnav_stop_radius = 0.2

    def set_env(self, env):
        self.env = env

    def reset(self, instruction: str, init_yaw: float = None, initial_height: float = 0.0 , **kwargs):
        if instruction is None:
            instruction = ""
        self.instruction = instruction
        self.init_yaw = init_yaw
        self.initial_height = initial_height
        self.traj_client.reset(instruction)

        self.conversation = [
            {"from": "human", "value": self.base_prompt.replace("<instruction>", instruction)},
            {"from": "gpt", "value": ""},
        ]

        self.messages = []
        self.goal = None
        self.local_actions = []
        self.step_id = 0

        self.last_pixel_goal = None

    def act(self, obs: Observation) -> Action:

        if self.local_actions:
            self.last_pixel_goal = None
            return Action(self.local_actions.pop(0))

        req = build_traj_request(
            obs,
            self.instruction,
            obs.height - self.initial_height,
        )

        req["min_depth"] = self.min_depth
        req["max_depth"] = self.max_depth

        action_list = self.traj_client.query(req, update_history=True)
        actions = action_list.get("actions", [])
        self.last_pixel_goal = action_list.get("pixel_goal", None)

        if not actions:
            return Action.TURN_LEFT
        
        first_action = actions[0]

        # === [CRITICAL LOGIC] 原子操作：如果 Server 决定低头 (5) ===
        if first_action == 5:
            #print(f"[LLMAgent] Atomic Look Down triggered at step {obs.step_id}")
            
            # A. 内部执行两次低头 (Habitat 中低头一次是 30度，原代码通常做两次)
            # 注意：这里的 step 不会增加外部 Evaluator 的 step 计数，因为我们在 Agent 内部
            # 但我们需要从 env 获取新的 observation
            obs_down_1 = self.env.step(Action.LOOK_DOWN.value)
            obs_down_2 = self.env.step(Action.LOOK_DOWN.value) # 这张是地板图
            
            # B. 构造地板图请求
            # 我们需要把 obs_down_2 封装成 req 格式
            # 注意：这里需要重新从 env 获取当前的 info 来构建 req，或者直接复用 obs_down_2
            # 简单起见，我们手动构建一个类似 build_traj_request 的 payload
            
            # 计算新的相对高度 (低头不会变高度，但为了严谨)
            current_height = self.env.sim.get_agent_state().position[1]
            
            req_floor = {
                "rgb": obs_down_2["rgb"],
                "depth": obs_down_2["depth"],
                "gps": obs_down_2["gps"],
                "yaw": obs_down_2["compass"][0],
                "camera_height": current_height - self.initial_height,
                "instruction": self.instruction,
                "step_id": obs.step_id, # 保持原来的 step_id
                "min_depth": self.min_depth,
                "max_depth": self.max_depth
            }
            
            # C. 第二次查询 Server (强制 NavDP)
            # [关键] update_history=False !!! 不让地板图进历史 !!!
            # Server 会识别 force_navdp=True (由 Client 根据 update_history=False 自动推导)
            traj_result = self.traj_client.query(req_floor, update_history=False, do_resize=False)
            
            nav_actions = traj_result.get("actions", [])
            #print(f"[LLMAgent] NavDP returned actions: {nav_actions}")
            
            # D. 内部执行两次抬头 (恢复平视)
            self.env.step(Action.LOOK_UP.value)
            self.env.step(Action.LOOK_UP.value)
            
            # E. 处理返回的动作
            # Server 之前返回的是 [4, 4, move, move...] (在你的旧 Server 代码里)
            # 但既然我们在 Client 端已经手动做了抬头，我们需要把 Server 返回的 4,4 去掉
            # 或者是让 Server 别返回 4,4。
            # 为了兼容性，我们在这里过滤一下：
            
            # 过滤掉开头的 4 (Look Up)
            valid_actions = [a for a in nav_actions if a != 4 and a != 5]
            
            if not valid_actions:
                 # 如果过滤完没动作了，或者 NavDP 失败，给个默认动作防止死循环
                 valid_actions = [Action.TURN_LEFT.value]

            # F. 填充 Buffer
            self.local_actions = valid_actions
            
            # G. 返回第一个动作给 Evaluator
            return Action(self.local_actions.pop(0))

        # === 常规动作 (非 5) ===
        else:
            # 如果 Server 返回的是一串动作 (比如连续移动)，存入 Buffer
            self.local_actions = actions[1:]
            return Action(first_action)
        # self.local_actions = actions[:4]

        # act = self.local_actions.pop(0)

        # if act == Action.STOP.value:
        #     return Action.TURN_LEFT

        # return Action(act)

    def set_camera_params(self, params: dict):
        self.camera_height = params["camera_height"]
        self.min_depth = params["min_depth"]
        self.max_depth = params["max_depth"]

        hfov_rad = np.deg2rad(params["hfov"])
        self.fx = self.fy = params["width"] / (2 * np.tan(hfov_rad / 2))
        self.cx = (params["width"] - 1) / 2.0
        self.cy = (params["height"] - 1) / 2.0

        self.intrinsic = get_intrinsic_matrix(
            params["width"], params["height"], params["hfov"]
        )


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
        if self._last_goal is None:
            self._last_goal = goal
        else:
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                #print("Pointnav policy reset!")
            self._last_goal = goal
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                #print('Pointnav policy reset!')
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


class ShortestPathAgent(BaseAgent):
    def __init__(self, sim):
        self.agent = ShortestPathFollower(sim, 0.25, False)

    def reset(self, instruction: str, **kwargs):
        pass

    def act(self, obs: Observation):
        action = self.agent.get_next_action(obs.gps)
        return Action(action)


