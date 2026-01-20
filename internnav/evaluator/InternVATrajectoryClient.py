import requests
import numpy as np
from PIL import Image
from internnav.evaluator.final_habitat_vln_evaluator import BaseTrajectoryClient
import json_numpy
json_numpy.patch()

class InternVATrajectoryClient(BaseTrajectoryClient):
    def __init__(self, url, max_history=12): # [修正] 增大默认历史长度
        self.url = url
        self.max_history = max_history
        self.full_history = [] 
        self.instruction = ""
        self.last_decided_action = -1

        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',  # 第 7 个词，原代码里有，之前 Server 端漏了
        ]

    def reset(self, instruction: str, **kwargs):
        """每个 Episode 开始时被 LLMAgent 调用"""
        self.instruction = instruction
        self.full_history = [] # 清空历史
        self.last_decided_action = -1

    def query(self, obs: dict, update_history: bool = True, do_resize: bool = True) -> dict:
        # 1. 图片处理 (Resize 384)
        import random
        prompt_suffix = random.choice(self.conjunctions)

        raw_img = Image.fromarray(obs["rgb"])
        if do_resize:
            processed_img = raw_img.resize((384, 384), Image.BILINEAR)
        else:
            # [CRITICAL FIX] 俯视图：发送原图，不缩放！
            processed_img = raw_img
        current_frame = {
            "rgb": np.array(processed_img),
            "depth": obs["depth"],
            "gps": obs["gps"],
            "yaw": obs["yaw"],
            "step_id": obs["step_id"]
        }
        if update_history:
            self.full_history.append(current_frame)
        
        # 2. 采样逻辑 (复刻原代码的 linspace 采样)
        temp_full_history = self.full_history if update_history else self.full_history + [current_frame]
        curr_len = len(temp_full_history)
        sampled_history = []
        
        if curr_len <= self.max_history:
            sampled_history = temp_full_history
        else:
            # 始终保留第一帧(起点)和最后一帧(当前)
            # 中间均匀采样
            # 比如 max=10, 拿 0 到 curr-2 之间的 8 个点
            num_history_samples = self.max_history - 1
            history_indices = np.unique(np.linspace(0, curr_len - 2, num_history_samples, dtype=int))
            
            for idx in history_indices:
                sampled_history.append(temp_full_history[idx])        
            # 加上当前帧
            sampled_history.append(temp_full_history[-1])

        # 3. Payload
        payload = {
            "instruction": self.instruction,
            "history": sampled_history, 
            "current_step_id": obs["step_id"],
            "camera_height": obs["camera_height"],
            "previous_action": int(self.last_decided_action),
            "min_depth": obs.get("min_depth", 0.0),
            "max_depth": obs.get("max_depth", 10.0),
            "force_navdp": not update_history,
            "prompt_suffix": prompt_suffix,
        }

        try:
            resp = requests.post(
                self.url,
                data=json_numpy.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=120.0,
            )
            resp.raise_for_status()
            result = json_numpy.loads(resp.text)
            
            if result and "actions" in result and len(result["actions"]) > 0:
                self.last_decided_action = result["actions"][0]
            else:
                self.last_decided_action = 0
                
            return result
        except Exception as e:
            #print(f"[InternVA Client Error] {e}")
            self.last_decided_action = 0
            return {"actions": [0]}