import argparse
import copy
import io
import json
import os
import sys
import random
import re
from typing import Any, List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from PIL import Image, ImageDraw, ImageFont
from transformers.image_utils import to_numpy_array
from transformers import AutoProcessor

import json_numpy
json_numpy.patch()

PROJECT_ROOT = "/data/sjh/InternNav"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    import internnav.utils
    # print("✅ Successfully pre-imported internnav.utils")
except ImportError:
    pass

try:
    from depth_camera_filtering import filter_depth
except ImportError:
    def filter_depth(depth, blur_type=None): return depth

from internnav.model.utils.vln_utils import (
    chunk_token,
    image_resize,
    traj_to_actions,
    split_and_clean
)

# ==========================================
# GLOBAL SETTINGS
# ==========================================

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

global_model = None
global_processor = None

# [State Management]
GLOBAL_STATE = {
    "last_action": -1,     # Action executed in previous step
    "last_step_id": -1,    # To detect new episodes
    "action_queue": [],    # Action Buffer Queue
}

DEFAULT_IMAGE_TOKEN = "<image>"
CONJUNCTIONS = [
    'you can see ', 
    'in front of you is ', 
    'there is ',
    'you can spot ', 
    'you are toward the ', 
    'ahead of you is ',
    'in your sight is ',
]
ACTIONS2IDX = {
    'STOP': [0], "↑": [1], "←": [2], "→": [3], "↓": [5],
}

@app.on_event("startup")
def load_model():
    global global_model, global_processor
    # print(">>> [Server] Loading InternVLA-N1 Model...")
    MODEL_PATH = "/data/sjh/InternNav/checkpoints/InternVLA-N1"
    
    if not os.path.exists(MODEL_PATH):
        # print(f"⚠️ Warning: Checkpoint path {MODEL_PATH} does not exist")
        pass

    try:
        from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
        
        global_processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        global_model = InternVLAN1ForCausalLM.from_pretrained(
            MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device).eval()
        #print(f">>> [Server] Model loaded successfully on {device}!")
    except Exception as e:
        #print(e)
        raise e

# ... (Helper Functions 保持不变) ...
# 请确保包含 get_intrinsic_matrix, get_axis_align_matrix, xyz_yaw_to_tf_matrix, 
# xyz_pitch_to_tf_matrix, xyz_yaw_pitch_to_tf_matrix, pixel_to_gps, 
# preprocess_depth_image_v2, dot_matrix_two_dimensional, parse_actions

def get_intrinsic_matrix(width=224, height=224, hfov=90.0) -> np.ndarray:
    camera_fov_rad = np.deg2rad(hfov)
    fx = fy = width / (2 * np.tan(camera_fov_rad / 2))
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return np.array([[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

def get_axis_align_matrix():
    return np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

def xyz_yaw_to_tf_matrix(xyz: np.ndarray, yaw: float) -> np.ndarray:
    x, y, z = xyz
    return np.array([[np.cos(yaw), -np.sin(yaw), 0, x], [np.sin(yaw), np.cos(yaw), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

def xyz_pitch_to_tf_matrix(xyz: np.ndarray, pitch: float) -> np.ndarray:
    x, y, z = xyz
    return np.array([[np.cos(pitch), 0, np.sin(pitch), x], [0, 1, 0, y], [-np.sin(pitch), 0, np.cos(pitch), z], [0, 0, 0, 1]])

def xyz_yaw_pitch_to_tf_matrix(xyz: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    rot1 = xyz_yaw_to_tf_matrix(xyz, yaw)[:3, :3]
    rot2 = xyz_pitch_to_tf_matrix(xyz, pitch)[:3, :3]
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rot1 @ rot2
    transformation_matrix[:3, 3] = xyz
    return transformation_matrix

def pixel_to_gps(pixel, depth, intrinsic, tf_camera_to_episodic):
    v, u = pixel
    z = depth[v, u]
    x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
    point_camera = np.array([x, y, z, 1.0])
    point_episodic = tf_camera_to_episodic @ point_camera
    point_episodic = point_episodic[:3] / point_episodic[3]
    return point_episodic[0], point_episodic[1]

def preprocess_depth_image_v2(depth_image, target_height=224, target_width=224, do_depth_scale=True, depth_scale=1000):
    depth_array = np.array(depth_image)
    if depth_array.ndim == 3 and depth_array.shape[2] == 1:
        depth_array = depth_array.squeeze(2)
    if not isinstance(depth_image, Image.Image):
        try:
            depth_image = Image.fromarray(depth_array, mode='F')
        except:
            depth_image = Image.fromarray(depth_array)
    resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)
    img = to_numpy_array(resized_depth_image)
    if do_depth_scale:
        img = img / depth_scale
    return img

def dot_matrix_two_dimensional(img_array, pixel_goal=None, dots_size_w=8, dots_size_h=8):
    img = Image.fromarray(img_array).convert('RGB')
    draw = ImageDraw.Draw(img, 'RGB')
    width, height = img.size
    if pixel_goal is not None:
        y_pixel, x_pixel = pixel_goal[0], pixel_goal[1]
        circle_radius = width // 240 
        draw.ellipse([(x_pixel - circle_radius, y_pixel - circle_radius), (x_pixel + circle_radius, y_pixel + circle_radius)], fill=(255, 0, 0))
    return img

def parse_actions(output):
    action_patterns = '|'.join(re.escape(k) for k in ACTIONS2IDX.keys())
    regex = re.compile(action_patterns)
    matches = regex.findall(output)
    actions = [ACTIONS2IDX[match] for match in matches]
    import itertools
    return list(itertools.chain.from_iterable(actions))


@app.post("/act")
async def act(req: dict):
    global global_model, global_processor, GLOBAL_STATE
    
    instruction = req["instruction"]
    history = req["history"]
    step_id = req["current_step_id"]
    current_obs = history[-1]
    force_navdp = req.get("force_navdp", False)
    
    if step_id < GLOBAL_STATE["last_step_id"] or step_id == 0:
        #print(f"🔄 DEBUG: New Episode (Step {step_id}). Clearing Queue.")
        GLOBAL_STATE["last_action"] = -1
        GLOBAL_STATE["action_queue"] = []
    
    GLOBAL_STATE["last_step_id"] = step_id

    # [LOGIC] Buffer Check
    if force_navdp:
        #print(f"🚀 DEBUG: Force NavDP triggered at step {step_id}")
        # 这里的 trick 是：让代码以为上一步就是 Action 5 (Look Down)
        # 这样就会直接进入下面的 if GLOBAL_STATE["last_action"] == 5 分支
        GLOBAL_STATE["last_action"] = 5 
        # 清空 Buffer，因为我们要重新生成轨迹
        GLOBAL_STATE["action_queue"] = []
    else:
        # 只有在非强制模式下，才检查 Buffer
        if len(GLOBAL_STATE["action_queue"]) > 0:
            next_action = GLOBAL_STATE["action_queue"].pop(0)
            # print(f"⏩ [Buffer] Serving action {next_action}. Remaining: {len(GLOBAL_STATE['action_queue'])}")
            
            if next_action == 5:
                GLOBAL_STATE["last_action"] = 5
            elif next_action == 4:
                pass 
            else:
                GLOBAL_STATE["last_action"] = next_action
                
            return {"actions": [next_action]}

    # [LOGIC] Model Inference
    input_images = []
    for frame in history:
        input_images.append(Image.fromarray(frame['rgb']).convert('RGB'))
    
    current_depth_raw = np.array(current_obs['depth']) 
    if current_depth_raw.ndim == 3: current_depth_raw = current_depth_raw.squeeze(-1)
    
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    user_content_list = []
    
    # 1. Base Instruction (Always present!)
    base_instr = f"You are an autonomous navigation assistant. Your task is to {instruction}. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task."
    
    # 根据 Look Down 状态决定是否包含历史
    # 你的 Prompt 显示：Look Down 时依然有历史！
    # 所以我们统一处理历史，不管是普通模式还是低头模式
    
    if len(input_images) > 1:
        base_instr += " These are your historical observations:"
    
    user_content_list.append({"type": "text", "text": base_instr})
    
    # 2. History Images (All except last one)
    # 如果是 Look Down 模式 (last_action=5)，那么 history 里的最后一张图是“俯视图”。
    # 倒数第二张是“平视图”。
    # 普通模式：[历史..., 平视(Current)]
    # 低头模式：[历史..., 平视, 俯视(Current)]
    
    # 我们需要根据模式切分图片
    if GLOBAL_STATE["last_action"] == 5:
        # 此时 input_images[-1] 是俯视图，input_images[-2] 是平视图
        # User 1 应该包含到 平视图 为止
        history_end_idx = len(input_images) - 2 # 排除俯视图和本轮平视图(作为当前图)
        current_flat_idx = len(input_images) - 2
        current_down_idx = len(input_images) - 1
        
        # 边界检查：如果历史太短（例如刚开局就低头，虽然不太可能）
        if current_flat_idx < 0: current_flat_idx = 0
        if history_end_idx < 0: history_end_idx = 0
        
    else:
        # 普通模式
        history_end_idx = len(input_images) - 1
        current_flat_idx = len(input_images) - 1
        current_down_idx = -1 # 不存在

    # 插入历史图片
    for i in range(history_end_idx):
        user_content_list.append({"type": "image", "image": input_images[i]})
        
    # 3. Current Image (Flat View)
    # 不管什么模式，第一轮 User 结尾总是平视图
    prompt_suffix = req.get("prompt_suffix", "you can see ")
    
    # try:
    #     prompt_suffix = random.choice(CONJUNCTIONS)
    # except:
    #     pass
        
    # [Fix] 根据你的 step_0_prompt.txt，conjunction 前面没有点
    # 但 step_10_prompt.txt 里有点?
    # 你的 Prompt: `...task. These are your historical observations:<|vision|>...<|vision|>. you can spot...`
    # 注意那个 `.`
    
    # 让我们复刻原代码逻辑：
    # 如果有历史，加历史。
    # 然后 `sources[0]["value"] += f" {prompt}."`
    
    user_content_list.append({"type": "text", "text": f" {prompt_suffix}"})
    
    # 插入平视图
    user_content_list.append({"type": "image", "image": input_images[current_flat_idx]})
    user_content_list.append({"type": "text", "text": "."})
    
    messages.append({"role": "user", "content": user_content_list})
    
    # 4. [Multi-turn Logic] Append Downward View
    if GLOBAL_STATE["last_action"] == 5:
        #print("⚡ DEBUG: Last action was Look Down (5). Appending Downward View...")
        
        # Assistant: ↓
        messages.append({"role": "assistant", "content": [{"type": "text", "text": "↓"}]})
        
        # User: you are toward the <image>.
        # 注意：原代码用的也是 random.choice(conjunctions)
        # 你提供的 Prompt 里是 "you are toward the"
        prompt_suffix_2 = "you can spot " # 默认值
        try:
            prompt_suffix_2 = random.choice(CONJUNCTIONS)
        except:
            pass

        user_content_2 = [
            {"type": "text", "text": f"{prompt_suffix_2}"},
            {"type": "image", "image": input_images[current_down_idx]}, 
            {"type": "text", "text": "."}
        ]
        messages.append({"role": "user", "content": user_content_2})

    # 3. Generate Final Text Input
    text_input = global_processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # [DEBUG Log]
    DEBUG_DIR = "debug_logs/server"
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    # 传入 input_images，确保数量对齐。
    # 如果是多轮模式，我们上面已经用到了 input_images 的所有图（包括最后一张），所以直接传 input_images 即可。
    inputs = global_processor(text=[text_input], images=input_images, return_tensors="pt").to(device)
    
    # with open(os.path.join(DEBUG_DIR, f"step_{step_id}_prompt.txt"), "w") as f:
    #     f.write(global_processor.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False))

    # Inference
    with torch.no_grad():
        output_ids = global_model.generate(**inputs, max_new_tokens=128, do_sample=False)
    llm_outputs = global_processor.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    #print(f"[Server] Step: {step_id} | Output: {llm_outputs}")

    new_action_sequence = []
    pixel_goal_to_return = None

    # === NavDP Branch ===
    if bool(re.search(r'\d', llm_outputs)):
        #print("🎯 DEBUG: Coordinate detected! Triggering NavDP...")
        coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]
        pixel_goal = [int(coord[1]), int(coord[0])] 

        pixel_goal_to_return = pixel_goal       
        # NavDP Prep
        min_depth = req.get("min_depth", 0.0)
        max_depth = req.get("max_depth", 10.0)
        current_depth_filtered = filter_depth(current_depth_raw, blur_type=None)
        depth_in_meters = current_depth_filtered * (max_depth - min_depth) + min_depth
        processed_depth_float = depth_in_meters * 1000.0
        depth_uint16 = processed_depth_float.astype(np.uint16)
        depth_pil_uint16 = Image.fromarray(depth_uint16, mode='I;16')

        look_down_image = input_images[-1].copy() 
        #pix_goal_image_pil = dot_matrix_two_dimensional(np.array(look_down_image), pixel_goal=pixel_goal)
        img_dp_raw = look_down_image.resize((224, 224))
        # img_array_for_dot = np.array(img_dp_raw)
        # img_goal_pil = dot_matrix_two_dimensional(img_array_for_dot, pixel_goal=pixel_goal)
        # img_goal_raw = img_goal_pil
        # img_goal_raw = img_dp_raw.copy()
        image_dp = torch.tensor(np.array(img_dp_raw)).to(torch.bfloat16).to(device) / 255.0
        pix_goal_image = torch.tensor(np.array(img_dp_raw)).to(torch.bfloat16).to(device) / 255.0
        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0) 
        depth_processed = preprocess_depth_image_v2(depth_pil_uint16, 224, 224, True, 1000)
        depth_tensor = torch.as_tensor(np.ascontiguousarray(depth_processed)).float().to(device)
        depth_tensor[depth_tensor > 5.0] = 5.0
        depth_dp = depth_tensor.unsqueeze(-1).to(torch.bfloat16) 
        depths_dp = torch.stack([copy.copy(depth_dp), depth_dp]).unsqueeze(0)
        
        # [NavDP Fix] Use Correct Inputs for Latents (Now with Instruction!)
        with torch.no_grad():
            traj_latents = global_model.generate_latents(
                output_ids, inputs.pixel_values, 
                inputs.image_grid_thw if hasattr(inputs, 'image_grid_thw') else None
            )
            dp_actions = global_model.generate_traj(traj_latents, images_dp, depths_dp, use_async=True)
            
        raw_navdp_actions = traj_to_actions(dp_actions)

        if len(raw_navdp_actions) > 0 and raw_navdp_actions[0] == 0:
            # 注意：这里要非常小心，如果真的到了终点应该 stop。
            # 但原代码这里确实强制给了 action 2。如果你想完全对齐：
            #print("⚠️ NavDP returned STOP immediately. Fallback to Action 2 (Random Turn).")
            # 替换掉原本的 0，或者在抬头后执行
            # 考虑到你有 [4, 4] 前置，这里的 0 会被放到 [4, 4, 0]
            # 建议修改 raw_navdp_actions
            raw_navdp_actions[0] = 2 
            # 重新构建
            new_action_sequence =  raw_navdp_actions
        
        # [Fix] Pad if length < 8 (Match original logic)
        if len(raw_navdp_actions) < 8:
            raw_navdp_actions += [0] * (8 - len(raw_navdp_actions))
        #print(f"🎯 DEBUG: Raw NavDP Actions (Continuous): {raw_navdp_actions}")
        
        # [Fix] Action Slicing (Max 4 steps)
        if len(raw_navdp_actions) >= 4:
             #print(f"✂️ [Align] Slicing NavDP actions from {len(raw_navdp_actions)} to 4 steps.")
             raw_navdp_actions = raw_navdp_actions[:4]

        # [Critical Fix] Restore Original Logic: Double Look Up!
        new_action_sequence =  raw_navdp_actions
        
        #print(f"🎯 DEBUG: Final NavDP Actions: {new_action_sequence}")
        GLOBAL_STATE["last_action"] = 0 

    # === Text Actions Branch ===
    else:
        parsed_actions = parse_actions(llm_outputs)
        new_action_sequence = []
        
        # [Critical Fix] Restore Original Logic: Double Look Down!
        for a in parsed_actions:
            new_action_sequence.append(a)
            if a == 5:
                #print("⚡ DEBUG: Detected Action 5. Doubling it to match original logic.")
                new_action_sequence.append(5)

    # [LOGIC] Fill Buffer
    if len(new_action_sequence) == 0:
        new_action_sequence = [0]

    GLOBAL_STATE["action_queue"] = new_action_sequence
    
    next_action = GLOBAL_STATE["action_queue"].pop(0)
    
    if next_action == 5:
        GLOBAL_STATE["last_action"] = 5
    else:
        GLOBAL_STATE["last_action"] = next_action

    final_actions = [int(next_action)]
    response = {"actions": final_actions}
    if pixel_goal_to_return is not None:
        response["pixel_goal"] = pixel_goal_to_return
        
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)