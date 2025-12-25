import argparse
import json
import os
import numpy as np
import torch
from dataclasses import dataclass

from internnav.utils.dist import *
from internnav.evaluator.final_habitat_vln_evaluator import Evaluator
from internnav.evaluator.final_habitat_vln_evaluator import LLMAgent
###############################################################
# ### MODIFIED: 添加一个简单的 Gr00t HTTP 客户端
###############################################################
import requests
import json_numpy
json_numpy.patch()

class Gr00tHTTPClient:
    """Simple wrapper to call GR00T HTTP inference server."""
    def __init__(self, host="127.0.0.1", port=8000):
        self.url = f"http://{host}:{port}/act"

    def get_action(self, obs):
        resp = requests.post(self.url, json={"observation": obs})
        return resp.json()

###############################################################

def parse_args():

    parser = argparse.ArgumentParser(description='Evaluate InternVLA-N1 on Habitat')
    parser.add_argument("--mode", default='dual_system', type=str, help="inference mode: dual_system or system2")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='scripts/eval/configs/vln_r2r_no_oracle.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./logs/habitat/test')  #!
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--predict_step_nums", type=int, default=16)
    parser.add_argument("--continuous_traj", action="store_true", default=False)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int, help='rank')
    parser.add_argument('--gpu', default=0, type=int, help='gpu')
    parser.add_argument('--port', default='2333')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    parser.add_argument('--gr00t_host', default='127.0.0.1')
    parser.add_argument('--gr00t_port', default=8000, type=int)

    return parser.parse_args()

def main():
    args = parse_args()
    print("[[[[args.mode]]]]", args.mode)

    init_distributed_mode(args)
    local_rank = args.local_rank
    np.random.seed(local_rank)

    ###########################################################################
    # 🔥 MODIFIED: 删除 InternVLA 模型加载 + processor，改为 GR00T 客户端
    ###########################################################################
    print("### MODIFIED: Using GR00T HTTP Client instead of InternVLA model ###")

    model = Gr00tHTTPClient(host=args.gr00t_host, port=args.gr00t_port)
    processor = None     # Habitat evaluator仍然需要该参数，但我们传 None

    # No torch dtype, no flash attention
    ###########################################################################

    world_size = get_world_size()

    # * 2. initialize evaluator
    # ===== 1. 构建 Agent（用 Gr00t）=====
    agent = LLMAgent(
        model=model,
        processor=None,   # Gr00t 不需要 processor
        args=args,
        device="cuda",
    )

    # ===== 2. 构建 Evaluator =====
    evaluator = Evaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        output_path=args.output_path,
        args=args,
        agent=agent,
        max_steps=500,
        idx=get_rank(),
        env_num=get_world_size(),
    )

    # ===== 3. 运行评测 =====
    evaluator.run()

if __name__ == "__main__":
    main()

