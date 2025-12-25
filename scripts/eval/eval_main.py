#本文件是运行的主函数，可更改内容如下：
'''
1.    # ===== 1. Trajectory Client =====
    traj_client = Gr00tTrajectoryClient(
        url=f"http://{args.gr00t_host}:{args.gr00t_port}/act"
    )
    这部分是连接http server的方式，创建一个自己的client（例如已有的gr00t的InternNav/internnav/evaluator/HTTPTrajectoryClient.py）。
    从其中声明函数类在这里实现即可
2.    # * 2. initialize evaluator
    # ===== 1. 构建 Agent=====
    agent = LLMAgent(
        traj_client=traj_client,
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
    这部分是构建agent及Evaluator，便于之后的运行，相关参数大体更换为新的即可。
    如果有新的要声明的参数在InternNav/internnav/evaluator/final_habitat_vln_evaluator.py声明
'''
import argparse
import json
import os
import numpy as np
import torch
from dataclasses import dataclass

from internnav.utils.dist import *
from internnav.evaluator.final_habitat_vln_evaluator import Evaluator
from internnav.evaluator.final_habitat_vln_evaluator import LLMAgent
from internnav.evaluator.HTTPTrajectoryClient import Gr00tTrajectoryClient

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

    ###连接server_Gr00t，后续使用只需要换成正确的url即可，见main()的 ===== 1. Trajectory Client =====
    parser.add_argument('--gr00t_host', default='127.0.0.1')
    parser.add_argument('--gr00t_port', default=9000, type=int)  

    return parser.parse_args()

def main():
    args = parse_args()
    print("[[[[args.mode]]]]", args.mode)

    init_distributed_mode(args)
    local_rank = args.local_rank
    np.random.seed(local_rank)

    # ===== 1. Trajectory Client =====
    traj_client = Gr00tTrajectoryClient(
        url=f"http://{args.gr00t_host}:{args.gr00t_port}/act"
    )

    # * 2. initialize evaluator
    # ===== 1. 构建 Agent=====
    agent = LLMAgent(
        traj_client=traj_client,
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

    # ===== 3. 运行评测 =====  这部分一般不用改，除非有什么最后要加的除推理之外的逻辑，计算部分已经放在了Evaluator()
    evaluator.run()

if __name__ == "__main__":
    main()

