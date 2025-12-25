from internnav.model.utils.vln_utils import traj_to_actions_Gr00t

import torch
import numpy as np

from collections import Counter

def test_fake_turning_trajectory():
    """
    构造一条 L 型轨迹：
    - 前进 2.0m
    - 左转 90°
    - 再前进 1.5m
    """

    B = 1
    T = 32
    dp_actions = torch.zeros((B, T, 3), dtype=torch.float32)

    # InternVA / Gr00t 的 unnormalize: dp_actions[:, :, :2] /= 4
    scale = 4.0

    traj = []

    # -------- 1. 向前走 2m（x 方向）--------
    n1 = 16
    step1 = 2.0 / n1  # 每步 ~0.125m
    for _ in range(n1):
        traj.append([step1 * scale, 0.0, 0.0])

    # -------- 2. 左转（通过改变方向来体现）--------
    # 这里不直接用 dtheta，而是通过轨迹拐弯
    n2 = 16
    step2 = 1.5 / n2  # 每步 ~0.094m
    for _ in range(n2):
        traj.append([0.0, step2 * scale, 0.0])

    traj = np.array(traj)[:T]
    dp_actions[0, :len(traj), :] = torch.from_numpy(traj)

    print("Fake dp_actions:")
    print(dp_actions[0, :32])

    actions = traj_to_actions_Gr00t(dp_actions, use_discrate_action=True)

    print("\nOutput discrete actions:")
    print(actions)

    print("\nAction counts:")
    print(Counter(actions))


if __name__ == "__main__":
    # test_fake_turning_trajectory()
    action_indices = list(range(0, 64, 4))
    print(action_indices)
