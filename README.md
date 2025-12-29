#本项目以Gr00t轨迹预测模型为例，通过 HTTP Trajectory Server 的方式接入 InternNav 的 Habitat VLN 评测流程
#便于后期更换模型接入接口等
主函数：InternNav/scripts/eval/eval_main.py 用于启动整个推理
用于推理：InternNav/scripts/eval/server_Gr00t.py 完成构造输入，调用模型启动推理，返回结果action。更换模型时照着这个文件的内容仿写一个，换为自己的逻辑即可
用于“HTTP 插头”：InternNav/internnav/evaluator/HTTPTrajectoryClient.py 此文件只用于继承一个BaseTrajectoryClient类，是HTTP Trajectory Server 的 Client，更换模型时按照模型需要使用正确方式包裹发送即可
基本类即函数定义：InternNav/internnav/evaluator/final_habitat_vln_evaluator.py 更换模型后如果有新的参数或逻辑，可以在这里添补

环境配置：
# 1.创建环境
conda create -n <internnav> python=3.10 libxcb=1.14
conda activate <internnav>

# 2.安装pytorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118

# 3.安装依赖
pip install -e .[model] --no-build-isolation


# 4.安装habitat
conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab  # install habitat_lab
pip install -e habitat-baselines # install habitat_baselines

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
cd Path/to/InternNav/
pip install -e .[habitat]

运行方法：
1.首先运行：

uvicorn InternNav.scripts.eval.server_Gr00t:app \
    --host 127.0.0.1 \
    --port 9000
    
2.接着运行：
python scripts/eval/eval_main.py --model_path /data/sjh/GR00T-Internva/output_uav/checkpoint-300000 --continuous_traj --output_path result/Gr00t/val_unseen_32traj_8steps --save_video
