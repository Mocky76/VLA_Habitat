#本项目以Gr00t轨迹预测模型为例，通过 HTTP Trajectory Server 的方式接入 InternNav 的 Habitat VLN 评测流程
#便于后期更换模型接入接口等
主函数：InternNav/scripts/eval/eval_main.py 用于启动整个推理
用于推理：InternNav/scripts/eval/server_Gr00t.py 完成构造输入，调用模型启动推理，返回结果action。更换模型时照着这个文件的内容仿写一个，换为自己的逻辑即可
用于“HTTP 插头”：InternNav/internnav/evaluator/HTTPTrajectoryClient.py 此文件只用于继承一个BaseTrajectoryClient类，是HTTP Trajectory Server 的 Client，更换模型时按照模型需要使用正确方式包裹发送即可
基本类即函数定义：InternNav/internnav/evaluator/final_habitat_vln_evaluator.py 更换模型后如果有新的参数或逻辑，可以在这里添补
使用方法：
1.首先运行：
uvicorn InternNav.scripts.eval.server_Gr00t:app \
    --host 127.0.0.1 \
    --port 9000
2.接着运行：
python scripts/eval/eval_main.py --model_path /data/sjh/GR00T-Internva/output_uav/checkpoint-300000 --continuous_traj --output_path result/Gr00t/val_unseen_32traj_8steps --save_video