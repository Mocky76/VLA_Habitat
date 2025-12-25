#本文件仅用于连接，主要职责为将evaluator 里产生的observation dict，通过 HTTP 发给外部 Trajectory Server，并把返回结果还原成 Python 对象
import requests
from internnav.evaluator.final_habitat_vln_evaluator import BaseTrajectoryClient
import json_numpy
json_numpy.patch()

#更换模型只需要添加并调用一个新的类即可，以下面这个为例
class Gr00tTrajectoryClient(BaseTrajectoryClient):
    def __init__(self, url):
        self.url = url

    def reset(self, instruction: str, **kwargs):
        pass

    def query(self, obs: dict) -> list[int]:
        #包一层约定协议
        payload = {"observation": obs}
        #使用 HTTP 发送
        resp = requests.post(
            self.url,
            data=json_numpy.dumps(payload),  
            headers={"Content-Type": "application/json"},
            timeout=5.0,
        )
        resp.raise_for_status()

        #还原返回值
        return json_numpy.loads(resp.text)

