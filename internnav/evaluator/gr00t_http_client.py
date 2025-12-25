import requests
import numpy as np
import json_numpy

json_numpy.patch()
class Gr00tHTTPClient:
    def __init__(self, url="http://127.0.0.1:8000/act"):
        self.url = url

    def get_action(self, obs):
        # obs: dict with "video.ego_view", "state.drone", ...
        obs_json = json_numpy.dumps({"observation": obs})
        response = requests.post(self.url, data=obs_json, headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        return json_numpy.loads(response.text)
