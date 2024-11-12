# -*- coding: utf-8 -*-

import base64
import json
import http.client
from http import HTTPStatus
import requests
import argparse
import pandas as pd
import csv
import re
from tqdm import tqdm
import yaml
import time
import time


class CreateTaskExecutor:
    def __init__(self, host, client_id, client_secret, access_token=None):
        self._host = host
        # client_id and client_secret are used to issue access_token.
        # You should not share this with others.
        self._client_id = client_id
        self._client_secret = client_secret
        # Base64Encode(client_id:client_secret)
        self._encoded_secret = base64.b64encode(
            "{}:{}".format(self._client_id, self._client_secret).encode("utf-8")
        ).decode("utf-8")
        self._access_token = access_token

    def _refresh_access_token(self):
        headers = {"Authorization": "Basic {}".format(self._encoded_secret)}

        conn = http.client.HTTPSConnection(self._host)
        # If existingToken is true, it returns a token that has the longest expiry time among existing tokens.
        conn.request("GET", "/v1/auth/token?existingToken=true", headers=headers)
        response = conn.getresponse()
        body = response.read().decode()
        conn.close()

        token_info = json.loads(body)
        self._access_token = token_info["result"]["accessToken"]

    def _send_request(self, create_task_request):
        headers = {"Authorization": "Bearer {}".format(self._access_token)}
        files = {
            "trainingDataset": (
                create_task_request["trainingDataset"].split("/")[-1],
                open(create_task_request["trainingDataset"], "rb"),
            )
        }
        data = {
            key: value
            for key, value in create_task_request.items()
            if key != "trainingDataset"
        }

        response = requests.post(
            f"https://{self._host}/v2/tasks", headers=headers, data=data, files=files
        )
        return response.json()

    def execute(self, create_task_request):
        if self._access_token is None:
            self._refresh_access_token()

        res = self._send_request(create_task_request)
        if res["status"]["code"] == "40103":
            # Check whether the token has expired and reissue the token.
            self._access_token = None
            return self.execute(create_task_request)
        elif res["status"]["code"] == "20000":
            return res["result"]
        else:
            return "Error"


class FindTaskExecutor:
    def __init__(self, host, client_id, client_secret, task_id, access_token=None):
        self._host = host
        # client_id and client_secret are used to issue access_token.
        # You should not share this with others.
        self._client_id = client_id
        self._client_secret = client_secret
        # Base64Encode(client_id:client_secret)
        self._encoded_secret = base64.b64encode(
            "{}:{}".format(self._client_id, self._client_secret).encode("utf-8")
        ).decode("utf-8")
        self._access_token = access_token
        self._task_id = task_id

    def _refresh_access_token(self):
        headers = {"Authorization": "Basic {}".format(self._encoded_secret)}

        conn = http.client.HTTPSConnection(self._host)
        # If existingToken is true, it returns a token that has the longest expiry time among existing tokens.
        conn.request("GET", "/v1/auth/token?existingToken=true", headers=headers)
        response = conn.getresponse()
        body = response.read().decode()
        conn.close()

        token_info = json.loads(body)
        self._access_token = token_info["result"]["accessToken"]

    def _send_request(self, find_task_request):
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": "Bearer {}".format(self._access_token),
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request(
            "GET", f"/v2/tasks/{self._task_id}", json.dumps(find_task_request), headers
        )
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding="utf-8"))
        conn.close()
        return result

    def execute(self, find_task_request):
        if self._access_token is None:
            self._refresh_access_token()

        res = self._send_request(find_task_request)
        if res["status"]["code"] == "40103":
            # Check whether the token has expired and reissue the token.
            self._access_token = None
            return self.execute(find_task_request)
        elif res["status"]["code"] == "20000":
            return res["result"]
        else:
            return "Error"


def create_task(file_name: str):
    with open("api_info.yaml") as f:
        h_params = yaml.load(f, Loader=yaml.FullLoader)["hcx_tuning"]

    create_task_executor = CreateTaskExecutor(
        host="api-hyperclova.navercorp.com",
        client_id=h_params["id"],
        client_secret=h_params["secret"],
    )

    request_data = {
        "name": "prompt_workshop_practice",
        "model": h_params["model"],
        "method": "LORA",
        "taskType": "GENERATION",
        "trainEpochs": 4,
        "learningRate": 1e-4,
        "trainingDataset": f"tuning_data/{file_name}.csv",
    }

    response = create_task_executor.execute(request_data)
    print(response)


def find_task(task_id: str):
    with open("api_info.yaml") as f:
        h_params = yaml.load(f, Loader=yaml.FullLoader)["hcx_tuning"]

    find_task_executor = FindTaskExecutor(
        host="api-hyperclova.navercorp.com",
        client_id=h_params["id"],
        client_secret=h_params["secret"],
        task_id=task_id,
    )

    request_data = {}

    response = find_task_executor.execute(request_data)
    print(response)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="간단 설명")
    parser.add_argument("--name", "-n", required=True, help="task_id")
    parser.add_argument(
        "--type",
        "-t",
        required=True,
        choices=["create", "find"],
        help="choice from (create, find)",
    )
    parser.add_argument("--data", "-d", help="data source")
    parser.add_argument("--epoch", "-e", help="epoch", default=4)
    parser.add_argument("--lr", "-l", help="learning rate", default=1e-4)
    args = parser.parse_args()

    with open("api_info.yaml") as f:
        h_params = yaml.load(f, Loader=yaml.FullLoader)["hcx_tuning"]

    if args.type == "create":
        create_task_executor = CreateTaskExecutor(
            host="api-hyperclova.navercorp.com",
            client_id=h_params["id"],
            client_secret=h_params["secret"],
        )

        request_data = {
            "name": args.name,
            "model": h_params["model"],
            "method": "LORA",
            "taskType": "GENERATION",
            "trainEpochs": args.epoch,
            "learningRate": args.lr,
            "trainingDataset": args.data,
        }

        response = create_task_executor.execute(request_data)
        print(response)

    elif args.type == "find":
        find_task_executor = FindTaskExecutor(
            host="api-hyperclova.navercorp.com",
            client_id=h_params["id"],
            client_secret=h_params["secret"],
            task_id=args.name,
        )

        request_data = {}

        response = find_task_executor.execute(request_data)
        print(response)
