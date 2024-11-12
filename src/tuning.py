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
import hashlib
import hmac


class CreateTaskExecutor:
    def __init__(self, host, uri, method, iam_access_key, secret_key, request_id):
        self._host = host
        self._uri = uri
        self._method = method
        self._api_gw_time = str(int(time.time() * 1000))
        self._iam_access_key = iam_access_key
        self._secret_key = secret_key
        self._request_id = request_id

    def _make_signature(self):
        secret_key = bytes(self._secret_key, "UTF-8")
        message = (
            self._method
            + " "
            + self._uri
            + "\n"
            + self._api_gw_time
            + "\n"
            + self._iam_access_key
        )
        message = bytes(message, "UTF-8")
        signing_key = base64.b64encode(
            hmac.new(secret_key, message, digestmod=hashlib.sha256).digest()
        )
        return signing_key

    def _send_request(self, create_request):

        headers = {
            "X-NCP-APIGW-TIMESTAMP": self._api_gw_time,
            "X-NCP-IAM-ACCESS-KEY": self._iam_access_key,
            "X-NCP-APIGW-SIGNATURE-V2": self._make_signature(),
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
        }
        result = requests.post(
            self._host + self._uri, json=create_request, headers=headers
        ).json()
        return result

    def execute(self, create_request):
        res = self._send_request(create_request)
        if "status" in res and res["status"]["code"] == "20000":
            return res["result"]
        else:
            return res


class FindTaskExecutor:
    def __init__(self, host, uri, method, iam_access_key, secret_key, request_id):
        self._host = host
        self._uri = uri
        self._method = method
        self._api_gw_time = str(int(time.time() * 1000))
        self._iam_access_key = iam_access_key
        self._secret_key = secret_key
        self._request_id = request_id

    def _make_signature(self, task_id):
        secret_key = bytes(self._secret_key, "UTF-8")
        message = (
            self._method
            + " "
            + self._uri
            + task_id
            + "\n"
            + self._api_gw_time
            + "\n"
            + self._iam_access_key
        )
        message = bytes(message, "UTF-8")
        signing_key = base64.b64encode(
            hmac.new(secret_key, message, digestmod=hashlib.sha256).digest()
        )
        return signing_key

    def _send_request(self, task_id):
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "X-NCP-APIGW-TIMESTAMP": self._api_gw_time,
            "X-NCP-IAM-ACCESS-KEY": self._iam_access_key,
            "X-NCP-APIGW-SIGNATURE-V2": self._make_signature(task_id),
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
        }

        result = requests.get(self._host + self._uri + task_id, headers=headers).json()
        return result

    def execute(self, taskId):
        res = self._send_request(taskId)
        if "status" in res and res["status"]["code"] == "20000":
            return res["result"]
        else:
            return res


def create_task(file_name: str):
    with open("api_info.yaml") as f:
        h_params = yaml.load(f, Loader=yaml.FullLoader)["colab"]

    completion_executor = CreateTaskExecutor(
        host="https://clovastudio.apigw.ntruss.com",
        uri="/tuning/v2/tasks",
        method="POST",
        iam_access_key=h_params["access_key"],
        secret_key=h_params["secret_key"],
        request_id=h_params["request_id"][0],
    )

    request_data = {
        "name": "prompt_workshop_practice",
        "model": h_params["model"],
        "method": "LORA",
        "taskType": "GENERATION",
        "trainEpochs": 4,
        "learningRate": 1e-4,
        "trainingDatasetBucket": "dhlab.workshop",
        "trainingDatasetFilePath": file_name + ".csv",
        "trainingDatasetAccessKey": h_params["access_key"],
        "trainingDatasetSecretKey": h_params["secret_key"],
    }

    response = completion_executor.execute(request_data)
    print(response)
    return response


def find_task(task_id: str):
    with open("api_info.yaml") as f:
        h_params = yaml.load(f, Loader=yaml.FullLoader)["colab"]

    completion_executor = FindTaskExecutor(
        host="https://clovastudio.apigw.ntruss.com",
        uri="/tuning/v2/tasks/",
        method="GET",
        iam_access_key=h_params["access_key"],
        secret_key=h_params["secret_key"],
        request_id=h_params["request_id"][0],
    )

    response = completion_executor.execute(task_id)
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
