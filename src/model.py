import http.client
import base64
import json, jsonlines
import asyncio
import os
import re
import csv

import boto3
import requests
import pandas as pd
from openai import AsyncClient

from src.util import load_questions, load_yaml, load_kmle


class KMLE:
    def __init__(self, api_key: str, primary_val: str, request_id: str):
        self.kmle = load_kmle()
        self.prompts = self._set_prompt()
        self.api_info = load_yaml("api_info.yaml")
        self.h_params = self.api_info["colab"]

        self.hcx = CompletionExecutor(
            host=self.h_params["host"],
            api_key=api_key,
            api_key_primary_val=primary_val,
            request_id=request_id,
        )

    def show_questions(self):
        return pd.DataFrame(self.kmle)[
            ["problem_category", "question", "options", "answer"]
        ]

    def test(self, prompt: str):
        """
        Tests the model's response to a given prompt.

        Args:
            prompt (str): The input text prompt for the model.

        Returns:
            str: The response generated by the model.
        """
        request_data = {
            "messages": [{"role": "user", "content": prompt}],
            "maxTokens": self.h_params["max_tokens"],
            "topP": self.h_params["top_p"],
            "temperature": self.h_params["temperature"],
            "repeatPenalty": self.h_params["repeat_penalty"],
        }
        return self.hcx.execute(request_data) # ["content"]

    async def run_test(self, system_prompt: str, start: int, end: int):
        ans_pattern = r"\((\d+)\)"
        message = []

        async def execute_request(prompt: str):
            request_data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "maxTokens": self.h_params["max_tokens"],
                "topP": self.h_params["top_p"],
                "temperature": self.h_params["temperature"],
                "repeatPenalty": self.h_params["repeat_penalty"],
            }
            response_text = await self.hcx.execute_async(request_data)
            if "content" in response_text:
                return response_text["content"]

        tasks = [
            execute_request(prompt)
            for _, prompt in enumerate(self.prompts[start : end + 1])
        ]

        for i in range(0, len(tasks), 5):
            print(f"Started generating #{i}.")
            result = await asyncio.gather(*tasks[i : i + 5])
            message.extend(result)
            await asyncio.sleep(1)

        df = pd.DataFrame(
            {
                "question": [data["question"] for data in self.kmle[start : end + 1]],
                "options": [data["options"] for data in self.kmle[start : end + 1]],
                "answer": [
                    data["answer_idx"][0] for data in self.kmle[start : end + 1]
                ],
                "pred_ori": message,
            }
        )

        df["pred"] = df["pred_ori"].apply(
            lambda answer: (
                re.search(ans_pattern, answer).group(1)
                if re.search(ans_pattern, answer)
                else 0
            )
        )

        score = (df["pred"] == df["answer"]).sum()
        print(f"점수: {score}")

        return df

    async def run(self, system_prompt: str, file_name: str):
        """
        Generates results for a given set of questions using prompts, and outputs the results as an Excel file.

        Args:
            system_prompt (str): A string containing the system prompt that needs to be processed.
            file_name (str): The name of the Excel file where the results will be saved.

        Returns:
            df: generated Excel file containing the results.
        """
        os.makedirs("output", exist_ok=True)
        ans_pattern = r"\((\d+)\)"
        path = f"output/{file_name}.xlsx"
        message = [None] * len(self.prompts)

        async def execute_request(idx: int, prompt: str):
            request_data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "maxTokens": self.h_params["max_tokens"],
                "topP": self.h_params["top_p"],
                "temperature": self.h_params["temperature"],
                "repeatPenalty": self.h_params["repeat_penalty"],
            }
            response_text = await self.hcx.execute_async(request_data)
            if "content" in response_text:
                message[idx] = response_text["content"]

        tasks = [
            execute_request(idx, prompt) for idx, prompt in enumerate(self.prompts)
        ]

        for i in range(0, len(tasks), 5):
            print(f"Started generating #{i}.")
            await asyncio.gather(*tasks[i : i + 5])
            await asyncio.sleep(1)

        df = pd.DataFrame(
            {
                "question": [data["question"] for data in self.kmle],
                "options": [data["options"] for data in self.kmle],
                "answer": [data["answer_idx"][0] for data in self.kmle],
                "pred_ori": message,
            }
        )

        df["pred"] = df["pred_ori"].apply(
            lambda answer: (
                re.search(ans_pattern, answer).group(1)
                if re.search(ans_pattern, answer)
                else 0
            )
        )

        score = (df["pred"] == df["answer"]).sum()
        print(f"점수: {score}")

        df.to_excel(path, index=False)

        return df
    
    async def fill_nan(self, system_prompt: str, file_name: str):
        ans_pattern = r"\((\d+)\)"
        df = pd.read_excel(f"output/{file_name}.xlsx")
        idx = list(df[df["pred_ori"].isna()].index)
        if not idx:
            return
        message = []

        nan_prompt = [self.prompts[i] for i in idx]

        async def execute_request(user_prompt: str):
            request_data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "maxTokens": self.h_params["max_tokens"],
                "topP": self.h_params["top_p"],
                "temperature": self.h_params["temperature"],
                "repeatPenalty": self.h_params["repeat_penalty"],
            }
            response_text = await self.hcx.execute_async(request_data)
            if "content" in response_text:
                return response_text["content"]

        tasks = [execute_request(prompt) for _, prompt in enumerate(nan_prompt)]

        for i in range(0, len(tasks), 5):
            print(f"Started generating #{i}.")
            result = await asyncio.gather(*tasks[i : i + 5])
            message.extend(result)
            await asyncio.sleep(1)

        for i, x in zip(idx, message):
            df.loc[i, "pred_ori"] = x

        df["pred"] = df["pred_ori"].apply(
            lambda answer: (
                int(re.search(ans_pattern, answer).group(1))
                if re.search(ans_pattern, answer)
                else 0
            )
        )

        score = (df["pred"] == df["answer"]).sum()
        print(f"점수: {score}")

        path = f"output/{file_name}.xlsx"
        df.to_excel(path, index=False)

        return df

    def _set_prompt(self, kmle=None):
        if not kmle:
            kmle = self.kmle
        prompt_template = load_yaml("prompt/kmle.yaml")

        prompt = prompt_template["user_prompt"]
        prompts = [
            prompt.format(
                **{
                    "CONTEXT": data["context"],
                    "QUESTION": data["question"],
                    "OPTIONS": "\n".join(
                        [f"{key}. {value}" for key, value in data["options"].items()]
                    ),
                }
            )
            for data in kmle
        ]

        return prompts

    def get_questions(self, idx: int):
        data = self.kmle[idx]
        context = data["context"]
        question = data["question"]
        options = "\n".join(f"{key}. {value}" for key, value in data["options"].items())
        answer = data["answer_idx"][0]

        result = (
            f"[문제]\n{context}\n{question}\n\n[보기]\n{options}\n\n[정답]\n{answer}"
        )

        print(result)

    def get_user_prompt(self, idx: int):
        print(self.prompts[idx])

    def generate_tuning_data(self, system_prompt: str, file_name: str):
        os.makedirs("tuning_data", exist_ok=True)

        kmle = load_kmle(train=True)
        prompts = self._set_prompt(kmle)

        path = f"tuning_data/{file_name}.csv"
        f = open(path, "w")
        writer = csv.writer(f)

        writer.writerow(["System_Prompt", "C_ID", "T_ID", "Text", "Completion"])
        for idx, prompt in enumerate(prompts):
            label = f"[정답] ({', '.join(kmle[idx]['answer_idx'])}) {', '.join(kmle[idx]['answer'])}"
            writer.writerow([system_prompt, idx, 0, prompt, label])

        f.close()

        s3 = boto3.client(
            service_name="s3",
            endpoint_url="https://kr.object.ncloudstorage.com",
            region_name="kr-standard",
            aws_access_key_id=self.h_params["access_key"],
            aws_secret_access_key=self.h_params["secret_key"],
        )

        s3.upload_file(path, "dhlab.workshop", file_name + ".csv")


class BananaPunch:
    def __init__(self, api_key: str, apigw_api_key: str, request_id: str, gpt_key: str):
        self.prompts = load_yaml("prompt/banana.yaml")
        self.questions = load_questions()
        self.api_info = load_yaml("api_info.yaml")
        self.h_params = self.api_info["colab"]
        self.gpt_key = gpt_key

        self.hcx = CompletionExecutor(
            host=self.h_params["host"],
            api_key=api_key,
            api_key_primary_val=apigw_api_key,
            request_id=request_id,
        )
        self.gpt = AsyncClient(api_key=self.gpt_key)

    def test(self, prompt: str):
        """
        Tests the model's response to a given prompt.

        Args:
            prompt (str): The input text prompt for the model.

        Returns:
            str: The response generated by the model.
        """
        request_data = {
            "messages": [{"role": "user", "content": prompt}],
            "maxTokens": self.h_params["max_tokens"],
            "topP": self.h_params["top_p"],
            "temperature": self.h_params["temperature"],
            "repeatPenalty": self.h_params["repeat_penalty"],
        }
        return self.hcx.execute(request_data)["content"]

    async def run_test(self, system_prompt: str, section: str, start: int, end: int):
        user_prompt = self.prompt_preprocessing(section)[start : end + 1]
        message = []

        async def execute_request(user_prompt: str):
            request_data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "maxTokens": self.h_params["max_tokens"],
                "topP": self.h_params["top_p"],
                "temperature": self.h_params["temperature"],
                "repeatPenalty": self.h_params["repeat_penalty"],
            }
            response_text = await self.hcx.execute_async(request_data)
            if "content" in response_text:
                return response_text["content"]

        tasks = [execute_request(prompt) for _, prompt in enumerate(user_prompt)]

        for i in range(0, len(tasks), 5):
            print(f"Started generating #{i}.")
            result = await asyncio.gather(*tasks[i : i + 5])
            message.extend(result)
            await asyncio.sleep(1)

        df = self.questions[section][start : end + 1]
        df.loc[:, "pred"] = message

        if section == "intent_classifier":
            score = (df["pred"] == df["의도 분류"]).sum()
            print(f"점수: {score}")

        return df

    async def run(self, system_prompt: str, section: str, file_name: str):
        """
        Generates results for a given set of questions using prompts, and outputs the results as an Excel file.

        Args:
            system_prompt (str): A string containing the system prompt that needs to be processed.
            section (str): Specifies which section of the questions to run.
                        Options include:
                        - 'intent_classifier': Processes questions related to intent classification.
                        - 'store_inquiry_handler': Handles questions pertaining to store inquiries.
                        - 'product_inquiry_handler': Manages questions about product inquiries.
                        - 'unwanted_topic_blocker': Blocks questions related to unwanted topics.

            file_name (str): The name of the Excel file where the results will be saved.

        Returns:
            DataFrame: The file to the generated Excel file containing the results.
        """
        os.makedirs("output", exist_ok=True)
        user_prompt = self.prompt_preprocessing(section)
        message = [None] * len(user_prompt)

        async def execute_request(idx: int, user_prompt: str):
            request_data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "maxTokens": self.h_params["max_tokens"],
                "topP": self.h_params["top_p"],
                "temperature": self.h_params["temperature"],
                "repeatPenalty": self.h_params["repeat_penalty"],
            }
            response_text = await self.hcx.execute_async(request_data)
            if "content" in response_text:
                message[idx] = response_text["content"]

        tasks = [execute_request(idx, prompt) for idx, prompt in enumerate(user_prompt)]

        for i in range(0, len(tasks), 5):
            print(f"Started generating #{i}.")
            await asyncio.gather(*tasks[i : i + 5])
            await asyncio.sleep(1)

        df = self.questions[section]
        df["pred"] = message

        if section == "intent_classifier":
            score = (df["pred"] == df["의도 분류"]).sum()
            print(f"점수: {score}")

        path = f"output/{file_name}.xlsx"
        df.to_excel(path, index=False)

        return df

    async def evaluate(self, file_name: str):
        """
        Evaluates the results in an Excel file generated by the `run` function and updates the file with scores.

        Args:
            file_name (str): The name of the Excel file to be evaluated and updated.

        Returns:
            str: The path to the updated Excel file containing the scores.
        """
        path = f"output/{file_name}.xlsx"
        df = pd.read_excel(path)
        message = [None] * len(df)
        prompts = [
            self.prompts["evaluate_prompt"].format(**{"data": f"질문: {q}\n답변: {p}"})
            for q, p in zip(df["질문"], df["pred"])
        ]
        print(f"Started evaluating #{len(message)}.")

        async def execute_request(idx, prompt, model="gpt-4o-mini"):
            completion = await self.gpt.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=256,
            )
            message[idx] = completion.choices[0].message.content

        tasks = [execute_request(idx, prompt) for idx, prompt in enumerate(prompts)]

        await asyncio.gather(*tasks)
        message = [int(meg.split("score:")[-1].strip()) for meg in message]
        print(f"점수: {sum(message) / len(message):.2f}")
        df["score"] = message
        df.to_excel(path, index=False)

        return path

    async def fill_nan(self, system_prompt: str, section: str, file_name: str):
        df = pd.read_excel(f"output/{file_name}.xlsx")
        idx = list(df[df["pred"].isna()].index)
        if not idx:
            return
        message = []

        user_prompt = self.prompt_preprocessing(section)
        nan_prompt = [user_prompt[i] for i in idx]

        async def execute_request(user_prompt: str):
            request_data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "maxTokens": self.h_params["max_tokens"],
                "topP": self.h_params["top_p"],
                "temperature": self.h_params["temperature"],
                "repeatPenalty": self.h_params["repeat_penalty"],
            }
            response_text = await self.hcx.execute_async(request_data)
            if "content" in response_text:
                return response_text["content"]

        tasks = [execute_request(prompt) for _, prompt in enumerate(nan_prompt)]

        for i in range(0, len(tasks), 5):
            print(f"Started generating #{i}.")
            result = await asyncio.gather(*tasks[i : i + 5])
            message.extend(result)
            await asyncio.sleep(1)

        for i, x in zip(idx, message):
            df.loc[i, "pred"] = x

        if section == "intent_classifier":
            score = (df["pred"] == df["의도 분류"]).sum()
            print(f"점수: {score}")

        path = f"output/{file_name}.xlsx"
        df.to_excel(path, index=False)

        return df

    def prompt_preprocessing(self, section: str):
        questions = self.questions[section]["질문"]
        return [
            self.prompts["user_prompt"].format(**{"question": q}) for q in questions
        ]

    def get_questions(self, section: str, idx: int):
        data = self.questions[section]
        question = data.iloc[idx, 0]
        sample_answer = data.iloc[idx, 1]

        print(f"{data.columns[0]}: {question}")
        print(f"{data.columns[1]}: {sample_answer}")

    def get_user_prompt(self, section: str, idx: int):
        user_prompt = self.prompt_preprocessing(section)
        print(user_prompt[idx])

    def __repr__(self):
        return "바나나펀치에 오신 여러분 환영합니다!"


class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def execute(self, completion_request):
        headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": self._api_key,
            "X-NCP-APIGW-API-KEY": self._api_key_primary_val,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
            "Content-Type": "application/json; charset=utf-8",
        }

        with requests.post(
            self._host + "/testapp/v1/chat-completions/HCX-003",
            headers=headers,
            json=completion_request,
        ) as r:
            response = r.content.decode("utf-8")
            try:
                result = eval(response)["result"]["message"]
            except:
                result = {"error": response}
            return result

    async def execute_async(self, completion_request):
        max_tries = 5
        try_cnt = 0

        while try_cnt < max_tries:
            headers = {
                "X-NCP-CLOVASTUDIO-API-KEY": self._api_key,
                "X-NCP-APIGW-API-KEY": self._api_key_primary_val,
                "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
                "Content-Type": "application/json; charset=utf-8",
            }
            with requests.post(
                self._host + "/testapp/v1/chat-completions/HCX-003",
                headers=headers,
                json=completion_request,
            ) as r:
                response = r.content.decode("utf-8")
                try:
                    # result = eval(response)["result"]["message"]
                    result = eval(response)
                    if result["status"]["code"] in [40100, 40101, 40102, 40103, 40104]:
                        raise ValueError("Authorization Error. Please check API Key")
                    elif result["status"]["code"] in [
                        "40400",
                        "42900",
                        "42901",
                        "50000",
                    ]:
                        try_cnt += 1
                    elif result["status"]["message"] != "OK":
                        raise ValueError(
                            f"Error: {result['status']['message']}({result['status']['code']})"
                        )
                    elif "content" not in result["result"]["message"]:
                        try_cnt += 1
                    else:
                        return result["result"]["message"]
                except:
                    try_cnt += 1
                await asyncio.sleep(2)

        return {"error": response}
