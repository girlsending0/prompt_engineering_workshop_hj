import http.client
import base64
import json, jsonlines
import asyncio
import os
import re
import csv

import pandas as pd
from openai import AsyncClient

from src.util import load_questions, load_yaml, load_kmle


class KMLE:
    def __init__(self):
        self.kmle = load_kmle()
        self.prompts = self._set_prompt()
        self.api_info = load_yaml("api_info.yaml")
        self.h_params = self.api_info["hcx"]

        self.hcx = ChatCompletionExecutor(
            host=self.h_params["host"],
            client_id=self.h_params["id"],
            client_secret=self.h_params["secret"],
        )

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

    async def run(self, system_prompt: str, file_name: str):
        """
        Generates results for a given set of questions using prompts, and outputs the results as an Excel file.

        Args:
            system_prompt (str): A string containing the system prompt that needs to be processed.
            file_name (str): The name of the Excel file where the results will be saved.

        Returns:
            str: The path to the generated Excel file containing the results.
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

        for i in range(0, len(tasks), 10):
            print(f"Started generating #{i}.")
            await asyncio.gather(*tasks[i : i + 10])
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

        return file_name

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

        kmle = load_kmle(train=True, sampling=True)
        prompts = self._set_prompt(kmle)

        f = open(f"tuning_data/{file_name}.csv", "w")
        writer = csv.writer(f)

        writer.writerow(["System_Prompt", "C_ID", "T_ID", "Text", "Completion"])
        for idx, prompt in enumerate(prompts):
            label = f"[정답] ({', '.join(kmle[idx]['answer_idx'])}) {', '.join(kmle[idx]['answer'])}"
            writer.writerow([system_prompt, idx, 0, prompt, label])

        f.close()


class BananaPunch:
    def __init__(self):
        self.prompts = load_yaml("prompt/banana.yaml")
        self.questions = load_questions()
        self.api_info = load_yaml("api_info.yaml")
        self.h_params = self.api_info["hcx"]

        self.hcx = ChatCompletionExecutor(
            host=self.h_params["host"],
            client_id=self.h_params["id"],
            client_secret=self.h_params["secret"],
        )
        self.gpt = AsyncClient(api_key=self.api_info["gpt"]["api_key"])

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
            str: The file name to the generated Excel file containing the results.
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

        for i in range(0, len(tasks), 10):
            print(f"Started generating #{i}.")
            await asyncio.gather(*tasks[i : i + 10])
            await asyncio.sleep(1)

        df = self.questions[section]
        df["pred"] = message

        if section == "intent_classifier":
            score = (df["pred"] == df["의도 분류"]).sum()
            print(f"점수: {score}")

        path = f"output/{file_name}.xlsx"
        df.to_excel(path, index=False)

        return file_name

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


class ChatCompletionExecutor:
    def __init__(
        self, host, client_id, client_secret, task_id="HCX-003", access_token=None
    ):
        self._host = host
        self._client_id = client_id
        self._client_secret = client_secret
        self._encoded_secret = base64.b64encode(
            "{}:{}".format(self._client_id, self._client_secret).encode("utf-8")
        ).decode("utf-8")
        self._access_token = access_token
        self._task_id = task_id

    def _refresh_access_token(self):
        headers = {"Authorization": "Basic {}".format(self._encoded_secret)}

        conn = http.client.HTTPSConnection(self._host)
        conn.request("GET", "/v1/auth/token?existingToken=true", headers=headers)
        response = conn.getresponse()
        body = response.read().decode()
        conn.close()

        token_info = json.loads(body)
        self._access_token = token_info["result"]["accessToken"]

    def _send_request(self, chat_completion_request):
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "text/event-stream",
            "Authorization": "Bearer {}".format(self._access_token),
        }

        conn = http.client.HTTPSConnection(self._host)
        if self._task_id == "HCX-003" or self._task_id == "HCX-DASH-001":
            conn.request(
                "POST",
                f"/v1/chat-completions/{self._task_id}",
                json.dumps(chat_completion_request),
                headers,
            )
        else:
            conn.request(
                "POST",
                f"/v2/tasks/{self._task_id}/chat-completions",
                json.dumps(chat_completion_request),
                headers,
            )
        response = conn.getresponse()
        answer = response.read().decode("utf-8")
        answers = answer.split("\n\n")
        for answer in answers:
            if "event:result" in answer:
                break
        result = answer.split('"message":')[1]
        result = result.split("},")[0] + "}"
        result = result.replace("null", '""')
        try:
            result = eval(result)
        except:
            result = {"error": answer}
        conn.close()
        return result

    def execute(self, chat_completion_request):
        if self._access_token is None:
            self._refresh_access_token()

        res = self._send_request(chat_completion_request)
        return res

    async def execute_async(self, chat_completion_request):
        if self._access_token is None:
            self._refresh_access_token()

        res = self._send_request(chat_completion_request)
        return res
