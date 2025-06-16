import os
import time
import json
from openai import OpenAI

import numpy as np
from collections import defaultdict

import inference

# Pricing and default n_epochs estimate
MAX_TOKENS_PER_EXAMPLE = 16385

TARGET_EPOCHS = 3
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25

client = OpenAI(
        api_key = ""
    )


# データファイルのアップロード
def upload_file(file_name: str, purpose: str) -> str:
    with open(file_name, "rb") as file_fd:
        response = client.files.create(file=file_fd, purpose=purpose)
    return response.id


# ジョブの進行状況の確認
def check_job_status(job_id):
    job = client.fine_tuning.jobs.retrieve(job_id)
    print(f"ステータス: {job.status}")
    if job.status == "succeeded":
        print(f"ファインチューニング済みモデル: {job.fine_tuned_model}")
        exit()
    elif job.status == "failed":
        print(f"ジョブ失敗。エラー: {job.error}")

def fine_tuning(jsonl_file_path):
    
    with open(jsonl_file_path, "rb") as file_fd:
        response = client.files.create(file=file_fd, purpose="fine-tune")
    
    job = client.fine_tuning.jobs.create(
        training_file=response.id,
        model="gpt-4o-mini-2024-07-18"
    )
    
    print(f"ファインチューニングジョブID: {job.id}")
    
    # 定期的に状態をチェック
    while True:
        check_job_status(job.id)
        time.sleep(30)  # 60秒ごとにチェック


def get_sta_output(response):
    output = response
    return output

def get_pro_output(strategy, act, response):
    output = f"The most appropriate set of negotiation strategy is ['{strategy}'] and the most appropriate dialogue act is ['{act}']. Based on the selected negotiation strategy and dialogue act, the response is \"{response}\""
    return output

def get_procot_output(process, strategy, act, response):
    output = f"### Analysis\n{process} To reach this goal, the most appropriate set of negotiation strategy is ['{strategy}'] and the most appropriate dialogue act is ['{act}']. Based on the selected negotiation strategy and dialogue act, the response is \"{response}\""
    return output

def get_star_output(process, response):
    output = f"### Analysis\n{process} Based on the analysis, the response is \"{response}\""
    return output

def get_openai_instruct(input_file_path, output_file_path, prompt_type):
    
    # JSONファイルを読み込む
    with open(input_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    prompt_list = []
    
    for one_data in data:
        if prompt_type == "standard":
            prompt = inference.get_zs_sta_prompt(one_data["item_description"], one_data["traget_price"], one_data["dialogue_history"])
            gold_respose = get_sta_output(one_data["gold_respose"])
        elif prompt_type == "proactive":
            prompt = inference.get_zs_pro_prompt(one_data["item_description"], one_data["traget_price"], one_data["dialogue_history"])
            gold_respose = get_pro_output(one_data["nego_strategy"], one_data["dialogue_act"], one_data["gold_respose"])
        elif prompt_type == "procot":
            prompt = inference.get_zs_procot_prompt(one_data["item_description"], one_data["traget_price"], one_data["dialogue_history"])
            gold_respose = get_procot_output(one_data["gold_process"], one_data["nego_strategy"], one_data["dialogue_act"], one_data["gold_respose"])
        elif prompt_type == "star":
            prompt = inference.get_star_prompt(one_data["item_description"], one_data["traget_price"], one_data["dialogue_history"])
            gold_respose = get_star_output(one_data["gold_process"],  one_data["gold_respose"])
            
        prompt_pair = {"messages": [{"role": "system", "content": prompt}, {"role": "assistant", "content": gold_respose}]}
        prompt_list.append(prompt_pair)
    
    # jsonlファイルの作成
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for item in prompt_list:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

    
if __name__ == "__main__":
    
    input_file_path = "../datasets/negotiate_labeled_train.json"
    
    #prompt_type = "standard"
    #prompt_type = "proactive"
    #prompt_type = "procot"
    prompt_type = "star"
    
    if prompt_type == "standard":
        jsonl_file_path = "../datasets/train_openai_standard.jsonl"
    elif prompt_type == "proactive":
        jsonl_file_path = "../datasets/train_openai_proactive.jsonl"
    elif prompt_type == "procot":
        input_file_path = "../datasets/procot_train.json"
        jsonl_file_path = "../datasets/train_openai_procot.jsonl"
    elif prompt_type == "star":
        input_file_path = "../datasets/star_train.json"
        jsonl_file_path = "../datasets/train_openai_star.jsonl"
    
    #get_openai_instruct(input_file_path, jsonl_file_path, prompt_type)
    
    fine_tuning(jsonl_file_path)
