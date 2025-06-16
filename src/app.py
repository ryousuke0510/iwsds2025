import requests
import streamlit as st

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

import random

import inference

# .envファイルの読み込み
load_dotenv()

client = OpenAI(
        api_key = os.environ['OPENAI_API_KEY']
    )



# omniの通常出力
def translate_en_to_ja(message_arg):
    message=[{"role": "system", "content": "あなたはプロの翻訳家です。次の英語で記述された商品説明を日本語で要約しながら翻訳してください。要約の仕方は、まず商品名を記述し、その後に商品説明を箇条書でお願いします。要約文以外の不要なものは出力してはいけません。"},
             {"role": "user", "content": message_arg},]
    response = client.chat.completions.create(
        model="gpt-4o-mini", # model = "deployment_name".
        messages = message,
    )
    return response.choices[0].message.content

def translate_en_to_ja_with_history(message_arg, dialogue_history):
    message=[{"role": "system", "content": f"あなたはプロの翻訳家です。次の対話履歴に続く英語を日本語に翻訳してください。翻訳結果以外の不要なものは出力してはいけません。 ### 対話履歴\n{dialogue_history}"},
             {"role": "user", "content": message_arg},]
    response = client.chat.completions.create(
        model="gpt-4o-mini", # model = "deployment_name".
        messages = message,
    )
    return response.choices[0].message.content

def translate_ja_to_en(message_arg):
    message=[{"role": "system", "content": "You are a professional translator. Please translate my Japanese into English."},
             {"role": "user", "content": message_arg},]
    response = client.chat.completions.create(
        model="gpt-4o-mini", # model = "deployment_name".
        messages = message,
    )
    return response.choices[0].message.content

def translate_ja_to_en_with_history(message_arg, dialogue_history):
    message=[{"role": "system", "content": f"You are a professional translator. Please translate my Japanese into English. Do not output anything unnecessary other than the translation result. ### Dialogue History\n{dialogue_history}"},
             {"role": "user", "content": message_arg},]
    response = client.chat.completions.create(
        model="gpt-4o-mini", # model = "deployment_name".
        messages = message,
    )
    return response.choices[0].message.content


# チャットログを保存したセッション情報を初期化
if "dialogue_history" not in st.session_state:
    st.session_state.dialogue_history = []
    
# チャットログを保存したセッション情報を初期化
if "dialogue_history_ja" not in st.session_state:
    st.session_state.dialogue_history_ja = []

# チャットログを保存したセッション情報を初期化
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# アプリ起動時の処理
if "instruction" not in st.session_state:
    with open("human_eval_data.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    
    data_num = len(data)
    # 1から100までのランダムな整数を選択
    random_integer = random.randint(0, data_num)
    
    st.session_state.instruction = {"item_description_ja":translate_en_to_ja(data[random_integer]["item_description"]), 
                                    "item_description_en":data[random_integer]["item_description"], 
                                    "seller_traget":data[random_integer]["traget_price"],
                                    "buyer_traget":data[random_integer]["buyer_price"],
                                    "ID":data[random_integer]["ID"]}

# サイドバーにモデル選択用のセレクトボックスを追加
st.sidebar.title("チャット設定")
model_num = st.sidebar.selectbox(
    "使用する言語モデルを選択してください:",
    ["FIRST", "SECOUND", "THIRD", "FOURTH", "FIFTH", "FINISH"],
    index=0
)

if "model_num" not in st.session_state:
    st.session_state.model_num = model_num

#　モデルを変更した時の処理
if st.session_state.model_num != model_num:
    
    if st.session_state.model_num == "FIRST":
        model_name = "Standard_Finetune"
    elif st.session_state.model_num == "SECOUND":
        model_name = "Pro_CoT_Finetune"
    elif st.session_state.model_num == "THIRD":
        model_name = "STaR_Finetune"
    elif st.session_state.model_num == "FOURTH":
        model_name = "gpt_4o_mini"
    elif st.session_state.model_num == "FIFTH":
        model_name = "Pro_Finetune"
    elif st.session_state.model_num == "FINISH":
        model_name = "FINISH"
        
    file_path = "./human_eval_result/" + str(st.session_state.instruction["ID"]) + "_" + model_name + ".json"
    
    data = [
                {
                    "ID":st.session_state.instruction["ID"], 
                    "MODEL":model_name,
                    "dialogue_history":st.session_state.dialogue_history_ja
                }
            ]
    
    
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
        
    del st.session_state["initial_utterance"]
    st.session_state.dialogue_history = []
    st.session_state.dialogue_history_ja = []
    st.session_state.chat_history = []
    st.session_state.model_num = model_num
    print(model_num)
    
    if model_name == "FINISH":
        exit()

# モデル設定
if model_num == "FIRST":
    # Standard-Finetune
    model = ""                          # Job名を記述する
    model_name = "Standard-Finetune"
elif model_num == "SECOUND":
    # Pro-CoT-Finetune
    model = ""                          # Job名を記述する
    model_name = "Pro-CoT-Finetune"
elif model_num == "THIRD":
    # STaR-Finetune
    model = ""                          # Job名を記述する
    model_name = "STaR-Finetune"
elif model_num == "FOURTH":
    # gpt-4o-mini
    model = "gpt-4o-mini"
    model_name = "gpt-4o-mini"
elif model_num == "FIFTH":
    # Pro-Finetune
    model = ""                          # Job名を記述する
    model_name = "Pro-Finetune"


def get_completion(item_description_en, seller_target, dialogue_hisotry):
    if model_name == "Standard-Finetune":
        prompt = inference.get_zs_sta_prompt(item_description_en, seller_target, dialogue_hisotry)
        utterance = inference.get_completion_standard(prompt, model)
    
    elif model_name == "Pro-CoT-Finetune":
        prompt = inference.get_zs_procot_prompt(item_description_en, seller_target, dialogue_hisotry)
        predict_process, predict_act, predict_strategy, utterance = inference.get_completion_procot(prompt, model)
        
    elif model_name == "STaR-Finetune":
        prompt = inference.get_star_prompt(item_description_en, seller_target, dialogue_hisotry)
        predict_process, utterance = inference.get_completion_star(prompt, model)
        
    elif model_name == "gpt-4o-mini":
        prompt = inference.get_zs_sta_prompt(item_description_en, seller_target, dialogue_hisotry)
        utterance = inference.get_completion_standard(prompt, model)
        
    elif model_name == "Pro-Finetune":
        prompt = inference.get_zs_pro_prompt(item_description_en, seller_target, dialogue_hisotry)
        predict_act, predict_strategy, utterance = inference.get_completion_proactive(prompt, model)
        
    #print(prompt)
    return utterance



################ UI部分

st.markdown(f"""
### 実験のお願い
あなたは商品を買うお客さんです。

「商品説明」とその商品の「販売価格」が提示されています。
商品をできるだけ「目標価格」になるようにこのシステムと会話を行ってください。

### 商品説明
{st.session_state.instruction["item_description_ja"]}

### 販売価格
{st.session_state.instruction["seller_traget"]} ドル

### あなたの目標価格
{st.session_state.instruction["buyer_traget"]} ドル
""")


CONTAINER_HEIGHT = 500

# 初期の発話
if "initial_utterance" not in st.session_state:
    
    agent_en = get_completion(st.session_state.instruction["item_description_en"], st.session_state.instruction["seller_traget"], "")
    
    st.session_state.initial_utterance = agent_en
    st.session_state.dialogue_history.append(f"'seller': '{agent_en}'")
    agent_ja = translate_en_to_ja_with_history(agent_en, "")
    st.session_state.chat_history.append({"role": "assistant", "content": agent_ja})
    st.session_state.dialogue_history_ja.append(f"'seller': '{agent_ja}'")
    with st.container(height = CONTAINER_HEIGHT):
        # 以前のチャットログを表示
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.write(chat["content"])

user_ja = st.chat_input("ここにメッセージを入力")

# チャット部分
if user_ja:
    st.session_state.chat_history.append({"role": "user", "content": user_ja})
    user_en = translate_ja_to_en_with_history(user_ja, ", ".join(st.session_state.dialogue_history))
    st.session_state.dialogue_history.append(f"'buyer': '{user_en}'")
    st.session_state.dialogue_history_ja.append(f"'user': '{user_ja}'")
        
    with st.container(height = CONTAINER_HEIGHT):
        # 以前のチャットログを表示
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.write(chat["content"])
        
        # 翻訳処理を実行
        with st.spinner('考え中...'):
            
            agent_en = get_completion(st.session_state.instruction["item_description_en"], st.session_state.instruction["seller_traget"], ", ".join(st.session_state.dialogue_history))
            
            st.session_state.dialogue_history.append(f"'seller': '{agent_en}'")
            agent_ja = translate_en_to_ja_with_history(agent_en, ", ".join(st.session_state.dialogue_history_ja))
            st.session_state.chat_history.append({"role": "assistant", "content": agent_ja})
            st.session_state.dialogue_history_ja.append(f"'seller': '{agent_ja}'")

            with st.chat_message("assistant"):
                st.write(agent_ja)
    
