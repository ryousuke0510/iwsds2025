import json
import re
import time
import openai_dialogue
from collections import Counter
MAX_API_CALL = 10

def extract_string(text, pattern):
    # 正規表現を使って発話部分を抽出
    match = re.search(pattern, text, re.DOTALL)  # 最初の一致を検索
    if match:
        return match.group(1)  # 一致した部分を返す
    return ""  # 一致しない場合はNoneを返す


def get_act_prompt(gold_respose, dialogue_history):
    return f'''Which dialogue act among the "dialogue acts" is the most appropriate for the next statement? Please select one.
### utterance
{gold_respose}

### dialogue acts
- intro, Meaning:Greetings, Example:I would love to buy
- inquiry, Meaning:Ask a question, Example:Sure, what's your price
- init-price, Meaning:Propose the first price, Example:I'm on a budget so I could do $5
- counter-price, Meaning:Proposing a counter price, Example:How about $15 and I'll waive the deposit
- others, Meaning:others
- agree, Meaning:Agree with the proposal , Example:That works for me
- disagree, Meaning:Disagree with a proposal, Example:Sorry, I can't agree to that
- inform, Meaning:Answer a question, Example: 	This bike is brand new
- vague-price, Meaning:Using comparatives with existing price, Example:That offer is too low
- insist, Meaning:Insist on an offer, Example:Still can I buy it for $5

### output format
Please enclose the dialogue act with [act] and [/act] tags. Do not output anything unnecessary other than the tags and the dialogue act.

### output exaple
If you select "intro" as the label, output:
[act]intro[/act]
For other dialogue strategies, enclose only the label name with [act] and [/act] tags in the same manner.

### dialogue_hisotry
{dialogue_history}
'''

def get_negotiate_prompt(gold_respose, dialogue_history):
    return f'''Which negotiation strategy among the "negotiation strategies" is the most appropriate for the following statement? 
First, answer the number of appropriate negotiation strategy.
Secound, answer the negotiation strategy.

### following statement
{gold_respose}

### negotiate strategies
- Describe-Product, Example:The car has leather seats
- Rephrase-Product, Example:45k miles −→ less than 50k miles
- Embellish-Product, Example:a luxury car with attractive leather seats
- Address-Concerns, Example:I've just taken it to maintenance
- Communicate-Interests, Example:I'd like to sell it asap
- Propose-Price, Example:How about 9k?
- Do-Not-Propose-First, Example:n/a
- Negotiate-Side-Offers, Example:I can deliver it for you
- Hedge, Example:I could come down a bit
- Communicate-Politely, Example:Greetings, gratitude, apology, please 
- Build-Rapport, Example:My kid really liked this bike, but he outgrew it
- Talk-Informally, Example:Absolutely, ask away!
- Show-Dominance, Example:The absolute highest I can do is 640
- Negative-Sentiment, Example:Sadly, I simply cannot go under 500
- Certainty-Words, Example:It has always had a screen protector
- Others

### output format
Please enclose the final negotiation strategies with [strategy] and [/strategy] tags. Do not include anything unnecessary other than the tags and the negotiation strategies.
If you select two or more strategies, please use ', ' as in [strategy]Propose-Price, Communicate-Interests[/strategy].

### dialogue_hisotry
{dialogue_history}

'''

def get_act_completion(prompt):
    
    strategy_list = ["intro", "inquiry", "init-price", "counter-price", "ohters", "agree", "disagree", "inform", "vague-price", "insist"]
    
    for i in range(MAX_API_CALL):
        try:
            openai_completion = openai_dialogue.chatgpt_4_omni(prompt)
            pattern = r'\[act\](.*?)\[/act\]'
            completion = extract_string(openai_completion, pattern)
            #print(completion)
            if completion in strategy_list:
                break
        except:
            print("API ERROR")
            time.sleep(1)
    return completion

def get_nego_completion(prompt):
    
    strategy_list = ["Describe-Product", "Rephrase-Product", "Embellish-Product", "Address-Concerns", "Communicate-Interests", 
                     "Propose-Price", "Do-Not-Propose-First", "Negotiate-Side-Offers", "Hedge", "Communicate-Politely", "Build-Rapport",
                     "Talk-Informally", "Show-Dominance", "Negative-Sentiment", "Certainty-Words", "Others"]
    
    for i in range(MAX_API_CALL):
        try:
            openai_completion = openai_dialogue.chatgpt_4_omni(prompt)
            print(openai_completion)
            pattern = r'\[strategy\](.*?)\[/strategy\]'
            completion = extract_string(openai_completion, pattern)
            #print(completion)
            split_list = list(dict.fromkeys([item.strip() for item in completion.split(',')]))
            count = 0
            for split in split_list:
                if split in strategy_list:
                    count+=1
            if count == len(split_list):
                string = ', '.join(split_list)
                break
        except:
            print("API ERROR")
            time.sleep(1)
    return string

def main(input_file_path, output_file_path_train, output_file_path_valid):
    
    # JSONファイルを読み込む
    with open(input_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    data_lebaled = []
    
    data_num = 1100
    train_num = 1000
    label_count = 0
        
    for one_data in data[:data_num]:
        
        gold_respose = one_data["gold_respose"]
        dialogue_history = one_data["dialogue_history"]
        
        act_prompt = get_act_prompt(gold_respose, dialogue_history)
        negotiate_prompt = get_negotiate_prompt(gold_respose, dialogue_history)
        
        
        dialogue_act = get_act_completion(act_prompt)
        nego_strategy = get_nego_completion(negotiate_prompt)
        
        label_count += 1
        
        one_data_lebaled = {"ID":label_count, "item_description":one_data["item_description"], "traget_price":one_data["traget_price"], "gold_respose":one_data["gold_respose"], "dialogue_act":dialogue_act, "nego_strategy":nego_strategy, "dialogue_history":one_data["dialogue_history"]}
        data_lebaled.append(one_data_lebaled)
        
        print("================================")
        print(f"ACT         :{dialogue_act}")
        print(f"NEGTOTIATION:{nego_strategy}")
        print(f"ID          :{label_count}")
        print(f"FINISHED %  :{int((label_count/data_num)*100)}%")
        
    
    # JSONL形式でファイルに書き戻す
    with open(output_file_path_train, 'w', encoding='utf-8') as file:
        json.dump(data_lebaled[:train_num], file, ensure_ascii=False, indent=4)
        
    # JSONL形式でファイルに書き戻す
    with open(output_file_path_valid, 'w', encoding='utf-8') as file:
        json.dump(data_lebaled[train_num:], file, ensure_ascii=False, indent=4)


def analyze_data(file_path):
    # JSONファイルを読み込む
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    act_list = ["intro", "inquiry", "init-price", "counter-price", "others", "agree", "disagree", "inform", "vague-price", "insist"]
    nego_list = ["Describe-Product", "Rephrase-Product", "Embellish-Product", "Address-Concerns", "Communicate-Interests", 
                 "Propose-Price", "Do-Not-Propose-First", "Negotiate-Side-Offers", "Hedge", "Communicate-Politely", 
                 "Build-Rapport", "Talk-Informally", "Show-Dominance", "Negative-Sentiment", "Certainty-Words", "Others"]

    act_count = [0]*len(act_list)
    nego_count = [0]*len(nego_list)
    
    # 各データを走査してカウントを増やす
    for one_data in data:
        act = one_data["dialogue_act"]
        nego = one_data["nego_strategy"]
        
        for i in range(len(act_list)):
            if act == act_list[i]:
                act_count[i] +=1
        
        for i in range(len(nego_list)):
            if nego == nego_list[i]:
                nego_count[i] +=1
     
    print(act_count)
    print(nego_count)   


def label_strategy(file_path):
    # JSONファイルを読み込む
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    data_lebaled = []
    
    data_num = 10
    label_count = 0
        
    for one_data in data[:data_num]:
        gold_respose = one_data["gold_respose"]
        dialogue_history = one_data["dialogue_history"]
        
        negotiate_prompt = get_negotiate_prompt(gold_respose, dialogue_history)
        
        nego_strategy = get_nego_completion(negotiate_prompt)
        
        label_count += 1
        
        one_data_lebaled = {"ID":one_data["ID"], "item_description":one_data["item_description"], "traget_price":one_data["traget_price"], "gold_respose":one_data["gold_respose"], "dialogue_act":one_data["dialogue_act"], "nego_strategy":nego_strategy, "dialogue_history":one_data["dialogue_history"]}
        data_lebaled.append(one_data_lebaled)
        
        print("================================")
        print(f"NEGTOTIATION:{nego_strategy}")
        print(f"ID          :{label_count}")
        print(f"FINISHED %  :{int((label_count/data_num)*100)}%")
        
    
    # JSONL形式でファイルに書き戻す
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data_lebaled, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    
    input_file_path = "../datasets/negotiate_data.json"
    output_file_path_train = "../datasets/negotiate_labeled_train.json"
    output_file_path_valid = "../datasets/negotiate_labeled_valid.json"
    
    #main(input_file_path, output_file_path_train, output_file_path_valid)
    
    #analyze_data(output_file_path_valid)
    
    # 複数の交渉戦略ラベル付け
    #label_strategy(output_file_path_train)
    #label_strategy(output_file_path_valid)