import json
import re
import time
import openai_dialogue
from collections import Counter
MAX_API_CALL = 10

def extract_string(text, pattern):
    # 正規表現を使って発話部分を抽出
    match = re.search(pattern, text)
    string = match.group(1) if match else ""
    return string

def get_zs_procot_process_prompt(item_description, target_price, dialogue_history, dialogue_act, nego_strategy, gold_response):
    return f'''### Instruction
Assume you are the seller.
Given the item description, the target selling price, and the conversation history, in order to reach a better deal with the buyer, first analyse the current negotiation progress and consider an appropriate goal, then select the most appropriate negotiation strategy and the most appropriate dialogue act to reach the goal.
Based on the selected one negotiation strategy and one dialogue act, generate a response.
The reply should start with the analysis of the current negotiation progress and an appropriate goal, and then follow by 'To reach this goal, the most appropriate negotiation strategy is [] and the most appropriate dialogue act is []. Based on the selected negotiation strategy and dialogue act, the response is' </s>

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

The item description is '{item_description}'.

The target selling price is {target_price}. 

The conversation history is {dialogue_history}

### Hints
I will give you hits.
the most appropriate negotiation strategy is '{nego_strategy}'
the most appropriate dialogue act is '{dialogue_act}'
the response is only "{gold_response}"

Please generate the response: ### Analysis
To reach this goal, the most appropriate negotiation strategy is [] and the most appropriate dialogue act is []. Based on the selected negotiation strategy and dialogue act, the response is ""
'''


def get_completion_procot(prompt, one_data):

    for i in range(MAX_API_CALL):
        try:
            
            openai_completion = openai_dialogue.chatgpt_4_omni(prompt)
            #print(f"openai_completion:{openai_completion}")
            # process
            try:
                process = openai_completion.split("### Analysis")[1].split("To reach this goal")[0].strip()
            except:
                process = openai_completion.split("To reach this goal")[0].strip()
            #print(f"process:{process}")
            
            '''
            # act
            try:
                dialogue_act_text = openai_completion.split("dialogue act is ['")[1].split("']")[0]
            except:
                dialogue_act_text = openai_completion.split("dialogue act is '")[1].split("'")[0]
            act = dialogue_act_text.strip('"')  # リストではなく単一の値
            print(f"act:{act}")
            
            # strategies
            try:
                strategies_text = openai_completion.split("negotiation strategy is ['")[1].split("']")[0]
            except:
                strategies_text = openai_completion.split("negotiation strategy is '")[1].split("'")[0]
            strategy = strategies_text.strip('"')  # リストではなく単一の値
            print(f"strategy:{strategy}")
            
            # utterance
            utterance = openai_completion.split("response is")[1].split('"')[1]
            '''
            print("============================================")
            #print(f"openai_completion:{openai_completion}")
            print(f"process:{process}")
            break
            #print(f"act:{act}")
            #print(f"strategy:{strategy}")
            #print(f"utterance:{utterance}")
            
        except:
            print("API ERROR")
            time.sleep(1)
    return process


def main(input_file_path, output_file_path):
    
    # JSONファイルを読み込む
    with open(input_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        
    data_inference = []

    count = 0
    data_num = len(data)
    
        
    for one_data in data:
 
        prompt = get_zs_procot_process_prompt(one_data["item_description"], one_data["traget_price"], one_data["dialogue_history"], one_data["dialogue_act"], one_data["nego_strategy"], one_data["gold_respose"])
        
        gold_process = get_completion_procot(prompt, one_data)
        
        one_data_lebaled = {"ID":one_data["ID"], "item_description":one_data["item_description"], "traget_price":one_data["traget_price"], 
                        "gold_respose":one_data["gold_respose"],
                        "gold_process":gold_process,
                        "dialogue_act":one_data["dialogue_act"],
                        "nego_strategy":one_data["nego_strategy"],
                        "dialogue_history":one_data["dialogue_history"]}
        
        count += 1
        
        data_inference.append(one_data_lebaled)
        
        print("================================")
        print(f"PRPCESS     :{gold_process}")
        print(f"ID          :{one_data["ID"]}")
        print(f"FINISHED %  :{int((count/data_num)*100)}%")
        
    
    # JSONL形式でファイルに書き戻す
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data_inference, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    
    input_file_path = "../datasets/negotiate_labeled_train.json"
    output_file_path = "../datasets/procot_train.json"
    
    #input_file_path = "../datasets/negotiate_labeled_valid.json"
    #output_file_path = "../datasets/procot_valid.json"
    
    main(input_file_path, output_file_path)
    