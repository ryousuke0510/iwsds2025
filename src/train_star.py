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

def get_zs_star_process_prompt(item_description, target_price, dialogue_history, dialogue_act, nego_strategy, gold_response):
    return f'''### Instruction
Assume you are the seller.
Given the item description, the target selling price, and the conversation history, in order to reach a better deal with the buyer, first analyse the current negotiation progress and consider an appropriate goal.
Based on the analysis, generate a response.
The reply should start with the analysis of the current negotiation progress and an appropriate goal, and then follow by 'Based on the analysis, the response is' </s>

The item description is '{item_description}'.

The target selling price is {target_price}. 

The conversation history is {dialogue_history}

### Hint
I will give you hit.
the response is only "{gold_response}"

Please generate the response: ### Analysis
Based on the analysis, the response is ""
'''


def get_completion_procot_star(prompt, one_data):

    for i in range(MAX_API_CALL):
        try:
            
            openai_completion = openai_dialogue.chatgpt_4_omni(prompt)
            #print(f"openai_completion:{openai_completion}")
            # process
            try:
                process = openai_completion.split("### Analysis")[1].split("Based on the analysis")[0].strip()
            except:
                process = openai_completion.split("Based on the analysis")[0].strip()
            #print(f"process:{process}")
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
 
        prompt = get_zs_star_process_prompt(one_data["item_description"], one_data["traget_price"], one_data["dialogue_history"], one_data["dialogue_act"], one_data["nego_strategy"], one_data["gold_respose"])
        
        gold_process = get_completion_procot_star(prompt, one_data)
        
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
    output_file_path = "../datasets/star_train.json"
    
    #input_file_path = "../datasets/negotiate_labeled_valid.json"
    #output_file_path = "../datasets/procot_valid.json"
    
    main(input_file_path, output_file_path)
    