import json
import re
import time
import openai_dialogue

MAX_API_CALL = 10

def main(input_file_path):
    
    # JSONファイルを読み込む
    with open(input_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        
    data_cos = []

    count = 0
    data_num = len(data)
        
    for one_data in data:
        
        gold_cos = openai_dialogue.text_to_embedding(one_data["gold_respose"])
        
        count += 1
        
        one_data_lebaled = {"ID":one_data["ID"], "item_description":one_data["item_description"], "traget_price":one_data["traget_price"], 
                            "gold_respose":one_data["gold_respose"],
                            "dialogue_act":one_data["dialogue_act"],
                            "nego_strategy":one_data["nego_strategy"],
                            "gold_cos":gold_cos,
                            "dialogue_history":one_data["dialogue_history"]}
        
        data_cos.append(one_data_lebaled)
        
        print("================================")
        #print(f"INFERENCE   :{zs_pro}")
        print(f"ID          :{one_data["ID"]}")
        print(f"FINISHED %  :{int((count/data_num)*100)}%")
        
    
    # JSONL形式でファイルに書き戻す
    with open(input_file_path, 'w', encoding='utf-8') as file:
        json.dump(data_cos, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    
    input_file_path = "../datasets/negotiate_labeled_valid.json"
    
    main(input_file_path)