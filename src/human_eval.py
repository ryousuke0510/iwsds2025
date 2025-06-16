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


def main(input_file_path, output_file_path):
    
    # JSONファイルを読み込む
    with open(input_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    data_lebaled = []
    
    data_num = 1100
    label_count = data_num
        
    for one_data in data[data_num:]:
    
        label_count += 1
        
        one_data_lebaled = {"ID":label_count, "item_description":one_data["item_description"], "traget_price":one_data["traget_price"], "buyer_price":one_data["buyer_price"], "gold_respose":one_data["gold_respose"], "dialogue_history":one_data["dialogue_history"]}
        data_lebaled.append(one_data_lebaled)   
    
    # JSONL形式でファイルに書き戻す
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data_lebaled, file, ensure_ascii=False, indent=4)
    

if __name__ == '__main__':
    
    input_file_path = "../src/negotiate_data.json"
    output_file_path = "../datasets/human_eval_data.json"

    
    main(input_file_path, output_file_path)
    