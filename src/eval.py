import os
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

import openai_dialogue

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def cal_bleu(reference: str, candidate: str) -> float:
    """
    Calculate BLEU score between a reference and a candidate string.

    Args:
        reference (str): The reference text.
        candidate (str): The candidate text to evaluate.

    Returns:
        float: The BLEU score (between 0 and 1).
    """
    # Tokenize the input strings
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()

    # BLEU score calculation with smoothing to avoid zero scores
    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)

    return bleu_score

def cal_cos(predict_utterance, gold_respose):
    gold_cos = openai_dialogue.text_to_embedding(gold_respose)
    predict_cos = openai_dialogue.text_to_embedding(predict_utterance)
    cos_score = cos_sim(gold_cos, predict_cos)
    return cos_score

def cal_nego(y_true, y_pred):
        # クラスリスト
    nego_list = ["Describe-Product", "Rephrase-Product", "Embellish-Product", "Address-Concerns", "Communicate-Interests", 
                 "Propose-Price", "Do-Not-Propose-First", "Negotiate-Side-Offers", "Hedge", "Communicate-Politely", 
                 "Build-Rapport", "Talk-Informally", "Show-Dominance", "Negative-Sentiment", "Certainty-Words", "Others"]

    # F1-macro スコアの計算
    f1_macro = f1_score(y_true, y_pred, labels=nego_list, average='macro', zero_division=0)

    return f1_macro

def cal_act(y_true, y_pred):
    # クラスリスト
    act_list = ["intro", "inquiry", "init-price", "counter-price", "others", "agree", "disagree", "inform", "vague-price", "insist"]

    # F1-macro スコアの計算
    f1_macro = f1_score(y_true, y_pred, labels=act_list, average='macro', zero_division=0)

    return f1_macro

def get_mean(list):
    return sum(list)/len(list)

def main(input_file_path):
    
    # JSONファイルを読み込む
    with open(input_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    bleu_score_list = []
    cos_score_list = []
    
    act_gold_list = []
    act_predict_list = []
    
    nego_gold_list = []
    nego_predict_list = []
    
    for one_data in data:
        
        predict_utterance = one_data["predict_utterance"]
        gold_respose = one_data["gold_respose"]
        
        act_gold_list.append(one_data["dialogue_act"])
        act_predict_list.append(one_data["predict_act"])
        if one_data["dialogue_act"] == one_data["predict_act"]:
            is_act = "TRUE"
        else:
            is_act = "FALSE"
        
        nego_gold_list.append(one_data["nego_strategy"])
        nego_predict_list.append(one_data["predict_strategy"])
        if one_data["nego_strategy"] == one_data["predict_strategy"]:
            is_nego = "TRUE"
        else:
            is_nego = "FALSE"
        
        bleu_score = cal_bleu(predict_utterance, gold_respose)
        bleu_score_list.append(bleu_score)
        
        cos_score = cal_cos(predict_utterance, gold_respose)
        cos_score_list.append(cos_score)
        
        print(f"is_nego   :{is_act}")
        print(f"is_act    :{is_nego}")
        print(f"bleu_score:{bleu_score}")
        print(f"cos_score :{cos_score}")
        print("============================")
    
    
    nego_f1 = cal_nego(nego_gold_list, nego_predict_list)
    act_f1 = cal_act(act_gold_list, act_predict_list)
    bleu = get_mean(bleu_score_list)
    cos = get_mean(cos_score_list)
    
    print(f"FINAL nego_f1   :{nego_f1}")
    print(f"FINAL act_f1    :{act_f1}")
    print(f"FINAL bleu_score:{bleu}")
    print(f"FINAL cos_score :{cos}")
    
    eval_result = {"nego_f1":nego_f1, "act_f1":act_f1, "bleu_score":bleu, "cos_score":cos}
    
    data.insert(0, eval_result)
    
        # JSONL形式でファイルに書き戻す
    with open(input_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    
    # standard
    #input_file_path = "../datasets/valid_ze_sta_original.json"
    input_file_path = "../datasets/valid_fs_sta_original.json"
    #input_file_path = "../datasets/valid_ze_sta_finetuned.json"
    
    # proactive
    #input_file_path = "../datasets/valid_ze_pro_original.json"
    #input_file_path = "../datasets/valid_fs_pro_original.json"
    #input_file_path = "../datasets/valid_ze_pro_finetuned.json"
    
    # proactive-cot
    #input_file_path = "../datasets/valid_ze_procot_original.json"
    #input_file_path = "../datasets/valid_fs_procot_original.json"
    #input_file_path = "../datasets/valid_ze_procot_finetuned.json"
    
    # star
    #input_file_path = "../datasets/valid_fs_star_original.json"
    #input_file_path = "../datasets/valid_star_finetuned.json"
    
    main(input_file_path)