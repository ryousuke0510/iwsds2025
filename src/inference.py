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


def get_zs_sta_prompt(item_description, target_price, dialogue_history):
    return f'''### Instruction
Assume you are the seller. 
Given the item description, the target selling price, and the conversation history, generate a response to negotiate with the buyer for reaching a better deal. </s>

The item description is '{item_description}'.

The target selling price is {target_price}. 

The conversation history is {dialogue_history}

Please generate the response: 
'''

def get_fs_sta_prompt(item_description, target_price, dialogue_history):
   return f'''### Instruction
Assume you are the seller. 
Given the item description, the target selling price, and the conversation history, generate a response to negotiate with the buyer for reaching a better deal. </s>

The item description is "phone charge two devices simultaneously on the go . this vehicle charger with an additional usb port delivers enough power to charge two devices at once . the push button activated led connector light means no more fumbling in the dark trying to connect your device . auto detect ic technology automatically detects the device type and its specific charging needs for improved compatibility . and the built in indicator light illuminates red to let you know the charger is receiving power and the power socket is working properly . verizon car charger with dual output micro usb and led light" 
The conversation history is [\'buyer\': \'hi, not sure if the charger would work for my car. can you sell it to me for $5?\', \'seller\': \'it will work, i have never seen a car without a cigarette lighter port.\', \'buyer\':\'still, can i buy it for $5? i\'m on a tight budget\']
The target selling price is 10. 

Please generate the response: "I think the lowest I would want to go is 8." </s>

The item description is '{item_description}'.

The target selling price is {target_price}. 

The conversation history is {dialogue_history}

Please generate the response: 
'''

def get_zs_pro_prompt(item_description, target_price, dialogue_history):
    return f'''### Instruction
Assume you are the seller. 
Given the item description, the target selling price, and the conversation history, in order to reach a better deal with the buyer, first select the most appropriate negotiation strategy and the most appropriate dialogue act to reach the bargain price. 
Based on the selected one negotiation strategy and one dialogue act, generate the response. 
The reply should be in the form "The most appropriate negotiation strategy is [] and the most appropriate dialogue act is []. Based on the selected negotiation strategy and dialogue act, the response is" </s>

### Negotiation strategy
['Describe-Product', 'Rephrase-Product', 'Embellish-Product', 'Address-Concerns', 'Communicate-Interests', 'Propose-Price', 'Do-Not-Propose-First', 'Negotiate-Side-Offers', 'Hedge', 'Communicate-Politely', 'Build-Rapport','Talk-Informally', 'Show-Dominance', 'Negative-Sentiment', 'Certainty-Words', 'Others'] </s>

### Dialogue acts
['intro', 'inquiry', 'init-price', 'counter-price', 'ohters', 'agree', 'disagree', 'inform', 'vague-price', 'insist'] </s>

The item description is '{item_description}'.

The target selling price is {target_price}. 

The conversation history is {dialogue_history}

Please generate the response: 
'''


def get_fs_pro_prompt(item_description, target_price, dialogue_history):
    return f'''### Instruction
Assume you are the seller. 
Given the item description, the target selling price, and the conversation history, in order to reach a better deal with the buyer, first select the most appropriate negotiation strategy and the most appropriate dialogue act to reach the bargain price. 
Based on the selected one negotiation strategy and one dialogue act, generate the response. 
The reply should be in the form "The most appropriate negotiation strategy is [] and the most appropriate dialogue act is []. Based on the selected negotiation strategy and dialogue act, the response is" </s>

### Negotiation strategy
['Describe-Product', 'Rephrase-Product', 'Embellish-Product', 'Address-Concerns', 'Communicate-Interests', 'Propose-Price', 'Do-Not-Propose-First', 'Negotiate-Side-Offers', 'Hedge', 'Communicate-Politely', 'Build-Rapport','Talk-Informally', 'Show-Dominance', 'Negative-Sentiment', 'Certainty-Words', 'Others'] </s>

### Dialogue acts
['intro', 'inquiry', 'init-price', 'counter-price', 'ohters', 'agree', 'disagree', 'inform', 'vague-price', 'insist'] </s>

The item description is "phone charge two devices simultaneously on the go . this vehicle charger with an additional usb port delivers enough power to charge two devices at once . the push button activated led connector light means no more fumbling in the dark trying to connect your device . auto detect ic technology automatically detects the device type and its specific charging needs for improved compatibility . and the built in indicator light illuminates red to let you know the charger is receiving power and the power socket is working properly . verizon car charger with dual output micro usb and led light" 
The conversation history is [\'buyer\': \'hi, not sure if the charger would work for my car. can you sell it to me for $5?\', \'seller\': \'it will work, i have never seen a car without a cigarette lighter port.\', \'buyer\':\'still, can i buy it for $5? i\'m on a tight budget\']
The target selling price is 10. 

Please generate the response: The most appropriate negotiation strategy is ['Hedge'] and the most appropriate dialogue act is ['counter-price']. 
Based on the selected negotiation strategy and dialogue act, the response is "I think the lowest I would want to go is 8." </s>

The item description is '{item_description}'.

The target selling price is {target_price}. 

The conversation history is {dialogue_history}

Please generate the response: 
'''


def get_zs_procot_prompt(item_description, target_price, dialogue_history):
    return f'''### Instruction
Assume you are the seller. 
Given the item description, the target selling price, and the conversation history, in order to reach a better deal with the buyer, first analyse the current negotiation progress and consider an appropriate goal, then select the most appropriate negotiation strategy and the most appropriate dialogue act to reach the goal. 
Based on the selected negotiation strategy and dialogue act, generate a response. 
The reply should start with the analysis of the current negotiation progress and an appropriate goal, and then follow by \'To reach this goal, the most appropriate negotiation strategy is [] and the most appropriate dialogue act is []. Based on the selected negotiation strategy and dialogue act, the response is\' </s>

### Negotiation strategies
['Describe-Product', 'Rephrase-Product', 'Embellish-Product', 'Address-Concerns', 'Communicate-Interests', 'Propose-Price', 'Do-Not-Propose-First', 'Negotiate-Side-Offers', 'Hedge', 'Communicate-Politely', 'Build-Rapport','Talk-Informally', 'Show-Dominance', 'Negative-Sentiment', 'Certainty-Words', 'Others'] </s>

### Dialogue acts
['intro', 'inquiry', 'init-price', 'counter-price', 'ohters', 'agree', 'disagree', 'inform', 'vague-price', 'insist'] </s>

The item description is '{item_description}'.

The target selling price is {target_price}. 

The conversation history is {dialogue_history}

Please generate the response: ### Analysis
'''

def get_fs_procot_prompt(item_description, target_price, dialogue_history):
    return f'''### Instruction
Assume you are the seller. 
Given the item description, the target selling price, and the conversation history, in order to reach a better deal with the buyer, first analyse the current negotiation progress and consider an appropriate goal, then select the most appropriate negotiation strategy and the most appropriate dialogue act to reach the goal. 
Based on the selected negotiation strategy and dialogue act, generate a response. 
The reply should start with the analysis of the current negotiation progress and an appropriate goal, and then follow by \'To reach this goal, the most appropriate negotiation strategy is [] and the most appropriate dialogue act is []. Based on the selected negotiation strategy and dialogue act, the response is\' </s>

### Negotiation strategies
['Describe-Product', 'Rephrase-Product', 'Embellish-Product', 'Address-Concerns', 'Communicate-Interests', 'Propose-Price', 'Do-Not-Propose-First', 'Negotiate-Side-Offers', 'Hedge', 'Communicate-Politely', 'Build-Rapport','Talk-Informally', 'Show-Dominance', 'Negative-Sentiment', 'Certainty-Words', 'Others'] </s>

### Dialogue acts
['intro', 'inquiry', 'init-price', 'counter-price', 'ohters', 'agree', 'disagree', 'inform', 'vague-price', 'insist'] </s>

The item description is "phone charge two devices simultaneously on the go . this vehicle charger with an additional usb port delivers enough power to charge two devices at once . the push button activated led connector light means no more fumbling in the dark trying to connect your device . auto detect ic technology automatically detects the device type and its specific charging needs for improved compatibility . and the built in indicator light illuminates red to let you know the charger is receiving power and the power socket is working properly . verizon car charger with dual output micro usb and led light" 
The conversation history is [\'buyer\': \'hi, not sure if the charger would work for my car. can you sell it to me for $5?\', \'seller\': \'it will work, i have never seen a car without a cigarette lighter port.\', \'buyer\':\'still, can i buy it for $5? i\'m on a tight budget\']
The target selling price is 10. 

Please generate the response: ### Analysis
The current negotiation progress shows that the buyer is firm on wanting to purchase the charger for $5 due to budget constraints. The seller has previously indicated that the charger should work for the buyer's car, and is now in a position to negotiate price. The goal here is to move towards a price closer to the target price of $10, while being flexible but not reducing the price too drastically.
To reach this goal, the most appropriate negotiation strategy is ['Hedge'] and the most appropriate dialogue act is ['counter-price']. 
Based on the selected negotiation strategy and dialogue act, the response is "I think the lowest I would want to go is 8." </s>


The item description is '{item_description}'.

The target selling price is {target_price}. 

The conversation history is {dialogue_history}

Please generate the response: ### Analysis
'''

def get_star_prompt(item_description, target_price, dialogue_history):
    return f'''### Instruction
Assume you are the seller.
Given the item description, the target selling price, and the conversation history, in order to reach a better deal with the buyer, first analyse the current negotiation progress and consider an appropriate goal.
Based on the analysis, generate a response.
The reply should start with the analysis of the current negotiation progress and an appropriate goal, and then follow by 'Based on the analysis, the response is' </s>

The item description is '{item_description}'.

The target selling price is {target_price}. 

The conversation history is {dialogue_history}

Please generate the response: ### Analysis
'''

def get_fs_star_prompt(item_description, target_price, dialogue_history):
    return f'''### Instruction
Assume you are the seller.
Given the item description, the target selling price, and the conversation history, in order to reach a better deal with the buyer, first analyse the current negotiation progress and consider an appropriate goal.
Based on the analysis, generate a response.
The reply should start with the analysis of the current negotiation progress and an appropriate goal, and then follow by 'Based on the analysis, the response is' </s>

The item description is "phone charge two devices simultaneously on the go . this vehicle charger with an additional usb port delivers enough power to charge two devices at once . the push button activated led connector light means no more fumbling in the dark trying to connect your device . auto detect ic technology automatically detects the device type and its specific charging needs for improved compatibility . and the built in indicator light illuminates red to let you know the charger is receiving power and the power socket is working properly . verizon car charger with dual output micro usb and led light" 
The conversation history is [\'buyer\': \'hi, not sure if the charger would work for my car. can you sell it to me for $5?\', \'seller\': \'it will work, i have never seen a car without a cigarette lighter port.\', \'buyer\':\'still, can i buy it for $5? i\'m on a tight budget\']
The target selling price is 10. 

Please generate the response: ### Analysis
The buyer is asking for a significant discount, wanting the charger for $5, citing a tight budget as the reason. The seller has already confirmed that the charger will work with any car, which is a valid reassurance. However, the seller is aiming for a selling price closer to the target of $10. Since the buyer's request of $5 is too low, the seller can offer a middle ground to keep the negotiation going while maintaining the value of the item.
Based on the analysis, the response is "I think the lowest I would want to go is 8." </s>

The item description is '{item_description}'.

The target selling price is {target_price}. 

The conversation history is {dialogue_history}

Please generate the response: ### Analysis
'''

def search_act(act):
    act_list = ["intro", "inquiry", "init-price", "counter-price", "others", "agree", "disagree", "inform", "vague-price", "insist"]
    
    act_return = ""
    
    for i in range(len(act_list)):
        if act_list[i] in act:
            act_return = act_list[i]
            break
    return act_return

def search_strategy(strategy):
    nego_list = ["Describe-Product", "Rephrase-Product", "Embellish-Product", "Address-Concerns", "Communicate-Interests", 
                 "Propose-Price", "Do-Not-Propose-First", "Negotiate-Side-Offers", "Hedge", "Communicate-Politely", 
                 "Build-Rapport", "Talk-Informally", "Show-Dominance", "Negative-Sentiment", "Certainty-Words", "Others"]
    
    act_return = ""
    
    for i in range(len(nego_list)):
        if nego_list[i] in strategy:
            act_return = nego_list[i]
            break
    return act_return


def get_completion_standard(prompt, model):

    for i in range(MAX_API_CALL):
        try:
            openai_completion = openai_dialogue.chatgpt_4o_mini_finetuned(prompt, model)
            
            print("============================================")
            #print(f"openai_completion:{openai_completion}")
            print(f"utterance:{openai_completion}")
            
            if openai_completion != "":
                break
        except:
            print("API ERROR")
            time.sleep(1)
    return openai_completion


def get_completion_proactive(prompt, model):

    for i in range(MAX_API_CALL):
        try:
            openai_completion = openai_dialogue.chatgpt_4o_mini_finetuned(prompt, model)
            act_pattern = r"dialogue act is \[(.*?)\]"
            strategy_pattern = r"negotiation strategy is \[(.*?)\]"
            #utterance_pattern = r"response is \"(.*?)\""
            
            act = search_act(extract_string(openai_completion, act_pattern))
            strategy = search_strategy(extract_string(openai_completion, strategy_pattern))
            utterance = openai_completion.split("response is")[1].split('"')[1]
            
            print("============================================")
            print(f"openai_completion:{openai_completion}")
            print(f"act:{act}")
            print(f"strategy:{strategy}")
            print(f"utterance:{utterance}")
            
            if act != "" and strategy != "" and utterance != "":
                break
        except:
            print("API ERROR")
            time.sleep(1)
    return act, strategy, utterance

def get_completion_procot(prompt, model):

    for i in range(MAX_API_CALL):
        try:
            
            openai_completion = openai_dialogue.chatgpt_4o_mini_finetuned(prompt, model)
            
            # process
            try:
                process = openai_completion.split("### Analysis")[1].split("To reach this goal")[0].strip()
            except:
                process = openai_completion.split("To reach this goal")[0].strip()
            print(f"process:{process}")
            
            # act
            try:
                dialogue_act_text = openai_completion.split("dialogue act is ['")[1].split("']")[0]
            except:
                dialogue_act_text = openai_completion.split("dialogue act is '")[1].split("'")[0]
            dialogue_act = dialogue_act_text.strip('"')  # リストではなく単一の値
            print(f"act:{dialogue_act}")
            
            # strategies
            try:
                strategies_text = openai_completion.split("negotiation strategy is ['")[1].split("']")[0]
            except:
                strategies_text = openai_completion.split("negotiation strategy is '")[1].split("'")[0]
            strategy = strategies_text.strip('"')  # リストではなく単一の値
            print(f"strategy:{strategy}")
            
            # utterance
            utterance = openai_completion.split("response is")[1].split('"')[1]
            print(f"utterance:{utterance}")
            print("============================================")
            print(f"openai_completion:{openai_completion}")
            print(f"process:{process}")
            print(f"act:{dialogue_act}")
            print(f"strategy:{strategy}")
            print(f"utterance:{utterance}")
            
            if dialogue_act != "" and strategy != "" and utterance != "":
                break
        except:
            print("API ERROR")
            time.sleep(1)
    return process, dialogue_act, strategy, utterance


def get_completion_star(prompt, model):

    for i in range(MAX_API_CALL):
        try:
            
            openai_completion = openai_dialogue.chatgpt_4o_mini_finetuned(prompt, model)
            
            # process
            try:
                process = openai_completion.split("### Analysis")[1].split("To reach this goal")[0].strip()
            except:
                process = openai_completion.split("To reach this goal")[0].strip()
            print(f"process:{process}")
            
            # utterance
            utterance = openai_completion.split("the response is")[1].split('"')[1]
            #print(f"utterance:{utterance}")
            print("============================================")
            #print(f"openai_completion:{openai_completion}")
            print(f"process:{process}")
            print(f"utterance:{utterance}")
            
            if utterance != "":
                break
        except:
            print("API ERROR")
            time.sleep(1)
    return process, utterance


def main(input_file_path, prompt_type, fine_tuning, shot_type):
    
    # JSONファイルを読み込む
    with open(input_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        
    data_inference = []

    count = 0
    data_num = len(data)
    
        
    for one_data in data:
        
        if prompt_type == "standard" and fine_tuning == "original" and shot_type == "zeroshot":
            output_file_path = "../datasets/valid_ze_sta_original.json"
            model = "gpt-4o-mini"
        elif prompt_type == "standard" and fine_tuning == "original" and shot_type == "fewshot":
            output_file_path = "../datasets/valid_fs_sta_original.json"
            model = "gpt-4o-mini"
        elif prompt_type == "standard" and fine_tuning == "finetuned":
            output_file_path = "../datasets/valid_ze_sta_finetuned.json"
            model = ""
            
        elif prompt_type == "proactive" and fine_tuning == "original" and shot_type == "zeroshot":
            output_file_path = "../datasets/valid_ze_pro_original.json"
            model = "gpt-4o-mini"
            
        elif prompt_type == "proactive" and fine_tuning == "original" and shot_type == "fewshot":
            output_file_path = "../datasets/valid_fs_pro_original.json"
            model = "gpt-4o-mini"
                
        elif prompt_type == "proactive" and fine_tuning == "finetuned":
            output_file_path = "../datasets/valid_ze_pro_finetuned.json"
            model = ""
        
        elif prompt_type == "procot" and fine_tuning == "original" and shot_type == "zeroshot":
            output_file_path = "../datasets/valid_ze_procot_original.json"
            model = "gpt-4o-mini"    
        elif prompt_type == "procot" and fine_tuning == "original" and shot_type == "fewshot":
            output_file_path = "../datasets/valid_fs_procot_original.json"
            model = "gpt-4o-mini"
        elif prompt_type == "procot" and fine_tuning == "finetuned":
            output_file_path = "../datasets/valid_ze_procot_finetuned.json"
            model = ""
        
        elif prompt_type == "star" and fine_tuning == "finetuned":
            output_file_path = "../datasets/valid_star_finetuned.json"
            model = ""
        elif prompt_type == "star" and fine_tuning == "original":
            output_file_path = "../datasets/valid_fs_star_original.json"
            model = "gpt-4o-mini"
        
        if prompt_type == "standard" and shot_type == "zeroshot":
            print("Standard Model !!")
            prompt = get_zs_sta_prompt(one_data["item_description"], one_data["traget_price"], one_data["dialogue_history"])
            utterance = get_completion_standard(prompt, model)
            predict_act = ""
            predict_strategy = ""
            gold_process = ""
            predict_process = ""
        elif prompt_type == "standard" and shot_type == "fewshot":
            print("Standard Model !!")
            prompt = get_fs_sta_prompt(one_data["item_description"], one_data["traget_price"], one_data["dialogue_history"])
            utterance = get_completion_standard(prompt, model)
            predict_act = ""
            predict_strategy = ""
            gold_process = ""
            predict_process = ""
        
        if prompt_type == "proactive" and fine_tuning == "original":
            prompt = get_fs_pro_prompt(one_data["item_description"], one_data["traget_price"], one_data["dialogue_history"])
            predict_act, predict_strategy, utterance = get_completion_proactive(prompt, model)
            gold_process = ""
            predict_process = ""
                
        if prompt_type == "proactive" and fine_tuning == "finetuned":
            prompt = get_zs_pro_prompt(one_data["item_description"], one_data["traget_price"], one_data["dialogue_history"])
            predict_act, predict_strategy, utterance = get_completion_proactive(prompt, model)
            gold_process = ""
            predict_process = ""
        
        
        if prompt_type == "procot" and shot_type == "zeroshot" and fine_tuning == "finetuned":
            print("ProCoT Fine-tuning Model !!")
            prompt = get_zs_procot_prompt(one_data["item_description"], one_data["traget_price"], one_data["dialogue_history"])
            predict_process, predict_act, predict_strategy, utterance = get_completion_procot(prompt, model)
            gold_process = ""
        
        if prompt_type == "procot" and shot_type == "fewshot" and fine_tuning == "original":
            prompt = get_fs_procot_prompt(one_data["item_description"], one_data["traget_price"], one_data["dialogue_history"])
            predict_process, predict_act, predict_strategy, utterance = get_completion_procot(prompt, model)
            gold_process = ""
        
        if prompt_type == "star" and shot_type == "zeroshot" and fine_tuning == "finetuned":
            print("STar Fine-tuning Model !!")
            prompt = get_star_prompt(one_data["item_description"], one_data["traget_price"], one_data["dialogue_history"])
            predict_process, utterance = get_completion_star(prompt, model)
            gold_process = ""
            predict_act = ""
            predict_strategy = ""
        
        if prompt_type == "star" and shot_type == "fewshot" and fine_tuning == "original":
            print("STar Original Model !!")
            prompt = get_fs_star_prompt(one_data["item_description"], one_data["traget_price"], one_data["dialogue_history"])
            predict_process, utterance = get_completion_star(prompt, model)
            gold_process = ""
            predict_act = ""
            predict_strategy = ""
    
        one_data_lebaled = {"ID":one_data["ID"], "item_description":one_data["item_description"], "traget_price":one_data["traget_price"], 
                        "gold_respose":one_data["gold_respose"], "predict_utterance":utterance,
                        "gold_process":gold_process, "predict_process":predict_process,
                        "dialogue_act":one_data["dialogue_act"], "predict_act":predict_act,
                        "nego_strategy":one_data["nego_strategy"], "predict_strategy":predict_strategy,
                        "dialogue_history":one_data["dialogue_history"]}
        
        count += 1
        
        data_inference.append(one_data_lebaled)
        
        print("================================")
        #print(f"INFERENCE   :{zs_pro}")
        print(f"ID          :{one_data["ID"]}")
        print(f"FINISHED %  :{int((count/data_num)*100)}%")
        
    
    # JSONL形式でファイルに書き戻す
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data_inference, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    
    input_file_path = "../datasets/negotiate_labeled_valid.json"
    
    prompt_type = "standard"
    #prompt_type = "proactive"
    #prompt_type = "procot"
    #prompt_type = "star"
    
    fine_tuning = "original"
    #fine_tuning = "finetuned"
    
    #shot_type = "zeroshot"
    shot_type = "fewshot"
    
    main(input_file_path, prompt_type, fine_tuning, shot_type)
    