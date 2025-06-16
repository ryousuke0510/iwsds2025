import os
from openai import OpenAI

client = OpenAI(
        api_key = ""
    )

def chatgpt_4o_mini_finetuned(message_arg, model):
    
    message=[{"role": "assistant", "content": message_arg},]
        
    response = client.chat.completions.create(
        model=model, # model = "deployment_name".
        messages = message,
    )
    
    return response.choices[0].message.content

# omniの通常出力
def chatgpt_4_omni(message_arg):
     
    message=[{"role": "user", "content": message_arg},]
        
    response = client.chat.completions.create(
        model="gpt-4o-mini", # model = "deployment_name".
        messages = message,
    )
    
    return response.choices[0].message.content

# omniの通常出力
def chatgpt_4_omni_formatted(message):
        
    response = client.chat.completions.create(
        model="gpt-4o-mini", # model = "deployment_name".
        messages = message,
    )
    
    return response.choices[0].message.content

def text_to_embedding(text):
    try:
        response = client.embeddings.create(
            model= "text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding
    except:
        response = client.embeddings.create(
            model= "text-embedding-3-small",
            input="hello"
        )
        return response.data[0].embedding
    
if __name__ == "__main__":
    print(chatgpt_4_omni("Hello!"))