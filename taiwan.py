from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import tqdm

device = 'cuda:0'

file_path = 'qs_gen_followup.txt'
questions = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        clean_line = line.strip()
        if clean_line:
            questions.append(clean_line)


model = AutoModelForCausalLM.from_pretrained(
    "yentinglin/Taiwan-LLM-7B-v2.0.1-chat",
    device_map="auto",
    torch_dtype=torch.bfloat16
).to(device)  # Ensure the model is on the GPU

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("yentinglin/Taiwan-LLM-7B-v2.0.1-chat")
f = open('qs_gen_followup_yentinglin.txt', 'w')

for question in tqdm.tqdm(range(len(questions))):
    chat = [
    {"role": "user", "content": "你好，請問你可以完成什麼任務？"},
    {"role": "assistant", "content": "你好，我可以回答您提出的各種問題、提供資訊和協助您完成許多不同的任務。例如：回答技術問題、提供建議、翻譯文字、尋找資料或協助您安排行程等。請告訴我如何能幫助您。"},
    {"role": "user", "content": questions[question]},
    ]
    inputs = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True).to(device)
    outputs = model.generate(
        inputs,
        max_new_tokens=1024,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        temperature=0.9
    )
    # print(outputs)
    text = tokenizer.decode(outputs[0])
    # print(text)
    if 'ASSISTANT:' in text:
        index = text.index('ASSISTANT:')
        text = text[index+10:]
        index = text.index('ASSISTANT:')
        text = text[index+10:]
    clean_text = text.replace('\n', ' ').replace('\r', ' ')
    # print(clean_text)
    f.write(clean_text)
    f.write('\n')
    # print('=====================')
