from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import tqdm

device = 'cuda:0'

file_path = 'qs_gen.txt'
questions = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        clean_line = line.strip()
        if clean_line:
            questions.append(clean_line)


model = AutoModelForCausalLM.from_pretrained(
    "MediaTek-Research/Breeze-7B-Instruct-v0_1",
    device_map="auto",
    torch_dtype=torch.bfloat16
).to(device)  # Ensure the model is on the GPU

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("MediaTek-Research/Breeze-7B-Instruct-v0_1")
f = open('qs_gen_breeze.txt', 'w')

for question in tqdm.tqdm(range(len(questions))):
    chat = [
    {"role": "user", "content": "你好，請問你可以完成什麼任務？"},
    {"role": "assistant", "content": "你好，我可以回答您提出的各種問題、提供資訊和協助您完成許多不同的任務。例如：回答技術問題、提供建議、翻譯文字、尋找資料或協助您安排行程等。請告訴我如何能幫助您。"},
    {"role": "user", "content": questions[question]},
    ]
    inputs = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_new_tokens=1024,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        temperature=0.9
    )
    text = tokenizer.decode(outputs[0])
    index = text.index('[/INST]')
    text = text[index+7:]
    index = text.index('[/INST]')
    text = text[index+7:]
    # index = text.index('</s>')
    # text = text[:index]
    # print(text)
    clean_text = text.replace('\n', ' ').replace('\r', ' ')
    f.write(clean_text)
    f.write('\n')
    # print('=====================')
