import re
import tqdm
import json

device = 'cuda:0'

file_path = 'qs_gen.txt'
file_path2 = 'qs_gen_taide.txt' # yentinglin
file_path3 = 'judge_qs_gen_taide.txt'
questions = []
replys = []
judges = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        clean_line = line.strip()
        if clean_line:
            questions.append(clean_line)

with open(file_path2, 'r', encoding='utf-8') as file:
    for line in file:
        clean_line = line.strip()
        if clean_line:
            replys.append(clean_line)

with open(file_path3, 'r', encoding='utf-8') as file:
    for line in file:
        clean_line = line.strip()
        if clean_line:
            judges.append(clean_line)

data = []

for i in tqdm.tqdm(range(len(replys))):
    if 'unsafe' in judges[i]:
        # print(i+1)
        # a = input()
        data.append({
            "question": questions[i],
            "reply": replys[i],
            "line number": i
        })

with open('extract_qs_gen_taide.json', 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False) 
print(len(data))