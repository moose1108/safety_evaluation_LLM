import openai
import os
import dotenv
import tqdm

dotenv.load_dotenv()
openai.api_key = os.getenv('API_KEY')
openai.organization = os.getenv('ORGANIZATION_LLM')

client = openai.OpenAI(
    organization=openai.organization,
    api_key=openai.api_key
)

file_path = 'qs_gen.txt'
questions = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        clean_line = line.strip()
        if clean_line:
            questions.append(clean_line)

f = open('qs_gen_gpt4.txt', 'w')

for i in tqdm.tqdm(range(len(questions))):
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "你是博學多聞的人，協助回答有疑問的人的提問，且你的回答很精準"},
        {"role": "user", "content": questions[i]}
        ]
    )
    text = completion.choices[0].message.content
    clean_text = text.replace('\n', ' ').replace('\r', ' ')
    # print(clean_text)
    f.write(clean_text)
    f.write('\n')
