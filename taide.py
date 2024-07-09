from vllm import LLM, SamplingParams
import tqdm

llm = LLM(model="../models/llama2-70b-ccw_cp-l-v2_tv_noemb_xwin", tensor_parallel_size=8, dtype='float16')

file_path = 'qs_gen.txt'
questions = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        clean_line = line.strip()
        if clean_line:
            questions.append(clean_line)

sampling_params = SamplingParams(temperature=0.75, top_p=0.9, max_tokens=1024)
# full_prompt = '<s> {} USER: {} ASSISTANT: '

opt = open('qs_gen_taide.txt', 'w')
for i in tqdm.tqdm(range(len(questions))):
    outputs = llm.generate([questions[i]], sampling_params)
    # print(outputs)
    # print('**************')
    content = outputs[0].outputs[0].text
    content = " ".join(content.split())
    opt.write(content)
    opt.write('\n')
    # print(content)
    # print('--------------------')
    # a = input()