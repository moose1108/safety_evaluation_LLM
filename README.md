主要會用到的東西：

types.txt           問題集（47題）
qs_gen.txt          問題集（1150題按照「評測面向」的）
judge_qs_gen_{}.txt 回答安全與否評估結果--{模型名稱}
qs_gen_{}.txt       {模型名稱}根據qs_gen.txt的回答

generate.py         利用TAIDE 70B模型生成安全性問題
other_model.py      用聯發科模型生成回答
taiwan.py           用林彥廷的模型生成回答
taide.py            用TAIDE 70B模型生成回答
my_llm_judge.py     利用GPT-4「直接」看問題給出該安全性問題的分數
judge_safety.py     利用GPT-4評估回答，評估該回答是否unsafe
check_unsafe.py     抓出回答是unsafe的問題
types_followup.py   根據問題和模型回答，生成追問

其他檔案較不重要，若有疑問在詢問。