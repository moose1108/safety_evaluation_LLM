# LLM 安全性評測（TAIDE）

以繁中安全性問題評測語言模型是否會產出歧視、違法教學、不雅用詞等 unsafe 回答。

## 主要結論（精簡）

在 **1150 題主基準**、GPT-4 判定下（以 `analysis/failures/*.json` 為準）：

| 模型 | unsafe | 比率 |
|------|--------|------|
| GPT-3.5-turbo（檔名 `gpt4`） | 12 | **1.0%** |
| Breeze-7B（one-shot 題庫） | 10 | **0.9%** |
| Breeze-7B | 21 | **1.8%** |
| TAIDE 70B | 32 | **2.8%** |
| Taiwan-LLM-7B | 40 | **3.5%** |
| Follow-up → Breeze | 51 | **4.4%** |
| Follow-up → Taiwan-LLM | 56 | **4.9%** |

**重點：** 開源繁中模型 unsafe 率高於 GPT-3.5；**追問會提高 unsafe 率**；失敗案例集中在**毒品／非法交易、贓物／盜竊**等面向。

完整表格與說明見 [RESULTS.md](RESULTS.md)。

## 目錄結構

```
benchmarks/     問題集（種子 47 題、主基準 1150 題、追問等）
responses/      各模型回答（main / followup / oneshot / types）
judgments/      GPT-4（或 GPT-3.5）安全判定
analysis/       失敗案例 JSON、評分、統計腳本
scripts/        產生問題／回答／評分的 pipeline
archive/        空檔、實驗稿、不完整產物
```

## 評測流程

```
types.txt (47) ──► 擴寫／生成 ──► qs_gen.txt (1150, 23 面向×50)
                                      │
                                      ▼
                         模型回答 → responses/**/qs_gen_*.txt
                                      │
                                      ▼
                         GPT-4 judge → judgments/**/judge_*.txt
                                      │
                                      ▼
                         抽取 unsafe → analysis/failures/*.json
```

## 常用檔案

| 路徑 | 說明 |
|------|------|
| `benchmarks/types.txt` | 種子問題 47 題 |
| `benchmarks/qs_gen.txt` | **主基準** 1150 題 |
| `benchmarks/extract_qs_gen.txt` | 從主基準篩出的 135 題子集 |
| `benchmarks/CATEGORIES.md` | 23 個評測面向說明 |
| `responses/main/qs_gen_{model}.txt` | 主基準回答 |
| `judgments/main/judge_qs_gen_{model}.txt` | 主基準判定 |
| `analysis/failures/extract_qs_gen_{model}.json` | unsafe 案例（含 Q/A/行號） |

## 腳本（在 `scripts/` 下執行）

```bash
cd scripts
python check_unsafe.py --responses ../responses/main/qs_gen_breeze.txt \
                       --judgments ../judgments/main/judge_qs_gen_breeze_4.txt \
                       --output ../analysis/failures/extract_qs_gen_breeze.json
python combine.py ../analysis/failures/extract_qs_gen_taide.json
python ../analysis/summarize_results.py
```

| 腳本 | 用途 |
|------|------|
| `taide.py` / `other_model.py` / `qs_gen_gpt4.py` | 主基準回答（TAIDE / Breeze / **實際為 GPT-3.5**） |
| `taiwan.py` | Follow-up 回答（Taiwan-LLM） |
| `types_followup.py` | 依 Q+A 產生追問 |
| `judge_safety.py` | GPT-4 判定 safe/unsafe |
| `check_unsafe.py` | 抽出 unsafe 案例為 JSON |
| `combine.py` | 依 23 面向統計 unsafe 分佈 |
| `generate.py` / `gen_from_reply.py` | 從有毒回覆反推問題 |
| `gen_from_topic.py` / `improvement.py` | 題庫生成／複雜化 |

## 環境

```bash
pip install -r requirements.txt
# OpenAI 腳本需在專案根目錄放置 .env：
# API_KEY=...
# ORGANIZATION_LLM=...
```

本地 TAIDE 推論依賴 vLLM 與 `../models/...` 路徑。

## 命名注意

- `qs_gen_gpt4.txt`：**回答模型是 GPT-3.5-turbo**，檔名有誤導。
- `judge_*_4.txt`：短格式（每行僅 `safe`/`unsafe`），比冗長解釋版好統計。
- `judgments/main/judge_qs_gen.txt`：無對應回答檔，來源不明，**不納入正式比較**。
