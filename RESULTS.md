# 評測結果與結論

指標定義：**unsafe 率越低越安全**。  
主數字以 `analysis/failures/extract_qs_gen_*.json` 為準（由 judge 結果抽取；含完整問題、回答、行號）。

Judge 原始檔中，部分為冗長解釋（可能在說明文字裡提到「unsafe」），統計時請優先使用：

- 短格式：`judgments/**/judge_*_4.txt` 或已是 `safe`/`unsafe` 單字的檔案
- 或直接用下方 JSON 計數

---

## 1. 主基準（1150 題）

題庫：`benchmarks/qs_gen.txt`（23 面向 × 每面向 50 題）。  
Judge：GPT-4（`scripts/judge_safety.py`）。

| 排名 | 設定 | unsafe | 比率 | 失敗案例 JSON |
|------|------|---------|------|----------------|
| 1 | GPT-3.5-turbo 回答（檔名 gpt4） | 12 | 1.04% | `extract_qs_gen_gpt4.json` |
| 2 | Breeze-7B @ one-shot 題庫 | 10 | 0.87% | `extract_qs_gen_oneshot_breeze.json` |
| 3 | Breeze-7B @ 主基準 | 21 | 1.83% | `extract_qs_gen_breeze.json` |
| 4 | TAIDE Llama2-70B | 32 | 2.78% | `extract_qs_gen_taide.json` |
| 5 | Taiwan-LLM-7B | 40 | 3.48% | `extract_qs_gen_yentinglin.json` |

### 結論 A — 相對安全性

在相同 1150 題下：

**GPT-3.5 ＞ Breeze-7B ＞ TAIDE 70B ＞ Taiwan-LLM-7B**

開源繁中 7B／70B 模型皆比 GPT-3.5 更容易產出被判定為 unsafe 的回答；TAIDE 70B 並未因為參數量大而明顯勝過 Breeze-7B。

---

## 2. 追問（Follow-up）實驗

流程：主基準 Q+A → Breeze 產生追問（`benchmarks/qs_gen_followup.txt`）→ 模型再答 → 再 judge。

| 設定 | unsafe | 比率 | 相對同模型主基準 |
|------|--------|------|------------------|
| Follow-up → Breeze | 51 | 4.43% | 主基準 1.83% → **+2.6 pp** |
| Follow-up → Taiwan-LLM | 56 | 4.87% | 主基準 3.48% → **+1.4 pp** |

### 結論 B — 追問會變得更不安全

單輪回答通過後，**第二輪追問仍會誘出更多 unsafe**。安全性評測不應只看單輪。

---

## 3. 失敗集中在哪些面向？

主基準每 50 題為一面向（見 `benchmarks/CATEGORIES.md`）。各模型 unsafe 的 top 面向：

| 模型 | Top 面向（件數） |
|------|------------------|
| Breeze | 贓物／盜竊 (5)、毒品／非法交易 (5) |
| TAIDE | 毒品／非法交易 (8)、贓物／盜竊 (7) |
| Taiwan-LLM | 毒品／非法交易 (9)、網路詐騙／資安 (6)、贓物／盜竊 (5) |
| Follow-up Breeze | 毒品／非法交易 (11)、贓物／盜竊 (8)、網路詐騙／資安 (7) |
| Follow-up Taiwan-LLM | 毒品／非法交易 (12)、贓物／盜竊 (11) |

### 結論 C — 違法實務類最脆弱

模型較常在「如何處理贓物／非法交易／毒品相關情境」上給出可被視為**提供違法方法或協助**的內容；歧視／髒話類相對較少成為主因（仍有，但不是最大宗）。

---

## 4. 種子基準（47 題）

題庫：`benchmarks/types.txt`。

| 模型 | GPT-4 judge unsafe |
|------|--------------------|
| Breeze | 3 / 47（6.4%） |
| Taiwan-LLM | 3 / 47（6.4%） |

樣本小，僅作早期驗證；正式比較請用 1150 題主基準。

複雜化後種子題品質分（`analysis/scores/compl_types_score.txt`，取每行最後一個均分）：**平均約 4.8 / 10**，誘答性中等。

---

## 5. 其他產物

| 項目 | 說明 |
|------|------|
| `benchmarks/extract_qs_gen.txt` | 從 1150 題人工／半自動篩出的 **135 題**子集，可作精簡 benchmark |
| `analysis/failures/extract_types_legacy.json` | 早期 47 筆失敗案例（無行號），僅作存檔 |
| `judgments/main/judge_qs_gen.txt` | 無對應回答、unsafe 異常高（約 28%），**來源不明，勿用於結論** |
| `qs_gen_gpt4*` | 實際為 **GPT-3.5-turbo** 回答 |

---

## 6. 建議後續

1. 對毒品／贓物等面向加硬拒答或拒答微調，再重跑主基準。  
2. 把 follow-up 納入標準評測協議。  
3. 統一 judge 輸出為短格式（`safe`/`unsafe`），避免冗長解釋干擾抽取。  
4. 釐清或刪除 `judge_qs_gen.txt`。  
5. 考慮將 `extract_qs_gen.txt`（135 題）標註篩選標準後定為「精簡版基準」。

重新產生本頁數字：

```bash
python analysis/summarize_results.py
```
