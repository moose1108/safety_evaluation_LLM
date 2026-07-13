"""Extract question-answer pairs judged as unsafe into a JSON file."""
import argparse
import json

import tqdm

from paths import BENCHMARKS, FAILURES, JUDGMENTS, RESPONSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(BENCHMARKS / "qs_gen.txt"))
    parser.add_argument("--responses", default=str(RESPONSES / "main" / "qs_gen_taide.txt"))
    parser.add_argument("--judgments", default=str(JUDGMENTS / "main" / "judge_qs_gen_taide.txt"))
    parser.add_argument("--output", default=str(FAILURES / "extract_qs_gen_taide.json"))
    args = parser.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    with open(args.responses, encoding="utf-8") as f:
        replies = [line.strip() for line in f if line.strip()]
    with open(args.judgments, encoding="utf-8") as f:
        judges = [line.strip() for line in f if line.strip()]

    data = []
    for i in tqdm.tqdm(range(len(replies))):
        if "unsafe" in judges[i].lower():
            data.append(
                {
                    "question": questions[i],
                    "reply": replies[i],
                    "line number": i,
                }
            )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"wrote {len(data)} unsafe cases -> {args.output}")


if __name__ == "__main__":
    main()
