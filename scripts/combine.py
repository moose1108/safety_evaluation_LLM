"""Count unsafe failures by the 23 evaluation categories (50 questions each)."""
import argparse
import json

from paths import FAILURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(FAILURES / "extract_qs_gen_gpt4.json"),
        help="failure JSON from check_unsafe.py",
    )
    args = parser.parse_args()

    with open(args.json_file, encoding="utf-8") as f:
        data = json.load(f)

    counts = [0] * 23
    for item in data:
        counts[int(item["line number"]) // 50] += 1

    print("per-category counts:", counts)
    print("total:", sum(counts), "(file has", len(data), "items)")


if __name__ == "__main__":
    main()
