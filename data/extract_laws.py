# [HOST] Ubuntu host, Python 3 venv at .venv/ (MANUAL_SETUP.md step 1; pyarrow comes with `datasets`).
#
# Physical setup: none. Needs the QA dataset in data/raw/qa/ (MANUAL_SETUP.md step 3).
#
# The QA dataset already holds the text of every article its questions are about
# (doc_name + article_content). This writes each article once to data/raw/laws/qa_articles.jsonl,
# the input of data/build_index.py. Nothing is downloaded.
#   {"law": "<doc_name>", "article": "5", "title": "Điều 5. <title>", "text": "<body>"}
#
# Usage: python data/extract_laws.py
import glob, json, os, re, sys

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA_DIR = os.path.join(ROOT, "data/raw/qa")
OUT = os.path.join(ROOT, "data/raw/laws/qa_articles.jsonl")


def main():
    files = glob.glob(os.path.join(QA_DIR, "**", "*.parquet"), recursive=True)
    if not files:
        sys.exit(f"missing QA parquet files in {QA_DIR}, see MANUAL_SETUP.md step 3")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    seen = set()  # (law, article number): many QA rows share one article
    with open(OUT, "w", encoding="utf-8") as out:
        for f in files:
            for row in pq.read_table(f, columns=["doc_name", "article_content"]).to_pylist():
                title, _, text = (row["article_content"] or "").strip().partition("\n")  # "Điều 5. <title>" / body
                match = re.match(r"\s*Điều\s+(\d+)", title)
                if not match:
                    continue
                key = (row["doc_name"], match.group(1))
                if key in seen:
                    continue
                seen.add(key)
                article = {"law": key[0], "article": key[1], "title": title.strip(), "text": text.strip()}
                out.write(json.dumps(article, ensure_ascii=False) + "\n")

    lawCount = len({law for law, _ in seen})
    print(f"{len(seen)} articles from {lawCount} laws -> {OUT}")


if __name__ == "__main__":
    main()
