import argparse, glob, json, os, random, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import app  # noqa: E402

RAW = os.path.join(ROOT, "data/raw/qa")
MAX_CHAT_TOKENS = 750   # system + 2 articles + question + answer; leaves room for 1 turn of history at 1024
MAX_ANSWER_TOKENS = app.MAX_TOKENS
CITE_RE = re.compile(r"điều\s+(\d+)", re.I)


def load_rows(args):
    from datasets import load_dataset
    kinds = {"json": ["*.json", "*.jsonl"], "parquet": ["*.parquet"], "csv": ["*.csv"]}
    rows = []
    for builder, pats in kinds.items():
        files = [f for p in pats for f in glob.glob(os.path.join(RAW, "**", p), recursive=True)]
        if files:
            rows += list(load_dataset(builder, data_files=files, split="train"))
    if not rows:
        sys.exit(f"missing QA files in {RAW}")
    if args.explode:  # one row holds a list of QA dicts
        rows = [{**r, **item} for r in rows for item in (r.get(args.explode) or [])]
    missing = {args.q_field, args.a_field} - set(rows[0])
    if missing:
        sys.exit(f"fields {missing} not in data; available: {sorted(rows[0])}. Pass --q-field/--a-field.")
    return rows


def parse_gold(v):
    """Gold article from a dataset field -> {"law", "article"} or None."""
    # shortcut: law name = text left after removing "Điều N"; good enough for substring matching in bench.
    if isinstance(v, dict):
        art = str(v.get("article") or v.get("article_id") or "")
        law = str(v.get("law") or v.get("law_id") or "")
        return {"law": law, "article": art} if art else None
    if isinstance(v, list):
        return parse_gold(v[0]) if v else None
    s = str(v or "")
    m = CITE_RE.search(s) or re.fullmatch(r"\s*(\d+)\s*", s)
    if not m:
        return None
    law = re.sub(r"[,.;:\s]+", " ", (s[:m.start()] + s[m.end():])).strip()
    return {"law": law, "article": m.group(1)}


def norm(s):
    return " ".join(re.findall(r"\w+", s.lower()))


def shingles(s):
    w = norm(s).split()
    return {" ".join(w[i:i + 3]) for i in range(max(1, len(w) - 2))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q-field", default="question")
    ap.add_argument("--a-field", default="answer")
    ap.add_argument("--gold-field", default=None, help="field holding the source article, e.g. 'Điều 5 Luật X'")
    ap.add_argument("--law-field", default=None, help="field holding the law name; replaces the law parsed from --gold-field")
    ap.add_argument("--explode", default=None, help="field holding a list of QA dicts per row")
    ap.add_argument("--test-size", type=int, default=150)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    base = os.path.join(ROOT, "train/base")
    if not os.path.isdir(base):
        sys.exit(f"missing {base}")
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    sys_n = ntok(app.SYSTEM_PROMPT)
    assert sys_n <= 100, f"system prompt is {sys_n} tokens, trim it in app.py"
    conn = app.open_db(os.path.join(ROOT, "data/laws.db"))

    # clean + dedupe
    seen, data = set(), []
    for r in load_rows(args):
        q = " ".join(str(r[args.q_field] or "").split())
        a = re.sub(r"[ \t]+", " ", str(r[args.a_field] or "")).strip()
        k = norm(q)
        if not q or not a or k in seen:
            continue
        seen.add(k)
        gold = parse_gold(r.get(args.gold_field)) if args.gold_field else None
        if gold and args.law_field:
            gold["law"] = str(r.get(args.law_field) or "")
        if gold and not CITE_RE.search(a):  # teach the model to cite
            a += f"\n\nCăn cứ: Điều {gold['article']}" + (f", {gold['law']}" if gold["law"] else "")
        if ntok(a) <= MAX_ANSWER_TOKENS:
            data.append({"question": q, "answer": a, "gold": gold})
    print(f"{len(data)} clean QA pairs")
    random.Random(args.seed).shuffle(data)

    # held-out test set, then drop near-duplicates of it from train/eval (3-gram Jaccard > 0.7)
    test, rest = data[:args.test_size], data[args.test_size:]
    index = {}
    for i, t in enumerate(test):
        for s in shingles(t["question"]):
            index.setdefault(s, set()).add(i)
    test_sh = [shingles(t["question"]) for t in test]
    kept = []
    for r in rest:
        sh = shingles(r["question"])
        cands = set().union(*(index.get(s, set()) for s in sh)) if sh else set()
        if not any(len(sh & test_sh[i]) / len(sh | test_sh[i]) > 0.7 for i in cands):
            kept.append(r)
    print(f"dropped {len(rest) - len(kept)} near-duplicates of test questions")

    # chat samples with retrieved articles, same format as app.py
    samples = []
    for i, r in enumerate(kept):
        arts = app.retrieve(conn, r["question"])
        msgs = [{"role": "system", "content": app.SYSTEM_PROMPT},
                {"role": "user", "content": app.format_user(arts, r["question"])},
                {"role": "assistant", "content": r["answer"]}]
        if len(tok.apply_chat_template(msgs, tokenize=True)) <= MAX_CHAT_TOKENS:
            samples.append({"messages": msgs})
        if i % 2000 == 0:
            print(f"\rretrieval {i}/{len(kept)}", end="", flush=True)
    print()
    n_eval = max(1, int(len(samples) * args.eval_frac))
    evals, train = samples[:n_eval], samples[n_eval:]

    def dump(path, rows):
        with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def text_file(path, rows, cap):
        with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
            size = 0
            for r in rows:
                t = tok.apply_chat_template(r["messages"], tokenize=False)
                f.write(t)
                size += len(t.encode())
                if size > cap:
                    break

    dump("data/train.jsonl", train)
    dump("data/eval.jsonl", evals)
    dump("bench/testset.jsonl", test)
    text_file("data/calib.txt", train, 300_000)
    text_file("data/eval_text.txt", evals, 150_000)

    # self-checks
    test_keys = {norm(t["question"]) for t in test}
    assert train and evals and test, "empty split"
    assert not any(norm(s["messages"][1]["content"].split("Câu hỏi: ")[-1]) in test_keys for s in train), "test leak"
    assert all(s["messages"][0]["content"] == app.SYSTEM_PROMPT for s in train[:100])
    print(f"train {len(train)}, eval {len(evals)}, test {len(test)} (system prompt {sys_n} tokens)")


if __name__ == "__main__":
    main()
