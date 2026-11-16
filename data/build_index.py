import glob, json, os, random, re, sqlite3, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import app  # noqa: E402

RAW = os.path.join(ROOT, "data/raw/laws")
OUT = os.path.join(ROOT, "data/laws.db")
MAX_TOK = 150


def load_articles():
    files = glob.glob(os.path.join(RAW, "*.jsonl"))
    if not files:
        sys.exit(f"missing law corpus in {RAW}")
    for f in files:
        for line in open(f, encoding="utf-8"):
            if line.strip():
                o = json.loads(line)
                yield o["law"], str(o["article"]), o.get("title", ""), o["text"]


def chunks(header, text, ntok):
    """Split article text into header-prefixed chunks of <= MAX_TOK tokens, on clause lines first."""
    room = MAX_TOK - ntok(header) - 1
    pieces = []
    for p in [p.strip() for p in re.split(r"\n(?=\s*\d+\.\s)", text) if p.strip()] or [text]:
        if ntok(p) <= room:
            pieces.append(p)
            continue
        cur, n = [], 0  # clause too long: hard-split by words (per-word counts are a close upper bound)
        for w in p.split():
            wn = ntok(" " + w)
            if cur and n + wn > room:
                pieces.append(" ".join(cur))
                cur, n = [], 0
            cur.append(w)
            n += wn
        if cur:
            pieces.append(" ".join(cur))
    out, cur = [], []
    for p in pieces:  # group clauses greedily up to the cap
        if cur and ntok("\n".join(cur + [p])) > room:
            out.append("\n".join(cur))
            cur = []
        cur.append(p)
    if cur:
        out.append("\n".join(cur))
    return [header + "\n" + c for c in out]


def main():
    from transformers import AutoTokenizer
    base = os.path.join(ROOT, "train/base")
    if not os.path.isdir(base):
        sys.exit(f"missing {base}")
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731

    if os.path.exists(OUT):
        os.remove(OUT)
    conn = sqlite3.connect(OUT)
    conn.execute("CREATE VIRTUAL TABLE articles USING fts5(law UNINDEXED, article UNINDEXED, text,"
                 " tokenize='unicode61 remove_diacritics 0')")
    rows = []
    for law, art, title, text in load_articles():
        num = (re.findall(r"\d+", art) or [art])[0]
        title = (title or "").strip()
        header = title if title.lower().startswith("điều") else f"Điều {num}. {title}".strip()
        for c in chunks(header, (text or "").strip(), ntok):
            rows.append((law, num, c))
        if len(rows) % 10000 < 5:
            print(f"\r{len(rows)} chunks", end="", flush=True)
    conn.executemany("INSERT INTO articles VALUES (?, ?, ?)", rows)
    conn.commit()
    print(f"\nwrote {len(rows)} chunks to {OUT}")

    # self-check: the first words of random chunks retrieve their article (same law + number) in the top 2.
    # Not the exact chunk: chunks of one long article share the header, so a sibling chunk can win.
    assert rows, "no articles loaded"
    sample = random.Random(0).sample(rows, min(50, len(rows)))
    hits = sum(any(r[:2] == c[:2] for r in app.retrieve(conn, " ".join(c[2].split()[:15]))) for c in sample)
    assert all(ntok(r[2]) <= MAX_TOK + 20 for r in sample), "chunk over token cap"  # +20: header slack
    print(f"self-check: {hits}/{len(sample)} chunks found their article by their own opening words")
    assert hits >= 0.8 * len(sample), "retrieval self-check failed"


if __name__ == "__main__":
    main()
