import argparse, csv, json, os, platform, socket, statistics, sys, threading, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root (~/beagle on the BBB), like `cd` in the shell scripts
sys.path.insert(0, ".")
import app  # noqa: E402

TESTSET = "bench/testset.jsonl"
RESULTS = "bench/results"
SPEED_KEYS = ["retrieval_ms", "ttft_s", "e2e_s", "prefill_tps", "decode_tps"]
CSV_FIELDS = ["i"] + SPEED_KEYS + ["prompt_n", "predicted_n", "question"]


def rss_mb(pid):
    for line in open(f"/proc/{pid}/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return 0.0


def machine_used_mb():
    m = app.meminfo()
    return (m["MemTotal"] - m["MemAvailable"]) / 1024


class MemSampler(threading.Thread):
    """Samples llama-server RSS and machine RAM use every 200 ms and keeps the peaks."""

    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid = pid
        self.running = True
        self.peakProc = 0.0
        self.peakMachine = 0.0

    def run(self):
        while self.running:
            try:
                self.peakProc = max(self.peakProc, rss_mb(self.pid))
            except FileNotFoundError:  # server exited
                return
            self.peakMachine = max(self.peakMachine, machine_used_mb())
            time.sleep(0.2)


def load_questions(n):
    if not os.path.exists(TESTSET):
        sys.exit(f"missing {TESTSET}: run data/prepare.py (MANUAL_SETUP.md step 8), "
                 "and on the BBB copy it over (step 16)")
    rows = [json.loads(line) for line in open(TESTSET, encoding="utf-8") if line.strip()]
    return rows if n == "all" else rows[:int(n)]


def rounded(x, digits):
    return None if x is None else round(x, digits)


def ask_one(base, conn, question, cache):
    t0 = time.time()
    arts = app.retrieve(conn, question) if conn else []
    retrievalMs = (time.time() - t0) * 1000 if conn else None
    msgs = [{"role": "system", "content": app.SYSTEM_PROMPT},
            {"role": "user", "content": app.format_user(arts, question)}]
    r = app.ask(base, msgs, cache=cache, t0=t0)  # ttft and e2e include retrieval
    t = r["timings"]
    prefill = t["prompt_n"] / (t["prompt_ms"] / 1000) if t.get("prompt_ms") else None
    decode = t["predicted_n"] / (t["predicted_ms"] / 1000) if t.get("predicted_ms") else None
    if decode is None and r["chunks"] > 1:  # server sent no timings: client-side estimate
        decode = (r["chunks"] - 1) / (r["end_t"] - r["first_t"])
    return {"retrieval_ms": rounded(retrievalMs, 1), "ttft_s": round(r["ttft_s"], 3),
            "e2e_s": round(r["total_s"], 3), "prefill_tps": rounded(prefill, 2),
            "decode_tps": rounded(decode, 2), "prompt_n": t.get("prompt_n"), "predicted_n": t.get("predicted_n")}


def summarize(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {}
    last = len(values) - 1
    return {"mean": round(statistics.mean(values), 3), "p50": values[round(0.5 * last)],
            "p90": values[round(0.9 * last)], "max": values[-1]}


def main():
    ap = argparse.ArgumentParser(description="Speed + memory benchmark on this machine")
    ap.add_argument("--n", default="20", help="number of test questions, or 'all'")
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--threads", type=int, default=os.cpu_count(), help="llama-server threads (BBB: 1 core)")
    args = ap.parse_args()
    flags = [("no-rag", args.no_rag), ("no-spec", args.no_spec), ("no-cache", args.no_cache)]
    config = "+".join(name for name, on in flags if on) or "default"
    tag = time.strftime("%Y%m%d-%H%M%S")
    cache = not args.no_cache

    questions = load_questions(args.n)
    conn = None if args.no_rag else app.open_db()
    print(f"bench [{config}] on {socket.gethostname()}: {len(questions)} questions, "
          f"{args.threads} threads, model {app.MODEL}", flush=True)
    proc, base, _ = app.start_server(threads=args.threads, spec=not args.no_spec, cache=cache)
    sampler = MemSampler(proc.pid)
    sampler.start()
    rows = []
    try:
        for i, q in enumerate(questions):
            r = ask_one(base, conn, q["question"], cache)
            rows.append({"i": i, **r, "question": q["question"]})
            print(f"[{i + 1}/{len(questions)}] {r['e2e_s']:.1f}s  {q['question'][:60]}", flush=True)
    finally:
        sampler.running = False
        proc.terminate()
        proc.wait()

    metrics = {}
    for k in SPEED_KEYS:
        for stat, v in summarize([r[k] for r in rows]).items():
            metrics[f"{k}_{stat}"] = v
    metrics["peak_proc_rss_mb"] = round(sampler.peakProc, 1)
    metrics["peak_machine_used_mb"] = round(sampler.peakMachine, 1)

    out = os.path.join(RESULTS, f"{socket.gethostname()}_{config}_{tag}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "questions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    summary = {"machine": socket.gethostname(), "arch": platform.machine(), "config": config, "tag": tag,
               "n": len(rows), "threads": args.threads, "model": os.path.basename(app.MODEL), "metrics": metrics}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    print(f"\n== bench [{config}] n={len(rows)} -> {out}")
    for k, v in metrics.items():
        print(f"  {k:28} {v}")


if __name__ == "__main__":
    main()
