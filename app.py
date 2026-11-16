import argparse, fcntl, hashlib, json, os, re, socket, sqlite3, subprocess, sys, time, urllib.request

os.chdir(os.path.dirname(os.path.abspath(__file__)))
LOCK_FILE = "/tmp/beagle-app.lock"
lockHandle = None
CTX, MAX_TOKENS, TOP_K = 1024, 256, 2

SYSTEM_PROMPT = (
    "Bạn là luật sư tư vấn pháp luật Việt Nam. Trả lời ngắn gọn, chính xác bằng tiếng Việt, "
    "dựa trên các điều luật được cung cấp và nêu rõ Điều, Luật áp dụng. "
    "Nếu không đủ căn cứ, hãy nói rõ là không chắc chắn. Không bịa điều luật."
)
DISCLAIMER = "Thông tin tham khảo từ chó săn (aka Beagle), không thay thế tư vấn của luật sư thật."


def find(*names):
    for n in names:
        if os.path.exists(n):
            return n
    return names[0]


MODEL = find("beagle-q4_k_m.gguf", "build/out/beagle-q4_k_m.gguf")
SERVER_BIN = find("llama-server", "build/llama.cpp/build-host/bin/llama-server")
DB = find("laws.db", "data/laws.db")
SLOT_DIR = "slots"


# ---------- prompt + retrieval ----------

def format_user(articles, question):
    if not articles:
        return f"Câu hỏi: {question}"
    lines = [f"[{i}] {law} – Điều {art}: {text}" for i, (law, art, text) in enumerate(articles, 1)]
    return "Điều luật liên quan:\n" + "\n".join(lines) + f"\n\nCâu hỏi: {question}"


def fts_query(text):
    # shortcut: plain OR of quoted words, no Vietnamese stopword list; BM25 IDF down-weights common words.
    words = list(dict.fromkeys(re.findall(r"\w+", text.lower())))[:32]
    return " OR ".join(f'"{w}"' for w in words)


def retrieve(conn, question, k=TOP_K):
    q = fts_query(question)
    if not q:
        return []
    return conn.execute(
        "SELECT law, article, text FROM articles WHERE articles MATCH ? ORDER BY bm25(articles) LIMIT ?",
        (q, k)).fetchall()


def open_db(path=DB):
    if not os.path.exists(path):
        sys.exit(f"missing {path}, run data/build_index.py")
    return sqlite3.connect(path)


# ---------- llama-server ----------

def http(base, path, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def take_lock():
    """One app.py or bench run per machine at a time: they share port 8080 and the BBB has RAM for one model."""
    global lockHandle
    if lockHandle:
        return
    handle = open(LOCK_FILE, "a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("app.py or a bench run is already running on this machine. "
                 "Find it with: pgrep -af 'app.py|bench/'")
    lockHandle = handle


def start_server(model=MODEL, binary=SERVER_BIN, port=8080, host="127.0.0.1", threads=1,
                 spec=True, cache=True, log="server.log"):
    """Start llama-server. Returns (proc, base_url, load_seconds)."""
    take_lock()
    for p in (model, binary):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    with socket.socket() as s:  # a second server can't bind, but /health would answer from the first one
        if s.connect_ex(("127.0.0.1", port)) == 0:
            sys.exit(f"port {port} is busy: another llama-server is running (a bench run or another app.py). "
                     "Wait for it or stop it: pkill llama-server")
    helptext = subprocess.run([binary, "--help"], capture_output=True, text=True).stdout
    cmd = [binary, "-m", model, "-c", str(CTX), "-n", str(MAX_TOKENS), "-t", str(threads),
           "-b", "64", "-ub", "64", "-np", "1", "--host", host, "--port", str(port)]
    if "--cache-ram" in helptext:
        cmd += ["--cache-ram", "0"]  # newer servers keep extra prompt caches in host RAM; BBB can't afford it
    if spec:
        if "--spec-type" in helptext:
            cmd += ["--spec-type", "ngram-simple"]
        else:
            print("warning: this llama-server has no --spec-type, speculative decoding off", file=sys.stderr)
    if cache:
        os.makedirs(SLOT_DIR, exist_ok=True)
        cmd += ["--slot-save-path", SLOT_DIR]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    while True:
        if proc.poll() is not None:
            sys.exit(f"llama-server exited ({proc.returncode}), see {log}")
        try:
            if http(base, "/health", timeout=5).status == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    load_s = time.time() - t0
    if cache:
        prime_system_slot(base, model)
    return proc, base, load_s


def prime_system_slot(base, model):
    """Restore the system-prompt KV from disk, or compute and save it once."""
    st = os.stat(model)  # a new prompt or a redeployed/different model must not reuse an old KV file
    key = f"{SYSTEM_PROMPT}|{os.path.basename(model)}|{st.st_size}|{int(st.st_mtime)}"
    name = "sys-" + hashlib.sha1(key.encode()).hexdigest()[:10] + ".bin"
    if os.path.exists(os.path.join(SLOT_DIR, name)):
        http(base, "/slots/0?action=restore", {"filename": name}).read()
        return
    # Qwen2.5 ChatML prefix, identical to what the chat template renders for the system message.
    prefix = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    http(base, "/completion", {"prompt": prefix, "n_predict": 0, "cache_prompt": True}).read()
    http(base, "/slots/0?action=save", {"filename": name}).read()


def ask(base, messages, cache=True, t0=None, on_token=None):
    """Stream one chat completion. Returns dict(text, ttft_s, total_s, timings, chunks)."""
    t0 = t0 or time.time()
    body = {"messages": messages, "stream": True, "temperature": 0, "max_tokens": MAX_TOKENS,
            "cache_prompt": cache}
    text, first, timings, chunks = [], None, None, 0
    with http(base, "/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            timings = ev.get("timings", timings)
            for ch in ev.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if first is None:
                        first = time.time()
                    chunks += 1
                    text.append(piece)
                    if on_token:
                        on_token(piece)
    end = time.time()
    return {"text": "".join(text), "ttft_s": (first or end) - t0, "total_s": end - t0,
            "first_t": first, "end_t": end, "timings": timings or {}, "chunks": chunks}


def count_tokens(base, s):
    with http(base, "/tokenize", {"content": s}) as r:
        return len(json.load(r)["tokens"])


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        d[k] = int(v.split()[0])  # kB
    return d


def pgmajfault():
    for line in open("/proc/vmstat"):
        if line.startswith("pgmajfault "):
            return int(line.split()[1])
    return 0


# ---------- chat REPL ----------

def chat(args):
    avail = meminfo().get("MemAvailable", 0) // 1024
    if avail < 450:
        print(f"warning: only {avail} MB available, the model may thrash")
    conn = None if args.no_rag else open_db()
    print("Loading...")
    proc, base, load_s = start_server(spec=not args.no_spec, cache=not args.no_cache)
    print(f"Ready ({load_s:.0f}s). {DISCLAIMER}\nType /reset to delete chat history, Ctrl-D to quit.\n")
    sys_n = count_tokens(base, SYSTEM_PROMPT) + 5
    history = []  # [(question, answer, n_tokens)] stored WITHOUT retrieved articles
    prev_q = ""
    try:
        while True:
            try:
                q = input("You ").strip()
            except EOFError:
                break
            if not q:
                continue
            if q == "/reset":
                history, prev_q = [], ""
                continue
            t0 = time.time()
            arts = [] if conn is None else retrieve(conn, q + " " + prev_q)
            user = format_user(arts, q)
            budget = CTX - MAX_TOKENS - sys_n - count_tokens(base, user) - 5
            while history and sum(h[2] for h in history) > budget:
                history.pop(0)  # drop oldest pair, system prefix stays cached
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
            for hq, ha, _ in history:
                msgs += [{"role": "user", "content": hq}, {"role": "assistant", "content": ha}]
            msgs.append({"role": "user", "content": user})
            f0 = pgmajfault()
            print("Luật sư: ", end="", flush=True)
            r = ask(base, msgs, cache=not args.no_cache, t0=t0,
                    on_token=lambda s: print(s, end="", flush=True))
            tm = r["timings"]
            speed = f", {tm['predicted_per_second']:.1f} tok/s" if "predicted_per_second" in tm else ""
            print(f"\n  [{r['total_s']:.0f}s{speed}]")
            if arts:
                print("  Nguồn: " + "; ".join(f"{law} – Điều {a}" for law, a, _ in arts))
            if pgmajfault() - f0 > 2000:
                print("  warning: model is thrashing; free RAM or use Q3_K_M")
            n = count_tokens(base, q) + count_tokens(base, r["text"]) + 10
            history.append((q, r["text"], n))
            prev_q = q
            print()
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        proc.wait()


def main():
    ap = argparse.ArgumentParser(description="Better call Beagle")
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--serve", action="store_true", help="only run the server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    take_lock()  # before the memory check and "loading" message, so a second app.py stops at once
    if args.serve:
        proc, _, load_s = start_server(port=args.port, host=args.host,
                                       spec=not args.no_spec, cache=not args.no_cache)
        print(f"server up on {args.host}:{args.port} ({load_s:.0f}s), Ctrl-C to stop")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        return
    chat(args)


if __name__ == "__main__":
    main()
