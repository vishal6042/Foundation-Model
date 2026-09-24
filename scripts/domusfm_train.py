"""Start a DomusFM run in the background and watch its progress (Windows / Linux / macOS).

    .venv/Scripts/python scripts/domusfm_train.py start    # launch configs/domusfm_corpus_fixed.yaml, then watch
    .venv/Scripts/python scripts/domusfm_train.py watch    # re-attach the live view (Ctrl+C leaves training running)
    .venv/Scripts/python scripts/domusfm_train.py stop     # stop the training process
    .venv/Scripts/python scripts/domusfm_train.py watch --once   # print one snapshot and exit

Options: --overlay <yaml> (default configs/domusfm_corpus_fixed.yaml), --set key=value ... (passed to the run),
--every <seconds> (refresh interval, default 10).

Training runs detached from this window. Log: runs/<name>.log, process id: runs/<name>.pid. Running `start`
again after a stop or crash resumes: pretraining from its last checkpoint, fine-tuning from results.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "domusfm.yaml"
DEFAULT_OVERLAY = "configs/domusfm_corpus_fixed.yaml"
LAST_RUN = ROOT / "results" / "domusfm_corpus" / "results.json"  # paper-masking run, for comparison
HEADER = "##### domusfm_train start"

PRE = re.compile(r"pretrain\[(\w+)\] step\s+(\d+) loss ([\d.]+)"
                 r"(?: \(contrastive ([\d.]+) mlm ([\d.]+) z_std ([\d.]+)\))? \((\d+)s\)")


# ---- config ------------------------------------------------------------------------------------

def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_cfg(overlay: str, sets: list[str]) -> dict:
    cfg = yaml.safe_load(BASE.read_text())
    if overlay:
        cfg = _merge(cfg, yaml.safe_load((ROOT / overlay).read_text()))
    for ov in sets:
        key, val = ov.split("=", 1)
        node, (*parents, leaf) = cfg, key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


def paths(cfg: dict) -> tuple[Path, Path, Path]:
    out = ROOT / cfg["out_dir"]
    return out, out.with_suffix(".log"), out.with_suffix(".pid")


# ---- process -----------------------------------------------------------------------------------

def alive(pid: int) -> bool:
    if os.name == "nt":  # os.kill(pid, 0) would terminate the process on Windows
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid(pid_path: Path) -> int | None:
    try:
        return int(pid_path.read_text().strip())
    except (OSError, ValueError):
        return None


def start(args, cfg):
    out, log, pid_path = paths(cfg)
    pid = read_pid(pid_path)
    if pid and alive(pid):
        print(f"Already running (pid {pid}). Watching it instead.")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", "homefm.baselines.domusfm.run"]
    if args.overlay:
        cmd += ["--overlay", args.overlay]
    if args.set:
        cmd += ["--set", *args.set]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore",
               HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    with open(log, "ab") as f:
        f.write(f"\n{HEADER} {dt.datetime.now():%Y-%m-%d %H:%M:%S}: {' '.join(cmd)}\n".encode())
        f.flush()
        kw = dict(cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        if os.name == "nt":
            kw["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        p = subprocess.Popen(cmd, **kw)
    pid_path.write_text(str(p.pid))
    print(f"Started training (pid {p.pid}). Log: {log.relative_to(ROOT)}")


def stop(cfg):
    _, _, pid_path = paths(cfg)
    pid = read_pid(pid_path)
    if not pid or not alive(pid):
        print("Not running.")
        return
    os.kill(pid, signal.SIGTERM)
    print(f"Stopped pid {pid}. Run `start` again to resume from the last checkpoint.")


# ---- parsing -----------------------------------------------------------------------------------

def read_log(log: Path) -> str:
    if not log.exists():
        return ""
    raw = log.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):  # PowerShell `>` writes UTF-16
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def parse(text: str) -> dict:
    seg = text[text.rfind(HEADER):] if HEADER in text else text  # the current launch only
    losses = {}  # (phase, step) -> row, across launches so a resumed run keeps its history
    for m in PRE.finditer(text):
        ph, st = m.group(1), int(m.group(2))
        losses[(ph, st)] = dict(phase=ph, step=st, loss=float(m.group(3)),
                                contrastive=float(m.group(4)) if m.group(4) else float(m.group(3)),
                                mlm=float(m.group(5)) if m.group(5) else None,
                                z_std=float(m.group(6)) if m.group(6) else None)
    recent = [(m.group(1), int(m.group(2)), int(m.group(7))) for m in PRE.finditer(seg)]
    lines = [ln for ln in seg.splitlines() if ln.strip()]
    return dict(
        losses=[losses[k] for k in sorted(losses, key=lambda k: (k[0] != "attribute", k[1]))],
        recent=recent,
        reused="[domusfm] reusing" in seg,
        done="[domusfm] saved" in seg,
        error=next((ln for ln in reversed(lines) if re.match(r"\w*(Error|Exception)\b", ln)), None)
        if "Traceback" in seg else None,
        device=m.groups() if (m := re.search(r"device=(\S+) params=([\d,]+)", seg)) else (None, None),
        last_line=lines[-1][:110] if lines else "",
    )


# ---- display -----------------------------------------------------------------------------------

BARS = "▁▂▃▄▅▆▇█"


def spark(vals: list[float], width: int = 40) -> str:
    vals = [math.log10(max(v, 1e-4)) for v in vals][-width:]  # log scale: a collapse to 1e-3 is obvious
    if not vals:
        return ""
    lo, hi = -4.0, max(max(vals), 0.7)
    return "".join(BARS[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


def bar(frac: float, width: int = 30) -> str:
    n = int(round(frac * width))
    return "[" + "#" * n + "-" * (width - n) + f"] {100 * frac:5.1f}%"


def fmt_s(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def health(row: dict) -> str:
    if row["step"] < 1000:
        return "warming up"
    c, z = row["contrastive"], row["z_std"]
    if c < 0.01 or (z is not None and z < 0.02):
        return "COLLAPSED: game too easy (stop, raise attr/event_mask_p or set homes_per_batch=1)"
    if c < 0.05:
        return "LOW: close to collapsing, keep an eye on it"
    return "OK: still learning"


def last_run_deltas() -> dict:
    try:
        res = json.loads(LAST_RUN.read_text())
    except (OSError, ValueError):
        return {}
    return _deltas(res)


def _deltas(res: dict) -> dict:
    out = {}
    for r in res.values():
        for key, v in r["results"].items():
            task, pct, name = key.split("|")
            if name == "DomusFM" and f"{task}|{pct}|w/o Pretrain" in r["results"]:
                out.setdefault(f"{task} {pct}", []).append(v["mean"] - r["results"][f"{task}|{pct}|w/o Pretrain"]["mean"])
    return {k: sum(v) / len(v) for k, v in out.items()}


def render(cfg: dict, overlay: str) -> tuple[str, bool]:
    out, log, pid_path = paths(cfg)
    pt, ft = cfg["pretrain"], cfg["finetune"]
    info = parse(read_log(log))
    pid = read_pid(pid_path)
    running = bool(pid and alive(pid))
    age = time.time() - log.stat().st_mtime if log.exists() else None

    try:
        results = json.loads((out / "results.json").read_text())
    except (OSError, ValueError):
        results = {}
    held = cfg.get("held_out") or []
    names = ["DomusFM"] + (["w/o Pretrain"] if cfg.get("ablation_no_pretrain") else [])
    n_total = len(held) * len(ft["tasks"]) * len(ft["train_pcts"]) * len(names)
    n_done = sum(len(r["results"]) for r in results.values())

    if info["done"]:
        status = "FINISHED"
    elif running:
        status = f"RUNNING (pid {pid})"
    elif info["error"]:
        status = f"CRASHED: {info['error'][:90]}"
    else:
        status = "STOPPED (run `start` to resume)" if log.exists() else "NOT STARTED (run `start`)"

    L = [f"DomusFM training  {overlay}  ->  {cfg['out_dir']}", "=" * 78,
         f"Status   {status}"]
    if age is not None:
        L.append(f"Log      {log.relative_to(ROOT)}  (updated {fmt_s(age)} ago)"
                 + ("  <- no output for a while" if running and age > 1200 else ""))
    if info["device"][0]:
        L.append(f"Device   {info['device'][0]}   params {info['device'][1]}")

    # pretraining
    s1, s2 = pt["steps_phase1"], pt["steps_phase2"]
    L += ["", "PRETRAINING"]
    ckpt = out / "pretrained_shared.pt"
    if info["reused"] or (ckpt.exists() and not info["recent"]):
        L.append("  done (saved backbone reused)")
    elif info["losses"]:
        last = info["losses"][-1]
        overall = last["step"] + (s1 if last["phase"] == "event" else 0)
        pre_done = n_done > 0 or "=== held-out" in read_log(log)[-20000:]
        frac = 1.0 if pre_done else min(1.0, overall / max(1, s1 + s2))
        L.append(f"  {bar(frac)}  phase {1 if last['phase'] == 'attribute' else 2}/2 ({last['phase']}), "
                 f"step {last['step']:,}/{s1 if last['phase'] == 'attribute' else s2:,}")
        rec = [r for r in info["recent"] if r[0] == last["phase"]]
        if not pre_done and len(rec) >= 2 and rec[-1][2] > rec[0][2]:
            rate = (rec[-1][1] - rec[0][1]) / (rec[-1][2] - rec[0][2])
            L.append(f"  speed {rate:.1f} steps/s   ETA pretraining {fmt_s((s1 + s2 - overall) / rate)}")
        mlm = f"   mlm {last['mlm']:.4f}" if last["mlm"] is not None else ""
        z = f"   z_std {last['z_std']:.3f}" if last["z_std"] is not None else ""
        L.append(f"  contrastive {last['contrastive']:.4f}{mlm}{z}   (chance = {math.log(pt['batch_size']):.2f})")
        L.append(f"  health: {health(last)}")
        for ph in ("attribute", "event"):
            c = [r["contrastive"] for r in info["losses"] if r["phase"] == ph]
            if c:
                L.append(f"  {ph:9s} contrastive (log scale) {spark(c)}  {c[0]:.3f} -> {c[-1]:.4f}")
        L.append("  last run (paper masking) collapsed to 0.0005 by step 5,000")
    else:
        L.append("  waiting for the first step (loading the homes can take a few minutes)")

    # fine-tuning
    L += ["", f"FINE-TUNING + TEST   {bar(n_done / max(1, n_total))}  {n_done}/{n_total} settings"]
    secs = sum(r.get("seconds", 0) for r in results.values())
    if n_done and running:
        L.append(f"  ETA fine-tuning {fmt_s(secs / n_done * (n_total - n_done))}")
    if results:
        cols = [(t, f"{int(p * 100)}%") for t in ft["tasks"] for p in ft["train_pcts"]]
        L.append("  " + f"{'home':8s}" + "".join(f"{t + ' ' + p:>18s}" for t, p in cols))
        for home in held:
            r = results.get(home, {}).get("results", {})
            cells = []
            for t, p in cols:
                a, b = r.get(f"{t}|{p}|DomusFM"), r.get(f"{t}|{p}|w/o Pretrain")
                if a and b:
                    cells.append(f"{a['mean']:.2f}/{b['mean']:.2f} {a['mean'] - b['mean']:+.2f}")
                elif a:
                    cells.append(f"{a['mean']:.2f}/ ...")
                else:
                    cells.append("...")
            L.append("  " + f"{home:8s}" + "".join(f"{c:>18s}" for c in cells))
        L.append("  cells: pretrained / no pretraining  difference (positive = pretraining helps)")
        now, before = _deltas(results), last_run_deltas()
        if now:
            L.append("  mean difference so far: " + "   ".join(
                f"{k} {v:+.3f}" + (f" (last run {before[k]:+.3f})" if k in before else "") for k, v in now.items()))
    if info["last_line"] and not info["done"]:
        L += ["", f"last log line: {info['last_line']}"]
    finished = info["done"] or (not running and log.exists())
    return "\n".join(L), finished


def watch(cfg, overlay, every: float, once: bool):
    if os.name == "nt":
        os.system("")  # enable ANSI escape codes in the classic Windows console
    try:
        while True:
            text, finished = render(cfg, overlay)
            if once:
                print(text)
                return
            sys.stdout.write("\033[2J\033[H" + text + f"\n\nrefresh every {every:.0f}s - Ctrl+C to leave "
                             "(training keeps running)\n")
            sys.stdout.flush()
            if finished:
                return
            time.sleep(every)
    except KeyboardInterrupt:
        print("\nLeft the watcher. Training keeps running; `watch` re-attaches, `stop` stops it.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["start", "watch", "stop"])
    ap.add_argument("--overlay", default=DEFAULT_OVERLAY)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--every", type=float, default=10)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.overlay, args.set)
    if args.command == "stop":
        return stop(cfg)
    if args.command == "start":
        start(args, cfg)
        time.sleep(2)
    watch(cfg, args.overlay, args.every, args.once)


if __name__ == "__main__":
    main()
