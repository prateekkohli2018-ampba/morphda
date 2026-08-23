#!/usr/bin/env python3
"""
Quota monitor and experiment watchdog for MORPH-DA.

Checks API availability every N minutes and reports which experiments
need re-queuing.

Usage:
    python scripts/monitor_quota.py [--interval 5]

Configuration:
    Set ANTHROPIC_API_KEY (or MORPH_DA_API_KEY) environment variable.
    Models to check are configured in configs/models.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_model_status(api_key: str, base_url: str, ca: str | None) -> dict[str, str]:
    """Ping each model and return status."""
    import requests
    models = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-5"]
    status = {}
    for model in models:
        try:
            r = requests.post(
                base_url,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "Hi"}]},
                verify=ca, timeout=10,
            )
            if r.status_code == 200:
                status[model] = f"OK → {r.json().get('model', '?')}"
            else:
                msg = r.json().get("error", {}).get("message", "")[:50]
                status[model] = f"{r.status_code}: {msg}"
        except Exception as e:
            status[model] = f"ERROR: {str(e)[:40]}"
    return status


def experiment_status() -> dict[str, dict]:
    """Summarize all agent run files."""
    results = {}
    agent_dir = Path("runs/natural_agents")
    if not agent_dir.exists():
        return results

    for model_dir in sorted(agent_dir.iterdir()):
        prog_file = model_dir / "programs.jsonl"
        if not prog_file.exists():
            continue
        rows = [json.loads(l) for l in open(prog_file) if l.strip().startswith("{")]
        seen = set(); deduped = []
        for r in rows:
            k = (r.get("task_id"), r.get("data_seed"))
            if k not in seen: seen.add(k); deduped.append(r)

        n = len(deduped)
        target = 303  # 101 tasks × 3 seeds
        corr = sum(1 for r in deduped if r.get("gold_correct"))
        exe  = sum(1 for r in deduped if r.get("execution_success"))
        wrong = sum(1 for r in deduped if r.get("execution_success") and not r.get("gold_correct"))
        results[model_dir.name] = {
            "n": n, "target": target, "pct": n / target,
            "accuracy": corr / n if n else 0,
            "esr": exe / n if n else 0,
            "wer": wrong / exe if exe else 0,
            "complete": n >= target,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=5,
                        help="Check interval in minutes (0 = one-shot)")
    args = parser.parse_args()

    api_key  = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("MORPH_DA_API_KEY", "")
    base_url = os.environ.get("MORPH_DA_API_URL", "https://api.anthropic.com/v1/messages")
    ca       = os.environ.get("REQUESTS_CA_BUNDLE")

    if not api_key:
        print("Warning: no API key set. Set ANTHROPIC_API_KEY to enable model status checks.")

    while True:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*60}")
        print(f"MORPH-DA Monitor — {now}")
        print(f"{'='*60}")

        if api_key:
            print("\n📡 Model status:")
            statuses = check_model_status(api_key, base_url, ca)
            for model, status in statuses.items():
                icon = "✓" if "OK" in status else "✗"
                print(f"  {icon} {model:<22} {status}")

        print("\n📊 Experiment progress:")
        exp = experiment_status()
        for model, info in exp.items():
            bar  = "█" * int(info["pct"] * 20) + "░" * (20 - int(info["pct"] * 20))
            done = "✓ DONE" if info["complete"] else f"  {info['n']}/{info['target']}"
            print(f"  {model:<28} [{bar}] {done}")
            if info["n"] > 0:
                print(f"    acc={info['accuracy']:.0%} esr={info['esr']:.0%} wer={info['wer']:.0%}")

        incomplete = [(m, i) for m, i in exp.items() if not i["complete"] and i["n"] > 0]
        if incomplete:
            print("\n⚠️  Incomplete runs:")
            for model, info in incomplete:
                model_id = model.replace("_", "-")
                print(f"  → python scripts/run_natural_agents.py --model {model_id} --seeds 42 7 123")

        if args.interval == 0:
            break
        print(f"\n⏱  Next check in {args.interval} min. Ctrl+C to stop.")
        time.sleep(args.interval * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
