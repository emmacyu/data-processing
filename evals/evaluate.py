"""src/evaluate.py
- run the extractor on the labelled dev set and report:
  field accuracy, row accuracy, needs_review rate, cost, latency
- every run is appended to out/experiments.jsonl
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
project_root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root_dir / "src"))

from extract import extract_items
from llm import build_system_prompt

FIELDS = ["brand", "size_value", "size_unit", "pack_count", "deal_type",
          "deal_qty", "deal_price", "reg_price", "save_amount"]

# 当前模型的单价（美元 / 百万 token），按官网价格填写
PRICE_IN_PER_MTOK = float(os.getenv("PRICE_IN_PER_MTOK", 1.0))
PRICE_OUT_PER_MTOK = float(os.getenv("PRICE_OUT_PER_MTOK", 5.0))


def same(pred, gold) -> bool:
    """数字允许 0.005 的误差，其他值必须完全相同。"""
    if isinstance(pred, (int, float)) and isinstance(gold, (int, float)):
        return abs(pred - gold) < 0.005
    return pred == gold


def score_accuracy(records: list[dict], gold: list[dict]) -> dict:
    by_sku = {r["sku"]: r for r in records}
    errors = []          # 给人看的错例
    wrong = {f: 0 for f in FIELDS}
    wrong_skus = set()
    for g in gold:
        pred = by_sku.get(g["sku"], {})
        if pred.get("status") != "ok":
            # 没有提取结果：所有字段都算错，但错例里只列一行
            errors.append({"sku": g["sku"], "field": "*", "pred": pred.get("status", "missing"), "gold": "-"})
            wrong_skus.add(g["sku"])
            for f in FIELDS:
                wrong[f] += 1
            continue
        for f in FIELDS:
            if not same(pred.get(f), g[f]):
                errors.append({"sku": g["sku"], "field": f, "pred": pred.get(f), "gold": g[f]})
                wrong_skus.add(g["sku"])
                wrong[f] += 1

    n = len(gold)
    return {
        "n": n,
        "field_accuracy": {f: round(1 - wrong[f] / n, 3) for f in FIELDS},
        "row_accuracy": round(1 - len(wrong_skus) / n, 3),
        "errors": errors,
    }


def score_operations(records: list[dict]) -> dict:
    tokens_in = sum(r.get("input_tokens", 0) for r in records)
    tokens_out = sum(r.get("output_tokens", 0) for r in records)
    latencies = [r["latency_s"] for r in records if "latency_s" in r]   # needs_review 的记录没有耗时
    needs_review = sum(r["status"] != "ok" for r in records)
    cost = (tokens_in * PRICE_IN_PER_MTOK + tokens_out * PRICE_OUT_PER_MTOK) / 1e6
    return {
        "needs_review": needs_review,
        "needs_review_rate": round(needs_review / len(records), 3) if records else 0,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(cost, 4),
        "cost_per_item_usd": round(cost / len(records), 5) if records else 0,
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "max_latency_s": max(latencies) if latencies else None,
    }


def log_run(entry: dict) -> None:
    out_dir = project_root_dir / "out"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "experiments.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_eval(n_examples: int, df: pd.DataFrame, gold: list[dict]) -> dict:
    records = extract_items(df, build_system_prompt(n_examples=n_examples))
    acc = score_accuracy(records, gold)
    ops = score_operations(records)

    print(f"\n===== n_examples={n_examples}  model={os.getenv('PRIMARY_MODEL')} =====")
    print(f"row accuracy:    {acc['row_accuracy']}")
    print(f"field accuracy:  {acc['field_accuracy']}")
    print(f"needs_review:    {ops['needs_review']}/{len(records)}")
    print(f"tokens in/out:   {ops['tokens_in']}/{ops['tokens_out']}   cost: ${ops['cost_usd']}")
    print(f"latency avg/max: {ops['avg_latency_s']}s / {ops['max_latency_s']}s")
    for e in acc["errors"]:
        print(f"  x {e['sku']}  {e['field']:<12} pred={e['pred']!r:<22} gold={e['gold']!r}")

    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": os.getenv("PRIMARY_MODEL"),
             "n_examples": n_examples, "row_accuracy": acc["row_accuracy"],
             "field_accuracy": acc["field_accuracy"], **ops}
    log_run(entry)
    return entry


if __name__ == "__main__":
    gold = json.loads((project_root_dir / "data" / "gold_labels.json").read_text())
    df = pd.read_csv(project_root_dir / "data" / "promo_feed_clean.csv", dtype=str, keep_default_na=False)
    dev = df[df["sku"].isin([g["sku"] for g in gold])]          # 只跑有标准答案的商品
    dev = dev.drop_duplicates(subset="sku", keep="first")
    # 用法：python src/evaluate.py 0 2 4   → 依次比较 0、2、4 个例子
    variants = [int(x) for x in sys.argv[1:]] or [2]
    results = run_eval(1, dev, gold)

    if len(results) > 1:
        cols = ["n_examples", "row_accuracy", "needs_review", "tokens_in", "cost_usd", "avg_latency_s"]
        print("\n", pd.DataFrame(results)[cols].to_string(index=False))
