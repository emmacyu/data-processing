# src/extract.py
import logging
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from llm import call_claude
from llm import build_system_prompt
from schemas import PromoItem
from verify import load_brand_index, verify_item

logger = logging.getLogger(__name__)

project_root_dir = Path(__file__).resolve().parent.parent


def row_to_text(row) -> str:
    return (f"description: {row['description']}\n"
            f"promo: {row['promo']}\n"
            f"reg_price: {row['reg_price'] or 'MISSING'}")


def extract_item(row, system_prompt, max_attempts=3, brand_index=None, verify=True):
    sku = row["sku"]
    row_text = row_to_text(row)
    user_prompt = row_text
    issues = []

    logger.debug("sku=%s 开始抽取 | %s", sku, row_text.replace("\n", " | "))
    for attempt in range(1, max_attempts + 1):
        try:
            result = call_claude(user_prompt=user_prompt, system_prompt=system_prompt, output_format=PromoItem)
        except ValidationError as e:
            # schema 保证不了的业务规则（save_amount 一致性、size 成对出现）在这里被挡住
            msg = e.errors()[0]["msg"]
            issues.append(msg)
            logger.warning("sku=%s 第 %s/%s 次校验失败: %s", sku, attempt, max_attempts, msg)
            user_prompt = (f"{row_text}\n\nYour previous answer was rejected:\n{e}\n"
                           "Fix the problem and answer again.")
        else:
            if attempt > 1:
                logger.info("sku=%s 第 %s 次重试后通过", sku, attempt)
            # 第三层核验：拿确定性代码复核 LLM 说的每一个可推导字段。
            # verify=False 时原样返回 LLM 的说法——用来单独测量 prompt 的效果。
            parsed = result.parsed.model_dump()
            if verify:
                if brand_index is None:
                    brand_index = load_brand_index()
                fields, verify_issues, corrections = verify_item(parsed, row, brand_index)
            else:
                fields, verify_issues, corrections = parsed, [], []
            return {
                "sku": sku,
                "description": row["description"],      # judge 和排错都要看原文
                "promo": row["promo"],
                # status 只表示「有没有产出可用记录」；要不要人看由 issues 决定。
                # 两者混在一起会让 evaluate.py 把修好的行判成全错。
                "status": "ok",
                "attempts": attempt,
                "issues": issues + verify_issues,
                "corrections": corrections,
                "model": result.model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_s": result.latency_s,
                **fields,
            }

    logger.error("sku=%s 重试 %s 次仍未通过校验，转人工: %s", sku, max_attempts, issues)
    return {"sku": sku, "status": "needs_review", "attempts": max_attempts, "issues": issues}


def extract_items(df, system_prompt, max_attempts=3, verify=True):
    logger.info("开始抽取 %s 行（max_attempts=%s）", len(df), max_attempts)
    brand_index = load_brand_index() if verify else None      # 只读一次 brands.csv
    results = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        logger.debug("进度 %s/%s", i, len(df))
        results.append(extract_item(row, system_prompt, max_attempts, brand_index, verify))

    ok = sum(r["status"] == "ok" for r in results)
    retried = sum(r["attempts"] > 1 for r in results)
    tokens_in = sum(r.get("input_tokens", 0) for r in results)
    tokens_out = sum(r.get("output_tokens", 0) for r in results)
    flagged = sum(bool(r["issues"]) for r in results)
    corrected = sum(bool(r.get("corrections")) for r in results)
    logger.info("抽取完成: ok=%s/%s  重试过=%s  代码纠正过=%s  待人工=%s  tokens in/out=%s/%s",
                ok, len(results), retried, corrected, flagged, tokens_in, tokens_out)
    return results


LLM_FIELDS = tuple(PromoItem.model_fields)      # LLM 负责产出的字段


def verify_records(records, df, brand_index=None):
    """对已抽取的结果补做核验——不重新调用 API。

    这样一次 API 开销就能同时得到「LLM 裸分」和「加核验后的分」两个数，
    否则对比变体时每个变体要跑两遍，成本翻倍。
    """
    if brand_index is None:
        brand_index = load_brand_index()
    rows = {str(r["sku"]).strip(): r for _, r in df.iterrows()}

    out = []
    for rec in records:
        if rec["status"] != "ok":
            out.append(dict(rec))
            continue
        parsed = {k: rec[k] for k in LLM_FIELDS if k in rec}
        fields, issues, corrections = verify_item(parsed, rows[rec["sku"]], brand_index)
        out.append({**rec, **fields,
                    "issues": list(rec.get("issues", [])) + issues,
                    "corrections": corrections})

    corrected = sum(bool(r.get("corrections")) for r in out)
    logger.info("补做核验: %s/%s 行被代码纠正过", corrected, len(out))
    return out


if __name__ == "__main__":
    from logging_setup import setup_logging

    setup_logging()

    df = pd.read_csv(project_root_dir / "data" / "promo_feed_clean.csv", dtype=str, keep_default_na=False)
    system_prompt = build_system_prompt(n_examples=2)

    items = extract_items(df.head(5), system_prompt)      # 在这里决定处理哪些行
    result_df = pd.DataFrame(items)
    print(result_df[["sku", "status", "brand", "deal_type", "deal_price", "reg_price", "save_amount"]])