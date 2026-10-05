"""Minimal tests for the deterministic parts of the pipeline. No real API calls.
Run: python -m pytest -q tests
"""
import os
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

from src import extract                          # noqa: E402
from src.schemas import PromoItem           # noqa: E402
from src.preprocess import to_date            # noqa: E402


GOOD = {"brand": "Kraft Dinner", "product_name": "Mac & Cheese", "size_value": 225, "size_unit": "g",
        "pack_count": 1, "deal_type": "multi_buy", "deal_qty": 2, "deal_price": 3.0,
        "reg_price": 1.99, "save_amount": 0.98}
BAD = {**GOOD, "save_amount": 1.00}     # wrong arithmetic


@pytest.mark.parametrize("text, expected", [
    ("2026-10-01", date(2026, 10, 1)),
    ("10/01/2026", date(2026, 10, 1)),   # V202: month/day
    ("01/10/2026", date(2026, 10, 1)),   # V303: day/month
    ("1 oct. 2026", date(2026, 10, 1)),
])

def test_to_date(text, expected):
    assert to_date(text) == expected


def test_schema_rejects_wrong_savings():
    PromoItem.model_validate(GOOD)
    with pytest.raises(ValidationError):
        PromoItem.model_validate(BAD)


def test_extract_retries_then_succeeds(monkeypatch):
    answers = [BAD, GOOD]

    def fake_call_claude(user_prompt, system_prompt=None, output_format=None):
        parsed = output_format.model_validate(answers.pop(0))   # raises on BAD, like the real parse
        return SimpleNamespace(parsed=parsed, model="fake", input_tokens=0, output_tokens=0, latency_s=0)

    monkeypatch.setattr(extract, "call_claude", fake_call_claude)
    row = pd.Series({"sku": "100231", "description": "KD MAC&CHS ORIGINAL 225G", "promo": "2/$3", "reg_price": "1.99"})
    record = extract.extract_item(row, "system")
    assert record["status"] == "ok" and record["attempts"] == 2


# ----------------------------------------------------------- 核验层（不打 API）
from src.verify import (normalize_price, parse_promo, parse_size,     # noqa: E402
                        load_brand_index, ground_brand, verify_item)


@pytest.mark.parametrize("raw, expected", [
    ("599", 5.99),        # 无小数点 -> 以分为单位（feed 里 V303 的写法）
    ("1.99", 1.99),
    ("", None),
    ("15.99", 15.99),
])
def test_normalize_price(raw, expected):
    assert normalize_price(raw) == expected


@pytest.mark.parametrize("promo, reg, expected", [
    ("2/$3",        1.99,  {"deal_type": "multi_buy",   "deal_qty": 2, "deal_price": 3.0,   "save_amount": 0.98}),
    ("3/$4 LIMIT 6", 1.79, {"deal_type": "multi_buy",   "deal_qty": 3, "deal_price": 4.0,   "save_amount": 1.37}),
    ("2 FOR 5.00",  3.49,  {"deal_type": "multi_buy",   "deal_qty": 2, "deal_price": 5.0,   "save_amount": 1.98}),
    ("SAVE $4",     15.99, {"deal_type": "save_amount", "deal_qty": 1, "deal_price": 11.99, "save_amount": 4.0}),
    ("BOGO",        4.79,  {"deal_type": "bogo",        "deal_qty": 2, "deal_price": 4.79,  "save_amount": 4.79}),
    ("30% OFF",     5.49,  {"deal_type": "percent_off", "deal_qty": 1, "deal_price": 3.84,  "save_amount": 1.65}),
    ("$1.99/LB",    None,  {"deal_type": "per_weight",  "deal_qty": 1, "deal_price": 1.99,  "save_amount": None}),
    ("ACHETEZ 1 OBTENEZ 1 GRATUIT", 4.79,
     {"deal_type": "bogo", "deal_qty": 2, "deal_price": 4.79, "save_amount": 4.79}),
])
def test_parse_promo(promo, reg, expected):
    assert parse_promo(promo, reg) == expected


def test_parse_promo_returns_none_when_unsure():
    """核验器拿不准时必须沉默，不能瞎猜——误报比漏报更糟。"""
    assert parse_promo("WHILE QUANTITIES LAST", 4.99) is None


@pytest.mark.parametrize("description, expected", [
    ("KD MAC&CHS ORIGINAL 225G", {"size_value": 225,  "size_unit": "g",  "pack_count": 1}),
    ("COKE 12X355ML",            {"size_value": 355,  "size_unit": "mL", "pack_count": 12}),
    ("TIDE PODS 42CT",           {"size_value": 42,   "size_unit": "ct", "pack_count": 1}),
    ("NATREL 2% LAIT 2L",        {"size_value": 2000, "size_unit": "mL", "pack_count": 1}),
    ("CHEERIOS CEREAL 0.43KG",   {"size_value": 430,  "size_unit": "g",  "pack_count": 1}),
    ("DOVE BODY WASH 354 ML",    {"size_value": 354,  "size_unit": "mL", "pack_count": 1}),
])
def test_parse_size(description, expected):
    assert parse_size(description) == expected


def test_ground_brand_maps_alias_to_official():
    index = load_brand_index()
    assert ground_brand("KD", index) == ("Kraft Dinner", None)
    assert ground_brand("IOGO", index) == ("iögo", None)      # 别名 -> 带变音符的官方拼写
    assert ground_brand(None, index) == (None, None)


def test_ground_brand_flags_unknown_but_keeps_it():
    """README：不在参考表里的品牌要保留并 flag，不能丢。"""
    brand, issue = ground_brand("Northshore Bakehouse", load_brand_index())
    assert brand == "Northshore Bakehouse"
    assert issue and issue.startswith("brand_not_in_reference")


def test_verify_item_overrides_wrong_llm_values():
    """LLM 把分当成元、算错 save_amount、漏了单位换算——三处都该被代码改回来。"""
    row = pd.Series({"sku": "100236", "description": "NATREL 2% LAIT 2L",
                     "promo": "2/$9", "reg_price": "599"})
    llm_said = {"brand": "NATREL", "product_name": "2% Milk", "size_value": 2, "size_unit": "mL",
                "pack_count": 1, "deal_type": "multi_buy", "deal_qty": 2, "deal_price": 9.0,
                "reg_price": 599.0, "save_amount": 1189.0}

    fields, issues, corrections = verify_item(llm_said, row, load_brand_index())

    assert fields["reg_price"] == 5.99          # 599 分 -> 5.99 元
    assert fields["save_amount"] == 2.98        # 5.99*2 - 9.00
    assert fields["size_value"] == 2000         # 2L -> 2000 mL
    assert fields["brand"] == "Natrel"          # 官方拼写（NATREL -> Natrel 也算一次纠正）
    assert set(corrections) == {"brand", "reg_price", "save_amount", "size_value"}
    assert issues == []                         # 全部能确定性裁决 -> 不需要人工


# ----------------------------------------------------- LLM-as-a-judge（打分逻辑，不打 API）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
from judge import ProductNameVerdict, passed, CRITERIA, score_product_names   # noqa: E402

ALL_OK = dict(same_product=True, brand_removed=True, size_removed=True, in_english=True, reason="ok")


def test_passed_requires_all_four_criteria():
    assert passed(ProductNameVerdict(**ALL_OK))
    for c in CRITERIA:
        verdict = ProductNameVerdict(**{**ALL_OK, c: False, "reason": f"{c} failed"})
        assert not passed(verdict), f"{c} 为 false 时不应通过"


def test_score_product_names_aggregates(monkeypatch):
    """两条记录：一条全过，一条 brand_removed 不过 -> 总分 0.5，该项通过率 0.5。"""
    verdicts = [
        ProductNameVerdict(**ALL_OK),
        ProductNameVerdict(**{**ALL_OK, "brand_removed": False, "reason": "brand still present"}),
    ]
    import judge as judge_mod
    monkeypatch.setattr(judge_mod, "judge_product_name",
                        lambda *a, **k: (verdicts.pop(0), 50, 20))

    records = [
        {"sku": "1", "status": "ok", "description": "KD MAC&CHS 225G", "brand": "Kraft Dinner",
         "product_name": "Macaroni & Cheese Original"},
        {"sku": "2", "status": "ok", "description": "TIDE PODS 42CT", "brand": "Tide",
         "product_name": "Tide Pods"},
    ]
    out = judge_mod.score_product_names(records)

    assert out["judged"] == 2
    assert out["product_name_accuracy"] == 0.5
    assert out["product_name_criteria"]["brand_removed"] == 0.5
    assert out["product_name_criteria"]["same_product"] == 1.0
    assert out["judge_tokens_in"] == 100 and out["judge_tokens_out"] == 40   # 计入成本
    assert [f["sku"] for f in out["product_name_failures"]] == ["2"]


def test_score_product_names_skips_failed_rows(monkeypatch):
    """抽取失败的行没有 product_name，不该送去 judge（白花钱且没意义）。"""
    import judge as judge_mod
    monkeypatch.setattr(judge_mod, "judge_product_name",
                        lambda *a, **k: pytest.fail("不该对失败行调用 judge"))
    out = judge_mod.score_product_names([{"sku": "1", "status": "needs_review", "issues": ["x"]}])
    assert out == {"judged": 0}


def test_verify_records_brand_grounding_counts_as_correction():
    """别名映射到官方拼写必须记进 corrections——否则实跑时这项纠正在指标里是隐形的。"""
    row = pd.Series({"sku": "100231", "description": "KD MAC&CHS ORIGINAL 225G",
                     "promo": "2/$3", "reg_price": "1.99"})
    llm_said = {**GOOD, "brand": "KD"}          # 别名，不是官方拼写

    fields, issues, corrections = verify_item(llm_said, row, load_brand_index())

    assert fields["brand"] == "Kraft Dinner"
    assert "brand" in corrections
    assert issues == []
