"""src/verify.py — LLM 结果的确定性核验层（Part A 的第三道关）

三道关，逐层收紧：
  ① API 约束解码        保证字段名/类型/枚举合法        —— 不可能违规
  ② PromoItem 校验器    保证算术自洽                    —— 失败则重试（extract.py）
  ③ 本文件              保证和原始数据、brands.csv 一致 —— 失败则纠正或转人工

对每个字段的三种处置：
  corrections  代码能确定答案 → 直接改，不重试（重试花钱且不一定更对）
  issues       代码也定不了 → 交人工（README 的 issues 字段）
  沉默         核验器自己拿不准 → 信 LLM

设计原则：只在有把握时开口。误报会把正确的 LLM 结果改坏，比不核验更糟，
所以每个解析器拿不准时都返回 None。

调用点（只有 verify_item 一个入口）：
  extract.py:50   抽取时同步核验 —— 生产路径
  extract.py:110  verify_records()，对已有结果离线补做 —— eval 路径
"""
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

project_root_dir = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- 价格 / 规格

def normalize_price(raw) -> float | None:
    """'599' -> 5.99（分），'1.99' -> 1.99（元），'' -> None。

    判据：feed 里所有无小数点的纯数字都是分（389/429/479/599/649/699），
    其余全部带小数点。见 data/promo_feed.csv 的 reg_price 列。
    """
    text = str(raw).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+", text):          # 无小数点 -> 以分为单位
        return round(int(text) / 100, 2)
    try:
        return round(float(text), 2)
    except ValueError:
        return None


SIZE_RE = re.compile(r"""
    (?:(?P<pack>\d+)\s*[xX]\s*)?            # 12X355ML 的 12
    (?P<value>\d+(?:\.\d+)?)\s*
    (?P<unit>KG|G|ML|L|CT)\b
""", re.VERBOSE | re.IGNORECASE)

UNIT_SCALE = {"G": (1, "g"), "KG": (1000, "g"), "ML": (1, "mL"), "L": (1000, "mL"), "CT": (1, "ct")}


def parse_size(description: str) -> dict | None:
    """从 description 里解析规格。拿不准就返回 None（保持沉默）。"""
    match = SIZE_RE.search(description or "")
    if not match:
        return None
    scale, unit = UNIT_SCALE[match["unit"].upper()]
    value = float(match["value"]) * scale
    return {
        "size_value": round(value, 2) if value % 1 else int(value),
        "size_unit": unit,
        "pack_count": int(match["pack"]) if match["pack"] else 1,
    }


# ---------------------------------------------------------------- promo 解析

def parse_promo(promo: str, reg_price: float | None) -> dict | None:
    """确定性解析 promo，返回 deal_type / deal_qty / deal_price / save_amount。

    覆盖 feed 里全部 7 个语法家族。解析不出来返回 None —— 这时就只能信 LLM。
    """
    text = (promo or "").strip().upper()
    if not text:
        return None

    out: dict | None = None

    if m := re.search(r"SAVE\s*\$?\s*(\d+(?:\.\d+)?)", text):            # SAVE $4 / SAVE 2.00
        save = float(m[1])
        out = {"deal_type": "save_amount", "deal_qty": 1,
               "deal_price": round(reg_price - save, 2) if reg_price else None}

    elif m := re.search(r"ACHETEZ\s*(\d+)\s*[ÉE]CONOMISEZ\s*\$?(\d+(?:\.\d+)?)", text):   # 买 N 省 $X
        qty, save = int(m[1]), float(m[2])
        out = {"deal_type": "save_amount", "deal_qty": qty,
               "deal_price": round(reg_price * qty - save, 2) if reg_price else None}

    elif re.search(r"\bBOGO\b|ACHETEZ\s*1\s*OBTENEZ\s*1", text):         # 买一赠一
        out = {"deal_type": "bogo", "deal_qty": 2, "deal_price": reg_price}

    elif m := re.search(r"(\d+(?:\.\d+)?)\s*%", text):                   # 30% OFF / 25%
        pct = float(m[1])
        out = {"deal_type": "percent_off", "deal_qty": 1,
               "deal_price": round(reg_price * (1 - pct / 100), 2) if reg_price else None}

    elif m := re.search(r"\$?(\d+(?:\.\d+)?)\s*/\s*LB\b", text):         # $1.99/LB
        # per_weight 的 deal_price 是每磅价，且不做 save_amount 计算
        return {"deal_type": "per_weight", "deal_qty": 1,
                "deal_price": float(m[1]), "save_amount": None}

    elif m := re.match(r"(\d+)\s*(?:/|\s+FOR\s+)\s*\$?(\d+(?:\.\d+)?)\s*\$?", text):   # 2/$3, 2/7$, 2 FOR 5.00
        out = {"deal_type": "multi_buy", "deal_qty": int(m[1]), "deal_price": float(m[2])}

    elif m := re.fullmatch(r"\$?(\d+(?:\.\d+)?)\s*(?:EA|CH)?", text):    # $5.99 / 9.99 / 3.49 EA / 5.49 CH
        out = {"deal_type": "sale_price", "deal_qty": 1, "deal_price": float(m[1])}

    if out is None:
        logger.debug("promo 无法确定性解析，交给 LLM: %r", promo)
        return None

    if reg_price is not None and out["deal_price"] is not None:
        out["save_amount"] = round(reg_price * out["deal_qty"] - out["deal_price"], 2)
    else:
        out["save_amount"] = None
    return out


# ---------------------------------------------------------------- brand 接地

def load_brand_index() -> dict[str, str]:
    """别名（小写）-> 官方拼写。iögo 的 IOGO/IÖGO 之类都在这里收敛。"""
    brands = pd.read_csv(project_root_dir / "data" / "brands.csv", dtype=str, keep_default_na=False)
    index = {}
    for _, row in brands.iterrows():
        official = row["brand"].strip()
        index[official.casefold()] = official
        for alias in row["aliases"].split("|"):
            if alias.strip():
                index[alias.strip().casefold()] = official
    logger.debug("brands.csv: %s 个品牌，%s 条别名", len(brands), len(index))
    return index


def ground_brand(brand: str | None, index: dict[str, str]) -> tuple[str | None, str | None]:
    """返回 (接地后的品牌, issue)。不在参考表里的保留原值并打 flag（README 第 63 行）。"""
    if brand is None:
        return None, None
    official = index.get(brand.strip().casefold())
    if official is None:
        return brand, f"brand_not_in_reference:{brand}"
    return official, None


# ---------------------------------------------------------------- 主入口

# 这些字段代码算得比 LLM 准，不一致时直接以代码为准
DERIVED_FIELDS = ("deal_type", "deal_qty", "deal_price", "save_amount",
                  "size_value", "size_unit", "pack_count")


def verify_item(parsed: dict, row, brand_index: dict[str, str]) -> tuple[dict, list[str], list[str]]:
    """核验并修正一条 LLM 结果。返回 (最终记录字段, issues, corrections)。

    issues      需要人看的（README 的 issues 字段语义）
    corrections 代码确定性地改掉了 LLM 的值——是质量指标，不是人工项

    parsed 是 PromoItem.model_dump() 的结果；row 是原始 feed 行。
    """
    item = dict(parsed)
    issues: list[str] = []
    corrections: list[str] = []
    sku = row["sku"]

    # 1) reg_price 根本不该问 LLM——feed 里本来就有这一列
    feed_reg = normalize_price(row.get("reg_price"))
    if feed_reg != item.get("reg_price"):
        logger.info("sku=%s reg_price 以 feed 为准: LLM=%r -> %r", sku, item.get("reg_price"), feed_reg)
        corrections.append("reg_price")
        item["reg_price"] = feed_reg

    # 2) brand 接地：别名 -> 官方拼写算一次纠正，查不到则保留并转人工
    before = item.get("brand")
    item["brand"], brand_issue = ground_brand(before, brand_index)
    if brand_issue:
        logger.warning("sku=%s %s（保留但需人工确认）", sku, brand_issue)
        issues.append(brand_issue)
    elif item["brand"] != before:
        logger.info("sku=%s brand 接地: %r -> %r", sku, before, item["brand"])
        corrections.append("brand")

    # 3) promo / 规格：能确定性推导的就以代码为准
    truth = parse_promo(row.get("promo"), item["reg_price"]) or {}
    if size := parse_size(row.get("description")):
        truth.update(size)
    if not truth:
        issues.append("not_verified")          # 核验器全程沉默 -> 这行只有 LLM 的说法
        logger.warning("sku=%s 无法确定性核验，完全依赖 LLM", sku)

    for field in DERIVED_FIELDS:
        if field not in truth:
            continue
        if item.get(field) != truth[field]:
            logger.info("sku=%s %s 以代码为准: LLM=%r -> %r", sku, field, item.get(field), truth[field])
            corrections.append(field)
            item[field] = truth[field]

    # 4) reg_price 缺失本身就值得人看一眼
    if item["reg_price"] is None and item.get("deal_type") != "per_weight":
        issues.append("missing_reg_price")

    return item, issues, corrections
