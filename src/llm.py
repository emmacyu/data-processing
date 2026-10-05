"""src/llm.py
- wrapper for Anthropic API: failover between models, timing, token usage
- optional structured output via a pydantic model
"""
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel
from schemas import PromoItem

logger = logging.getLogger(__name__)

project_root_dir = Path(__file__).resolve().parent.parent
load_dotenv(project_root_dir / ".env")

MODEL_LIST = [m for m in [os.getenv("PRIMARY_MODEL"), os.getenv("FALLBACK_MODEL")] if m]
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1000))

client = anthropic.Anthropic(max_retries=3, timeout=60)


@dataclass
class LLMResult:
    text: str
    parsed: Any            # 传了 output_format 时是 pydantic 对象，否则是 None
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float


FEW_SHOT_EXAMPLES = [
    ("description: QUAKER CHEWY BARS 156G\npromo: 3/$10\nreg_price: 3.99",
     PromoItem(brand="Quaker", product_name="Chewy Bars", size_value=156, size_unit="g", pack_count=1,
               deal_type="multi_buy", deal_qty=3, deal_price=10.00, reg_price=3.99, save_amount=1.97)),
    ("description: COKE 6X222ML\npromo: BOGO\nreg_price: 4.49",
     PromoItem(brand="Coca-Cola", product_name="Cola", size_value=222, size_unit="mL", pack_count=6,
               deal_type="bogo", deal_qty=2, deal_price=4.49, reg_price=4.49, save_amount=4.49)),
    ("description: FRESH ATLANTIC SALMON FILLETS\npromo: $9.99/LB\nreg_price: MISSING",
     PromoItem(brand=None, product_name="Atlantic Salmon Fillets", size_value=None, size_unit=None, pack_count=1,
               deal_type="per_weight", deal_qty=1, deal_price=9.99, reg_price=None, save_amount=None)),
    ("description: NATREL LAIT 1% 4L\npromo: 20%\nreg_price: 7.49",
     PromoItem(brand="Natrel", product_name="1% Milk", size_value=4000, size_unit="mL", pack_count=1,
               deal_type="percent_off", deal_qty=1, deal_price=5.99, reg_price=7.49, save_amount=1.50)),
]

RULES = """You normalize rows of a Canadian grocery promo feed into structured fields.

Rules:
- Convert sizes to base units: kg -> g, L -> mL. "4X106G" means pack_count 4, size 106 g.
- deal_price is the TOTAL paid for deal_qty units: "3/$10" -> deal_qty 3, deal_price 10.00.
- BOGO -> deal_qty 2, deal_price = one regular price.
- For percent_off and "SAVE $X", compute deal_price from reg_price, rounded to the cent.
- save_amount = reg_price * deal_qty - deal_price. Use null if reg_price is missing.
- Never invent a price. If a value is unknown, use null.
- Fresh produce and meat are often unbranded; a place name (e.g. ONTARIO) is not a brand.
- Descriptions may be in French: write product_name in English."""


def call_claude(user_prompt: str, system_prompt: str | None = None,
                output_format: type[BaseModel] | None = None,
                models: list[str] | None = None) -> LLMResult:
    """models 留空则用 .env 的 PRIMARY/FALLBACK；judge 需要换成别的模型时显式传入。"""
    kwargs = {"system": system_prompt} if system_prompt else {}
    last_error = None

    for model in models or MODEL_LIST:
        start = time.perf_counter()
        logger.debug("calling %s (max_tokens=%s, output_format=%s)",
                     model, MAX_TOKENS, output_format.__name__ if output_format else None)
        try:
            response = client.messages.parse(
                model=model, max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": user_prompt}],
                output_format=output_format, **kwargs)
            result = LLMResult(
                # 不能假设 content[0] 是文本：开了思考的模型（如 sonnet-5 默认自适应思考）
                # 第一个 block 是 ThinkingBlock，取 [0].text 会 AttributeError
                text=next((b.text for b in response.content if b.type == "text"), ""),
                parsed=response.parsed_output,
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_s=round(time.perf_counter() - start, 2),
            )
        except anthropic.APIConnectionError as e:          # 网络问题、超时
            logger.warning("%s connection error (%s), trying next model", model, e)
            last_error = e
        except anthropic.RateLimitError as e:              # 429
            retry_after = e.response.headers.get("retry-after", "?")
            logger.warning("%s rate limited (retry-after=%ss), trying next model", model, retry_after)
            last_error = e
        except anthropic.APIStatusError as e:
            if e.status_code < 500:                        # 4xx：请求或账户问题，换模型也没用
                logger.error("%s returned %s: %s — 不重试，4xx 换模型也没用",
                             model, e.status_code, e.message)
                raise
            logger.warning("%s server error %s, trying next model", model, e.status_code)
            last_error = e
        else:
            # 成本和延迟的唯一可审计来源：model 取自 response，不是取自配置
            logger.info("ok  %s  in=%s out=%s  %.2fs",
                        result.model, result.input_tokens, result.output_tokens, result.latency_s)
            if response.stop_reason == "max_tokens":
                logger.warning("响应被 max_tokens=%s 截断，结果可能不完整", MAX_TOKENS)
            return result

    tried = models or MODEL_LIST
    logger.error("all models failed: %s", tried)
    raise RuntimeError(f"All models in {tried} failed") from last_error


def build_system_prompt(n_examples=1):
    if n_examples==0:
        return RULES

    examples = "\n\n".join(
        f"Input:\n{row}\nOutput:\n{item.model_dump_json()}"
        for row, item in FEW_SHOT_EXAMPLES[:n_examples]
    )
    return f"{RULES}\n\nExamples:\n\n{examples}"


if __name__ == "__main__":
    from logging_setup import setup_logging

    setup_logging()

    result = call_claude(
        "description: KD MAC&CHS ORIGINAL 225G | promo: 2/$3 | reg_price: 1.99",
        system_prompt=build_system_prompt(n_examples=1),
        output_format=PromoItem,
    )
    print(result.parsed.model_dump())
    print(result.model, result.input_tokens, result.output_tokens, result.latency_s)