import pandas as pd
from anthropic import Anthropic
from typing import Literal, Optional
from pydantic import BaseModel, Field

# 1. 定义数据模型
DealType = Literal["multi_buy", "sale_price", "save_amount", "bogo", "percent_off", "per_weight"]

class PromoItem(BaseModel):
    """Structured fields extracted from one promo feed row."""
    brand: Optional[str] = Field(description="Official brand name, e.g. 'Kraft Dinner' for KD. null if unbranded")
    product_name: str = Field(description="Clean English product name, without brand or size")
    size_value: Optional[float] = Field(description="Size converted to base units: kg -> g, L -> mL. null if no size")
    size_unit: Optional[Literal["g", "mL", "ct"]]
    pack_count: int = Field(ge=1, description="12 for '12X355ML', otherwise 1")
    deal_type: DealType
    deal_qty: int = Field(ge=1, description="Number of units the deal price applies to")
    deal_price: Optional[float] = Field(description="Total customer pays for deal_qty units")
    reg_price: Optional[float] = Field(description="Regular price of ONE unit in dollars")
    save_amount: Optional[float] = Field(description="reg_price * deal_qty - deal_price, rounded to cent")


class ScoreReason(BaseModel):
    score: int
    reason: str


class JudgeEvaluation(BaseModel):
    tone_match: ScoreReason
    language_correct: ScoreReason
    formatting_ok: ScoreReason
    completeness: ScoreReason
    overall_pass: bool


# For mvp case study
class CopyVariant(BaseModel):
    simplified: str = Field(description="精简版宣传文案")
    complicated: str = Field(description="详细/故事化宣传文案")

class GenerationBatch(BaseModel):
    example_1: CopyVariant
    example_2: CopyVariant
    example_3: CopyVariant

class JudgeEvaluation(BaseModel):
    clarity_score: int = Field(description="清晰度评分 (1-5)")
    brand_alignment_score: int = Field(description="品牌契合度评分 (1-5)")
    overall_pass: bool = Field(description="综合评估是否通过")
    reasoning: str = Field(description="评审意见与理由")