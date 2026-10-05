from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

DealType = Literal["multi_buy", "sale_price", "save_amount", "bogo", "percent_off", "per_weight"]


class PromoItem(BaseModel):
    """Structured fields extracted from one promo feed row."""

    brand: Optional[str] = Field(description="Official brand name, e.g. 'Kraft Dinner' for KD. null if unbranded (e.g. fresh produce)")
    product_name: str = Field(description="Clean English product name, without brand or size")
    size_value: Optional[float] = Field(description="Size converted to base units: kg -> g, L -> mL. null if no size")
    size_unit: Optional[Literal["g", "mL", "ct"]]
    pack_count: int = Field(ge=1, description="12 for '12X355ML', otherwise 1")
    deal_type: DealType = Field(description=(
        "multi_buy: 'N for $X' like 2/$5. sale_price: a single price like $3.99. "
        "save_amount: 'SAVE $X'. bogo: buy one get one free. percent_off: 'X% OFF'. per_weight: '$X/LB'."))
    deal_qty: int = Field(ge=1, description="Number of units the deal price applies to. 2 for '2/$5' and for BOGO")
    deal_price: Optional[float] = Field(description="Total the customer pays for deal_qty units. null if unknown")
    reg_price: Optional[float] = Field(description="Regular price of ONE unit in dollars. null if missing")
    save_amount: Optional[float] = Field(description="reg_price * deal_qty - deal_price, rounded to the cent. null if unknown")

    @model_validator(mode="after")
    def check_consistency(self):
        if (self.size_value is None) != (self.size_unit is None):
            raise ValueError("size_value and size_unit must both be set or both be null")
        if self.deal_type != "per_weight" and None not in (self.reg_price, self.deal_price, self.save_amount):
            expected = round(self.reg_price * self.deal_qty - self.deal_price, 2)
            if abs(expected - self.save_amount) > 0.01:
                raise ValueError(f"save_amount {self.save_amount} != reg_price*deal_qty - deal_price = {expected}")
        return self