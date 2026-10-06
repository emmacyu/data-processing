import os
import re
import csv
import json
import pandas as pd
from pathlib import Path
from typing import Optional, List
import sys

sys.path.append(str(Path(__file__).resolve().parent))
# 导入结构化 Schema 与 LLM 模块
from src.schema import CopyVariant, GenerationBatch, JudgeEvaluation, PromoItem
from src.llm import llm


def load_style_guides(styles_dir: str = "styles") -> dict:
    """读取 styles/ 目录下所有客户的风格指南"""
    styles = {}
    if not os.path.exists(styles_dir):
        return styles
    for filename in os.listdir(styles_dir):
        if filename.endswith("_style.md"):
            client = filename.replace("_style.md", "")
            with open(os.path.join(styles_dir, filename), "r", encoding="utf-8") as f:
                styles[client] = f.read()
    return styles


def generate_copy_batch(row: dict, style_guide: str) -> GenerationBatch:
    """调用真实 LLM 生成包含 3 组 CopyVariant 的 GenerationBatch"""
    prompt = f"""
You are an expert copywriter. Generate 3 pairs of promotional copy variants (simplified & complicated) for the following product.

Product Details:
- Title: {row.get('title')}
- Description: {row.get('description')}
- Original Price: ${row.get('orig_price')}
- Promotional Price: ${row.get('promo_price')}
- Category: {row.get('category')}
- Max Characters Allowed: {row.get('max_copy_chars', 200)}

Brand Style Guide:
{style_guide}
"""
    return llm.structured_call(
        messages=[{"role": "user", "content": prompt}],
        output_format=GenerationBatch
    )


def rule_checks(copy_text: str, row: dict, client: str) -> dict:
    """第一层确定性规则校验"""
    result = {}
    max_chars = int(row.get('max_copy_chars', 200))
    
    # 1. 通用基础规则
    result['length_ok'] = len(copy_text) <= max_chars
    result['char_count'] = len(copy_text)
    result['price_mentioned'] = any(c in copy_text.lower() for c in ['$', 'price', 'save'])
    
    # 2. Emoji 检测
    emoji_pattern = r'[\U0001F300-\U0001FAFF]'
    has_emoji = bool(re.search(emoji_pattern, copy_text))
    
    # 3. 客户专属规则
    if client == "TechCorp":
        result['price_bolded'] = '**$' in copy_text       # 价格必须加粗
        result['no_emoji'] = not has_emoji               # 禁止使用 Emoji
        required_keys = ['length_ok', 'price_mentioned', 'price_bolded', 'no_emoji']
    elif client == "FashionCo":
        result['has_emoji'] = has_emoji                  # 必须使用 Emoji
        result['price_has_exclamation'] = re.search(r'\$[\d.]+\!', copy_text) is not None  # 价格后必须有感叹号
        required_keys = ['length_ok', 'price_mentioned', 'has_emoji', 'price_has_exclamation']
    else:
        required_keys = ['length_ok', 'price_mentioned']
        
    result['all_rules_pass'] = all(result.get(k, False) for k in required_keys)
    result['failed_rules'] = [k for k in required_keys if not result.get(k, False)]
    return result


def llm_judge(copy_text: str, row: dict, style_guide: str) -> JudgeEvaluation:
    """第二层真实 LLM-as-a-Judge 评估 (使用最新的 JudgeEvaluation Schema)"""
    prompt = f"""
You are a strict QA reviewer evaluating promotional copy.

Product: {row.get('title')}
Copy text to evaluate: {copy_text}
Max allowed length: {row.get('max_copy_chars', 200)} characters

Style Guide Constraints:
{style_guide}

Please evaluate clarity_score (1-5), brand_alignment_score (1-5), overall_pass (boolean), and provide detailed reasoning.
"""
    return llm.structured_call(
        messages=[{"role": "user", "content": prompt}],
        output_format=JudgeEvaluation
    )


def generate_html_flyer(results: list, output_path: str = "outputs/flyer_preview.html"):
    """导出格式化的 HTML 预览页面"""
    tile_html_template = """
    <div style="border:1px solid #e0e0e0;border-radius:8px;padding:14px;width:220px;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;box-shadow:0 2px 4px rgba(0,0,0,0.05);background:#fff">
      <span style="font-size:10px;font-weight:bold;background:#f0f0f0;color:#555;padding:2px 6px;border-radius:4px">{client}</span>
      <h4 style="margin:8px 0 4px;font-size:14px;color:#333">{title}</h4>
      <p style="font-size:12px;color:#666;margin:0 0 10px;line-height:1.4">{copy_text}</p>
      <div style="color:#d32f2f;font-weight:bold;font-size:16px">
        ${promo_price} <span style="color:#999;font-weight:normal;font-size:12px;text-decoration:line-through">${orig_price}</span>
      </div>
    </div>
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    valid_cards = [tile_html_template.format(**r) for r in results if r["status"] == "✅ PASS"]

    html_content = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Flyer Preview</title></head>
<body style="background:#f9f9f9;padding:20px">
  <h2 style="font-family:sans-serif;color:#333">Promotional Copy Flyer Preview</h2>
  <div style="display:flex;gap:16px;flex-wrap:wrap">
    {"".join(valid_cards)}
  </div>
</body>
</html>
"""
    path = Path(output_path)
    path.write_text(html_content, encoding="utf-8")
    print(f"\nHTML 预览文件已保存至: {path.resolve()}")


def main():
    # 1. 加载风格指南与数据
    styles = load_style_guides("styles")
    csv_path = "data/promotions.csv" if os.path.exists("data/promotions.csv") else "promotions.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: 找不到数据文件 {csv_path}")
        return

    df = pd.read_csv(csv_path)
    rows = df.to_dict(orient="records")
    results = []

    print(f"开始处理 {len(rows)} 条促销数据...\n")

    # 2. 执行生成与两层校验 Pipeline
    for i, row in enumerate(rows, start=1):
        client = row.get("client", "")
        style_guide = styles.get(client, "")
        
        # 尝试实例化 PromoItem 校验数据完整性
        try:
            promo_item = PromoItem(
                brand=client,
                product_name=str(row.get("title", "")),
                deal_type="sale_price",
                deal_qty=1,
                pack_count=1,
                deal_price=float(row.get("promo_price", 0.0)),
                reg_price=float(row.get("orig_price", 0.0))
            )
        except Exception:
            promo_item = None

        print(f"[{i}/{len(rows)}] 正在为 SKU: {row.get('sku')} ({client}) 生成文案...")
        batch: GenerationBatch = generate_copy_batch(row, style_guide)
        examples = [batch.example_1, batch.example_2, batch.example_3]

        for idx, ex in enumerate(examples, start=1):
            for copy_type, copy_text in [("simplified", ex.simplified), ("complicated", ex.complicated)]:
                # 第一层规则校验
                rule_res = rule_checks(copy_text, row, client)
                
                # 第二层 LLM Judge 评估
                judge_res: JudgeEvaluation = llm_judge(copy_text, row, style_guide)
                
                final_pass = rule_res["all_rules_pass"] and judge_res.overall_pass
                
                results.append({
                    "sku": row["sku"],
                    "client": client,
                    "example_num": idx,
                    "copy_type": copy_type,
                    "copy_text": copy_text,
                    "rule_pass": rule_res["all_rules_pass"],
                    "judge_pass": judge_res.overall_pass,
                    "clarity_score": judge_res.clarity_score,
                    "brand_alignment_score": judge_res.brand_alignment_score,
                    "status": "✅ PASS" if final_pass else "❌ FAIL",
                    "failed_rules": rule_res["failed_rules"],
                    "judge_reasoning": judge_res.reasoning,
                    "orig_price": row["orig_price"],
                    "promo_price": row["promo_price"],
                    "title": row["title"]
                })

    # 3. 输出统计结果
    results_df = pd.DataFrame(results)
    print("\n================ Pipeline 执行结果概览 ================")
    print(results_df[["sku", "client", "copy_type", "status", "clarity_score", "brand_alignment_score", "rule_pass", "judge_pass"]].to_string())

    # 4. 生成 HTML 预览
    generate_html_flyer(results)


if __name__ == "__main__":
    main()