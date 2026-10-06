import csv
import os
import json
from pathlib import Path
from llm import llm

def load_styles(styles_dir):
    styles = {}
    for filename in os.listdir(styles_dir):
        if filename.endswith("_style.md"):
            client_name = filename.replace("_style.md", "")
            with open(os.path.join(styles_dir, filename), 'r', encoding='utf-8') as f:
                styles[client_name] = f.read()
    return styles

def generate_promotions(csv_path, styles_dir, output_path):
    styles = load_styles(styles_dir)
    results = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sku = row['sku']
            client = row['client']
            style_guide = styles.get(client, "No style guide found.")
            
            # Each promotion item needs 3 examples
            item_results = {
                "sku": sku,
                "client": client,
                "examples": []
            }

            for i in range(1, 4):
                # Generate Simplified Copy
                simplified_prompt = f"""
                Generate a SIMPLIFIED promotional copy for {sku}.
                Original Price: {row['orig_price']}, Promo Price: {row['promo_price']}
                Valid: {row['promo_start']} to {row['promo_end']}
                Category: {row['category']}
                
                Style Guidelines:
                {style_guide}
                """
                simplified_copy = llm.chat(simplified_prompt)

                # Generate Complicated Copy
                complicated_prompt = f"""
                Generate a COMPLICATED promotional copy for {sku}.
                Original Price: {row['orig_price']}, Promo Price: {row['promo_price']}
                Valid: {row['promo_start']} to {row['promo_end']}
                Category: {row['category']}
                
                Style Guidelines:
                {style_guide}
                """
                complicated_copy = llm.chat(complicated_prompt)

                item_results["examples"].append({
                    "example_id": i,
                    "simplified": simplified_copy,
                    "complicated": complicated_copy
                })
            
            results.append(item_results)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print(f"Successfully generated copy for {len(results)} items.")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    CSV_FILE = BASE_DIR / "data" / "promotions.csv"
    STYLES_DIR = BASE_DIR / "styles"
    OUTPUT_FILE = BASE_DIR / "data" / "generated_copies.json"

    generate_promotions(CSV_FILE, STYLES_DIR, OUTPUT_FILE)
