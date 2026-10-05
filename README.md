# Mock Technical Test: Promo Copy Pipeline

**Time limit: 60 minutes** (plus 5 minutes to read this page before you start the timer).
You are not expected to finish everything. We care more about how you prioritise, structure
the code and reason about quality than about completeness. Commit to a working end-to-end
slice first, then improve it.

Allowed: Python docs, library docs, Anthropic API docs. Not allowed: asking an AI assistant
to write the solution for you (you *will* call Claude from your code; that's the point).

---

## Context

We produce weekly promotional flyers for grocery retailers. Each retailer sends a **promo feed**
(one row per advertised item). Our platform turns each row into structured data, then
generates short **bilingual (English / French) flyer copy** that follows each retailer's style rules. Two retailers are onboarded for this exercise: `northmart` and `boreal`.

The feed comes from several vendors and is not clean. Nobody has written down all of its
conventions — part of the job is to work them out from the data.

## Files

| Path | What it is |
|---|---|
| `data/promo_feed.csv` | This week's promo feed (28 rows) |
| `data/brands.csv` | Curated brand reference: official spelling, known aliases, French name |
| `data/gold_labels.json` | Hand-labelled correct parse for 8 rows (dev set) |
| `config/clients.yaml` | Style rules for each retailer |
| `starter/` | Optional skeleton: a thin Claude client wrapper and function stubs |

Set `ANTHROPIC_API_KEY` in your environment. Default model: `claude-sonnet-5`
(override with `CLAUDE_MODEL`).

---

## Tasks

### Part A — Normalise the feed (≈ 20 min)

Turn every feed row into a record with (at least) these fields:

| Field | Meaning |
|---|---|
| `sku` | string |
| `brand` | official brand spelling from `brands.csv`, or `null` if the item has no brand |
| `product_name` | clean, human-readable English product name without brand or size (e.g. `Macaroni & Cheese Original`) |
| `size_value`, `size_unit` | normalised to base units: `g`, `mL` or `ct` |
| `pack_count` | e.g. `12` for `12X355ML`, otherwise `1` |
| `deal_type` | one of `multi_buy`, `sale_price`, `save_amount`, `bogo`, `percent_off`, `per_weight` |
| `deal_qty` | how many units the deal price applies to |
| `deal_price` | what the customer pays for `deal_qty` units (for `per_weight`: price per lb) |
| `reg_price` | regular price per unit in dollars, or `null` |
| `save_amount` | `reg_price * deal_qty - deal_price`, rounded to the cent, or `null` |
| `issues` | list of data-quality flags for anything a human should look at |

Rules:
- Use deterministic code wherever it is reliable. Use Claude only where it adds value
  (for example cleaning product names, or brands that are not in the reference). Be ready to
  justify each choice.
- Any LLM call that returns data must return **structured output** that you validate.
- Brands must be **grounded** in `brands.csv`. If Claude suggests a brand that is not in the
  reference, keep it but flag it.

Output: `out/normalized.json`

### Part B — Generate flyer copy (≈ 20 min)

For each normalised item and for a chosen retailer (`--client northmart` or `--client boreal`),
produce one entry per language listed in that retailer's config:

```json
{"sku": "100231", "lang": "fr", "headline": "...", "price_line": "2 pour 3 $"}
```

- `headline`: short marketing headline written by Claude, following the retailer's `tone`,
  `headline.case`, `headline.max_chars`, `brand_casing` and `forbidden_words`.
- `price_line`: the price text as it will be printed, in the correct format for the language.
- Retailer rules must come from `config/clients.yaml`, not be hard-coded. Adding a third
  retailer should require no code change.
- Use the French brand name from `brands.csv` in French copy.

Output: `out/copy_<client>.json`

### Part C — Validate and evaluate (≈ 15 min)

1. Write **automatic checks** for the generated copy. Every row that fails a check must say
   which rule it broke. Decide which failures should trigger a retry and which need a human.
2. Write an **evaluation script** that scores your Part A output against
   `data/gold_labels.json`, field by field. Use it to compare at least two variants
   (e.g. prompt with vs. without few-shot examples, or two models). Report the numbers.
   Note: we hold back a larger labelled set and will run your pipeline against it.

### Part D — Notes (≈ 5 min)

In `NOTES.md`, briefly write:
1. The feed conventions you inferred from the data.
2. Any **conflicts between retailer rules**, or between rules and the data, and the trade-off you
   would propose to the business stakeholder.
3. What you would do next with more time, and how you'd run this in production.

---

## What we look at

- Correctness on real, messy data (including the held-out set)
- Division of labour between code and LLM, and grounding
- Structured outputs, validation, retries
- Measured evaluation rather than eyeballing
- Configuration-driven design, readability, small sensible abstractions
- Clear written reasoning about ambiguous rules
