"""
Synthetic data generator for the Retail Insights project.

Creates one month (October 2026) of apparel-store sales in the exact format
of DATA_CONTRACT.md, with realistic patterns:
  - weekend spikes
  - pre-festival boost in the last week
  - a few "star" products, many average ones, and some dead stock
  - a mid-month discount campaign on selected products
  - end-of-month stock snapshot (hot items low, dead items high)

Run from the project root:
    python src/generate_data.py
"""

import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # fixed seed = same data every run

# ---------------------------------------------------------------- settings
START = "2026-10-01"
END = "2026-10-31"
N_PRODUCTS = 120
BASE_BILLS_PER_DAY = 70
OUTPUT_PATH = "data/synthetic/sales_oct_2026.csv"

# category -> (product types, MRP range, cost as share of MRP, sizes)
CATEGORIES = {
    "Men's Topwear": (["Round Neck T-Shirt", "Polo T-Shirt", "Casual Shirt", "Formal Shirt", "Hoodie"],
                      (499, 1999), 0.45, ["S", "M", "L", "XL"]),
    "Men's Bottomwear": (["Slim Fit Jeans", "Chinos", "Cargo Pants", "Joggers", "Shorts"],
                         (799, 2999), 0.48, ["30", "32", "34", "36"]),
    "Women's Topwear": (["Printed Top", "Kurti", "Crop Top", "Casual Shirt", "Sweatshirt"],
                        (499, 2499), 0.42, ["S", "M", "L", "XL"]),
    "Women's Bottomwear": (["Skinny Jeans", "Palazzo", "Leggings", "Straight Pants", "Skirt"],
                           (599, 2499), 0.44, ["S", "M", "L", "XL"]),
    "Kids Wear": (["Kids T-Shirt", "Kids Jeans", "Kids Dress", "Kids Shorts", "Kids Hoodie"],
                  (299, 1299), 0.40, ["2-3Y", "4-5Y", "6-7Y", "8-9Y"]),
    "Footwear": (["Sneakers", "Sandals", "Formal Shoes", "Slides", "Running Shoes"],
                 (799, 3999), 0.50, ["6", "7", "8", "9", "10"]),
    "Accessories": (["Belt", "Wallet", "Cap", "Backpack", "Sunglasses"],
                    (299, 1999), 0.38, ["Free Size"]),
}
BRANDS = ["Urban Edge", "Levi's", "Roadster", "H&W Basics", "Trendy Co", "FlexFit", "Classic Line", "Vibe"]
PAYMENTS = ["UPI", "Card", "Cash"]
PAYMENT_PROBS = [0.5, 0.3, 0.2]


# ---------------------------------------------------------------- products
def build_products():
    rows = []
    cat_names = list(CATEGORIES.keys())
    for i in range(N_PRODUCTS):
        cat = cat_names[i % len(cat_names)]
        types, (lo, hi), cost_ratio, sizes = CATEGORIES[cat]
        ptype = types[rng.integers(len(types))]
        brand = BRANDS[rng.integers(len(BRANDS))]
        mrp = int(round(rng.uniform(lo, hi) / 50) * 50) - 1  # prices like 1349, 799
        rows.append({
            "product_id": f"SKU-{i + 1:04d}",
            "product_name": f"{brand} {ptype}",
            "category": cat,
            "brand": brand,
            "mrp": mrp,
            "cost_price": int(mrp * cost_ratio),
            "sizes": sizes,
        })
    products = pd.DataFrame(rows)

    # Popularity: skewed, so a few products dominate sales (like real stores)
    popularity = rng.lognormal(mean=0, sigma=1.0, size=N_PRODUCTS)
    dead = rng.choice(N_PRODUCTS, size=12, replace=False)  # dead stock
    popularity[dead] = 0.0
    products["popularity"] = popularity
    products["is_dead"] = products.index.isin(dead)

    # Discount campaign: 20 random live products, mid-month
    live = products.index[~products["is_dead"]].to_numpy()
    campaign = rng.choice(live, size=20, replace=False)
    products["on_campaign"] = products.index.isin(campaign)
    products["campaign_discount"] = np.where(products["on_campaign"],
                                             rng.choice([20, 30, 40], size=N_PRODUCTS), 0)
    return products


# ---------------------------------------------------------------- sales
def day_multiplier(day):
    mult = 1.0
    if day.weekday() >= 5:      # Saturday, Sunday
        mult *= 1.5
    if day.day >= 24:           # pre-festival shopping
        mult *= 1.4
    return mult


def generate_sales(products):
    days = pd.date_range(START, END)
    records = []
    invoice_no = 10000

    for day in days:
        n_bills = rng.poisson(BASE_BILLS_PER_DAY * day_multiplier(day))
        campaign_active = 15 <= day.day <= 25

        # today's selection weights (campaign products sell more while active)
        weights = products["popularity"].to_numpy().copy()
        if campaign_active:
            weights = np.where(products["on_campaign"], weights * 1.7, weights)
        probs = weights / weights.sum()

        for _ in range(n_bills):
            invoice_no += 1
            basket_size = min(1 + rng.poisson(0.8), 5)
            items = rng.choice(len(products), size=basket_size, replace=False, p=probs)
            pay = rng.choice(PAYMENTS, p=PAYMENT_PROBS)

            for idx in items:
                p = products.iloc[idx]
                if campaign_active and p["on_campaign"]:
                    disc = int(p["campaign_discount"])
                else:
                    disc = int(rng.choice([0, 0, 0, 10]))   # occasional small discount
                selling_price = round(p["mrp"] * (1 - disc / 100))
                qty = int(rng.choice([1, 1, 1, 2, 3], p=[0.55, 0.2, 0.1, 0.1, 0.05]))

                records.append({
                    "invoice_id": f"INV-{invoice_no}",
                    "date": day.date(),
                    "product_id": p["product_id"],
                    "product_name": p["product_name"],
                    "category": p["category"],
                    "brand": p["brand"],
                    "size": rng.choice(p["sizes"]),
                    "quantity": qty,
                    "mrp": p["mrp"],
                    "selling_price": selling_price,
                    "discount_pct": disc,
                    "cost_price": p["cost_price"],
                    "payment_type": pay,
                })
    return pd.DataFrame(records)


def add_stock(sales, products):
    """End-of-month stock snapshot: hot items nearly sold out, dead items pile up."""
    sold = sales.groupby("product_id")["quantity"].sum()
    stock = {}
    for _, p in products.iterrows():
        units = int(sold.get(p["product_id"], 0))
        if p["is_dead"]:
            stock[p["product_id"]] = int(rng.integers(25, 80))        # unsold pile
        else:
            stock[p["product_id"]] = int(units * rng.uniform(0.0, 1.2) + rng.integers(0, 10))
    sales["current_stock"] = sales["product_id"].map(stock)
    return sales


# ---------------------------------------------------------------- main
def main():
    products = build_products()
    sales = generate_sales(products)
    sales = add_stock(sales, products)

    # Dead products never appear in a sales file, so also save a product master
    # (every SKU with its stock, including the ones that sold nothing).
    master = products[["product_id", "product_name", "category", "brand", "mrp", "cost_price"]].copy()
    sold_stock = sales.groupby("product_id")["current_stock"].first()
    master["current_stock"] = master["product_id"].map(sold_stock)
    dead_ids = products.loc[products["is_dead"], "product_id"]
    master.loc[master["product_id"].isin(dead_ids), "current_stock"] = rng.integers(25, 80, size=len(dead_ids))

    # Column order from the data contract
    cols = ["invoice_id", "date", "product_id", "product_name", "category", "brand", "size",
            "quantity", "mrp", "selling_price", "discount_pct", "cost_price",
            "current_stock", "payment_type"]
    sales = sales[cols]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    sales.to_csv(OUTPUT_PATH, index=False)
    master.to_csv("data/synthetic/product_master.csv", index=False)

    revenue = (sales["quantity"] * sales["selling_price"]).sum()
    print(f"Saved {len(sales):,} rows to {OUTPUT_PATH}")
    print(f"Bills: {sales['invoice_id'].nunique():,} | Products sold: {sales['product_id'].nunique()}"
          f" of {N_PRODUCTS} | Revenue: Rs {revenue:,.0f}")


if __name__ == "__main__":
    main()