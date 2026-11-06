"""
Recommendation engine for the Retail Insights project.

Turns the numbers from metrics.py into plain-English actions.
Every rule returns a list of recommendations, each one a dict with:
    priority (High / Medium / Low), type, item, message

Test it from the project root:
    python src/insights.py
"""

import pandas as pd

# Works both when run as a script (python src/insights.py)
# and when imported by the dashboard (from src.insights import ...)
try:
    from src.metrics import (load_sales, add_columns, product_performance,
                             abc_analysis, dead_stock, category_summary)
except ImportError:
    from metrics import (load_sales, add_columns, product_performance,
                         abc_analysis, dead_stock, category_summary)


# ---------------------------------------------------------------- RULE 1: reorder
def reorder_alerts(abc, days):
    """A and B products that will run out within a week at the current sales rate."""
    top = abc[abc["abc_class"].isin(["A", "B"])].copy()
    top["daily_rate"] = top["units_sold"] / days
    top["days_cover"] = top["current_stock"] / top["daily_rate"]
    low = top[top["days_cover"] < 7]

    recs = []
    for _, r in low.iterrows():
        recs.append({
            "priority": "High" if r["days_cover"] < 3 else "Medium",
            "type": "Reorder",
            "item": r["product_name"],
            "message": (f"{r['product_name']} ({r['product_id']}): only {r['current_stock']} units left, "
                        f"about {r['days_cover']:.1f} days of stock at the current sales rate. Reorder now."),
        })
    return recs


# ---------------------------------------------------------------- RULE 2: slow movers
def slow_mover_alerts(abc):
    """Products in the bottom 25% of units sold that still hold a lot of stock."""
    threshold = abc["units_sold"].quantile(0.25)
    slow = abc[(abc["units_sold"] <= threshold) &
               (abc["current_stock"] >= abc["units_sold"] * 2)]

    recs = []
    for _, r in slow.iterrows():
        recs.append({
            "priority": "Medium",
            "type": "Slow mover",
            "item": r["product_name"],
            "message": (f"{r['product_name']} ({r['product_id']}): sold only {r['units_sold']} units but "
                        f"{r['current_stock']} are in stock ({r['sell_through_pct']}% sell-through). "
                        f"Consider a discount, a bundle, or a better shelf position."),
        })
    return recs


# ---------------------------------------------------------------- RULE 3: dead stock
def dead_stock_alerts(dead):
    """Products with zero sales and stock on hand."""
    if dead.empty:
        return []
    big = dead["stock_value_at_cost"].quantile(0.75)

    recs = []
    for _, r in dead.iterrows():
        recs.append({
            "priority": "High" if r["stock_value_at_cost"] >= big else "Medium",
            "type": "Dead stock",
            "item": r["product_name"],
            "message": (f"{r['product_name']} ({r['product_id']}): zero sales this month, "
                        f"{r['current_stock']} units (about Rs {r['stock_value_at_cost']:,.0f} at cost) are stuck. "
                        f"Mark it down, bundle it, or return it to the vendor."),
        })
    return recs


# ---------------------------------------------------------------- RULE 4: category margin
def category_margin_alerts(cat):
    """Categories whose profit margin is well below the store average."""
    if "profit" not in cat.columns:
        return []
    overall = cat["profit"].sum() / cat["revenue"].sum() * 100

    recs = []
    for _, r in cat.iterrows():
        margin = r["profit"] / r["revenue"] * 100
        if margin < overall - 3:
            recs.append({
                "priority": "Medium",
                "type": "Low margin",
                "item": r["category"],
                "message": (f"{r['category']}: profit margin is {margin:.1f}% against a store average of "
                            f"{overall:.1f}%. Review pricing, discount depth, or sourcing cost."),
            })
    return recs


# ---------------------------------------------------------------- RULE 5: discount impact
def discount_impact(df, days):
    """
    For products that sold at 20%+ discount, compare units per day
    on discounted days vs the other days of the month.
    """
    disc_rows = df[df["discount_pct"] >= 20]
    rows = []
    for pid, group in disc_rows.groupby("product_id"):
        product_df = df[df["product_id"] == pid]
        disc_days = group["date"].nunique()
        disc_units = group["quantity"].sum()
        other_days = days - disc_days
        other_units = product_df["quantity"].sum() - disc_units

        disc_rate = disc_units / disc_days
        other_rate = other_units / other_days if other_days > 0 else float("nan")
        lift = disc_rate / other_rate if other_rate and other_rate > 0 else float("nan")

        other_rows = product_df[product_df["discount_pct"] < 20]
        profit_disc = group["profit"].sum() / disc_units if "profit" in group.columns else float("nan")
        profit_other = (other_rows["profit"].sum() / other_rows["quantity"].sum()
                        if "profit" in df.columns and not other_rows.empty else float("nan"))

        rows.append({
            "product_id": pid,
            "product_name": group["product_name"].iloc[0],
            "discount_pct": round(group["discount_pct"].mean()),
            "units_per_day_discounted": round(disc_rate, 2),
            "units_per_day_other": round(other_rate, 2) if other_days > 0 else float("nan"),
            "sales_lift": round(lift, 2) if lift == lift else float("nan"),
            "profit_per_unit_discounted": round(profit_disc, 0),
            "profit_per_unit_other": round(profit_other, 0) if profit_other == profit_other else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("sales_lift", ascending=False).reset_index(drop=True)


def discount_alerts(impact):
    """Flag discounts that did not lift sales enough to be worth the lost margin."""
    recs = []
    for _, r in impact.iterrows():
        if r["sales_lift"] != r["sales_lift"]:      # NaN check
            continue
        if r["sales_lift"] < 1.1:
            recs.append({
                "priority": "Medium",
                "type": "Weak discount",
                "item": r["product_name"],
                "message": (f"{r['product_name']} ({r['product_id']}): the {r['discount_pct']}% discount raised "
                            f"daily sales only {r['sales_lift']}x. Margin was given away for little gain, "
                            f"so avoid repeating this discount."),
            })
        elif r["sales_lift"] >= 1.5:
            recs.append({
                "priority": "Low",
                "type": "Strong discount",
                "item": r["product_name"],
                "message": (f"{r['product_name']} ({r['product_id']}): the {r['discount_pct']}% discount lifted "
                            f"daily sales {r['sales_lift']}x. Worth repeating, ideally during peak days."),
            })
    return recs


# ---------------------------------------------------------------- combine everything
def generate_recommendations(sales, master):
    """Run every rule and return one table, High priority first."""
    days = (sales["date"].max() - sales["date"].min()).days + 1

    perf = product_performance(sales)
    abc = abc_analysis(perf)
    dead = dead_stock(sales, master)
    cat = category_summary(sales)
    impact = discount_impact(sales, days)

    recs = []
    recs += reorder_alerts(abc, days)
    recs += slow_mover_alerts(abc)
    recs += dead_stock_alerts(dead)
    recs += category_margin_alerts(cat)
    recs += discount_alerts(impact)

    table = pd.DataFrame(recs, columns=["priority", "type", "item", "message"])
    order = {"High": 0, "Medium": 1, "Low": 2}
    table["_rank"] = table["priority"].map(order)
    table = table.sort_values(["_rank", "type"]).drop(columns="_rank").reset_index(drop=True)
    return table, impact


# ---------------------------------------------------------------- quick test
if __name__ == "__main__":
    sales = add_columns(load_sales("data/synthetic/sales_oct_2026.csv"))
    master = pd.read_csv("data/synthetic/product_master.csv")

    recs, impact = generate_recommendations(sales, master)

    print("=== Recommendations by type and priority ===")
    print(recs.groupby(["type", "priority"]).size().to_string())

    print("\n=== Top 8 High-priority actions ===")
    for msg in recs[recs["priority"] == "High"]["message"].head(8):
        print("-", msg)

    print("\n=== Discount impact (best 8 by sales lift) ===")
    print(impact.head(8).to_string(index=False))

    recs.to_csv("data/processed/recommendations.csv", index=False)
    print(f"\nSaved {len(recs)} recommendations to data/processed/recommendations.csv")