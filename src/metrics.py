"""
Core metrics for the Retail Insights project.

Each function takes a pandas DataFrame and returns a result, so the
Streamlit dashboard (app.py) can reuse them later.

Test it from the project root:
    python src/metrics.py
"""

import pandas as pd


# ---------------------------------------------------------------- loading
def load_sales(path):
    """Read a sales CSV and make sure the date column is a real date."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_columns(df):
    """Add revenue, cost, profit and discount amount to every row."""
    df = df.copy()
    df["revenue"] = df["quantity"] * df["selling_price"]
    df["discount_amount"] = df["quantity"] * (df["mrp"] - df["selling_price"])

    # cost_price is optional in the data contract, so handle its absence
    if "cost_price" in df.columns:
        df["cost"] = df["quantity"] * df["cost_price"]
        df["profit"] = df["revenue"] - df["cost"]
    return df


# ---------------------------------------------------------------- KPIs
def kpis(df):
    """Headline numbers for the month."""
    total_revenue = df["revenue"].sum()
    bills = df["invoice_id"].nunique()
    total_mrp_value = (df["quantity"] * df["mrp"]).sum()

    result = {
        "total_revenue": total_revenue,
        "total_units": int(df["quantity"].sum()),
        "total_bills": bills,
        "avg_bill_value": total_revenue / bills,
        "avg_discount_pct": df["discount_amount"].sum() / total_mrp_value * 100,
    }
    if "profit" in df.columns:
        result["total_profit"] = df["profit"].sum()
        result["profit_margin_pct"] = df["profit"].sum() / total_revenue * 100
    return result


# ---------------------------------------------------------------- products
def product_performance(df):
    """One row per product: units, revenue, profit, stock, sell-through."""
    agg = {
        "units_sold": ("quantity", "sum"),
        "revenue": ("revenue", "sum"),
        "avg_discount_pct": ("discount_pct", "mean"),
        "current_stock": ("current_stock", "last"),
    }
    if "profit" in df.columns:
        agg["profit"] = ("profit", "sum")

    perf = (df.groupby(["product_id", "product_name", "category"])
              .agg(**agg)
              .reset_index()
              .sort_values("revenue", ascending=False)
              .reset_index(drop=True))

    # Sell-through = share of available stock that was sold
    perf["sell_through_pct"] = (perf["units_sold"] /
                                (perf["units_sold"] + perf["current_stock"]) * 100).round(1)
    return perf


def top_products(perf, n=10, by="revenue"):
    """Best sellers by revenue, units_sold or profit."""
    return perf.sort_values(by, ascending=False).head(n)


def slow_movers(perf, n=10):
    """Products that sold, but the least (dead stock is handled separately)."""
    return perf.sort_values("units_sold", ascending=True).head(n)


def dead_stock(sales_df, master_df):
    """Products in the catalogue with zero sales this month, but stock on hand."""
    sold_ids = set(sales_df["product_id"])
    dead = master_df[~master_df["product_id"].isin(sold_ids)].copy()
    dead = dead[dead["current_stock"] > 0]
    dead["stock_value_at_cost"] = dead["current_stock"] * dead["cost_price"]
    return dead.sort_values("stock_value_at_cost", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- ABC analysis
def abc_analysis(perf):
    """
    Classify products by their share of revenue:
      A = the products making up the first 80% of revenue
      B = the next 15%
      C = the last 5%
    """
    abc = perf.sort_values("revenue", ascending=False).reset_index(drop=True).copy()
    total = abc["revenue"].sum()
    abc["revenue_share_pct"] = abc["revenue"] / total * 100
    abc["cumulative_pct"] = abc["revenue_share_pct"].cumsum()

    # Use the cumulative share BEFORE this product, so the product that
    # crosses the 80% line is still counted as A.
    before = abc["cumulative_pct"] - abc["revenue_share_pct"]
    abc["abc_class"] = "C"
    abc.loc[before < 95, "abc_class"] = "B"
    abc.loc[before < 80, "abc_class"] = "A"
    return abc


# ---------------------------------------------------------------- summaries
def category_summary(df):
    """Revenue, units, profit and share per category."""
    agg = {
        "revenue": ("revenue", "sum"),
        "units_sold": ("quantity", "sum"),
    }
    if "profit" in df.columns:
        agg["profit"] = ("profit", "sum")

    cat = df.groupby("category").agg(**agg).reset_index()
    cat["revenue_share_pct"] = (cat["revenue"] / cat["revenue"].sum() * 100).round(1)
    return cat.sort_values("revenue", ascending=False).reset_index(drop=True)


def daily_trend(df):
    """Revenue, units and bills per day."""
    return (df.groupby("date")
              .agg(revenue=("revenue", "sum"),
                   units_sold=("quantity", "sum"),
                   bills=("invoice_id", "nunique"))
              .reset_index())


# ---------------------------------------------------------------- quick test
if __name__ == "__main__":
    sales = add_columns(load_sales("data/synthetic/sales_oct_2026.csv"))
    master = pd.read_csv("data/synthetic/product_master.csv")

    print("=== KPIs ===")
    for name, value in kpis(sales).items():
        print(f"{name}: {value:,.2f}")

    perf = product_performance(sales)

    print("\n=== Top 10 products by revenue ===")
    print(top_products(perf)[["product_name", "units_sold", "revenue", "current_stock"]].to_string(index=False))

    print("\n=== 10 slowest movers ===")
    print(slow_movers(perf)[["product_name", "units_sold", "revenue", "current_stock"]].to_string(index=False))

    print("\n=== Dead stock (top 5 by stock value) ===")
    print(dead_stock(sales, master).head(5)[["product_name", "current_stock", "stock_value_at_cost"]].to_string(index=False))

    abc = abc_analysis(perf)
    print("\n=== ABC analysis ===")
    print(abc.groupby("abc_class").agg(products=("product_id", "count"),
                                       revenue_share=("revenue_share_pct", "sum")).round(1))

    print("\n=== Category summary ===")
    print(category_summary(sales).to_string(index=False))