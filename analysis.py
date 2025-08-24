
"""
BlinkIT Grocery Data — Python Analysis
--------------------------------------
This script mirrors common Power BI dashboard analyses using pure Python.

How to run:
    python analysis.py --data "DATASET.xlsx" --outdir "figures"

Outputs:
- Cleaned dataset (CSV): ./clean_blinkit.csv
- Figures: ./figures/*.png

Rules followed:
- Matplotlib only (no seaborn)
- One chart per figure (no subplots)
- No explicit colors/styles set
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        default = "DATASET.xlsx"
        if os.path.exists(default):
            path = default
        else:
            raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_excel(path, sheet_name="BlinkIT Grocery Data")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Item Fat Content" in df.columns:
        df["Item Fat Content"] = (
            df["Item Fat Content"]
            .replace({"LF":"Low Fat","low fat":"Low Fat","reg":"Regular","low Fat":"Low Fat"})
            .fillna("Unknown")
        )
    if "Item Weight" in df.columns:
        df["Item Weight"] = df.groupby("Item Type")["Item Weight"].transform(
            lambda s: s.fillna(s.median())
        )
        df["Item Weight"] = df["Item Weight"].fillna(df["Item Weight"].median())
    numeric_cols = ["Item Visibility","Item Weight","Sales","Rating"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Sales" in df.columns:
        df = df[df["Sales"].notna() & (df["Sales"] >= 0)]
    return df

def kpi_summary(df: pd.DataFrame) -> pd.DataFrame:
    kpis = {
        "total_sales": float(df["Sales"].sum() if "Sales" in df.columns else np.nan),
        "avg_sale": float(df["Sales"].mean() if "Sales" in df.columns else np.nan),
        "median_sale": float(df["Sales"].median() if "Sales" in df.columns else np.nan),
        "items": int(df.shape[0]),
        "unique_products": int(df["Item Identifier"].nunique() if "Item Identifier" in df.columns else 0),
        "unique_outlets": int(df["Outlet Identifier"].nunique() if "Outlet Identifier" in df.columns else 0),
        "avg_rating": float(df["Rating"].mean() if "Rating" in df.columns else np.nan),
    }
    return pd.DataFrame([kpis])

def plot_save(fig, outdir: str, filename: str):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path

def chart_sales_by_item_type(df, outdir):
    g = df.groupby("Item Type", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    fig = plt.figure()
    plt.bar(g["Item Type"], g["Sales"])
    plt.title("Total Sales by Item Type")
    plt.xlabel("Item Type"); plt.ylabel("Total Sales")
    plt.xticks(rotation=45, ha="right")
    return plot_save(fig, outdir, "sales_by_item_type.png")

def chart_sales_by_outlet_type(df, outdir):
    g = df.groupby("Outlet Type", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    fig = plt.figure()
    plt.bar(g["Outlet Type"], g["Sales"])
    plt.title("Total Sales by Outlet Type")
    plt.xlabel("Outlet Type"); plt.ylabel("Total Sales")
    plt.xticks(rotation=45, ha="right")
    return plot_save(fig, outdir, "sales_by_outlet_type.png")

def chart_sales_by_outlet_size(df, outdir):
    order = ["Small","Medium","High"]
    g = df.groupby("Outlet Size", as_index=False)["Sales"].sum()
    g["order"] = g["Outlet Size"].apply(lambda x: order.index(x) if x in order else 99)
    g = g.sort_values("order")
    fig = plt.figure()
    plt.bar(g["Outlet Size"], g["Sales"])
    plt.title("Total Sales by Outlet Size")
    plt.xlabel("Outlet Size"); plt.ylabel("Total Sales")
    return plot_save(fig, outdir, "sales_by_outlet_size.png")

def chart_avg_sales_by_fat(df, outdir):
    g = df.groupby("Item Fat Content", as_index=False)["Sales"].mean().sort_values("Sales", ascending=False)
    fig = plt.figure()
    plt.bar(g["Item Fat Content"], g["Sales"])
    plt.title("Average Sales by Fat Content")
    plt.xlabel("Item Fat Content"); plt.ylabel("Average Sales")
    plt.xticks(rotation=45, ha="right")
    return plot_save(fig, outdir, "avg_sales_by_fat.png")

def chart_establishment_year_sales(df, outdir):
    if "Outlet Establishment Year" not in df.columns: return None
    g = df.groupby("Outlet Establishment Year", as_index=False)["Sales"].sum().sort_values("Outlet Establishment Year")
    fig = plt.figure()
    plt.plot(g["Outlet Establishment Year"], g["Sales"], marker="o")
    plt.title("Sales by Outlet Establishment Year")
    plt.xlabel("Outlet Establishment Year"); plt.ylabel("Total Sales")
    return plot_save(fig, outdir, "sales_by_establishment_year.png")

def chart_visibility_vs_sales(df, outdir):
    if "Item Visibility" not in df.columns: return None
    fig = plt.figure()
    plt.scatter(df["Item Visibility"], df["Sales"], s=8, alpha=0.6)
    plt.title("Item Visibility vs Sales")
    plt.xlabel("Item Visibility"); plt.ylabel("Sales")
    return plot_save(fig, outdir, "visibility_vs_sales.png")

def chart_top_products(df, outdir, n=10):
    g = df.groupby("Item Identifier", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False).head(n)
    fig = plt.figure()
    plt.bar(g["Item Identifier"], g["Sales"])
    plt.title(f"Top {n} Products by Sales")
    plt.xlabel("Item Identifier"); plt.ylabel("Total Sales")
    plt.xticks(rotation=45, ha="right")
    return plot_save(fig, outdir, "top_products.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="DATASET.xlsx")
    parser.add_argument("--outdir", type=str, default="figures")
    args = parser.parse_args()

    df = load_data(args.data)
    df = clean_data(df)

    df.to_csv("clean_blinkit.csv", index=False)
    summary = kpi_summary(df)
    summary.to_csv("kpi_summary.csv", index=False)
    print("KPI Summary:")
    print(summary.to_string(index=False))

    chart_paths = []
    chart_paths.append(chart_sales_by_item_type(df, args.outdir))
    chart_paths.append(chart_sales_by_outlet_type(df, args.outdir))
    chart_paths.append(chart_sales_by_outlet_size(df, args.outdir))
    chart_paths.append(chart_avg_sales_by_fat(df, args.outdir))
    p = chart_establishment_year_sales(df, args.outdir)
    if p: chart_paths.append(p)
    p = chart_visibility_vs_sales(df, args.outdir)
    if p: chart_paths.append(p)
    chart_paths.append(chart_top_products(df, args.outdir))

    print("\nSaved charts:")
    for p in chart_paths:
        print("-", p)

if __name__ == "__main__":
    main()
