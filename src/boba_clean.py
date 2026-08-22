"""
Clean the raw Snackpass exports into tidy, analysis-ready tables.

Input : data/raw/daily_financials_2025_12.csv
        data/raw/item_performance_2025_12.csv
Output: data/processed/daily.csv
        data/processed/items.csv
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Rows that are accounting artifacts, not menu items.
NON_MENU_CATEGORIES = {"Gift Cards", "Custom Discounts and Surcharges"}

MONEY_COLS = [
    "Subtotal",
    "Gross Sales",
    "Refunds",
    "Discounts",
    "Net Sales",
    "Cash",
    "Gift Card Redemption",
    "Store Credit Redemption",
    "Taxes You Owe",
    "Tips",
    "Total Sales",
    "Processing Fees",
    "Snackpass Fees",
]


def to_float(series: pd.Series) -> pd.Series:
    """'$1,234.56' -> 1234.56 ; handles negatives like '$-21.00'."""
    return (
        series.astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .str.strip()
        .replace({"": None})
        .astype(float)
    )


def clean_daily() -> pd.DataFrame:
    df = pd.read_csv(RAW / "daily_financials_2025_12.csv")

    for col in MONEY_COLS:
        df[col] = to_float(df[col])

    df["Orders"] = df["Orders"].astype(int)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date").reset_index(drop=True)

    # Derived fields — these are what the analysis actually leans on.
    df["day_of_week"] = df["Date"].dt.day_name()
    df["is_weekend"] = df["Date"].dt.dayofweek >= 5
    df["avg_order_value"] = df["Net Sales"] / df["Orders"]
    df["tip_rate"] = df["Tips"] / df["Net Sales"]
    df["discount_rate"] = df["Discounts"] / df["Gross Sales"]
    df["cash_share"] = df["Cash"] / df["Total Sales"]
    df["net_sales_ma7"] = df["Net Sales"].rolling(7, min_periods=1).mean()

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


def clean_items() -> pd.DataFrame:
    # Row 0 of the export is a title banner, not a header.
    df = pd.read_csv(RAW / "item_performance_2025_12.csv", skiprows=1)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    df["Net Sales"] = to_float(df["Net Sales"])
    df["Count"] = df["Count"].astype(int)

    df = df[~df["Category"].isin(NON_MENU_CATEGORIES)].copy()

    # Snackpass tags menu names with merchandising labels. Keep them as flags
    # rather than throwing them away — whether an item is tagged *POPULAR* is
    # itself a variable worth testing.
    df["is_tagged_popular"] = df["Item"].str.contains(r"\*POPULAR\*", regex=True)
    df["is_tagged_new"] = df["Item"].str.contains(r"\*NEW\*", regex=True)
    df["item_clean"] = (
        df["Item"].str.replace(r"\*[A-Z ]+\*", "", regex=True).str.strip()
    )
    df["category_clean"] = (
        df["Category"].str.replace(r"\*[A-Za-z ]+\*", "", regex=True).str.strip()
    )

    # No unit-price column in the export, so back it out. This is an effective
    # realized price (post-discount, blended across sizes and toppings), NOT a
    # menu board price.
    df["avg_realized_price"] = df["Net Sales"] / df["Count"]

    df = df.sort_values("Net Sales", ascending=False).reset_index(drop=True)
    df["sales_rank"] = df.index + 1
    df["cum_sales_share"] = df["Net Sales"].cumsum() / df["Net Sales"].sum()

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    daily = clean_daily()
    items = clean_items()

    daily.to_csv(PROCESSED / "daily.csv", index=False)
    items.to_csv(PROCESSED / "items.csv", index=False)

    print(f"daily.csv  {daily.shape[0]:>4} rows x {daily.shape[1]} cols")
    print(f"items.csv  {items.shape[0]:>4} rows x {items.shape[1]} cols")

    # Reconciliation check: the two exports should roughly agree on the month's
    # net sales. They won't match exactly — item net sales exclude tips, taxes,
    # and some order-level adjustments.
    gap = items["net_sales"].sum() - daily["net_sales"].sum()
    print(
        f"\nreconciliation: items ${items['net_sales'].sum():,.2f} vs "
        f"daily ${daily['net_sales'].sum():,.2f} "
        f"(gap ${gap:,.2f}, {gap / daily['net_sales'].sum():.2%})"
    )


if __name__ == "__main__":
    main()

