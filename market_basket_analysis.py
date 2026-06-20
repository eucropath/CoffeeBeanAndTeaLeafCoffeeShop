"""
Market Basket Analysis (association rule mining)
==================================================

Improvements over a typical apyori-based starter script (e.g. the
Apriori_MBA reference notebook):

  * Reads .xlsx / .xls / .csv directly (no manual CSV pre-export needed).
  * Handles THREE common transaction layouts and auto-detects which one
    you have:
      1. "wide"   - one row per transaction, one column per item *slot*
                    (e.g. Drinks | Customizations | Foods | Addons),
                    with blanks/NaN where a slot wasn't used.
                    <-- this is the shape of the coffee shop dataset.
      2. "long"   - one row per (transaction_id, item) pair.
      3. "basket" - one row per transaction, items comma-separated with
                    no header (the classic apyori-tutorial format).
  * Cleans data: trims whitespace, drops blanks/NaN, de-duplicates items
    within a transaction (so "Oat Milk" listed twice doesn't inflate
    counts), and optionally folds variant spellings together.
  * Flags likely typos / near-duplicate item names automatically (e.g.
    "Swicth to Soy Milk" vs "Switch to Soy Milk") so you can decide
    whether to merge them via SYNONYM_MAP instead of silently guessing.
  * Uses mlxtend (actively maintained) instead of apyori (unmaintained,
    last released years ago, awkward frozenset API), with the SAME
    default thresholds the reference notebook uses (min_support=0.02,
    min_confidence=0.3, min_lift=1.0). Outputs a tidy DataFrame with
    support, confidence, lift, leverage, and conviction.
  * Auto-relaxes min_support if your thresholds are too strict for a
    small dataset and would otherwise silently return zero rules.
  * Saves results as CSV + JSON, plus an optional bar chart of the
    top rules by lift.
  * Driven by argparse so it's reusable on any similar dataset without
    editing the file.

Usage
-----
    python market_basket_analysis.py data.xlsx --layout wide \
        --columns Drinks Customizations Foods Addons

    python market_basket_analysis.py transactions.csv --layout basket

    python market_basket_analysis.py orders.xlsx --layout long \
        --id-column OrderID --item-column Item

If --layout is omitted the script tries to guess it from the file
structure.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                            "mlxtend", "--break-system-packages"])
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder


# --------------------------------------------------------------------------
# Manual fixes for known spelling variants. Add to this as you spot them
# (the typo detector below will suggest candidates). Keys are matched
# case-insensitively after whitespace-trimming; values are the canonical
# form that will appear in the output.
# --------------------------------------------------------------------------
SYNONYM_MAP: dict[str, str] = {
    "swicth to soy milk": "Switch to Soy Milk",
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

# Encodings to try, in order, for CSV files that aren't UTF-8. Real-world
# exports (e.g. from Excel) are very commonly Windows-1252 / Latin-1, which
# breaks on the first non-ASCII character (accents, curly quotes, etc.)
# if you assume UTF-8 outright.
CSV_ENCODING_FALLBACKS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]


def _read_csv_with_fallback_encoding(path: Path, **kwargs) -> pd.DataFrame:
    """pd.read_csv that retries with common fallback encodings on failure."""
    last_err: Exception | None = None
    for enc in CSV_ENCODING_FALLBACKS:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    raise last_err


def read_table(path: Path, sheet: str | int | None = 0) -> pd.DataFrame:
    """Read a csv/xlsx/xls file into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path, sheet_name=sheet)
    if suffix == ".csv":
        return _read_csv_with_fallback_encoding(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def guess_layout(df: pd.DataFrame) -> str:
    """Best-effort guess at which transaction layout a DataFrame uses."""
    cols_lower = [str(c).lower() for c in df.columns]
    if any("id" in c for c in cols_lower) and df.shape[1] <= 3:
        return "long"
    # Wide layout: real (named) column headers, with blanks/NaN in the data.
    # Note: don't check `df.columns.dtype == object` here -- on newer pandas
    # versions (e.g. pandas >= 2.x/3.x with the string dtype backend),
    # string column labels report dtype 'str', not 'object', so that check
    # silently fails and every wide-layout file gets misclassified as
    # 'basket'. A RangeIndex (0, 1, 2, ...) is the real signal of "no
    # real headers"; anything else means the file has named columns.
    has_named_columns = not isinstance(df.columns, pd.RangeIndex)
    if has_named_columns and df.isna().any().any():
        return "wide"
    # A wide-format file where every row happens to be fully filled (no
    # NaNs anywhere) would otherwise be misclassified as "basket". Treat
    # any file with real, named, non-RangeIndex column headers as "wide"
    # by default -- "basket" is reserved for genuinely headerless files.
    if has_named_columns:
        return "wide"
    return "basket"


def transactions_from_wide(df: pd.DataFrame, columns: list[str] | None = None) -> list[list[str]]:
    """
    One row per transaction, one column per item slot
    (e.g. Drinks | Customizations | Foods | Addons).
    Blank/NaN cells just mean that slot wasn't used.
    """
    use_cols = columns or list(df.columns)
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in file: {missing}")

    transactions = []
    for _, row in df[use_cols].iterrows():
        items = [str(v) for v in row.tolist() if pd.notna(v) and str(v).strip()]
        transactions.append(items)
    return transactions


def transactions_from_long(df: pd.DataFrame, id_column: str, item_column: str) -> list[list[str]]:
    """One row per (transaction_id, item) pair -> group into baskets."""
    for c in (id_column, item_column):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found. Available: {list(df.columns)}")
    grouped = df.groupby(id_column)[item_column].apply(
        lambda s: [str(v) for v in s if pd.notna(v) and str(v).strip()]
    )
    return grouped.tolist()


def csv_rows_are_ragged(path: Path) -> bool:
    """
    Raw scan: do rows in this CSV have varying field counts?

    This exists to catch headerless 'basket' files (one row per
    transaction, comma-separated items, no padding) BEFORE handing the
    file to pandas. The original approach -- try pd.read_csv and catch
    pd.errors.ParserError -- only works if a later row has MORE fields
    than the first row, which is how older pandas behaved. On current
    pandas (tested here: 3.0.x), the C parser instead silently treats
    overflow fields on long rows as extra index columns and pads short
    rows with NaN, so it never raises -- it just returns a wrong-shaped
    DataFrame with item values misread as an index/header. Scanning raw
    row lengths first sidesteps that entirely.
    """
    last_err: Exception | None = None
    for enc in CSV_ENCODING_FALLBACKS:
        try:
            with open(path, newline="", encoding=enc) as f:
                lengths = {len(row) for row in csv.reader(f) if row}
            return len(lengths) > 1
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    raise last_err


def transactions_from_basket_csv(path: Path) -> list[list[str]]:
    """
    Classic apyori-tutorial format: headerless CSV, items comma-separated per row.
    Uses the csv module (not pandas.read_csv) because rows are commonly
    "ragged" -- different transactions have different numbers of items --
    which pandas' tabular reader rejects outright.
    """
    last_err: Exception | None = None
    for enc in CSV_ENCODING_FALLBACKS:
        try:
            with open(path, newline="", encoding=enc) as f:
                return [row for row in csv.reader(f) if row]
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    raise last_err


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def clean_transactions(
    transactions: list[list[str]],
    synonym_map: dict[str, str] | None = None,
) -> list[list[str]]:
    """Trim whitespace, drop empties, apply synonyms, de-duplicate per basket."""
    synonym_map = synonym_map or {}
    cleaned = []
    for basket in transactions:
        seen = []
        for raw in basket:
            item = str(raw).strip()
            if not item or item.lower() in ("nan", "none"):
                continue
            item = synonym_map.get(item.lower(), item)
            if item not in seen:
                seen.append(item)
        if seen:
            cleaned.append(seen)
    return cleaned


def detect_possible_typos(transactions: list[list[str]], cutoff: float = 0.82) -> list[tuple[str, str, float]]:
    """
    Flag pairs of distinct item names that are suspiciously similar
    (likely the same item, misspelled). Doesn't auto-merge anything --
    just surfaces candidates for you to add to SYNONYM_MAP.
    """
    items = sorted({item for basket in transactions for item in basket})
    flagged = []
    seen_pairs = set()
    for item in items:
        close = difflib.get_close_matches(item, items, n=3, cutoff=cutoff)
        for match in close:
            if match == item:
                continue
            pair = tuple(sorted((item, match)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            ratio = difflib.SequenceMatcher(None, item, match).ratio()
            flagged.append((pair[0], pair[1], round(ratio, 3)))
    return sorted(flagged, key=lambda x: -x[2])


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def perform_market_basket_analysis(
    transactions: list[list[str]],
    min_support: float = 0.02,
    min_confidence: float = 0.3,
    min_lift: float = 1.0,
    max_len: int | None = None,
    auto_relax: bool = True,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, float]:
    """
    Run apriori + association rule mining (mirrors the reference
    notebook's apyori call: min_support=0.02, min_confidence=0.3,
    min_lift=1.0, min_length=2 -- mlxtend's association_rules() already
    only emits rules with >=2 items total, so no extra min_length knob
    is needed here).

    Returns (rules_df, frequent_itemsets_df, min_support_used).
    If auto_relax is True and the given min_support yields zero frequent
    itemsets, the threshold is halved (down to a floor) until results
    appear, rather than failing silently.
    """
    if not transactions:
        print("No transactions to analyze.")
        return None, None, min_support

    encoder = TransactionEncoder()
    encoded_array = encoder.fit(transactions).transform(transactions)
    onehot = pd.DataFrame(encoded_array, columns=encoder.columns_)

    support_used = min_support
    floor = 1.0 / len(transactions)  # smallest possible non-zero support
    frequent_itemsets = pd.DataFrame()

    while True:
        frequent_itemsets = apriori(
            onehot, min_support=support_used, use_colnames=True, max_len=max_len
        )
        if not frequent_itemsets.empty or not auto_relax or support_used <= floor:
            break
        support_used = max(support_used / 2, floor)

    if frequent_itemsets.empty:
        print("No frequent itemsets found even after relaxing min_support. "
              "Your basket may be too sparse, or items too varied, for "
              "association rules to be meaningful.")
        return None, None, support_used

    # conviction = (1 - support(consequent)) / (1 - confidence). When a
    # rule has confidence == 1.0 the denominator is 0, so mlxtend
    # correctly returns inf -- but numpy prints a benign
    # "invalid value encountered in divide" RuntimeWarning every time it
    # happens. Suppress just that warning instead of the result.
    with np.errstate(divide="ignore", invalid="ignore"):
        rules = association_rules(
            frequent_itemsets, metric="confidence", min_threshold=min_confidence
        )
    rules = rules[rules["lift"] >= min_lift].copy()

    if rules.empty:
        print(f"{len(frequent_itemsets)} frequent itemsets found, but none "
              f"produced rules at confidence>={min_confidence}, lift>={min_lift}. "
              "Try lowering --min-confidence or --min-lift.")
        return None, frequent_itemsets, support_used

    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))

    keep_cols = ["antecedents", "consequents", "support", "confidence",
                 "lift", "leverage", "conviction"]
    rules = rules[keep_cols].rename(columns={
        "antecedents": "Antecedent",
        "consequents": "Consequent",
        "support": "Support",
        "confidence": "Confidence",
        "lift": "Lift",
        "leverage": "Leverage",
        "conviction": "Conviction",
    })
    for col in ("Support", "Confidence", "Lift", "Leverage", "Conviction"):
        rules[col] = rules[col].round(4)

    rules = rules.sort_values(by=["Lift", "Confidence"], ascending=False).reset_index(drop=True)
    return rules, frequent_itemsets, support_used


def item_frequency_table(transactions: list[list[str]]) -> pd.DataFrame:
    """Simple standalone-item popularity report (useful even with zero rules)."""
    counts: dict[str, int] = {}
    for basket in transactions:
        for item in basket:
            counts[item] = counts.get(item, 0) + 1
    n = len(transactions) or 1
    df = pd.DataFrame(
        [(item, c, round(c / n, 4)) for item, c in counts.items()],
        columns=["Item", "Count", "Support"],
    )
    return df.sort_values("Count", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def save_outputs(rules_df: pd.DataFrame, out_dir: Path, basename: str = "association_rules") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{basename}.csv"
    json_path = out_dir / f"{basename}.json"
    rules_df.to_csv(csv_path, index=False)
    rules_df.to_json(json_path, orient="records", indent=2)
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


def plot_top_rules(rules_df: pd.DataFrame, out_dir: Path, top_n: int = 10,
                    basename: str = "top_rules_by_lift") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping chart.")
        return

    top = rules_df.head(top_n).iloc[::-1]
    labels = [f"{a} -> {c}" for a, c in zip(top["Antecedent"], top["Consequent"])]

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(top) + 2))
    ax.barh(labels, top["Lift"], color="#6f4e37")
    ax.set_xlabel("Lift")
    ax.set_title(f"Top {len(top)} Association Rules by Lift")
    fig.tight_layout()

    out_path = out_dir / f"{basename}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Market basket analysis on transaction data.")
    p.add_argument("input", help="Path to .xlsx, .xls, or .csv transaction file")
    p.add_argument("--sheet", default=0, help="Excel sheet name or index (default: first sheet)")
    p.add_argument("--layout", choices=["wide", "long", "basket"], default=None,
                    help="Transaction layout. If omitted, the script guesses.")
    p.add_argument("--columns", nargs="*", default=None,
                    help="[wide layout] Which columns hold item slots (default: all columns)")
    p.add_argument("--id-column", default=None, help="[long layout] Transaction ID column name")
    p.add_argument("--item-column", default=None, help="[long layout] Item column name")
    p.add_argument("--min-support", type=float, default=0.02)
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--min-lift", type=float, default=1.0)
    p.add_argument("--max-len", type=int, default=None, help="Max items per rule itemset")
    p.add_argument("--no-auto-relax", action="store_true",
                    help="Disable automatic min_support relaxation when 0 rules are found")
    p.add_argument("--top-n", type=int, default=10, help="How many rules to print/plot")
    p.add_argument("--no-plot", action="store_true", help="Skip generating the bar chart")
    p.add_argument("--output-dir", default="./mba_output")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame | None:
    args = parse_args(argv)
    path = Path(args.input)
    out_dir = Path(args.output_dir)

    print(f"Loading: {path}")
    layout = args.layout
    df = None

    if layout is None and path.suffix.lower() == ".csv" and csv_rows_are_ragged(path):
        print("Detected rows with varying item counts (ragged CSV) - "
              "treating as 'basket' layout (headerless, comma-separated items).")
        layout = "basket"

    if layout != "basket":
        try:
            df = read_table(path, sheet=args.sheet)
        except pd.errors.ParserError:
            if layout is not None:
                raise
            print("File has rows of differing length (ragged CSV) - "
                  "treating as 'basket' layout (headerless, comma-separated items).")
            layout = "basket"

    if df is not None:
        print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns: {list(df.columns)}")
        if layout is None:
            layout = guess_layout(df)
        print(f"Using layout: '{layout}'" + ("" if args.layout else " (auto-detected)"))
    else:
        layout = "basket"
        print("Using layout: 'basket'" + ("" if args.layout else " (auto-detected)"))

    if layout == "wide":
        transactions = transactions_from_wide(df, args.columns)
    elif layout == "long":
        if not args.id_column or not args.item_column:
            sys.exit("Layout 'long' requires --id-column and --item-column.")
        transactions = transactions_from_long(df, args.id_column, args.item_column)
    else:  # basket
        if path.suffix.lower() == ".csv":
            transactions = transactions_from_basket_csv(path)
        else:
            if df is None:
                df = read_table(path, sheet=args.sheet)
            transactions = [list(row.dropna().astype(str)) for _, row in df.iterrows()]

    transactions = clean_transactions(transactions, SYNONYM_MAP)
    transactions = [t for t in transactions if t]  # drop empty baskets
    print(f"{len(transactions)} non-empty transactions after cleaning")

    typos = detect_possible_typos(transactions)
    if typos:
        print("\nPossible spelling variants (consider adding to SYNONYM_MAP):")
        for a, b, ratio in typos[:15]:
            print(f"  '{a}'  ~  '{b}'   (similarity {ratio})")
        print()

    freq_table = item_frequency_table(transactions)
    out_dir.mkdir(parents=True, exist_ok=True)
    freq_table.to_csv(out_dir / "item_frequency.csv", index=False)
    print("\nTop 10 most popular individual items:")
    print(freq_table.head(10).to_string(index=False))

    rules_df, frequent_itemsets, support_used = perform_market_basket_analysis(
        transactions,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        max_len=args.max_len,
        auto_relax=not args.no_auto_relax,
    )

    if support_used != args.min_support:
        print(f"\nNote: min_support auto-relaxed from {args.min_support} to "
              f"{round(support_used, 4)} to find results.")

    if rules_df is None:
        print("\nNo association rules to report. See item_frequency.csv for "
              "standalone item popularity instead.")
        return None

    print(f"\nFound {len(rules_df)} association rules")
    print(f"\nTop {args.top_n} rules by Lift:")
    print(rules_df.head(args.top_n).to_string(index=False))

    save_outputs(rules_df, out_dir)
    if not args.no_plot:
        plot_top_rules(rules_df, out_dir, top_n=args.top_n)

    return rules_df


if __name__ == "__main__":
    main()