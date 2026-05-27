import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from Data import FinancialDataCollector
except ImportError as e:
    print(f"Error: Could not import Data.py: {e}")
    print("Make sure Data.py is in the same folder as this script.")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Export all financial data to CSV")
parser.add_argument("--start", type=str, default="2010-01-01", help="Start date (default: 2010-01-01)")
parser.add_argument("--end",   type=str, default=None,         help="End date (default: today)")
parser.add_argument("--out",   type=str, default=os.path.join(SCRIPT_DIR, "financial_data.csv"), help="Output CSV path")
args = parser.parse_args()

print("=" * 60)
print("  X-INVEST  |  Financial Data Exporter")
print("=" * 60)
print(f"  Start  : {args.start}")
print(f"  End    : {args.end or 'today'}")
print(f"  Output : {args.out}")
print("=" * 60)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

collector = FinancialDataCollector(start_date=args.start, end_date=args.end)

all_stocks = collector.collect_all_data()

if not all_stocks:
    print("\nError: No data collected. Check your internet connection and try again.")
    sys.exit(1)

combined = collector.create_combined_dataset(all_stocks)

collector.save_to_csv(combined, filename=args.out, lowercase_cols=True)

print("\nDone!")
print(f"  Rows    : {len(combined):,}")
print(f"  Columns : {len(combined.columns)}")
print(f"  Range   : {combined.index.min().date()} -> {combined.index.max().date()}")
print(f"  Saved   : {args.out}")