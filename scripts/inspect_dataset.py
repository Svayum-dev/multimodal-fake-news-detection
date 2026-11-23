import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.file, sep="\t", low_memory=False)
    print(f"Rows: {len(df):,}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst rows:")
    print(df.head(args.rows).to_string())
    print("\nLabel distribution:")
    if "2_way_label" in df:
        print(df["2_way_label"].value_counts(dropna=False))
    if "image_url" in df:
        print(f"\nMissing image URLs: {df['image_url'].isna().sum():,}")
    if "hasImage" in df:
        print("\nhasImage distribution:")
        print(df["hasImage"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
