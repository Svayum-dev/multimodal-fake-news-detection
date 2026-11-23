import argparse
from pathlib import Path
import time

import pandas as pd
import requests
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--output-dir", default="data/images")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--pause", type=float, default=0.05)
    args = parser.parse_args()

    df = pd.read_csv(args.file, sep="\t", low_memory=False)
    if "id" not in df.columns or "image_url" not in df.columns:
        raise ValueError("Expected 'id' and 'image_url' columns in the TSV.")

    if args.limit:
        df = df.head(args.limit)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (academic dataset downloader)"})

    success, skipped, failed = 0, 0, 0
    failures = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_id = str(row["id"]).strip()
        url = row["image_url"]
        if not image_id or pd.isna(url) or not str(url).startswith(("http://", "https://")):
            failed += 1
            failures.append({"id": image_id, "url": str(url), "reason": "missing/invalid URL or id"})
            continue

        destination = output_dir / f"{image_id}.jpg"
        if destination.exists() and destination.stat().st_size > 0:
            skipped += 1
            continue

        try:
            response = session.get(str(url), timeout=args.timeout)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                raise ValueError(f"Response is not an image ({content_type})")
            destination.write_bytes(response.content)
            success += 1
        except Exception as exc:
            failed += 1
            failures.append({"id": image_id, "url": str(url), "reason": str(exc)[:200]})
        time.sleep(args.pause)

    failure_file = output_dir / "download_failures.csv"
    pd.DataFrame(failures).to_csv(failure_file, index=False)
    print(f"Downloaded: {success}; already existed: {skipped}; failed: {failed}")
    print(f"Failure log: {failure_file}")


if __name__ == "__main__":
    main()
