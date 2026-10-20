"""Prepare reproducible English-Spanish OPUS-100 text splits from its Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split(index: int, seed: int) -> str:
    value = hashlib.sha256(f"{seed}:{index}".encode()).digest()[0]
    return "valid" if value < 8 else "test" if value < 16 else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, nargs="?", default=Path("data/opus100_en-es_train.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if not args.parquet.is_file():
        raise FileNotFoundError(args.parquet)

    parquet = pq.ParquetFile(args.parquet)
    if "translation" not in parquet.schema_arrow.names:
        raise ValueError(f"Se esperaba una columna 'translation'; columnas: {parquet.schema_arrow.names}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: args.output_dir / f"opus100_en-es_{name}.txt" for name in ("train", "valid", "test")}
    counts = {name: 0 for name in paths}
    skipped = 0
    duplicates = 0
    row_index = 0
    seen: set[bytes] = set()
    with (paths["train"].open("w", encoding="utf-8", newline="\n") as train,
          paths["valid"].open("w", encoding="utf-8", newline="\n") as valid,
          paths["test"].open("w", encoding="utf-8", newline="\n") as test):
        handles = {"train": train, "valid": valid, "test": test}
        for batch in parquet.iter_batches(columns=["translation"], batch_size=4096):
            for pair in batch.column(0).to_pylist():
                split = _split(row_index, args.seed)
                row_index += 1
                if not isinstance(pair, dict):
                    skipped += 1
                    continue
                english, spanish = _clean(pair.get("en") or ""), _clean(pair.get("es") or "")
                if not english or not spanish:
                    skipped += 1
                    continue
                pair_hash = hashlib.sha256((english + "\0" + spanish).encode("utf-8")).digest()
                if pair_hash in seen:
                    duplicates += 1
                    continue
                seen.add(pair_hash)
                handles[split].write(f"Translate English to Spanish: English: {english} Spanish: {spanish}\n")
                counts[split] += 1

    manifest = {
        "dataset": "Helsinki-NLP/opus-100",
        "config": "en-es",
        "source_file": str(args.parquet),
        "source_rows": row_index,
        "skipped_empty_or_invalid": skipped,
        "exact_duplicate_pairs_removed": duplicates,
        "split_seed": args.seed,
        "split_method": "SHA-256(row_index, seed), approximately 94/3/3; stable and disjoint",
        "format": "Translate English to Spanish: English: {en} Spanish: {es}",
        "rows": counts,
        "files": {name: str(path) for name, path in paths.items()},
    }
    (args.output_dir / "opus100_en-es_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
