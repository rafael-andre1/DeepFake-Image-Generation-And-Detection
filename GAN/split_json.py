#!/usr/bin/env python
"""Split large JSONL files into ~90 MB chunks named pt1, pt2, ..."""

import os, glob

CHUNK_BYTES = 90 * 1024 * 1024  # 90 MB


def split_jsonl(src_path, out_dir):
    """Split a single JSONL file into pt1.jsonl, pt2.jsonl, ... in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    part = 1
    current_size = 0
    out_file = None

    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line_bytes = len(line.encode("utf-8"))

            if out_file is None or current_size + line_bytes > CHUNK_BYTES:
                if out_file:
                    out_file.close()
                out_path = os.path.join(out_dir, f"pt{part}.jsonl")
                out_file = open(out_path, "w", encoding="utf-8")
                print(f"  Writing {out_path}")
                part += 1
                current_size = 0

            out_file.write(line)
            current_size += line_bytes

    if out_file:
        out_file.close()

    print(f"  Done: {part - 1} part(s)\n")


def main():
    folder = input("Folder: (gNUMBER format) ").strip()

    if folder.startswith("g") and folder[1:].isdigit():
        base = os.path.dirname(os.path.abspath(__file__))
        outputs_dir = os.path.join(base, "GAN", "outputs")

        for item in os.listdir(outputs_dir):
            if item.startswith(folder):
                folder = item
                break

        json_dir = os.path.join(outputs_dir, folder, "json_samples")

        if not os.path.isdir(json_dir):
            print(f"No json_samples dir found at {json_dir}")
            return

        files = sorted(glob.glob(os.path.join(json_dir, "*.jsonl")))
        if not files:
            print("No .jsonl files found.")
            return

        print(f"Found {len(files)} file(s) in {json_dir}\n")

        split_dir = os.path.join(json_dir, "split")

        for src in files:
            size_mb = os.path.getsize(src) / (1024 * 1024)
            print(f"{os.path.basename(src)} ({size_mb:.1f} MB)")

            if size_mb <= 90:
                print("  Skipped (already under 90 MB)\n")
                continue

            split_jsonl(src, split_dir)

        print(f"All splits saved to: {split_dir}")
    else:
        print("Invalid format. Use gNUMBER (e.g. g9)")


if __name__ == "__main__":
    main()