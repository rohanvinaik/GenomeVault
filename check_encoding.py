#!/usr/bin/env python3
"""
Find and report problematic Python files
"""

import os
from pathlib import Path


def check_files():
def check_files():
    base_path = Path.home() / "genomevault"

    print("Checking Python files for issues...")

    for py_file in base_path.glob("**/*.py"):
        # Skip venv
        if "venv" in str(py_file) or "__pycache__" in str(py_file):
            continue

        try:
            with open(py_file, "rb") as f:
                content = f.read()

            # Try UTF-8
            try:
                content.decode("utf-8")
            except UnicodeDecodeError as e:
                print(f"\nUTF-8 Error in {py_file.relative_to(base_path)}:")
                print(f"  Position {e.start}: byte {content[e.start]:02x}")

                # Try other encodings
                for enc in ["latin-1", "cp1252"]:
                    try:
                        decoded = content.decode(enc)
                        print(f"  Can be decoded with {enc}")
                        break
                    except BaseException:
                        pass

        except Exception as e:
            print(f"Error reading {py_file}: {e}")


if __name__ == "__main__":
    check_files()
