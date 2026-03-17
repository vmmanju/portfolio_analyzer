#!/usr/bin/env python3
"""Save clipboard contents into workspace/chat_history as a timestamped file.

Usage:
  python scripts/save_chat_from_clipboard.py

Works on Windows (uses PowerShell Get-Clipboard). Falls back to tkinter or pyperclip if needed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def read_clipboard() -> str:
    # Try PowerShell (Windows)
    try:
        proc = subprocess.run(["powershell", "-Command", "Get-Clipboard"], capture_output=True, text=True, check=True)
        return proc.stdout
    except Exception:
        pass

    # Try tkinter (portable, but requires tkinter installed)
    try:
        import tkinter as tk

        r = tk.Tk()
        r.withdraw()
        text = r.clipboard_get()
        r.destroy()
        return text
    except Exception:
        pass

    # Try pyperclip if available
    try:
        import pyperclip

        return pyperclip.paste()
    except Exception:
        pass

    raise RuntimeError("Unable to read clipboard via PowerShell, tkinter, or pyperclip.")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "chat_history"
    out_dir.mkdir(exist_ok=True)

    try:
        text = read_clipboard()
    except Exception as e:
        print(f"Failed to read clipboard: {e}", file=sys.stderr)
        return 2

    if not text:
        print("Clipboard is empty. Nothing saved.")
        return 0

    out_file = out_dir / "chat_history.txt"
    header = f"\n--- Saved: {datetime.now().isoformat()} ---\n"
    with out_file.open("a", encoding="utf8") as fh:
        fh.write(header)
        fh.write(text)
        fh.write("\n")

    print(f"Appended clipboard to: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
