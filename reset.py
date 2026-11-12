"""Remove trained runs and generated maps, to get back to a clean checkout.

    python reset.py                     list what would go, ask, then remove it
    python reset.py --yes               remove without asking
    python reset.py --dry-run           only list what would go
    python reset.py --runs              only the trained runs
    python reset.py --maps              only the generated maps
    python reset.py --include-bundled   also remove the bundled models (stage1_navigate, stage2_defence)

What it removes:
  runs/<name>/          models you trained (the bundled ones are kept unless --include-bundled;
                        the app opens runs/stage1_navigate by default, so removing it needs
                        --model scripted or a new training run afterwards)
  cache/                maps built from configs, plus downloaded OpenStreetMap / satellite data
  worlds/custom_*/      cities made with "Generate" in the app's Scenario panel

What it keeps: the maps shipped in worlds/ (city_suburb, city_dense_core), configs/, assets/,
the code and the .venv. Removed files are deleted, not moved to the recycle bin.
"""
from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUNDLED_RUNS = ("stage1_navigate", "stage2_defence")


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def targets(runs: bool, maps: bool, include_bundled: bool) -> list[Path]:
    out: list[Path] = []
    if runs and (ROOT / "runs").is_dir():
        out += [p for p in sorted((ROOT / "runs").iterdir())
                if include_bundled or p.name not in BUNDLED_RUNS]
    if maps:
        if (ROOT / "cache").is_dir():
            out += sorted((ROOT / "cache").iterdir())
        if (ROOT / "worlds").is_dir():
            out += sorted(p for p in (ROOT / "worlds").glob("custom_*") if p.is_dir())
    return out


def _remove(path: Path):
    def retry_writable(func, p, _exc):          # Windows: read-only files refuse to be deleted
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if path.is_dir() and not path.is_symlink():
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=retry_writable)
        else:
            shutil.rmtree(path, onerror=retry_writable)
    else:
        path.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", action="store_true", help="only remove trained runs")
    ap.add_argument("--maps", action="store_true", help="only remove generated maps and the download cache")
    ap.add_argument("--include-bundled", action="store_true",
                    help=f"also remove the bundled models: {', '.join(BUNDLED_RUNS)}")
    ap.add_argument("--dry-run", action="store_true", help="list what would be removed and stop")
    ap.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()
    both = not (args.runs or args.maps)
    found = targets(args.runs or both, args.maps or both, args.include_bundled)
    for p in found:                              # never touch anything outside the project
        if ROOT not in p.resolve().parents:
            raise SystemExit(f"Refusing to remove {p}: it is outside {ROOT}")

    if not found:
        print("Nothing to remove.")
        return 0
    sizes = {p: _size(p) for p in found}
    print("Will remove:" if not args.dry_run else "Would remove:")
    for p in found:
        print(f"  {p.relative_to(ROOT).as_posix():<48} {_human(sizes[p]):>9}")
    print(f"  {'total':<48} {_human(sum(sizes.values())):>9}")
    kept = [n for n in BUNDLED_RUNS if (ROOT / "runs" / n).is_dir() and (args.runs or both)
            and not args.include_bundled]
    if kept:
        print(f"Keeping the bundled models: {', '.join(kept)} (add --include-bundled to remove them)")
    if args.dry_run:
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("Not removing anything: run with --yes to confirm when there is no terminal to ask in.")
            return 1
        if input("This cannot be undone. Remove these? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Nothing removed.")
            return 1

    failed = []
    for p in found:
        try:
            _remove(p)
        except OSError as e:                     # e.g. a file held open by the running app
            failed.append((p, e))
    print(f"Removed {len(found) - len(failed)} of {len(found)}.")
    for p, e in failed:
        print(f"  could not remove {p.relative_to(ROOT).as_posix()}: {e.strerror or e}")
    if failed:
        print("Close the control app (python main.py) if it is running, then try again.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
