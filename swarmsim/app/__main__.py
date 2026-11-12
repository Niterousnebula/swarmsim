import argparse

from .server import serve

ap = argparse.ArgumentParser(description="swarmsim control app (3-D view, piloting, training)")
ap.add_argument("--model", default="stage1_navigate", help="run folder in runs/, or scripted / random")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
args = ap.parse_args()
serve(args.model, args.port, not args.no_browser)
