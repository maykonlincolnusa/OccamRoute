"""Launch the OccamRoute KPI observability server or export a standalone HTML dashboard."""
from __future__ import annotations

import argparse
from pathlib import Path

from occamroute.dashboard import export_standalone_html


def main() -> None:
    parser = argparse.ArgumentParser(description="OccamRoute Observability Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--export-only", action="store_true", help="Export standalone HTML snapshot and exit")
    parser.add_argument("--output", default="artifacts/occamroute_dashboard.html")
    args = parser.parse_args()

    out = export_standalone_html(output_path=Path(args.output))
    print(f"Exported standalone dashboard snapshot -> {out}")
    if args.export_only:
        return

    import uvicorn

    print(f"Starting OccamRoute Dashboard at http://{args.host}:{args.port}")
    uvicorn.run("occamroute.dashboard.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
