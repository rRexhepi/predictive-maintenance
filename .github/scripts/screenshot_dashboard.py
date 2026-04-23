"""Playwright-driven screenshot capture of the Grafana dashboard.

Runs in CI after a compose-up step has brought the stack up and a brief
traffic burst has populated metrics. Headless chromium, deterministic
viewport, fixed output path.

Usage (inside the workflow):

    python .github/scripts/screenshot_dashboard.py \\
        --url "http://localhost:3000/d/predictive-maintenance" \\
        --output docs/grafana-dashboard.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Grafana dashboard URL.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--width", type=int, default=1680)
    parser.add_argument("--height", type=int, default=1400)
    parser.add_argument("--wait-ms", type=int, default=6000, help="Dwell before snapshot.")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=2,  # retina-ish so the image isn't mush
        )
        page = ctx.new_page()
        # kiosk=tv hides the Grafana chrome and sidebars; refresh=15s
        # forces a fresh render on load so panels populate before
        # the first screenshot.
        page.goto(
            f"{args.url}?kiosk=tv&refresh=15s",
            wait_until="networkidle",
            timeout=60_000,
        )
        page.wait_for_timeout(args.wait_ms)
        page.screenshot(path=str(out), full_page=True)
        browser.close()

    print(f"Wrote screenshot to {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
