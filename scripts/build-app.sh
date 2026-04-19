#!/usr/bin/env bash
# build-app.sh — produce a minimal "Voice Studio.app" on your Desktop
# via `osacompile`. Double-click it to launch the backend + frontend in
# the background and open http://localhost:3007 in the default browser.
#
# This complements install-autostart.sh — use the LaunchAgent if you want
# auto-start on login, use this .app if you prefer clicking an icon.
#
# Both can coexist. They don't fight each other — start.sh exits cleanly
# if another instance is already holding port 8000 or 3007.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$HOME/Desktop/Voice Studio.app"

# AppleScript body: spawn start.sh inside a Terminal window (so the user
# can see logs + hit Ctrl-C to stop), then open the UI. We use Terminal
# rather than a silent background process so the user has a clear way to
# observe / abort.
read -r -d '' APPLESCRIPT <<APPLESCRIPT || true
tell application "Terminal"
    do script "cd '${REPO_DIR}' && ./start.sh"
end tell
delay 3
tell application "Safari" to open location "http://localhost:3007/"
APPLESCRIPT

rm -rf "$APP_DIR"
osacompile -o "$APP_DIR" -e "$APPLESCRIPT"

echo
echo "✓ Built $APP_DIR"
echo "  Double-click to start Voice Studio + open the UI."
echo "  The backend runs in a Terminal window — Ctrl-C in there to stop."
echo
echo "If you'd rather have it auto-start on login without a Terminal"
echo "window, run scripts/install-autostart.sh instead."
