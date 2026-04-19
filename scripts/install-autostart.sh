#!/usr/bin/env bash
# install-autostart.sh — register a macOS LaunchAgent so Voice Studio
# starts automatically on login. Once installed, your Mac wakes up, logs
# you in, and ./start.sh begins running in the background. Reader's render
# queue starts draining without you doing anything.
#
# Uninstall: `launchctl unload ~/Library/LaunchAgents/com.danmarzari.voice-studio.plist`
# then delete the plist.
#
# Logs: ~/Library/Logs/voice-studio.{out,err}.log — tail those to debug.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST_LABEL="com.danmarzari.voice-studio"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"
LOG_DIR="$HOME/Library/Logs"
LOG_OUT="$LOG_DIR/voice-studio.out.log"
LOG_ERR="$LOG_DIR/voice-studio.err.log"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${PLIST_LABEL}</string>

  <!-- Use bash so start.sh's shebang behaves the same way it does in your
       terminal. We pass -lc so the user's login shell init runs (picks up
       nvm, pyenv, PATH tweaks, etc.). -->
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>cd '${REPO_DIR}' &amp;&amp; ./start.sh</string>
  </array>

  <!-- Run once at login + keep alive if it dies. -->
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>

  <!-- Throttle restarts so a persistent crash doesn't hammer the machine. -->
  <key>ThrottleInterval</key>
  <integer>30</integer>

  <!-- Capture logs. -->
  <key>StandardOutPath</key>
  <string>${LOG_OUT}</string>
  <key>StandardErrorPath</key>
  <string>${LOG_ERR}</string>

  <!-- Give the Python venv a PATH that includes brew (ffmpeg lives there). -->
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
</dict>
</plist>
PLIST

# Unload any previous version, then load the new one.
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load -w "$PLIST_PATH"

echo
echo "✓ Voice Studio LaunchAgent installed."
echo "  Plist:   $PLIST_PATH"
echo "  Logs:    $LOG_OUT"
echo "           $LOG_ERR"
echo
echo "It's running now (launchctl load -w), and will start on every login."
echo "Check status:   launchctl list | grep ${PLIST_LABEL}"
echo "Stop it:        launchctl unload '$PLIST_PATH'"
echo "Remove:         launchctl unload '$PLIST_PATH' && rm '$PLIST_PATH'"
