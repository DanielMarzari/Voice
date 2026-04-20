#!/usr/bin/env bash
# lightning-env.sh — source this in a new terminal to get the Lightning
# CLI on PATH + an echo of the current Studio's state.
#
# Usage:
#   source scripts/lightning-env.sh       (sets PATH, runs `lightning list studios`)
#   source scripts/lightning-env.sh quiet (sets PATH, prints nothing)
#
# Full walkthrough: see spikes/LIGHTNING_SETUP.md

# The pip --user install puts `lightning` here on macOS with Homebrew python.
export PATH="$HOME/Library/Python/3.11/bin:$PATH"

# Defaults for this project. Override in your shell if you spin up a
# different Studio.
export LIGHTNING_TEAMSPACE="${LIGHTNING_TEAMSPACE:-danmarzari/default-project}"
export LIGHTNING_STUDIO="${LIGHTNING_STUDIO:-sole-sapphire-nqrv}"
export LIGHTNING_STUDIO_ROOT="/teamspace/studios/this_studio"
export LIGHTNING_SSH_ALIAS="sole-sapphire-nqrv"

# Sanity check the CLI is installed.
if ! command -v lightning >/dev/null 2>&1; then
    echo "⚠  lightning CLI not on PATH. Install with:"
    echo "     python3.11 -m pip install --user lightning-sdk"
    return 1 2>/dev/null || exit 1
fi

# Print status unless caller said quiet.
if [[ "${1:-}" != "quiet" ]]; then
    echo "⚡  Lightning env ready"
    echo "   CLI:       $(command -v lightning) ($(lightning --version 2>&1 | head -1))"
    echo "   Teamspace: $LIGHTNING_TEAMSPACE"
    echo "   Studio:    $LIGHTNING_STUDIO"
    echo "   SSH:       ssh $LIGHTNING_SSH_ALIAS"
    echo
    lightning list studios --teamspace "$LIGHTNING_TEAMSPACE" 2>&1 | tail -n +1
fi
