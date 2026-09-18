#!/bin/bash
# Keep Mac awake for Grok Bot local-exec while allowing display sleep.
# Display off is OK; lid-close still often drops Wi‑Fi on MacBook Air.
set -euo pipefail
if pgrep -f 'caffeinate -ims' >/dev/null; then
  echo "caffeinate already running: $(pgrep -lf 'caffeinate -ims')"
  exit 0
fi
nohup caffeinate -ims >/tmp/grok_caffeinate.log 2>&1 &
echo "started caffeinate pid=$! (display may sleep; system stays up on AC/battery)"
