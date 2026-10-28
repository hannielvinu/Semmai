#!/usr/bin/env bash
# Semmai local demo (macOS / Linux). Needs only Python 3.9+.
cd "$(dirname "$0")"
exec python3 app/server.py "$@"
