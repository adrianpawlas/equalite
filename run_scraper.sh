#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run_scraper.sh — Cron-safe wrapper for the Equalité scraper
# =============================================================================
# This script resolves absolute paths so it works correctly when run by cron.
# It sources the .env file, activates any venv, and logs output with timestamps.
#
# Usage:
#   ./run_scraper.sh                    # Full scrape with embeddings
#   ./run_scraper.sh --skip-embeddings  # Scrape without embeddings (faster)
#   ./run_scraper.sh --limit 10         # Test with 10 products
#   ./run_scraper.sh --resume           # Resume interrupted scrape
#   ./run_scraper.sh --skip-existing    # Only new products
# =============================================================================

# --- Resolve project root (works when called from any location or by cron) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Logging setup ---
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/scraper_$(date +%Y%m%d_%H%M%S).log"

# --- Load .env ---
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
else
    echo "[ERROR] .env file not found at $SCRIPT_DIR/.env" | tee -a "$LOG_FILE"
    exit 1
fi

# --- Python environment detection ---
# Prefer a virtual environment if it exists, otherwise use system python3
if [ -d "$SCRIPT_DIR/venv" ]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python3"
elif command -v python3 &>/dev/null; then
    PYTHON="$(command -v python3)"
else
    echo "[ERROR] python3 not found" | tee -a "$LOG_FILE"
    exit 1
fi

# --- Verify dependencies are installed ---
"$PYTHON" -c "import requests, bs4, torch, transformers, supabase, dotenv, tqdm" 2>/dev/null || {
    echo "[ERROR] Missing Python dependencies. Run: pip3 install -r requirements.txt" | tee -a "$LOG_FILE"
    exit 1
}

# --- Run scraper ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Equalité scraper..." | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Command: $PYTHON $SCRIPT_DIR/scraper.py $*" | tee -a "$LOG_FILE"

"$PYTHON" "$SCRIPT_DIR/scraper.py" "$@" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scraper completed successfully." | tee -a "$LOG_FILE"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scraper exited with code $EXIT_CODE. Check log: $LOG_FILE" | tee -a "$LOG_FILE"
fi

exit $EXIT_CODE
