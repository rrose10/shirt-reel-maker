#!/bin/sh
# One-time setup for the "AI video" tab on the NAS.
# Run it with:   sudo sh setup-ai.sh      (inside the shirt-reel-maker folder)
# It asks you to paste your Google API key (the letters stay hidden), saves it into
# the private file .env (not part of git), and starts the containers.
set -e
cd "$(dirname "$0")"
[ -f .env.example ] || { echo "Run 'sudo git pull' first (.env.example is missing)."; exit 1; }

printf "Paste your Google API key, then press Enter (nothing will show while you paste): "
stty -echo 2>/dev/null || true
read -r KEY
stty echo 2>/dev/null || true
echo

KEY=$(printf '%s' "$KEY" | tr -d ' \t\r\n"'"'")
if [ -z "$KEY" ]; then echo "No key was entered. Run this again."; exit 1; fi

# keep the other lines of .env.example, put the key in
grep -v '^GEMINI_API_KEY=' .env.example > .env.tmp
printf 'GEMINI_API_KEY=%s\n' "$KEY" >> .env.tmp
mv .env.tmp .env
chmod 600 .env
echo "Key saved in .env (${#KEY} characters)."

docker compose up -d
sleep 3
echo
echo "Done. Open the tool, press Ctrl+F5, click the 'AI video' tab."
echo "At the bottom it should say: Videos today: 0 of 30."
