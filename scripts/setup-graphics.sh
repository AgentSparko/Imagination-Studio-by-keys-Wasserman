#!/usr/bin/env bash
# Motion graphics for the Edit room's Graphics tab, optional (oneshot-setup.sh --graphics and oneshot-setup-spark.sh
# --graphics run this): titles, lower thirds, kinetic type, captions, slideshows, music videos and explainers, rendered on
# this machine with HyperFrames (the open-source hyperframes npm package, Apache-2.0) and GSAP.
#
#   bash scripts/setup-graphics.sh            # Node 22 if this machine's is older, hyperframes and GSAP, a browser, a test render
#   bash scripts/setup-graphics.sh --verify   # check only: nothing downloaded or installed
#   bash scripts/setup-graphics.sh --no-test  # no test render at the end
#
# Everything goes in $PREFIX/graphics (~/.local/opt/tensorfold-studio/graphics; GRAPHICS_DIR names another), about 400 MB:
#   1. Node.js 22 (HyperFrames needs 22 or newer): the system's when it is new enough, else the official build from
#      nodejs.org for this machine (Linux arm64 or x64, macOS arm64 or x64) at the version below, its SHA-256 checked,
#      unpacked into $PREFIX/graphics/node; nothing is installed system-wide;
#   2. hyperframes and gsap from npm at pinned versions, into $PREFIX/graphics/node_modules;
#   3. the browser it renders with: a Chromium already in Playwright's cache (~/.cache/ms-playwright, or
#      ~/Library/Caches/ms-playwright on a Mac) when one is there and starts, else HyperFrames' own Chrome download, else
#      (Linux on arm64, a DGX Spark, which Chrome for Testing has no build for) Playwright's Chromium for arm64. The one
#      chosen is written to $PREFIX/graphics/browser-path. HYPERFRAMES_BROWSER_PATH names one of your own;
#   4. ffmpeg, which renders the frames to MP4 (sudo apt install ffmpeg, or brew install ffmpeg), checked only;
#   5. a test render: a three-second title at 320x180.
# Telemetry and update checks are switched off for every render (HYPERFRAMES_NO_TELEMETRY, DO_NOT_TRACK).
set -euo pipefail

NODE_VERSION="22.23.3"
node_sha256() {  # the official tarballs' SHA-256, from nodejs.org/dist/v$NODE_VERSION/SHASUMS256.txt
  case "$1" in
    linux-arm64) echo a44aeb94849a299b22df10b9e622ec2f605c2183501bc40590705131de7c740f;;
    linux-x64) echo df450af89261115ef9f9e3830c3eeb2cc9213b63c720b1af623cb5dcbe2e02de;;
    darwin-arm64) echo 23b25245dcfb9af7262f8ff142e9e2e0af025368117329e7a7458a51e5922f53;;
    darwin-x64) echo 8a677b0219178efd6eb0e475457c4afb452b521a92f6e67845a73bd85727f2a8;;
  esac
}
HYPERFRAMES_VERSION="0.8.186"   # app/graphics.py HYPERFRAMES says the same
GSAP_VERSION="3.14.2"
PLAYWRIGHT_VERSION="1.64.1"     # only for a Linux arm64 machine with no Chromium of its own
VERIFY=0; TEST=1
for arg in "$@"; do
  case "$arg" in
    --verify) VERIFY=1;;
    --no-test) TEST=0;;
    *) echo "unknown option $arg" >&2; exit 2;;
  esac
done
ok()  { echo "  ✓ $*"; }
bad() { echo "  ! $*"; MISSING=1; }
step(){ echo; echo "==> $*"; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
DIR="${GRAPHICS_DIR:-$PREFIX/graphics}"
export HYPERFRAMES_NO_TELEMETRY=1 DO_NOT_TRACK=1 HYPERFRAMES_NO_UPDATE_CHECK=1 HYPERFRAMES_SKIP_SKILLS=1 NO_UPDATE_NOTIFIER=1
export PATH="$PATH:/usr/bin:/usr/local/bin:/opt/homebrew/bin"
MISSING=0
major() { "$1" --version 2>/dev/null | sed -E 's/^v([0-9]+).*/\1/'; }

case "$(uname -s)-$(uname -m)" in
  Linux-aarch64|Linux-arm64) PLATFORM=linux-arm64; EXT=tar.xz;;
  Linux-x86_64) PLATFORM=linux-x64; EXT=tar.xz;;
  Darwin-arm64) PLATFORM=darwin-arm64; EXT=tar.gz;;
  Darwin-x86_64) PLATFORM=darwin-x64; EXT=tar.gz;;
  *) echo "FATAL: no Node.js build for $(uname -s) $(uname -m)" >&2; exit 1;;
esac

step "Node.js 22 or newer"
NODE=""
if [ -x "$DIR/node/bin/node" ] && [ "$(major "$DIR/node/bin/node")" -ge 22 ] 2>/dev/null; then
  NODE="$DIR/node/bin/node"; ok "$("$NODE" --version), installed here ($DIR/node)"
elif command -v node >/dev/null && [ "$(major "$(command -v node)")" -ge 22 ] 2>/dev/null; then
  NODE="$(command -v node)"; ok "$("$NODE" --version), this machine's own ($NODE)"
elif [ "$VERIFY" = 1 ]; then
  bad "no Node 22 ($(command -v node >/dev/null && node --version || echo "no node")): bash scripts/setup-graphics.sh installs one in $DIR/node"
else
  echo "  this machine has $(command -v node >/dev/null && node --version || echo "no node"); installing Node v$NODE_VERSION for HyperFrames only"
  NAME="node-v$NODE_VERSION-$PLATFORM"
  WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
  curl -fsSL --retry 3 -o "$WORK/$NAME.$EXT" "https://nodejs.org/dist/v$NODE_VERSION/$NAME.$EXT"
  GOT="$( (command -v sha256sum >/dev/null && sha256sum "$WORK/$NAME.$EXT" || shasum -a 256 "$WORK/$NAME.$EXT") | cut -d' ' -f1)"
  [ "$GOT" = "$(node_sha256 "$PLATFORM")" ] || { echo "FATAL: $NAME.$EXT does not match its pinned SHA-256 (got $GOT)" >&2; exit 1; }
  ok "downloaded $NAME.$EXT (SHA-256 checked)"
  mkdir -p "$WORK/x" && tar -xf "$WORK/$NAME.$EXT" -C "$WORK/x"
  rm -rf "$DIR/node"; mkdir -p "$DIR"; mv "$WORK/x/$NAME" "$DIR/node"
  NODE="$DIR/node/bin/node"; ok "$("$NODE" --version) in $DIR/node"
fi

step "HyperFrames $HYPERFRAMES_VERSION and GSAP $GSAP_VERSION -> $DIR/node_modules"
have_pkg() { [ "$(sed -nE 's/^ *"version": *"([^"]+)".*/\1/p' "$DIR/node_modules/$1/package.json" 2>/dev/null | head -1)" = "$2" ]; }
if have_pkg hyperframes "$HYPERFRAMES_VERSION" && have_pkg gsap "$GSAP_VERSION"; then ok "there"
elif [ "$VERIFY" = 1 ]; then bad "not installed at these versions in $DIR (installed: hyperframes $(sed -nE 's/^ *"version": *"([^"]+)".*/\1/p' "$DIR/node_modules/hyperframes/package.json" 2>/dev/null | head -1))"
elif [ -n "$NODE" ]; then
  mkdir -p "$DIR"
  [ -f "$DIR/package.json" ] || echo '{ "name": "studio-graphics", "private": true }' > "$DIR/package.json"
  NPM="$(dirname "$NODE")/npm"; [ -x "$NPM" ] || NPM="$(command -v npm)"
  PATH="$(dirname "$NODE"):$PATH" "$NPM" install --prefix "$DIR" --no-audit --no-fund --save-exact \
    "hyperframes@$HYPERFRAMES_VERSION" "gsap@$GSAP_VERSION" 2>&1 | tail -3 | sed 's/^/  /'
  if have_pkg hyperframes "$HYPERFRAMES_VERSION" && have_pkg gsap "$GSAP_VERSION"; then ok "installed"; else bad "npm did not install them (above)"; fi
fi
CLI="$DIR/node_modules/hyperframes/dist/cli.js"

step "A browser to render with"
starts() { [ -x "$1" ] && "$1" --headless --no-sandbox --dump-dom about:blank >/dev/null 2>&1; }
playwright_shell() {  # the newest Chromium in Playwright's cache that starts, the headless shell first
  local cache found
  for cache in "${PLAYWRIGHT_BROWSERS_PATH:-}" "$HOME/.cache/ms-playwright" "$HOME/Library/Caches/ms-playwright"; do
    [ -n "$cache" ] && [ -d "$cache" ] || continue
    while IFS= read -r found; do
      if starts "$found"; then echo "$found"; return 0; fi
    done < <( { ls -d "$cache"/chromium_headless_shell-*/*/headless_shell "$cache"/chromium_headless_shell-*/*/chrome-headless-shell 2>/dev/null | sort -r
                ls -d "$cache"/chromium-*/chrome-linux*/chrome "$cache"/chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium \
                  "$cache"/chromium-*/chrome-mac*/"Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing" 2>/dev/null | sort -r; } )
  done
  return 1
}
BROWSER=""
if [ -n "${HYPERFRAMES_BROWSER_PATH:-}" ]; then
  if starts "$HYPERFRAMES_BROWSER_PATH"; then BROWSER="$HYPERFRAMES_BROWSER_PATH"; ok "HYPERFRAMES_BROWSER_PATH: $BROWSER"
  else bad "HYPERFRAMES_BROWSER_PATH=$HYPERFRAMES_BROWSER_PATH does not start"; fi
elif [ -s "$DIR/browser-path" ] && starts "$(cat "$DIR/browser-path")"; then
  BROWSER="$(cat "$DIR/browser-path")"; ok "$BROWSER (chosen before)"
elif BROWSER="$(playwright_shell)"; then
  ok "Playwright's Chromium, already on this machine: $BROWSER"
elif [ "$VERIFY" = 1 ]; then
  if [ -n "$NODE" ] && [ -f "$CLI" ] && BROWSER="$("$NODE" "$CLI" browser path 2>/dev/null | tail -1)" && starts "$BROWSER"; then ok "HyperFrames' own: $BROWSER"
  else BROWSER=""; bad "no browser: bash scripts/setup-graphics.sh fetches one"; fi
elif [ -n "$NODE" ] && [ -f "$CLI" ]; then
  if [ "$PLATFORM" != linux-arm64 ] && "$NODE" "$CLI" browser ensure 2>&1 | tail -2 | sed 's/^/  /' \
      && BROWSER="$("$NODE" "$CLI" browser path 2>/dev/null | tail -1)" && starts "$BROWSER"; then
    ok "HyperFrames' own Chrome: $BROWSER"
  else
    # Linux on arm64: Chrome for Testing has no build, Playwright publishes one
    echo "  fetching Playwright's Chromium headless shell for $PLATFORM (Playwright $PLAYWRIGHT_VERSION, about 110 MB)"
    PATH="$(dirname "$NODE"):$PATH" "$(dirname "$NODE")/npx" --yes "playwright@$PLAYWRIGHT_VERSION" install chromium-headless-shell 2>&1 | tail -3 | sed 's/^/  /' || true
    if BROWSER="$(playwright_shell)"; then ok "$BROWSER"
    else BROWSER=""; bad "no browser starts here: the Chromium may lack system libraries (sudo npx playwright@$PLAYWRIGHT_VERSION install-deps chromium), or set HYPERFRAMES_BROWSER_PATH"; fi
  fi
fi
if [ -n "$BROWSER" ] && [ "$VERIFY" = 0 ]; then mkdir -p "$DIR" && printf '%s\n' "$BROWSER" > "$DIR/browser-path"; fi

step "ffmpeg"
if command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null; then ok "$(ffmpeg -version | head -1 | cut -d' ' -f1-3)"
else bad "no ffmpeg: $([ "$(uname -s)" = Linux ] && echo "sudo apt install ffmpeg" || echo "brew install ffmpeg")"; fi

if [ "$MISSING" = 0 ] && [ "$TEST" = 1 ] && [ "$VERIFY" = 0 ]; then
  step "A test render (a three-second title at 320x180)"
  OUT="$(mktemp -d)/graphics-test.mp4"
  GRAPHICS_DIR="$DIR" python3 -I "$HERE/app/graphics.py" selftest "$OUT" | sed 's/^/  /' || bad "the test render failed: run it again with HYPERFRAMES_DEBUG=1 to see why"
fi
echo
if [ "$MISSING" = 1 ]; then
  echo "  Not ready yet: run bash scripts/setup-graphics.sh (or oneshot-setup$([ "$(uname -s)" = Linux ] && echo -spark).sh --graphics) again for what is missing above."
  [ "$VERIFY" = 1 ] && exit 1
  exit 0
fi
echo "  Motion graphics ready: Edit -> Graphics. Rendered with HyperFrames (Apache-2.0) on this machine."
