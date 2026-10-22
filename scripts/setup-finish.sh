#!/usr/bin/env bash
# Finish (app/finish.py): the final cut and mix through Tesseract, and its converter to Premiere Pro and After Effects.
#   bash scripts/setup-finish.sh            # install or complete
#   bash scripts/setup-finish.sh --verify   # say what is there, download nothing
# Under $PREFIX/finish (PREFIX defaults to ~/.local/opt/tensorfold-studio) it adds:
#   bin/tsrct        Tesseract's command-line renderer, the pinned release from github.com/mirage-hq/tesseract
#   bin/tsrct-conv   Tesseract Converter, the pinned release from github.com/mirage-hq/Tesseract-Converter (MIT)
#   fonts/           Instrument Serif and Geist (SIL Open Font License), for the title card and captions
# Every download is checked against the SHA-256 published beside it (the fonts against the sums below) before
# anything is installed. Nothing here needs sudo, an account or a key.
# Licences: Tesseract is free to download and use but it is not open source: it comes under Mirage's own terms
# (TERMS.md in the bundle, mirage.app/legal), which leave out commercial use by a business with a yearly revenue of
# 1 M USD or more without a separate agreement, and do not allow passing the program on. The converter is MIT.
# Telemetry: Tesseract may send basic usage counts for some commands (no file names, paths or contents); it is on by
# default in its releases. This script never changes that setting. To turn it off: "$PREFIX/finish/bin/tsrct" telemetry disable
# Tesseract runs on macOS (Apple silicon or Intel), Windows and Linux x86_64; not on Linux ARM such as a DGX Spark.
# There this script installs only the fonts: run it on your Mac too and set the Mac as the remote computer in
# Settings > Connections (Brains & tools); the Studio then finishes the cut on the Mac over ssh.
set -euo pipefail
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
DEST="$PREFIX/finish"
TSRCT_VERSION="0.4.0"          # the release Tesseract's own skills pin (fxSchemaVersion 77)
CONV_VERSION="0.3.0"           # the converter release built on the same FX schema (77)
# TODO: pin the two releases' SHA-256 here, as the fonts' are. For now each download is checked against the .sha256
# file published in the same release, which proves the download is whole but not that the release is the one read.
FONTS_COMMIT="45e3d5269e4a17d823f6d84470bc084dc5ae46f6"   # google/fonts
INSTRUMENT_SHA="498efd461f6ddfcb7a111bf9a565709d2085d48201d501ead960d93e84ffbb88"
GEIST_SHA="73894e0448cae90a92b6c2f8732b7bb9acb7b94c418bff559dad4a18e1de9659"
VERIFY=0; [ "${1:-}" = "--verify" ] && VERIFY=1
export PATH="/opt/homebrew/bin:$PATH"
ok() { echo "  ✓ $*"; }
note() { echo "  ! $*"; }
sum256() { if command -v shasum >/dev/null; then shasum -a 256 "$1" | cut -d' ' -f1; else sha256sum "$1" | cut -d' ' -f1; fi; }
check_sums() { (cd "$1" && if command -v shasum >/dev/null; then shasum -a 256 -c "$2"; else sha256sum -c "$2"; fi); }

echo "Finish (Tesseract $TSRCT_VERSION, Tesseract Converter $CONV_VERSION)"
OS="$(uname -s)"; ARCH="$(uname -m)"
case "$OS-$ARCH" in
  Darwin-arm64) PLATFORM=darwin-arm64 ;;
  Darwin-x86_64) PLATFORM=darwin-x86_64
    [ "$(sysctl -in sysctl.proc_translated 2>/dev/null || true)" = 1 ] && PLATFORM=darwin-arm64 ;;  # Apple silicon under Rosetta
  Linux-x86_64) PLATFORM=linux-x86_64 ;;
  *) PLATFORM="" ;;
esac
mkdir -p "$DEST/bin" "$DEST/fonts" 2>/dev/null || true

# ---- the fonts (any machine: the Studio packs them into the project wherever it renders)
font() {  # path in google/fonts, file name here, sha256
  local target="$DEST/fonts/$2"
  if [ -f "$target" ] && [ "$(sum256 "$target")" = "$3" ]; then ok "font: $2"; return; fi
  [ "$VERIFY" = 0 ] || { note "font $2 is not installed"; return; }
  local scratch; scratch="$(mktemp -d)"
  curl -fsSL "https://raw.githubusercontent.com/google/fonts/$FONTS_COMMIT/ofl/$1" -o "$scratch/font"
  [ "$(sum256 "$scratch/font")" = "$3" ] || { rm -rf "$scratch"; echo "  font $2 does not match its checksum: not installed" >&2; exit 1; }
  mv "$scratch/font" "$target"; rm -rf "$scratch"
  curl -fsSL "https://raw.githubusercontent.com/google/fonts/$FONTS_COMMIT/ofl/${1%%/*}/OFL.txt" -o "$DEST/fonts/OFL-${2%%.*}.txt" || true
  ok "font: $2 (SIL Open Font License)"
}
font "instrumentserif/InstrumentSerif-Regular.ttf" "InstrumentSerif-Regular.ttf" "$INSTRUMENT_SHA"
font "geist/Geist%5Bwght%5D.ttf" "Geist[wght].ttf" "$GEIST_SHA"

if [ -z "$PLATFORM" ]; then
  note "Tesseract does not run on $OS $ARCH. Run this script on your Mac as well, then set the Mac as the remote"
  note "computer in Settings > Connections (Brains & tools): the Studio finishes the cut there over ssh."
  exit 0
fi

# ---- Tesseract
version_of() { "$1" --version 2>/dev/null | head -1 | awk '{print $2}'; }
TSRCT="$DEST/bin/tsrct"
if [ -x "$TSRCT" ] && [ "$(version_of "$TSRCT")" = "$TSRCT_VERSION" ]; then
  ok "Tesseract $TSRCT_VERSION: $TSRCT"
elif [ "$VERIFY" = 1 ]; then
  for other in "$HOME/Library/Application Support/Tesseract/bin/tsrct" "${XDG_DATA_HOME:-$HOME/.local/share}/Tesseract/bin/tsrct" "$(command -v tsrct || true)"; do
    if [ -n "$other" ] && [ -x "$other" ]; then ok "Tesseract $(version_of "$other") (installed by itself): $other"; break; fi
  done
  [ -x "$TSRCT" ] || note "Tesseract is not installed under $DEST"
else
  command -v curl >/dev/null || { echo "  curl is needed" >&2; exit 1; }
  [ "$PLATFORM" = linux-x86_64 ] && { command -v unzip >/dev/null || { echo "  unzip is needed: sudo apt install unzip" >&2; exit 1; }; }
  NAME="tesseract-$TSRCT_VERSION-$PLATFORM.zip"
  URL="https://github.com/mirage-hq/tesseract/releases/download/v$TSRCT_VERSION"
  scratch="$(mktemp -d)"; trap 'rm -rf "$scratch"' EXIT
  echo "  downloading $NAME (about 95 MB)"
  curl -fL --progress-bar "$URL/$NAME" -o "$scratch/$NAME"
  curl -fsSL "$URL/$NAME.sha256" -o "$scratch/$NAME.sha256"
  check_sums "$scratch" "$NAME.sha256" >/dev/null || { echo "  $NAME does not match its published checksum: not installed" >&2; exit 1; }
  ok "checksum of $NAME"
  mkdir -p "$scratch/x"
  if [ "$OS" = Darwin ]; then ditto -x -k "$scratch/$NAME" "$scratch/x"; else unzip -q "$scratch/$NAME" -d "$scratch/x"; fi
  INSTALLER="$(find "$scratch/x" -maxdepth 3 -name install.sh -type f | head -1)"
  [ -n "$INSTALLER" ] || { echo "  the bundle has no install.sh" >&2; exit 1; }
  # Tesseract's own installer checks the platform and the bundle's checksums, then links bin/tsrct under the folder given
  bash "$INSTALLER" "$DEST" | grep -v ': OK$' || true
  rm -rf "$scratch"; trap - EXIT
  [ -x "$TSRCT" ] || { echo "  Tesseract's installer did not leave $TSRCT" >&2; exit 1; }
  ok "Tesseract $(version_of "$TSRCT"): $TSRCT"
  echo "    Tesseract is free to use under Mirage's terms (not open source; see TERMS.md beside it)."
fi
if [ -x "$TSRCT" ]; then
  echo "    usage counts: $("$TSRCT" telemetry status 2>/dev/null || echo unknown) (left as you set them)"
fi
[ "$PLATFORM" = linux-x86_64 ] && { command -v ffmpeg >/dev/null || note "ffmpeg with libx264 is needed on Linux for MP4: sudo apt install ffmpeg"; }

# ---- Tesseract Converter (MIT)
CONV="$DEST/bin/tsrct-conv"
if [ -x "$CONV" ] && [ "$("$CONV" --version 2>/dev/null)" = "tsrct-conv $CONV_VERSION" ]; then
  ok "Tesseract Converter $CONV_VERSION: $CONV"
elif [ "$VERIFY" = 1 ]; then
  note "Tesseract Converter is not installed: Premiere and After Effects projects wait for it (Resolve does not)"
else
  STEM="Tesseract-Converter-$CONV_VERSION-$PLATFORM"
  URL="https://github.com/mirage-hq/Tesseract-Converter/releases/download/v$CONV_VERSION"
  scratch="$(mktemp -d)"; trap 'rm -rf "$scratch"' EXIT
  echo "  downloading $STEM.zip (about 50 MB)"
  curl -fL --progress-bar "$URL/$STEM.zip" -o "$scratch/$STEM.zip"
  curl -fsSL "$URL/$STEM.zip.sha256" -o "$scratch/$STEM.zip.sha256"
  check_sums "$scratch" "$STEM.zip.sha256" >/dev/null || { echo "  $STEM.zip does not match its published checksum: not installed" >&2; exit 1; }
  ok "checksum of $STEM.zip"
  if [ "$OS" = Darwin ]; then ditto -x -k "$scratch/$STEM.zip" "$scratch/x"; else unzip -q "$scratch/$STEM.zip" -d "$scratch/x"; fi
  [ -x "$scratch/x/$STEM/tsrct-conv" ] || { echo "  the converter bundle has no tsrct-conv" >&2; exit 1; }
  # the bundle's own manifest lists a checksum for each of its files
  python3 - "$scratch/x/$STEM" <<'PY' || { echo "  the converter bundle does not match its manifest: not installed" >&2; exit 1; }
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
files = json.loads((root / "native/manifest.json").read_text())["files"]
bad = [name for name, digest in files.items() if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
sys.exit(1 if bad or not files else 0)
PY
  [ "$("$scratch/x/$STEM/tsrct-conv" --version)" = "tsrct-conv $CONV_VERSION" ] || { echo "  the converter reports another version" >&2; exit 1; }
  mkdir -p "$DEST/converter"
  rm -rf "$DEST/converter/$CONV_VERSION-$PLATFORM"
  mv "$scratch/x/$STEM" "$DEST/converter/$CONV_VERSION-$PLATFORM"
  ln -sfn "$DEST/converter/$CONV_VERSION-$PLATFORM/tsrct-conv" "$CONV"
  rm -rf "$scratch"; trap - EXIT
  ok "Tesseract Converter $CONV_VERSION: $CONV (MIT)"
fi
echo "Done. The Studio finds it at $DEST; open Edit > Timeline > Finish."
