#!/usr/bin/env bash
#
# build_audiotee.sh — reproducibly build the audiotee binary from source
#
# Design:
#   * Pin a reviewed commit -> the build is exactly the code you audited.
#   * Work in a temp dir OUTSIDE the project -> the repo stays clean either way.
#   * trap EXIT auto-cleans the temp source (--keep-src keeps it for inspection).
#   * Grep for networking code before building; abort if found (a tripwire
#     against a future malicious upstream commit).
#
set -euo pipefail

# ── Pin the reviewed version (no release/tag upstream, so pin a commit) ──
AUDIOTEE_REPO="https://github.com/makeusabrew/audiotee.git"
AUDIOTEE_COMMIT="56ac954369a09318e46b88a6eec33c2d2b0d32a3"   # main @ 2026-06-25

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$PROJECT_ROOT/bin"
KEEP_SRC=0
[[ "${1:-}" == "--keep-src" ]] && KEEP_SRC=1

# ── Work in a temp dir outside the project ────────────────────────────
WORKDIR="$(mktemp -d)"
cleanup() {
  if [[ "$KEEP_SRC" == "1" ]]; then
    echo "🔎 kept source for inspection: $WORKDIR"
  else
    rm -rf "$WORKDIR"
    echo "🧹 removed temp source"
  fi
}
trap cleanup EXIT   # always clean on exit (success or failure), unless --keep-src

SRC="$WORKDIR/audiotee"

# ── 1. Fetch only the pinned commit (by SHA, no unrelated history) ────
echo "📥 fetching pinned audiotee commit $AUDIOTEE_COMMIT ..."
mkdir -p "$SRC"
git -C "$SRC" init --quiet
git -C "$SRC" remote add origin "$AUDIOTEE_REPO"
git -C "$SRC" fetch --quiet --depth 1 origin "$AUDIOTEE_COMMIT"
git -C "$SRC" checkout --quiet FETCH_HEAD
ACTUAL_HEAD="$(git -C "$SRC" rev-parse HEAD)"
if [[ "$ACTUAL_HEAD" != "$AUDIOTEE_COMMIT" ]]; then
  echo "❌ fetched commit ($ACTUAL_HEAD) does not match the pin; aborting" >&2
  exit 1
fi
echo "   HEAD: $ACTUAL_HEAD ✔"

# ── 2. Security tripwire: confirm this version has no networking code ──
# Scan all *.swift (incl. Package.swift / Plugins); .swift-only avoids
# false positives on https:// links in README/docs.
echo "🛡  auditing for networking code ..."
if grep -rniE 'URLSession|URLRequest|dataTask|NWConnection|CFSocket|socket\(|getaddrinfo|https?://' \
     "$SRC" --include='*.swift'; then
  echo "❌ networking code detected; aborting build for safety" >&2
  exit 1
fi
echo "   ✅ no networking code"

# ── 3. Build ──────────────────────────────────────────────────────────
echo "🔨 swift build -c release (first run is slow) ..."
( cd "$SRC" && swift build -c release )

# ── 4. Copy out only the built binary ─────────────────────────────────
# Remove the old file first (fresh inode), then re-sign the copy ad-hoc —
# otherwise an in-place overwrite leaves the kernel's cached cdhash
# mismatched against the new contents, and the first exec is CS_KILLed
# (exit 137).
mkdir -p "$BIN_DIR"
rm -f "$BIN_DIR/audiotee"
cp "$SRC/.build/release/audiotee" "$BIN_DIR/audiotee"
codesign --force --sign - "$BIN_DIR/audiotee"
echo "✅ done: $BIN_DIR/audiotee"
if "$BIN_DIR/audiotee" --help >/dev/null 2>&1; then
  echo "   binary runs ✔"
else
  echo "   ⚠️ binary self-test failed (--help)" >&2
fi
