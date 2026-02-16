#!/usr/bin/env bash
# Build the WebRTC APM C shim shared library.
#
# Usage: bash build_apm_shim.sh
#
# Produces: libwebrtc_apm_shim.so in the same directory as this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${SCRIPT_DIR}/apm_shim.cpp"
OUT="${SCRIPT_DIR}/libwebrtc_apm_shim.so"

echo "Building APM shim: ${SRC} -> ${OUT}"

CFLAGS=$(pkg-config --cflags webrtc-audio-processing-1)
LIBS=$(pkg-config --libs webrtc-audio-processing-1)

set -x
g++ -shared -fPIC -O2 -o "${OUT}" "${SRC}" ${CFLAGS} ${LIBS}
set +x

echo ""
echo "Build successful: ${OUT}"
ls -la "${OUT}"
