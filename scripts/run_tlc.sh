#!/usr/bin/env bash
# Model-check spec/tla/OrderLifecycle.tla with TLC.
# Safety (crashes, amends, duplicate and late fills) and liveness
# (no crash, no amend, weak fairness) are separate configs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JAR="${TLA_TOOLS_JAR:-$ROOT/.tla/tla2tools.jar}"
URL="https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar"
SHA="ab4694601923fd5ac06452abbf847c366a5054a3d739552085edd6ed986c29ec"

if [[ ! -f "$JAR" ]]; then
  mkdir -p "$(dirname "$JAR")"
  curl -fsSL -o "$JAR" "$URL"
fi
echo "${SHA}  ${JAR}" | sha256sum -c -

META="$(mktemp -d)"
trap 'rm -rf "$META"' EXIT

run_tlc() {
  local cfg="$1"
  local dir="$2"
  mkdir -p "$dir"
  java -Xmx1g -cp "$JAR" tlc2.TLC \
    -workers 2 \
    -cleanup \
    -metadir "$dir" \
    -config "$cfg" \
    "$ROOT/spec/tla/OrderLifecycle.tla"
}

echo "== TLC safety =="
run_tlc "$ROOT/spec/tla/OrderLifecycle.cfg" "$META/safety"
echo "== TLC liveness =="
run_tlc "$ROOT/spec/tla/OrderLifecycleLiveness.cfg" "$META/liveness"
