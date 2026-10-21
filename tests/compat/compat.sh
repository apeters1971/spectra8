#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 CERN
# Author: Andreas Joachim Peters <andreas.joachim.peters@cern.ch>
#
# Cross-platform format compatibility corpus for the Spectra8 container.
#
#   tests/compat/compat.sh encode  <dir>          build inputs, encode matrix
#   tests/compat/compat.sh decode  <dir>          decode frames, compare
#   tests/compat/compat.sh compare <dirA> <dirB>  encoder determinism report
#
# A corpus directory holds what one platform produced:
#   <dir>/MANIFEST            arch, commit, CLI version, one line per frame
#   <dir>/inputs/<name>       deterministic test inputs (seeded python + files
#                             of the producing commit)
#   <dir>/frames/<name>.<variant>.lhc4
#
# `decode` runs the local CLI over every frame and compares byte-for-byte with
# the input stored next to it, so a corpus produced on x86_64 can be checked
# on aarch64 / Apple silicon and vice versa. Exit 1 on any mismatch.
#
# After a format or BlockType change, add a variants() line below for any new
# --codec so a newly encoded corpus includes it. Frames encoded before the
# change must still decode.
#
# Environment: SPECTRA8            path to the CLI (default: spectra8 in PATH)
#              SPECTRA8_COMPAT_QUICK=1  smaller matrix (no L10/L11, no oracle)
set -euo pipefail
exec 2>&1

SPECTRA8=${SPECTRA8:-spectra8}
mode=${1:-}; dir=${2:-}
[ -n "$mode" ] && [ -n "$dir" ] || { echo "usage: $0 encode|decode <dir> | compare <dirA> <dirB>" >&2; exit 2; }
arch=$(uname -m)
here=$(cd "$(dirname "$0")" && pwd)
src_root=$(cd "$here/../.." && pwd)

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
if [ "$(uname -s)" = Darwin ]; then fsize() { stat -f %z "$1"; }; else fsize() { stat -c %s "$1"; }; fi
cli_version() { local h; h=$("$SPECTRA8" --help); echo "${h%%$'\n'*}"; }

# ---------------------------------------------------------------------------
# inputs: deterministic synthetic data (seeded python) + files of this commit
# ---------------------------------------------------------------------------
make_inputs() {
  local out=$1
  mkdir -p "$out"
  python3 - "$out" <<'PY'
import math, os, random, struct, sys
out = sys.argv[1]
def w(name, b):
    with open(os.path.join(out, name), "wb") as f: f.write(b)

rnd = random.Random(20260916)
# float32 detector-like page: smooth signal + noise + occasional spikes
n = 65536
vals = []
for i in range(n):
    v = 100.0 * math.sin(i / 700.0) + 3.0 * math.sin(i / 13.0) + rnd.gauss(0, 0.5)
    if rnd.random() < 0.002: v += rnd.uniform(500, 5000)
    vals.append(v)
w("f32_signal.bin", struct.pack("<%df" % n, *vals))
# float64: energies with an exponential distribution, some exact zeros
n = 32768
vals = [0.0 if rnd.random() < 0.05 else rnd.expovariate(1 / 37.5) for _ in range(n)]
w("f64_energy.bin", struct.pack("<%dd" % n, *vals))
# int32 timestamps: monotonically increasing, small deltas
n = 65536
t = 1_700_000_000
vals = []
for _ in range(n):
    t += rnd.choice([1, 1, 1, 2, 3, 25, 400]); vals.append(t)
w("i32_timestamps.bin", struct.pack("<%di" % n, *vals))
# int16 ADC samples: baseline + noise + pulses
n = 131072
vals = []
for i in range(n):
    v = 512 + int(rnd.gauss(0, 3))
    if (i % 4096) < 60: v += int(2000 * math.exp(-(i % 4096) / 15.0))
    vals.append(max(-32768, min(32767, v)))
w("i16_adc.bin", struct.pack("<%dh" % n, *vals))
# uint64 ids: sparse, sorted, high bits constant
n = 16384
vals = sorted(rnd.randrange(1 << 40, (1 << 40) + (1 << 30)) for _ in range(n))
w("u64_ids.bin", struct.pack("<%dQ" % n, *vals))
# interleaved records: int32 id, float32 pt, int16 charge, u8 flags, u8 pad
recs = []
for i in range(24576):
    recs.append(struct.pack("<ifhBB", 1000 + i, rnd.expovariate(1 / 12.0), rnd.choice([-1, 1]), rnd.randrange(8), 0))
w("records_12b.bin", b"".join(recs))
# text: pseudo log lines
lines = []
for i in range(6000):
    lines.append("2026-09-16T%02d:%02d:%02d run=%d evt=%d det=%s status=%s size=%d\n" % (
        (i // 3600) % 24, (i // 60) % 60, i % 60, 400000 + i // 1000, i,
        rnd.choice(["TRK", "CAL", "MUON", "PIX"]), rnd.choice(["ok", "ok", "ok", "warn", "retry"]), rnd.randrange(1 << 20)))
w("text_log.txt", "".join(lines).encode())
# incompressible + tiny edge cases
w("random_16k.bin", bytes(rnd.getrandbits(8) for _ in range(16384)))
w("tiny_400_f32.bin", struct.pack("<100f", *[rnd.uniform(-1, 1) for _ in range(100)]))
w("one_byte.bin", b"\x2a")
PY
  # files of the producing commit (text / source / png)
  cat "$src_root"/README.md "$src_root"/docs/*.md > "$out/text_docs.md" 2>/dev/null || cp "$src_root/README.md" "$out/text_docs.md"
  head -c 1048576 "$src_root"/src/encoder.cpp > "$out/source_cpp.txt"
  if [ -f "$src_root/docs/spectra8-logo.png" ]; then cp "$src_root/docs/spectra8-logo.png" "$out/logo.png"; fi
}

# ---------------------------------------------------------------------------
# encode matrix: "<variant-name>|<cli args>"
# ---------------------------------------------------------------------------
variants() {
  cat <<'EOF'
beam1|--codec beam -1
beam5|--codec beam -5
beam5f|--codec beam -5 --filter
beam9crc|--codec beam -9 --seek-table --block-crc --content-crc
beam5b16|--codec beam -5 --block-log=16 --filter
prism5|--codec prism -5
prism5f|--codec prism -5 --filter
crystal1|--codec crystal -1
crystal5|--codec crystal -5
crystal5f|--codec crystal -5 --filter
crystal9f|--codec crystal -9 --filter
mosaic5|--codec mosaic -5
mosaic5f|--codec mosaic -5 --filter
flash1|--codec flash -1
flash5|--codec flash -5
laser5|--codec laser -5
zstd5|--codec zstd -5
lzma5|--codec lzma -5
bzip3_5|--codec bzip3 -5
auto5|--codec auto -5
auto5f|--codec auto -5 --filter
auto9f|--codec auto -9 --filter
EOF
  if [ "${SPECTRA8_COMPAT_QUICK:-0}" != 1 ]; then
    cat <<'EOF'
ultra|--codec beam --ultra
crystal11f|--codec crystal --level=11 --filter
oracle5|--codec oracle -5
oracle5f|--codec oracle -5 --filter
EOF
  fi
}

do_encode() {
  local inputs=$dir/inputs frames=$dir/frames manifest=$dir/MANIFEST
  echo "== compat encode on $arch with $(cli_version)"
  if [ -e "$dir" ] && [ ! -f "$manifest" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "refusing to overwrite $dir: exists and is not a compat corpus" >&2; exit 2
  fi
  rm -rf "$dir"; mkdir -p "$frames"
  make_inputs "$inputs"
  {
    echo "# spectra8 compat corpus"
    echo "# arch=$arch os=$(uname -s) commit=${CI_COMMIT_SHA:-$(git -C "$src_root" rev-parse HEAD 2>/dev/null || echo unknown)}"
    echo "# cli=$(cli_version)"
    echo "# variant input in_bytes frame_bytes sha256(frame)"
  } > "$manifest"
  local n=0 fail=0 name args in base out
  while IFS='|' read -r name args; do
    [ -n "$name" ] || continue
    for in in "$inputs"/*; do
      base=$(basename "$in"); out="$frames/$base.$name.lhc4"
      # shellcheck disable=SC2086
      if "$SPECTRA8" -z -q -f $args "$in" "$out" 2>"$out.err"; then
        rm -f "$out.err"
        printf '%s %s %d %d %s\n' "$name" "$base" "$(fsize "$in")" "$(fsize "$out")" \
          "$(sha256 "$out")" >> "$manifest"
        n=$((n + 1))
      elif grep -qiE "not (linked|available|built|compiled)|unsupported|unknown codec" "$out.err"; then
        # backend not linked into this build (zstd/lzma/bzip3): not a failure
        echo "skip  $name $base: $(head -1 "$out.err")"; rm -f "$out" "$out.err"
      else
        echo "FAIL  encode $name $base: $(head -1 "$out.err")"; fail=$((fail + 1)); rm -f "$out" "$out.err"
      fi
    done
  done < <(variants)
  echo "== encoded $n frames into $dir ($fail encode failures)"
  [ "$fail" = 0 ]
}

tmp=""
trap '[ -n "$tmp" ] && rm -rf "$tmp"' EXIT

do_decode() {
  local inputs=$dir/inputs frames=$dir/frames f base in out n=0 fail=0 src
  [ -f "$dir/MANIFEST" ] || { echo "FAIL  $dir is not a compat corpus (no MANIFEST)"; exit 1; }
  src=$(sed -n 's/^# arch=\([^ ]*\).*/\1/p' "$dir/MANIFEST" | head -1)
  tmp=$(mktemp -d)
  echo "== compat decode on $arch with $(cli_version): corpus from ${src:-?} ($(sed -n '2p;3p' "$dir/MANIFEST" | tr '\n' ' '))"
  for f in "$frames"/*.lhc4; do
    [ -e "$f" ] || continue
    # "<input>.<variant>.lhc4" -> strip ".lhc4" and the variant component
    base=$(basename "$f" .lhc4); in="$inputs/${base%.*}"; out="$tmp/out"
    if ! "$SPECTRA8" -d -q -f "$f" "$out" 2>"$tmp/err"; then
      echo "FAIL  decode ${src}->$arch $(basename "$f"): $(head -1 "$tmp/err")"; fail=$((fail + 1)); continue
    fi
    if ! cmp -s "$in" "$out"; then
      echo "FAIL  mismatch ${src}->$arch $(basename "$f"): decoded $(fsize "$out") bytes, expected $(fsize "$in")"
      fail=$((fail + 1)); continue
    fi
    n=$((n + 1))
  done
  echo "== compat decode ${src}->$arch: $n frames ok, $fail failed"
  [ "$fail" = 0 ]
}

do_compare() {
  local a=$dir b=${3:-} same=0 diff=0 name base _ sum osum
  [ -n "$b" ] && [ -f "$a/MANIFEST" ] && [ -f "$b/MANIFEST" ] || { echo "usage: $0 compare <dirA> <dirB>" >&2; exit 2; }
  echo "== encoder determinism $(sed -n 's/^# arch=\([^ ]*\).*/\1/p' "$a/MANIFEST") vs $(sed -n 's/^# arch=\([^ ]*\).*/\1/p' "$b/MANIFEST") (informational)"
  while read -r name base _ _ sum; do
    case $name in \#*|'') continue;; esac
    osum=$(awk -v n="$name" -v b="$base" '$1==n && $2==b {print $5}' "$b/MANIFEST")
    [ -n "$osum" ] || continue
    if [ "$sum" = "$osum" ]; then same=$((same + 1)); else diff=$((diff + 1)); echo "   differs: $name $base"; fi
  done < "$a/MANIFEST"
  echo "   identical frames: $same, differing frames: $diff"
}

case $mode in
  encode)  do_encode ;;
  decode)  do_decode ;;
  compare) do_compare "$@" ;;
  *) echo "unknown mode $mode" >&2; exit 2 ;;
esac
