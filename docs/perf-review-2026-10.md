# spectra8 encoder/decoder review — ratio and speed headroom (October 2026)

Scope: every encoder and decoder path in `src/` (Flash, Beam LZ, entropy
coders, Prism BWT, Mosaic/Oracle, Crystal static/BWT/CM, byte filters, frame
layer), with reference numbers from `bench --verdict`, `bench --verdict
--no-filter` and `bench --verdict --text`, plus small out-of-tree experiments
to quantify the candidates that looked promising in the code. The review
itself changed no source files; the follow-up pass that implemented its
proposals is recorded in §0 (the format was still in beta, so the Flash
block layout and the entropy table headers changed — frame `kVersion` is now
2 — with no compatibility paths kept).

Machine / build: Apple M-series (arm64), macOS, `clang -O3 -std=c++20`,
`SPECTRA8_NATIVE=OFF`, no LTO, zstd 1.5.7 / xz 5.8.4 / bundled bzip3 linked.
All numbers single-threaded. Every "experiment" below was built against
`build/libspectra8.a` from a harness in `/tmp` (the harness includes
`cli/bench.cpp` so it uses the exact verdict datasets and the exact external
filter staging, `verdict_ext_prepare`, RLE off).

---

## 0. Implemented after the review (2026-10-13)

Everything in the §1 table except the two "leave as is" rows was
implemented. Numbers are from the same harnesses as the review (verdict
streams = the 22 datasets after the external filter, 128 KiB blocks, single
thread, arm64), before → after; §2 holds the re-run `bench --verdict`
tables.

| § | Change | Files | Measured |
|---|--------|-------|----------|
| 1, 2 | **Flash**: token `adv:3 \| wide:1 \| ml:4` with `adv = literal length + wide`, 1-byte offset when ≤ 255 else 2 bytes (`kMaxOffset` 65535), LZ4-style inc32/dec64 overlap copy (16-byte steps for long runs), short-token fast path (one 16-byte literal copy + one 16+8-byte match copy when `adv<7 && ml<15`), hash table sized to the input (2^10..2^14 slots, one per ~8 bytes) and zeroed per call | `flash.hpp`, `format.hpp` (`kVersion` 2) | verdict streams (filtered): ratio 2.405 → **2.491**, decode 4452 → **~6200 MB/s**, encode unchanged; raw 128 KiB blocks (`--no-filter` data, same-process A/B): ratio 1.366 → **1.408**, decode **+8 %** (4030 → 4367 MB/s), encode +1 % (ASan fuzz clean). Two things were learnt on the way, see §3.5: storing `lit_len + wide` instead of `lit_len` takes one add off the decoder's loop-carried input-pointer chain (**+15 % on short-match data**; with `ll:3` alone the decoder was −7 % vs the old 4-bit field on raw data), and a persistent table with a base offset (to skip the per-call memset) made the stale-entry branch unpredictable on noise (**−25 % encode** on incompressible input) — zeroing a right-sized table is both faster and simpler |
| 3 | **Beam serialize**: branchless MSB-first bit packer (`MsbPacker`, 4 symbols per flush; quad streams packed in one interleaved loop into a scratch buffer), two-queue Huffman length build (no heap), token arrays written by index instead of `push_back`, one-pass `off_low` histograms, RLE trial skipped when its floor already loses, `build_lengths` without a vector allocation | `bits.hpp`, `huffman.cpp`, `entropy.cpp`, `encoder.cpp` | Huffman encode 1086 → **2413 MB/s**; L1 serialize 538 → **1225 MB/s**; Beam L1 encode 424 → **703 MB/s**, L2 422 → 690 (ratio unchanged) |
| 4 | **Level ladder**: L3/L4 run a zstd-"dfast" parser (8-byte long table probed first, 4-byte short table checked against a long hit at ip+1, backward match extension — the latter also added to L1/L2); `hash_log` 16 / 18; the chain/row finder starts at L5. Also fixed a latent `insert_match` double insert for 2-byte matches | `encoder.cpp` (`parse_block_fast_impl<kDouble>`), `match_finder.cpp/.hpp` | verdict: L3 140 → **430 MB/s** (ratio 1.964 → 1.952, between L2 1.924 and L5 1.964), L4 112 → **430 MB/s** (1.967 → 1.952); dickens L3 55 → 200 MB/s (2.648 → 2.491), L4 43 → 200 (2.677 → 2.510); L1/L2 +0.1–0.2 % ratio from the backward extension. The ladder is monotonic in speed and ratio from L1 to L5 |
| 5 | **Crystal literal trial in Beam L5–8**: the screen's `std::log2` loops replaced by a float x·log₂x table (348 → 168 µs/block) and its threshold raised from "order-1 bound ≥ 0.87× required gain" to **≥ 2×** for the Crystal-L1 trial (every block that ended up taking Crystal on verdict/mozilla had ≥ 3.3×, median realised 0.7–0.85 of the bound). L9+/ultra (Crystal-L5 trial) keep the 0.87× screen: Crystal's text pipeline is not bounded by order-1, and the 2× screen cost 2 % of the L9/L11 ratio on enwik8 (3.101 → 3.035, 3.420 → 3.341) before this was split | `encoder.cpp` | L5 encode verdict 42 → **47 MB/s**, mozilla 51 → **66**, samba 109 → **144**, ratio identical on verdict/dickens/mozilla/samba/enwik8 at L5–L7 (the Crystal-L1 trial never wins on text, so the gate is free there); enwik8 L9 3.1008 → 3.1024, L11 3.4201 → 3.4225 (headers). The remaining L5 cost is Crystal's own L1 model search (~1.2 ms per 128 KiB block: 50 % greedy pre-binning, 30 % interval costs, 15 % boundary refinement). `top_singles` 2 → 1 would give another +20 % for −0.2 % ratio and was **not** taken (it would also change Crystal L1 proper) |
| 7 | **Crystal block size**: default 1 MiB → **2 MiB for L1–8, 4 MiB for L9** (CM tiers stay at 1 MiB) | `spectra8.cpp` (`crystal_block_log`) | 8 MiB inputs, L5: dickens 3.720 → **3.876**, webster 4.805 → **5.034**, xml 12.20 → 12.60, osdb 3.635 → **3.795**, sao 1.524 → 1.544, decode within noise (275–470 MB/s); L9 at 4 MiB: dickens 3.734 → **4.018**, webster 4.836 → 5.264, osdb 3.658 → 3.955, decode −10…−25 % (inverse BWT leaves L2). 8 MiB blocks would add another 2 % on text but halve decode. `kBwtGain` 3 % → 1.5 % / 0 % at L9: identical output on all nine corpus files — left at 3 % |
| 8 | **Compact entropy headers**: Huffman code lengths as a deflate-style run-length stream under a 15-symbol `SmallCode` (encoder keeps the 4-bit packing when it is smaller); rANS order-0 and order-1 tables as frequency bit-length classes under a 14-symbol `SmallCode` plus mantissa bits (O1 dropped its dense/sparse split; one code shared by the 8 classes) | `huffman.hpp/.cpp`, `tans.cpp`, `entropy.cpp` (O1 size estimate), `tests/api_test.cpp` (golden hashes) | 256 KiB blocks: verdict **+0.23…+0.45 %**, samba +0.18…+0.35 %, dickens/mozilla +0.1 %; serialize speed unchanged (the header is built once in `encoded_size` and reused) |
| 10, 12 | `crystal_bwt` walk: `walk_chains` + skewed scratch lines + final `memcpy` replaced by the shared `bwt_decode::walk_k_tab<16/8>` staging-line walker (`NavFused`) | `crystal_bwt.cpp` | decode neutral on arm64 (within 1 %), one less copy of the block; `lf_prefetch` left as measured |
| 11 | `delta_inv` prefix sums: NEON kernels for 16/32-bit elements; both NEON and SSE2 keep only `carry += broadcast(last lane)` on the loop-carried path (the SSE2 kernel used to round-trip the carry through a GPR) | `transform.hpp` | arm64: w=2 7.5 → **30 GB/s**, w=4 17 → **34 GB/s** (SSE2 kernel verified via sse2neon) |

Not changed, as the review recommended: Crystal `kPrec`/`kSymbolBits`/
`kTableBits`/`kBoundaryBits` (§8.2), Flash hash size/skip (§3.2),
`lf_prefetch` (§6).

Tests: `ctest` 4/4 (roundtrip fuzz, API incl. new golden payload hashes,
stream decoder, logfile).

### 0.1 Second round — Crystal lane modes (static coder format change)

Follow-up to §8.1's "I do not see a decoder-side change that would move this
materially without changing the model". The format was still open, so the
decode steps of the static coder were measured first (instrumented copy of
`crystal_static.cpp`, verdict streams, Crystal L5, the 32 KiB BWT-gate
window calls excluded):

| | filtered streams (3.97 b/B) | raw records (2.95 b/B) |
|---|---|---|
| bytes in constant lanes (`nctx == 1`, one symbol) | 0 % | **29.4 %** |
| bytes in lanes ≥ 7.9 b/B (noise to the model) | 0.8 % | **18.0 %** |
| bytes in runs ≥ 2 of the lane's top symbol | 31.5 % (0.715 steps/byte if a run were one step) | 39.3 % (0.632) |
| bytes in contexts with ≤ 16 symbols | 16.6 % | 52.3 % |

Three changes were proposed; two were implemented, the third was prototyped
and dropped on its measurement.

**Implemented — lane modes (`LaneMode` in `crystal_static.cpp`).** Every
lane of the lattice is now one of *coded* (as before), *constant* (every
byte of the lane equal: the model carries 2 mode bits + the value, no
table, no contexts) or *raw* (the lane's bytes travel verbatim in a section
after the model; no table, no contexts). The decoder fills constant and raw
lanes before the rANS pass — with ≥ 2 raw lanes and a power-of-two stride
through a 16-record byte interleave (`zip1/zip2` / `punpck` via the
GCC/clang vector extension; the strided byte scatter is store-bound at ~2
stores per cycle and was 14 % of the decode time of a float block with 6
raw lanes of 8), otherwise a periodic-pattern doubling `memcpy` plus the
scatter — and the lockstep loop only shifts their bytes into the byte
history (`skip_one`: 3 instructions per byte against ~30 for a coded byte;
in the AVX2 `V8` body `v8_skip_step`: no gather for a constant lane, one for
a raw lane, against five). The lane cycle, the stream split and the bounds
checks are unchanged, so the `LaneBody`/`GenericBody`/`S1Body`/`V8`
structure and the BMI2 dispatch are untouched; the stride-1 order-0 context
is hoisted out of `S1Body` on the way (`s1_ctx`: the byte stores could
alias the lane table, so the compiler reloaded it per byte). Encoder:
constant lanes are found by a strided scan before the model search (which
they then skip), raw lanes after the count pass when the lane's estimated
coded size (count entropy + table header) is not below 8 bits per byte —
strict, so the size can only shrink; slack of 1/2/5 % made no difference on
the verdict data (noise lanes sit far above 8 b/B, everything else far
below). The skipped lanes' bytes index a sentinel frequency row with
`f = kM`, which leaves the rANS state bit-exactly unchanged, so
`encode_streams` needs no special case. A block with no coded lane writes
K = 1.

Measured (same-process A/B on the 22 verdict datasets, 128 KiB, Crystal L5,
arm64; `/tmp/spectra8exp/crydec.cpp`):

| | before | after |
|---|---|---|
| raw records: size | 975 228 | **971 423** (−0.4 %) |
| raw records: decode | 539 MB/s | **~790 MB/s** (+45 %); columnar integers 700–740 → 1250–1650, `gau-f32` 805 → 1740, `gau-f64` 649 → 1675 |
| filtered streams: size | 969 070 | 969 078 (+8 bytes, the mode bits) |
| filtered streams: decode | 596 | 602 (noise) |

`bench --verdict --no-filter`: Crystal L1 2.758 / 513 MB/s → **2.769 / 762**,
L5 2.957 / 506 → **2.968 / 711**, L9 2.977 / 474 → **2.989 / 612**; the
codecs that route literals or tiles through the static coder follow
(Beam L5 2.541 → 2.547, L9 2.670 → 2.677, ultra 2.769 → 2.776, Mosaic L9
2.923 → 2.934, Oracle L9 2.967 → 2.979). `bench --verdict` (filtered):
Crystal L1 3.040 → **3.048**, L5 3.123 → **3.133**, L9 3.133 → **3.143**,
Beam L5 2.972 → 2.976, Mosaic L5 3.074 → 3.080, Oracle L5 3.084 → 3.092;
decode speeds there are within this machine's ±10 % run-to-run noise (the
same-process A/B above is the measurement). `bench --verdict --text`
(enwik8): Crystal L1/L5/L9 ratios identical (3.551 / 3.644 / 3.812), decode
within noise — the BWT pipeline's symbol streams have no constant or noise
lanes.

**Prototyped and dropped — run symbols for stride-1 lanes.** The plan was a
run code (an unused symbol value of the lane) for 2–8 repeats of the lane's
top symbol, per-stream output cursors and streams split by symbol count.
The rANS loop is throughput-bound, not latency-bound (16 streams hide the
~20-cycle chain; `S1Body` runs at ~22 instructions per byte on AArch64), so
what matters is instructions per byte, not steps. A run-aware `S1Body`
(cursor per stream, run length read from spare bits of the `freq|cum`
entry, 8-byte broadcast store, history select) was built with the length
fixed at 1, i.e. identical output on today's payloads, and measured
**−15 %** on the filtered streams (649 → 554 MB/s). With the measured
0.73 steps/byte the ceiling is therefore ~+16 % before the tail/fix-up
logic, realistically +5–10 %, for a format change and the same treatment in
the AVX2 body (whose `outp[i][step]` stores would have to become
per-stream cursors — no scatter in AVX2). Not worth it; the pair-coding
variant measured earlier (+7 %) falls under the same argument. What would
move the filtered path is a different symbol coder (tANS: one table load
returning symbol, bit count and next state, ~12 instructions per symbol
against rANS's two dependent loads + multiply + renormalise), which is a
rewrite of the static coder's encoder, decoder and vector body, not a
tuning.

Tests: `ctest` 4/4; ASan/UBSan roundtrip + truncation/corruption fuzz of
`build_crystal_static`/`decode_crystal_static` over strides 1–16, lane kinds
(constant, noise, skewed, lattice-delta, copy), sizes 1–200 000 and levels
1–9, native and with `-DCRY_V8_EMU` + `SPECTRA8_CRYSTAL_X86=avx2` (the vector
body's logic on this host; 180 vector-body calls, 0 failures); x86-64
compile-checked with the baseline ISA, `-mavx2 -mbmi2 -mpopcnt` and
`-march=x86-64-v3` (Rosetta is not installed on this machine, so no x86
execution).

### 0.2 Third round — executables (2026-10-15)

Trigger: a `bench --dir` over x86-64 shared libraries. lzma P6 3.66 at
26 MB/s encode; zstd L19 3.37; Beam L11 3.14, Prism L11 3.11 at 17 MB/s
decode, Crystal L9 2.51; `auto` picked lzma on 912 of 934 blocks and
finished at 3.65 — at 6 MB/s, the sum of running every encoder. Machine
code's redundancy is short-range and position-dependent (the same call
sequence or prologue with one operand byte changed, 4-byte-aligned tables);
LZMA models that directly (literal contexts on `pos & 3`, matched-literal
coding after a near-miss, four repeat offsets with a 1-byte short-rep,
adaptive binary coding), the BWT family tops out at 3.1–3.4 and the record
coders have no model for it at all. So the fix is routing, not modelling.

Changes (`transform.hpp`, `pack_filter.hpp`, `spectra8.cpp`, CLI):

* **Executable detector** `detail::detect_exec_arch`: ELF / Mach-O / PE
  header (e_machine / cputype / Machine → x86 or AArch64), else content on
  8 × 8 KiB windows spread over the block. x86: `0xE8/0xE9` followed by a
  displacement with top byte 0x00/0xFF, varied from hit to hit (a u64
  timestamp column whose byte 2 walks through 0xE8 sits four bytes before a
  zero lane and passed the range test with one repeated value — the
  verdict's `ts-u64` lost its `dshuf64` to `bcj_x86` in the first cut),
  either ≥ 2/3 of the `0xE8` bytes (scalar code) or ≥ 20 % with a REX
  prefix density ≥ 8 % (SIMD-heavy x86-64: BLAS kernels have `0xE8` as a
  ModRM byte in half the cases, REX at 11–23 % of all bytes against 1–6 %
  in data, text and random bytes). AArch64: ≥ 1/40 of the aligned words
  decode as BL / ADRP / RET / frame STP-LDP, ≥ 1/256 as BL, ≥ 90 % of the
  BL targets within 16 MiB (1/8 of random BL candidates pass that). Order-0
  entropy floors 4.5 (x86) and 3.0 bits (AArch64 words sit at 4.0–4.5).
  None of the 26 verdict datasets, enwik8, ROOT files, PNG, tar.gz,
  README or HTML triggers; 216 of 418 64 KiB pages of the x86 set and 74
  of 95 256 KiB pages of OpenBLAS do (the rest are symbol / string /
  relocation pages).
* **`detect_filter` takes the BCJ outright on code** and
  `detect_framed_filter` returns None: the adjacency proxy cannot value a
  branch-target rewrite (it leaves local statistics alone and creates
  long-range repeats), and before this `bcj_x86` was never chosen on a real
  binary — the detector proposed `dict32`/`dict64`/`shuffle64` and the raw
  block won every time, so `+flt` did nothing for executables.
* **`bcj_x86` with the xz range rule**: only displacements with top byte
  0x00/0xFF are converted, sign-folded to 25 bits; `0xE8` bytes inside data
  are left alone so repeated data stays repeated (unconditional: zstd L5
  3.679 / lzma 4.642; range-tested: 3.739 / 4.723 on the x86 set). SWAR
  8-byte skip over non-opcode bytes (`std::countr_zero` on the zero-byte
  mask), 2.7 → 4.4–5.8 GB/s.
* **`bcj_arm64` (filter id 40)**, xz's `--arm64` transform: BL imm26 and
  ADRP imm21 (|imm| < 2^17 pages, sign-folded to 18 bits so the decoder's
  range test sees exactly the converted words) become absolute. 6.7–7.4
  GB/s word loop. Public `FilterId::BcjArm64`, `StatsFilterKind` shifted
  (`Stored` 41), CLI name `bcj_arm64`.
* **Auto prunes the race to zstd + lzma on executables** (when lzma is
  available and in `auto_codecs`; the gain rule still decides between the
  two). CLI: a whole-file executable with no `--codec` takes lzma L9 with
  the BCJ filter, the same way text takes Prism; `--filter` is now accepted
  with `--codec zstd|lzma|bzip3` (L4EF frame), which the library always
  supported and the bench already used.
* **No raw keep-smaller trial for a BCJ stage** in the external-codec
  frame: the BCJ comes from the detector, not the proxy, and the raw block
  won on 3 of 41 / 3 of 34 files by 0.005 % in total (`fbtest.cpp`), while
  the trial doubled the lzma time. The in-frame Spectra8 path keeps its 16 KiB
  L1 screen + raw trial unchanged.

Measured (`bench --dir`, L5, `--auto-gain 1`, same session, before → after;
ratio columns from the full run, speeds from a quiet zstd/lzma/auto re-run):

| set | zstd L5 | zstd L19 | Beam L5 | Crystal L5 | lzma P6 | bzip3 | auto L5 g1% | lzma P5 enc | auto enc |
|---|---|---|---|---|---|---|---|---|---|
| 41 x86-64 `.so` (manylinux numpy/scipy, 24.8 MiB) | 3.525 → 3.739 | 4.102 → 4.338 | 3.317 → 3.532 | 3.005 → 3.158 | 4.485 → 4.740 | 3.956 → 4.207 | 4.478 → 4.723 | 6.3 → 4.1 MB/s | 1.1 → 4.0 MB/s |
| 34 arm64 Mach-O bundles (XRootD, 4 MiB) | 3.976 → 4.394 | 4.508 → 4.929 | 3.986 → 4.319 | 3.356 → 3.794 | 5.068 → 5.508 | 4.231 → 4.682 | 5.060 → 5.492 | 7.0 → 7.5 MB/s | 0.7 → 7.0 MB/s |

Auto now runs at the lzma row's speed. The x86 lzma row itself slowed
because xz compresses BCJ'd x86-64 ~30 % slower than the raw bytes (`xz -5`
0.155 s vs `xz -5 --x86` 0.206 s on a 1 MiB `.so`: more and longer matches
to walk); on the user's machine, where plain lzma encoded at 26 MB/s, auto
moves from 6 to ~18 MB/s with a 5 % better ratio. Decode speed of the fast
rows drops by the BCJ inverse pass (0.2–0.4 ms/MiB): irrelevant against
lzma, visible on Flash/zstd/Beam L1.

*Note (Laser)*: The subsequent introduction of **Laser** (`Codec::Laser = 10`, `--laser`)
provides Spectra8's native LZMA2-class binary compressor with 5-stream decoupled range
decoding, matching or beating LZMA P6 ratios while decoding 26–45% faster (120–185 MB/s)
and scaling to ~800 MB/s multi-threaded.

Tests: `ctest` 4/4 (api_test: BCJ x86 range rule, BCJ arm64 BL/ADRP
values, roundtrips at awkward sizes and at the ADRP range edge, detector
positives for ELF/Mach-O headers and headerless x86-64/AArch64 streams,
negatives for random, text, and a u64 column of 232); verdict re-run (only
`code-x86` detected as code, every other dataset keeps its filter); x86-64
compile-checked (`-arch x86_64`, baseline and `-mavx2 -mbmi2`).

Not done, noted for later: Beam L11 decodes these libraries at 371 MB/s
against 3.7 GB/s at L5 with the same decoder — the L9+ literal trial runs
Crystal L5 with a 1.5 % bar, and Crystal's BWT text pipeline on symbol /
string literals is the likely culprit (see §4); worth one `--stats` check.

---

## 1. Summary — what is worth doing

Ordered by (expected gain) / (risk + effort). "Measured" means the number
comes from an experiment on the verdict corpus; "estimated" means it comes
from reading the code.

| # | Path | Change | Effect | Cost / risk | Status |
|---|------|--------|--------|-------------|--------|
| 1 | **Flash decode** | LZ4-style overlap copy (inc32/dec64 pattern, 8-byte stores) for offsets 2..15 instead of the per-byte loop | **+35 % aggregate decode** (4452 → 5988 MB/s on the verdict streams; 2–3× on f32/f64 planes) | ~25 lines in `decode_flash_block`, bit-exact format, portable (plain `memcpy`) | measured |
| 2 | **Flash ratio** | Token `ll:3 | ml:4 | offflag:1`, offset is 1 byte when ≤ 255 else 2 bytes | **+3.6 % Flash ratio** (2.405 → 2.491 verdict) with a branchless 2-byte load + mask in the decoder | format change (new Flash block version), encoder trivially adapted | measured (simulated on the real parse) |
| 3 | **Beam L1/L2 encode** | Cut fixed per-array overhead in `encode_array` (token arrays), faster Huffman bit packer | serialize is **73 % of L1 encode time** (538 MB/s vs 1.46 GB/s parse); 3 token arrays of ~4.6 K symbols cost 52 µs/block, literal Huffman runs at 745 MB/s | encoder-only, no format change | measured |
| 4 | **Beam L3/L4 tuning** | L3/L4 (`lazy`, chain 16/32) parse at 175 MB/s and reach the *same* ratio as the L5 parse (chain 8, no lazy, 265–390 MB/s) on the verdict corpus | level ladder is non-monotonic: L5-without-Crystal-literals = L3 ratio at 1.7× the speed | re-tune `params_for_level` 3/4 (e.g. L3 = L5 params, L4 = chain 16 greedy) | measured |
| 5 | **Beam L5 encode** | The Crystal literal trial (`encode_array_crystal`, cry_level 1) is 85 % of L5 encode time: 214 → 42 MB/s for +2.6 % ratio | document / expose; consider gating it to L6+ or running Crystal's L1 search with a smaller `kAnalysisMax` sample for literals | tunable only | measured |
| 6 | **Crystal** | `kPrec` 12 → 13, `kSymbolBits`, `kTableBits`, `kBoundaryBits` sweeps | all **≤ ±0.05 %**; `kPrec`=13 is a net loss (−0.16 % ratio, −1.5 % decode) | — | measured: **leave as is** |
| 7 | Crystal text mode | 8 MiB enwik8 at L5 (1 MiB blocks): Crystal 3.498 vs bzip3-768k 3.722 vs Prism L5 3.522; decode 232 MB/s | the static coder's cost on BWT output (~6 % behind an adaptive coder at equal block size) is the design trade; the cheap lever is the default `block_log` ladder for Crystal on text (kBwtDefaultBlockLog style) | tunable | measured |
| 8 | Entropy headers | uncompressed Huffman (≤131 B) / rANS (≤515 B) tables | only **0.33–0.43 %** of Beam output on 128 KiB blocks; halving them gives +0.2 % | low priority (I expected more before measuring) | measured |
| 9 | Flash encode | bigger hash (2^15..2^17), slower skip | +0.1–0.2 % ratio for −4…−10 % encode speed | **not** worth it | measured |
| 10 | `crystal_bwt::walk_chains` | writes K skewed scratch lines then `memcpy`s n bytes to `dst` | one extra pass over the block (~0.2–0.5 % of BWT decode) — `bwt_decode.hpp` already has the 64-byte staging-line variant | small refactor | estimated |
| 11 | `transform.hpp` `delta_inv` | SSE2 prefix sum only; NEON falls back to scalar | not on the critical path for the verdict rows (filters are external there), but affects in-frame filter decode on ARM | small | estimated |
| 12 | `bwt_decode.hpp` `lf_prefetch(nav + np)` | prefetches the address that is loaded on the very next iteration — no distance; on x86 it is a wasted uop, on ARM compiled out | remove or prefetch 1 step ahead via a second chain | trivial | estimated |

Items 1 and 2 answer the "low-hanging fruit for Flash" question: #1 is pure
speed with zero format risk; #2 is the only ratio lever Flash has without an
entropy stage (72 % of Flash output is raw literals; offsets are 18 % of the
output and 38 % of them fit in one byte).

---

## 2. Reference benchmarks

Each table shows the review baseline (`before`, the tree the review was
written against) and the re-run after the §0 changes (`after`, same
machine, same build flags, `bench --verdict` default budget). Ratios are
exact; throughput is the better of two runs. The `after` runs were taken on
a loaded desktop (an IDE helper at 100 % of a core and a VM at ~75 % through
the whole session): the untouched external rows (zstd, lzma, bzip3) are
3–15 % slower in `after`, and the Beam/Crystal decode columns moved by the
same amount. A same-process A/B harness (`abdec`, best of 12) puts Beam L1/L5/L9
and Crystal L5 decode within ±3 % of the old tree on both the filtered and
the raw streams; the numbers in the tables below should be read with that
noise floor.

### 2.1 `bench --verdict` (22 datasets, 128 KiB each, external filter on)

```
 config                       ratio  before -> after     comp MB/s before -> after    dec MB/s before -> after
 spectra8 Flash +flt              2.404 ->  2.490         1394.4 ->  1379.2         3796.2 ->  4397.4
 spectra8 Beam L1 +flt            2.812 ->  2.821          377.8 ->   479.5         2272.5 ->  2088.4
 spectra8 Beam L5 +flt            2.965 ->  2.972           50.9 ->    55.4         1809.8 ->  1500.2
 spectra8 Beam L9 +flt            3.026 ->  3.034           13.9 ->    15.3          989.9 ->  1069.8
 spectra8 Beam ultra +flt         3.069 ->  3.076            4.1 ->     4.1         1346.5 ->  1249.0
 zstd     L22 +flt            2.973 ->  2.973           13.9 ->    13.2         2227.1 ->  2109.1
 zstd     L5 +flt             2.884 ->  2.884          597.9 ->   570.1         2298.3 ->  2241.0
 spectra8 Prism L5 +flt           3.012 ->  3.016            8.7 ->     9.5          206.7 ->   231.7
 spectra8 Prism L9 +flt           3.119 ->  3.119           13.5 ->    13.5           36.3 ->    37.0
 spectra8 Mosaic L5 +flt          3.073 ->  3.074           10.3 ->     9.8          654.9 ->   580.5
 spectra8 Mosaic L9 +flt          3.108 ->  3.109            3.1 ->     2.8          673.1 ->   560.5
 spectra8 Oracle L5 +flt          3.083 ->  3.084            6.4 ->     6.1          694.2 ->   629.3
 spectra8 Oracle L9 +flt          3.127 ->  3.128            1.3 ->     1.3          673.2 ->   640.7
 spectra8 Crystal L1 +flt         3.040 ->  3.040           33.8 ->    35.7          729.3 ->   776.4
 spectra8 Crystal L5 +flt         3.123 ->  3.123           20.1 ->    21.1          704.1 ->   752.6
 spectra8 Crystal L9 +flt         3.133 ->  3.133            3.9 ->     3.7          692.6 ->   610.2
 spectra8 Crystal L11 +flt        3.165 ->  3.165            6.5 ->     6.4            5.8 ->     5.6
 lzma     P9e +flt            3.002 ->  3.002           12.8 ->    12.2          127.2 ->   125.0
 lzma     P6 +flt             3.002 ->  3.002           18.0 ->    17.6          124.0 ->   124.9
 bzip3 65k +flt               2.962 ->  2.962           17.9 ->    18.0           29.6 ->    29.0
 bzip3 8M +flt                3.118 ->  3.118           18.6 ->    19.4           30.9 ->    30.1
 auto L5 g5%  +flt            3.119 ->  3.119            1.8 ->     1.7          745.0 ->   758.3
 auto L5 g2%  +flt            3.137 ->  3.137            1.8 ->     1.8          356.4 ->   339.6
 auto L5 g0%  +flt            3.167 ->  3.167            1.8 ->     1.5          137.6 ->   135.9
```

Filters chosen: binary=shuf64, bin-i16=dict16, bin-i32/f32=dict32,
bin-i64/f64=dict64, gau-i16=zzshuf64, gau-i32=dict32, gau-i64=dict64,
gau-f32=bitsh32, gau-f64=bitsh64, poi-i16=zzshuf64, poi-i32/f32=dict32,
poi-i64/f64=dict64, wave-f32=zzd2shuf32, wave-f64=d2shuf64, ramp-i32=d32+bit,
ts-u64=dshuf64, dec-f64=dict32, code-x86=bcj_x86 (unchanged).

What moved: Flash +3.6 % ratio and +16 % decode (the token layout and the
overlap copy, §3.5); Beam L1 +27 % encode (serialize path, §4.1) and +0.3 %
ratio (backward match extension, compact headers); Beam L5 +9 % encode
(Crystal-trial screen, §4.3); Beam L9 +10 % encode and +0.3 % ratio; Crystal
L1/L5 +5 % encode and +7 % decode (compact rANS headers in the token arrays
Crystal shares with Beam; the decode gain is noise-level). Everything else
is within the noise floor described above. Versus the README table that
preceded the review: Beam L1 2.794 → 2.821, Beam L5 2.952 → 2.972, Beam L9
2.984 → 3.034, Crystal L5 3.054 → 3.123, Crystal L9 3.055 → 3.133.

### 2.2 `bench --verdict --no-filter`

```
 config                       ratio  before -> after     comp MB/s before -> after    dec MB/s before -> after
 spectra8 Flash                   1.368 ->  1.410          940.7 ->   841.7         4213.0 ->  3951.4
 spectra8 Beam L1                 2.023 ->  2.031          183.6 ->   221.5         1442.7 ->  1209.7
 spectra8 Beam L5                 2.538 ->  2.541           39.9 ->    37.7          874.3 ->   700.1
 spectra8 Beam L9                 2.665 ->  2.670           12.2 ->     9.7          850.1 ->   659.1
 spectra8 Beam ultra              2.764 ->  2.769            3.2 ->     2.9          948.1 ->   810.2
 zstd     L22                 2.195 ->  2.195            6.9 ->     5.8         1142.7 ->  1058.4
 zstd     L5                  1.996 ->  1.996          220.9 ->   197.1         1071.9 ->   983.0
 spectra8 Prism L5                2.369 ->  2.376            8.9 ->     8.8          114.4 ->   129.2
 spectra8 Prism L9                2.514 ->  2.514           15.0 ->    12.6           27.3 ->    23.1
 spectra8 Mosaic L5               2.748 ->  2.748            8.5 ->     7.9          533.4 ->   457.7
 spectra8 Mosaic L9               2.923 ->  2.923            2.8 ->     2.5          566.6 ->   483.1
 spectra8 Oracle L5               2.776 ->  2.776            5.1 ->     4.9          535.0 ->   467.1
 spectra8 Oracle L9               2.967 ->  2.967            1.2 ->     1.1          509.5 ->   435.4
 spectra8 Crystal L1              2.758 ->  2.758           27.7 ->    24.5          591.3 ->   512.7
 spectra8 Crystal L5              2.957 ->  2.957           17.3 ->    15.5          574.7 ->   505.9
 spectra8 Crystal L9              2.977 ->  2.977            3.8 ->     3.3          531.8 ->   474.0
 spectra8 Crystal L11             2.989 ->  2.989            6.5 ->     5.0            6.0 ->     4.8
 lzma     P9e                 2.608 ->  2.608            8.4 ->     7.1           80.9 ->    73.6
 bzip3 8M                     2.515 ->  2.515           13.7 ->    14.5           22.2 ->    21.3
 auto L5 g5%                  2.957 ->  2.957            1.5 ->     1.5          501.9 ->   488.7
```

Flash on raw blocks: +3.1 % ratio. The table shows Flash decode −6 % here,
but this run is the one most affected by the machine load (zstd L5 decode,
untouched, is −8 % in the same table); the same-process A/B on exactly these
22 raw blocks measured old 4030 → new 4367 MB/s (+8 %) decode and 940 → 951
MB/s encode (§3.5). Beam L1 +21 % encode, +0.4 % ratio. Without the filter
Crystal's lattice model does the stride work itself and stays 16 % ahead of
Beam L5 and 35 % ahead of zstd L22 at 2.7× zstd L22's encode speed.

### 2.3 `bench --verdict --text` (enwik8, 8 MiB, filter = digit_norm)

```
 config                       ratio  before -> after     comp MB/s before -> after    dec MB/s before -> after
 spectra8 Flash +flt              1.842 ->  1.907          319.5 ->   294.9         1022.0 ->   912.0
 spectra8 Beam L1 +flt            2.393 ->  2.432          127.5 ->   164.8          567.7 ->   516.0
 spectra8 Beam L5 +flt            2.857 ->  2.871           64.4 ->    65.7          626.0 ->   571.5
 spectra8 Beam L9 +flt            3.108 ->  3.116           13.7 ->    11.4          509.2 ->   467.8
 spectra8 Beam ultra +flt         3.461 ->  3.433            2.4 ->     2.0          546.6 ->   500.5
 zstd     L22 +flt            3.631 ->  3.631            3.3 ->     3.3          735.5 ->   703.6
 zstd     L5 +flt             3.010 ->  3.010          150.9 ->   144.2          710.7 ->   670.3
 spectra8 Prism L5 +flt           3.522 ->  3.583            8.8 ->     8.9          145.9 ->   132.7
 spectra8 Prism L9 +flt           4.168 ->  4.168           25.6 ->    25.1           31.7 ->    32.9
 spectra8 Prism L11 128M +flt     4.300 ->  4.300           15.8 ->    14.8           30.8 ->    30.8
 spectra8 Mosaic L5 +flt          3.436 ->  3.436           13.5 ->    13.3          236.8 ->   222.6
 spectra8 Oracle L9 +flt          3.502 ->  3.502            2.7 ->     2.4          233.5 ->   196.1
 spectra8 Crystal L1 +flt         3.426 ->  3.551           38.4 ->    37.7          234.3 ->   197.9
 spectra8 Crystal L5 +flt         3.498 ->  3.644           18.0 ->    18.5          232.0 ->   194.3
 spectra8 Crystal L9 +flt         3.515 ->  3.812            5.8 ->    11.4          219.5 ->   173.6
 spectra8 Crystal L11 +flt        3.825 ->  3.825            3.0 ->     2.9            2.2 ->     2.3
 lzma     P9e +flt            3.701 ->  3.701            3.0 ->     3.1          123.4 ->   135.3
 bzip3 768k +flt              3.722 ->  3.722           13.5 ->    15.3           25.8 ->    24.5
 bzip3 64M +flt               4.300 ->  4.300           13.6 ->    15.0           17.2 ->    17.5
 auto L5 g5%  +flt            3.498 ->  3.644            1.4 ->     1.3          225.1 ->   213.2
 auto L5 g2%  +flt            3.685 ->  3.644            1.3 ->     1.2          120.3 ->   202.6
```

Crystal on text is the big mover: the 2 MiB / 4 MiB block ladder gives
L1 +3.6 %, L5 +4.2 %, L9 +8.4 % ratio (3.515 → 3.812; L9 now also encodes 2×
faster because the 4 MiB BWT replaces more of the lattice-model search), at
−15…−20 % decode (the inverse BWT working set leaves L2) — see §0 row 7 for
the per-corpus numbers behind that choice. `auto L5` follows Crystal
(3.498 → 3.644). Flash +3.5 %, Beam L1 +1.6 % (backward extension), Beam
L5/L9 +0.3…0.5 % (compact headers).

**Beam ultra 3.461 → 3.433** is not a parser or coder regression: bisecting
the tree on the staged `digit_norm` stream shows the whole 0.8 % comes from
one 2 MiB block whose Crystal literal array (192 968 bytes) now misses the
L9+ "must win ≥ 1.5 % of the block" budget by 17 bytes (193 014 → 192 951),
because the compact Huffman header made the rival order-0 literal array 63
bytes smaller. The gate is a deliberate decode-speed guard (Crystal literals
decode at ~600 MB/s vs multi-GB/s Huffman); with it disabled the new tree
gives 3.557 on the same stream (old tree: 3.555). Left as is; the gate's
granularity on 4-block inputs is the limitation, not the header change.

---

## 3. Flash (`src/flash/flash.hpp`)

### 3.1 What the stream looks like on the verdict corpus

Parsed from the actual Flash output of the 22 staged datasets (`flashstat`):

```
TOTAL out 1,198,969 B = literals 866,282 (72.3 %) + tokens 108,656 + offsets 217,268 + len-ext 6,763 (27.7 %)
sequences 108,634;   offsets: <=15 8.6 %  <=255 29.8 %  <=4095 40.5 %  >4095 21.0 %
rep-offset (same as previous) 0.2 %;  tokens with lit_len 0: 76.3 %
match length: 4: 48 %, 5: 21 %, 6: 11 %, 7: 13.5 % (8-byte stride), >= 19: 1.7 %
```

Consequences:

* **Rep-offset token would be useless** (0.2 % of matches repeat the offset);
  the shuffled/dict planes have no recurring offset structure.
* **A larger window is pointless**: the 2^14 table keeps only the latest
  position, candidates older than 64 KiB are overwritten anyway, and
  bigger tables barely help (below).
* **Short offsets are the only real redundancy**: 38 % of offsets fit in one
  byte, 76 % of tokens waste the literal-length nibble on 0.

### 3.2 Hash table / skip experiments (ratio = arithmetic mean over datasets, speed = total)

| variant | mean ratio | encode MB/s |
|---|---|---|
| `kHashLog` 14, skip 1<<6 (current) | 3.634 | 1887 |
| 15 | 3.638 | 1813 |
| 16 | 3.641 | 1763 |
| 17 | 3.642 | 1697 |
| 16, skip 1<<7 | 3.643 | 1588 |
| 14, skip 1<<8 | 3.635 | — |

+0.1–0.2 % ratio for −4…−10 % encode speed (the per-block `memset` of the
table grows 4–8×). Not a fruit. Keep 2^14.

### 3.3 Token layout experiments (simulated on the real parse, exact byte accounting)

| layout | verdict ratio | delta |
|---|---|---|
| current `ll:4 | ml:4`, 16-bit offset | 2.4051 | — |
| **A: `ll:3 | ml:4 | f:1`, offset 1 B if ≤ 255 else 2 B** | **2.4911** | **+3.58 %** |
| B: `ll:4 | ml:3 | f:1` | 2.4896 | +3.52 % |
| C: `ll:2 | ml:5 | f:1` | 2.4874 | +3.43 % |
| D: `ll:3 | ml:3 | f:2` (8/16/24-bit offsets) | 2.4889 | +3.49 % |

Layout A costs nothing in the decoder: always load 2 bytes (the stream has
slack), `off = v & (flag ? 0xFFFF : 0xFF); ip += 1 + flag` — branchless on both
ISAs. Literal lengths 7..14 now need an extension byte (histogram: only 1.3 %
of tokens), match length ≥ 19 keeps the nibble-15 extension (1.7 %). It is a
format change, so it needs a new Flash block version/flag in `format.hpp`; the
old decoder stays for existing frames. On text (enwik8) the gain will be
smaller (offsets are larger), on struct/record data larger.

A 24-bit offset variant (D) buys nothing on 128 KiB blocks and is not needed.

### 3.4 Decoder: short-offset copy

Offsets 2..15 currently fall into `for (k) op[k] = m[k]` — a dependent
byte loop. On the f32/f64 planes (35–42 % of matches at offset ≤ 15) this
dominates. Replacing it with the LZ4 pattern (`kInc32[8] = {0,1,2,1,0,4,4,4}`,
`kDec64[8] = {0,0,0,-1,-4,1,2,3}`, then 8-byte copies) — the same tables the
Beam decoder already uses in `copy_match_overlap_wild` — gives:

```
decode current            4452 MB/s
decode v2 (inc32/dec64)   5988 MB/s   (+35 %)
  gau-f32   10382 -> 26657    poi-f32   7529 -> 18914    wave-f64  6407 -> 24200
  binary     8690 -> 15346    bin-f32  34712 -> 61342    ramp-i32  4695 ->  7401
  bin-i32    3986 ->  3814    gau-i64   3474 ->  3474    code-x86  2513 ->  2644
```

(Per-dataset numbers are hot-cache 128 KiB loops, hence higher than the
verdict's harmonic mean. The i32/i64 planes are within noise.) Bit-exact,
no format change, pure `memcpy` so it compiles identically on ARM and x86.

Two further Flash decoder notes (not measured):

* The literal copy already does 16-byte wild copies; a "short token" fast path
  (lit ≤ 16 and match ≤ 16 and offset ≥ 16 → one `copy16` each, no length
  checks) as in LZ4's `shortcut` would help the 48 % of sequences with ML=4,
  but the gain is typically 5–10 % and it adds branches — try after #1.
* `encode_flash_block` memsets a 64 KiB table per 128 KiB block. LZ4 avoids
  this with a per-call base offset; it is ~3 % of encode time here.

### 3.5 Follow-up findings while implementing (same-process old/new A/B, 22 raw 128 KiB datasets)

Both notes above turned out to be wrong in an instructive way:

* **Token layout vs. decoder latency.** The first implementation stored the
  literal length in the 3-bit field (`ll:3 | wide:1 | ml:4`). On the raw
  (`--no-filter`) datasets it decoded 5–12 % *slower* than the old
  `ll:4 | ml:4` format, although >99 % of the tokens on those sets took the
  new fast path. The reason is the loop-carried input pointer: the next token
  is at `ip + 1 + lit_len + 1 + wide`, i.e. two field extractions and two
  adds after the token load, vs. `ip + 1 + lit_len + 2` (one extraction,
  constant folded into the next load) in the old layout. On an 8-byte-per-
  token stream that one extra cycle is ~12 % of the per-token cost. Storing
  `adv = lit_len + wide` in the field makes the next token `ip + adv + 1`
  (load → shift → add), which is one cycle *shorter* than the old layout:
  bin-i32 1932 → 2280 MB/s, bin-i64 3732 → 4333, poi-f64 3732 → 4172
  (+15–18 % over the first layout, +5–12 % over the old one). Cost: a
  wide-offset sequence carries at most 6 inline literals (code-x86 ratio
  1.312 → 1.306, everything else identical).
* **Persistent hash table.** Keeping the 2^14 table across calls with a base
  offset removed the memset but made `pos >= base` data-dependent on noise
  (a slot holds a stale entry or not at random) — the branch mispredicts
  ~50 % and encode dropped 25–28 % on gau-f32/f64, 10 % on the struct sets.
  Folding the check into the byte compare branchlessly fixed the noise case
  but cost ~10 % on compressible data (extra `csel` on the match pointer).
  Zeroing a table sized to the input (one slot per ~8 bytes, 1 Ki..16 Ki
  slots) is on par with the old code on every dataset (total +1 %) and
  keeps the memset at 4–16 KiB for the short filtered sub-streams, where the
  fixed 64 KiB clear had been the visible cost.

```
raw 128 KiB, same process      old      new
  Flash ratio (aggregate)     1.366    1.408
  encode MB/s                  940      951
  decode MB/s                 4030     4367
```

---

## 4. Beam LZ (`src/encoder.cpp`, `src/beam/match_finder.cpp`, `src/decoder.cpp`)

### 4.1 Where encode time goes

Parse vs serialize split on the verdict streams (same inputs the verdict
rows use):

| level | parse MB/s | serialize MB/s | serialize share | total MB/s |
|---|---|---|---|---|
| L1 | 1458 | 538 | **73 %** | 424 |
| L2 | 1392 | 551 | 72 % | 422 |
| L3 | 176 | 403 | 30 % | 125 |
| L5 | 265–390 | 49 | **85 %** | 42 |
| L5 without Crystal literal trial | 391 | 436 | 47 % | 214 |

Inside L1 serialize (166 µs per 128 KiB block): literal `encode_array`
55 µs (33 %; Huffman encode runs at ~745 MB/s on 41 KB of literals), the
three ~4.6 K-symbol token arrays 52 µs (32 % — i.e. ~17 µs *each*, which is
fixed overhead: histogram + code-length build + `encoded_size` estimates +
RLE trial + allocations, not bit output), the remainder is stream
construction, `off_low` selection (4 entropy evaluations) and offset
bit-packing.

Concrete encoder-only wins (no format change):

* Token arrays: skip the RLE trial unless `adj_eq*8 >= n` already does; the
  remaining cost is the Huffman code-length build (sort-based) and two size
  estimates. A radix/counting-sort length limiter and reusing one scratch
  `std::vector` across arrays would cut most of the 17 µs.
* Literal Huffman encoder: 745 MB/s is low for a 11-bit-max code; a 2- or
  4-stream encoder writing with 64-bit accumulators (the decoder already has
  the quad layout for n ≥ 1024) should reach 1.5–2 GB/s on both ISAs.
* Beam L1 would then land near 600+ MB/s, i.e. zstd L5 territory at +0.3 %
  better ratio than zstd L5 (2.812 vs 2.884 is zstd ahead by ratio but at
  598 MB/s; Beam L1 is the speed peer).

### 4.2 Level ladder (`params_for_level`)

```
L1 {hash 14, chain 4,  nice 16,  greedy}   L2 {15, 8,  24, greedy}
L3 {16, 16, 32, lazy}  L4 {17, 32, 48, lazy}
L5 {17, 8,  64, greedy, dense_short_insert=false}   (zstd-5 analogue)
L6 {20,128,96,lazy1} L7 {21,320,128,lazy1} L8 {22,640,192,lazy2} L9 {22,1280,256,lazy2}
```

Measured on the verdict corpus: L3 ratio 1.9637 at 125 MB/s; L5 with the
Crystal literal trial disabled 1.9635 at 214 MB/s (relative numbers from the
harness, which counts filtered input bytes). The L3/L4 lazy hash-chain
settings are dominated by the L5 greedy settings on this data (the lazy
evaluation + `dense_short_insert` costs 1.7× for no ratio). Suggest L3 :=
current L5 parser params, L4 := chain 16 greedy, and keep lazy for L6+. Needs
checking on text (lazy matters more there — Beam L5 text 2.857 vs L9 3.108).

### 4.3 Crystal literal arrays inside Beam

`serialize_parse_result` (encoder.cpp ~1317–1379) tries Crystal on the literal
array from L5 (gain bar 4 % of the block estimate; 1.5 % from L9). Measured
effect at L5: +2.6 % ratio, encode 214 → 42 MB/s (the Crystal L1 model search
is ~34 MB/s), decode of those blocks bounded by Crystal's ~700 MB/s. This is
the single biggest encode-speed cost in the Beam L5 path and the reason
Beam L5 is 12× slower to encode than zstd L5 for +2.8 % ratio. Options:

* Move the trial to L6 (L5 keeps the zstd-5 analogy also in encode speed), or
* Run the literal trial with a smaller analysis sample (`kAnalysisMax` is
  128 KiB; literals are ~40 KB so the whole array is modelled — a 16–32 KiB
  sample for the search and a full pass only for the final tables would cut
  most of the cost), or
* Require the `cond_gain` screen to predict ≥ 1.5× the bar instead of 1/1.15
  (the screen currently lets most planes through, see the L5 column:
  Crystal literals appear on 15 of 22 datasets).

### 4.4 Token-context conditioning

`kMinCondSeq = 4096` and L ≥ 6: unchanged recommendation, the measured header
tax (`kCondArrayTax = 24`) is correct (see §5.2, an extra array costs ~1 table).

### 4.5 Decoder

`decode_compressed_block` is in good shape: token arrays are fully decoded
before the sequence loop (cache-friendly on 128 KiB blocks), `off_low`
tokens are a separate byte array, 8-byte copy for short far matches,
`copy_match_overlap_wild` for near offsets, `__builtin_expect` on the 255
escapes. Decode speed on filtered data (1810 MB/s at L5, 2270 at L1) is
bounded by:

* Huffman literal decode (11-bit codes, quad streams) — comparable with
  zstd's 4-stream Huffman; zstd L5 decodes 2298 MB/s here, mostly because
  it has fewer, longer sequences on these planes.
* Crystal literal arrays (~700 MB/s) whenever selected — this is why Beam L5
  decode in the no-filter run is 874 MB/s vs 1443 for L1.

No decoder-side low-hanging fruit found in Beam. A possible medium-size item
is a 2-sequence-per-iteration unroll with the offset bit refill hoisted (the
`MsbReader::ensure/peek0/consume` triple per new offset is 3 dependent ops),
but it needs measurement on x86 where register pressure is the limit.

---

## 5. Entropy coders (`huffman.cpp`, `tans.cpp`, `arith.cpp`, `entropy.cpp`)

### 5.1 Selection logic

`encode_array` picks Huffman vs rANS from size estimates (`huff::encoded_size`,
`ans::encoded_size`) and builds only the winner; rANS needs ≥ 4 % (tokens,
L3+) or ≥ 1.5 % (literals, L9+) to be taken because its decode is slower.
The RLE trial runs when ≥ 12.5 % of bytes repeat. `encode_bwt_entropy`
(Prism) races adaptive O0, static O1 rANS, adaptive O1 and O0 rANS with
estimate-based pruning. The logic is sound; the only cost note is the fixed
per-call overhead on tiny token arrays (§4.1).

### 5.2 Header cost — measured, smaller than it looks

Tally of table bytes in the Beam payloads (tag byte + Huffman `u16 num_syms`
+ 4-bit lengths + mode byte, or rANS `mode, u16 num_syms, u16 freq[]` +
stream flushes):

| level | payload | header bytes | share | arrays | huff | ans | raw |
|---|---|---|---|---|---|---|---|
| L1 | 1,027,689 | 4,381 | 0.43 % | 89 | 53 | 0 | 18 |
| L5 | 972,300 | 3,214 | 0.33 % | 89 | 34 | 4 | 33 |
| L9 | 952,678 | 3,964 | 0.42 % | 106 | 51 | 2 | 34 |

Halving header size (e.g. Crystal-style bit-length + mantissa tables, or
FSE-coded weights) would give +0.2 % ratio. Worth doing only as part of a
format revision; not a priority. (Worst single case: ts-u64, 4.2 % of a
8.5 KB payload.)

### 5.3 Portability notes

* Huffman decode: 11-bit single + pair tables, table-driven; identical code on
  both ISAs, no intrinsics. Fine.
* rANS: 12-bit probabilities, 16-bit renorm, 2/4/8 interleaved ways chosen by
  n — the 8-way case keeps 8 states live, which is fine on AArch64 (31 GPRs)
  and still fits x86-64 (16 GPRs) because the per-state work is a table load +
  mul-add. No change needed.
* `arith.cpp` adaptive range coder uses one integer division per symbol —
  cheap on Apple silicon (as the comment says) but ~20–40 cycles on older x86;
  it is only used in Prism/BWT paths where the inverse BWT dominates anyway.

---

## 6. Prism (BWT) — `bwt.cpp`, `bwt_decode.hpp`, `bwt_cm.cpp`, `bwt_ultra.cpp`, `lzp.cpp`

Ratio-wise Prism L9 is the strongest fast-encode option on text (4.168 at
25.6 MB/s encode vs bzip3-768k 3.722); its decode (31.7 MB/s, 36 MB/s on the
verdict corpus) is the limiter and is dominated by the LF-mapping inverse,
which is a random-access chain (one cache miss per output byte unless K
chains are interleaved). The code already does:

* `build_lf_chunked`, K-chain interleaving (`walk_k_tab`) with 64-byte
  staging lines, 4K-aliasing avoidance, `NavFused` (symbol packed in the
  top byte of the link word — one load per step).

Observations:

* `lf_prefetch(nav + np)` in `NavFused::step`/`NavSplit::step` prefetches the
  address that is dereferenced on the very next iteration; with no distance
  it cannot hide latency. It is x86-only (`__builtin_prefetch`, compiled out
  on ARM). Either remove it or restructure so chain *c* prefetches its own
  *next-next* link (requires keeping `e` of the previous step) — that is the
  standard trick that gives 1.3–1.6× on x86 for K ≥ 8 chains.
* Prism L5 vs L9 decode (207 vs 36 MB/s on the verdict, 146 vs 32 on text) —
  L5 is the "LZ/zstd fast last step" segmentation mode; the L9 modes
  (CM/ultra) are inherently slow. No tunable found that changes this without a
  ratio cost.

---

## 7. Mosaic / Oracle (`mosaic.cpp`, `oracle.cpp`)

Both sit between Beam and Crystal in ratio (3.07–3.13) with 650–700 MB/s
decode and slow encode (1.3–10 MB/s). On the verdict corpus Crystal L5
(3.123, 20 MB/s encode, 704 MB/s decode) dominates Mosaic L9 and Oracle L9 on
all three axes, and `auto` never picks them with filters on (one Mosaic pick
without filters). No specific tunable to report; they are the research
siblings of Crystal.

---

## 8. Crystal (`crystal.cpp`, `crystal_static.cpp`, `crystal_bwt.cpp`)

### 8.1 Where Crystal stands

Verdict: Crystal L5 3.123 / 20 MB/s / 704 MB/s — best ratio of any sub-second
decoder, 5 % ahead of Beam L5 at 0.39× its decode speed, and `auto g5%` picks
it on 12 of 22 datasets (17 of 22 without filters). L9 adds +0.3 % for 5×
encode time; L11 (CM) adds +1.0 % for 120× slower decode.

Decode is bounded by the rANS loop's instruction count: ~1.4 ns/byte ≈ 4.5
cycles per symbol with 16 interleaved streams (`kMaxStreams = 16`), ~22
instructions per byte in `S1Body` on AArch64 — throughput-bound, the 16
streams hide the chain latency. The x86 path has the AVX2 `V8` body and
BMI2 dispatch; on AArch64 the scalar `LaneBody`/`S1Body` is the right
choice (no gather, and the 31 GPRs keep all 16 states in registers). At
the time of the review I did not see a decoder-side change that would move
this materially without changing the model; the follow-up in §0.1 took the
format change: lanes that are constant or noise now bypass the coder (raw
records +45 % decode, −0.4 % size), while the run-symbol / pair-coding
ideas for the filtered byte planes were measured and dropped (the per-step
bookkeeping costs what the saved steps gain in a throughput-bound loop).

### 8.2 Tunables tested (verdict streams, build-time constant sweeps)

Baseline: L1 3.0413, L5 3.1246, L9 3.1346 (sum ratio, harness); decode
836 / 813 / 804 MB/s (hot-cache).

| constant | tested | L1 | L5 | L9 | decode |
|---|---|---|---|---|---|
| `kPrec` 12 (12-bit rANS tables) | 13 | 3.0396 | 3.1192 | 3.1295 | −1.5 % | 
| `kSymbolBits` 7.5 (MDL per-symbol table cost) | 6.0 | 3.0422 | 3.1257 | 3.1338 | −1 % L9 |
|  | 9.0 | 3.0407 | 3.1249 | 3.1349 | = |
|  | 11.0 | 3.0394 | 3.1247 | 3.1328 | = |
| `kTableBits` 24 (fixed table cost) | 16 / 40 | 3.0413 | 3.1245 / 3.1246 | 3.1346 / 3.1347 | = |
| `kBoundaryBits` 9 (group boundary cost) | 6 / 13 | 3.0413 | 3.1245 / 3.1246 | 3.1346 / 3.1347 | = |

Everything is within ±0.05 % — the MDL model search is at a flat optimum on
this corpus, and `kPrec` 13 **loses** (header bits grow by one per present
symbol, which outweighs the precision gain at 128 KiB blocks, and the dense
tables double to 9 KiB per context). Recommendation: leave all of these.
If you want a knob with a visible effect: the L1→L5 gap (2.7 %) is the
`SearchCfg` ladder (`kmax` 16→32, `max_bins` 64→128, `top_singles` 2→5,
C2/Lat2/Trend/RecEq features from L4); an "L3 = L5 features with L1 kmax"
point might be a useful intermediate for the Beam literal trial (§4.3).

### 8.3 Crystal BWT / text mode

On enwik8 at the default 1 MiB block (L5): Crystal 3.498 vs Prism L5 3.522,
zstd L22 3.631, bzip3-768k 3.722 (≈ same block size), Crystal decode 232 MB/s
(7–9× faster than any of those with higher ratio). The 6 % gap to bzip3 at
equal block size is the price of a *static* coder on MTF/BWT output; the
sticky-MTF + F-class split already recovers most of what an order-2 adaptive
coder would. Levers:

* **Block size ladder** — bzip3 goes 3.72 → 4.30 from 768 KiB to 64 MiB; Prism
  uses `kBwtDefaultBlockLog`-style growth (1/2/4/8 MiB by level). Crystal in
  BWT mode uses the generic ladder (256 KiB / 1 MiB / 2 MiB / 4 MiB). Giving
  Crystal the BWT ladder when its text/BWT kind wins would likely add
  3–6 % on text at no decode cost per byte (the inverse BWT is already
  chain-interleaved: `kChains16Min` 256 KiB, `kChains8Min` 64 KiB).
* `kBwtGain = 0.03` (crystal.cpp:1222) — BWT must be predicted 3 % smaller
  than the lattice to be chosen; `consider_bwt = lat*100 >= n` plus the
  32 KiB window test (`est_w < 1.05 * win_lat`). These gates are
  encode-time guards; lowering `kBwtGain` to 1–2 % at L ≥ 9 is the only
  cheap change, and only affects borderline mixed blocks.
* `crystal_bwt::walk_chains<K>` writes K skewed scratch lines then copies n
  bytes into `dst` (lines 826–858). `bwt_decode.hpp::walk_k_tab` already
  solves this with 64-byte staging lines flushed straight into `dst`; porting
  that removes one full pass over the block (~0.3–0.5 % of decode).

### 8.4 Profiles

The pre-trained profile path (`crystal_profile != 0`) bypasses the model
search entirely; for repetitive record streams this is the right encode-speed
lever (model search is the whole encode cost: L1 34 MB/s, L5 20 MB/s, L9
4 MB/s). Nothing to change.

---

## 9. Filters (`transform.hpp`, `pack_filter.hpp`)

* Detection (`detect_filter`, 16 KiB mid-sample, 0.97/0.90 margins): the
  verdict picks sensible filters for all 22 datasets; the dict/FOR/ALP framed
  filters subsume the byte shuffles where applicable. No ratio fruit here
  on the verdict data — on real executables the proxy never nominated
  `bcj_x86`; fixed by direct executable detection in §0.2.
* `shuffle_inv` has NEON and SSE2 paths; `delta_inv` has an SSE2 prefix-sum
  only — on AArch64 it is scalar. Adding a NEON prefix-sum (shift-add tree,
  `vextq_u8`) is a 20-line change; it matters for in-frame filtered blocks on
  ARM (d2shuf/dshuf planes), not for the verdict rows (external filter timing
  is excluded there).
* `bitshuffle` transposes: both ISAs covered.

---

## 10. ARM vs x86_64 checklist for any of the above

* Flash #1 (overlap copy) and #2 (token layout): plain `memcpy`/mask code,
  identical on both.
* Beam serialize speedups: scalar integer work; a 64-bit bit accumulator is
  equally good on both.
* Crystal: the scalar lane body is register-bound on x86 (16 GPRs, hence the
  fused table layout and the `V8` AVX2 variant). Do not add live state to the
  decode loop (e.g. `kPrec` 13 or extra per-lane fields) without re-measuring
  on x86; this is also why the `kPrec` result above should settle that
  question.
* BWT prefetch: `__builtin_prefetch` is x86-only in `lf_prefetch`; if a
  distance-1 prefetch is added, enable it on AArch64 too (`prfm` is cheap and
  Apple/Neoverse cores benefit as much).
* `arith.cpp` division per symbol is fine on Apple silicon, slower on
  pre-Zen4/Icelake x86 — only matters for Prism's adaptive modes.

---

## Appendix A — experiment inventory (all in `/tmp`, nothing added to the repo)

| harness | what |
|---|---|
| `exp flash` | `kHashLog`/skip variants of `flash.hpp` compiled under distinct namespaces, ratio per dataset |
| `flashspeed` | encode MB/s per variant; decode current vs inc32/dec64 variant, per dataset, verified bit-exact |
| `flashstat` / `flashsim` | parse Flash output: literal/token/offset shares, offset & length histograms; exact byte simulation of token layouts A–D |
| `exp beamhdr` | parse Beam payloads, tally Huffman/rANS table bytes per array, per level |
| `beamsplit` | `parse_block` vs `serialize_parse_result` timing per level; `NOCRY` env to disable the Crystal literal trial |
| `litenc` | L1 serialize breakdown: literal `encode_array`, three token arrays, `hist_u8` |
| `exp crystal` + `exp13`, `exp_sb6/sb9/sb11/tb16/tb40/bb6/bb13` | `crystal_static.cpp` recompiled with one constant changed, linked ahead of the library; ratio + decode per dataset, roundtrip verified |
| `crystat` (`crystal_static_s.cpp`) | §0.1 step statistics: per lane bits/byte, top-symbol probability, run shares, constant / noise lanes, context alphabet sizes (`CRYSTAT` / `CRYRUN` lines) |
| `crydec_old/new/il*` (`crydec.cpp`, `buildvar.sh`) | Crystal L5 on the verdict streams (filtered / raw): size and best-of-N decode per dataset; variants via `sed` on `crystal_static.cpp` (raw-lane slack, scatter vs interleave, `CRYMODES` lane-mode dump) |
| `crydec_run` (`cs_run.cpp`, `mkrun.py`) | run-aware `S1Body` prototype with the run length fixed at 1: the cost side of the run-symbol idea (−15 %) |
| `cryfuzz_asan`, `cryfuzz_asan_emu` (`cryfuzz.cpp`, `buildfuzz.sh`) | ASan/UBSan roundtrip + corruption fuzz of the static coder, native and `-DCRY_V8_EMU` with `SPECTRA8_CRYSTAL_X86=avx2` |
| `execdet` / `execstat` (`execdet.cpp`, `execstat.cpp`) | §0.2 `detect_exec_arch` verdict per file and per fixed-size page; raw per-page statistics (entropy, `0xE8` density and pass rate, REX share, BL/ADRP/frame word shares) used to set the thresholds |
| `vdet` (`vdet.cpp`, includes `cli/bench.cpp`) | `detect_exec_arch` over the 26 verdict datasets at 128 KiB and 1 MiB |
| `bcjbench` (`bcjbench.cpp`, also `-arch x86_64` and ASan builds) | BCJ forward/inverse/detector GB/s on real binaries, roundtrip check |
| `fbtest` (`fbtest.cpp`) | lzma/zstd L5 `+flt` with and without the raw keep-smaller trial over a directory: ratio cost of skipping it on BCJ stages |
| `so/x86s`, `so/arm64` (`spectra8_old/new`) | the two library sets (manylinux `.so` from numpy/scipy wheels; XRootD arm64 bundles) and the before/after CLI binaries for `bench --dir` A/B |

Raw outputs: `/tmp/spectra8report/verdict_default.txt`,
`/tmp/spectra8report/verdict_nofilter.txt`, `/tmp/spectra8report/verdict_text.txt`,
`/tmp/spectra8exp/*.txt`.
