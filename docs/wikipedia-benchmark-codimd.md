---
title: Wikipedia (enwik8) Benchmark: Ratio, Speed, and ns/byte
tags: benchmark, compression, spectra8, zstd, lzma, bzip3, prism, crystal, wordnorm
robots: noindex, nofollow
---

# Wikipedia (enwik8) Benchmark Comparison

**Corpus**: Wikipedia English XML Dump (`enwik8`, Hutter Prize benchmark subset)  
**Uncompressed Size**: 100,000,000 bytes (100.00 MB / 95.37 MiB)  
**Execution Environment**: Single-threaded, Apple Silicon (Darwin 25.6.0, `clang -O3 -std=c++20`)  
**Verification**: 100% round-trip verified (`memcmp` bitwise match against original)

---

## 1. Summary Results Table (Raw / Unfiltered Baseline)

| Codec & Configuration | Compressed Size (Bytes) | Ratio | Bits / Char (bpc) | Encode Speed | Encode Latency (ns/byte) | Decode Speed | Decode Latency (ns/byte) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **prism** (128 MiB block, L11) | **20,742,407** | **4.821x** | **1.659** | 3.4 MB/s | **279.0 ns/B** | 4.7 MB/s | **201.2 ns/B** | OK |
| **bzip3** (128 MiB block) | 20,749,615 | 4.819x | 1.660 | 1.4 MB/s | 695.3 ns/B | 2.7 MB/s | 352.2 ns/B | OK |
| **lzma** (level 9) | 24,865,252 | 4.022x | 1.989 | 0.3 MB/s | 3,414.9 ns/B | 30.4 MB/s | 31.4 ns/B | OK |
| **zstd** (level 22) | 25,269,896 | 3.957x | 2.022 | 0.3 MB/s | 3,412.0 ns/B | **278.7 MB/s** | **3.4 ns/B** | OK |
| **crystal** (level 11) | 26,217,705 | 3.814x | 2.097 | 0.4 MB/s | 2,520.6 ns/B | 0.3 MB/s | 2,745.5 ns/B | OK |
| **crystal** (level 5) | 29,387,256 | 3.403x | 2.351 | 5.2 MB/s | 181.7 ns/B | 38.0 MB/s | 25.1 ns/B | OK |
| **zstd** (level 5) | 33,699,396 | 2.967x | 2.700 | **21.5 MB/s** | **44.4 ns/B** | 216.2 MB/s | 4.4 ns/B | OK |

> **Note on ns/byte**: Calculated as $\frac{T_{\text{seconds}} \times 10^9}{100{,}000{,}000 \text{ data bytes}} = T_{\text{seconds}} \times 10$. Represents the average single-thread compute cost per uncompressed input byte.

---

## 2. Word Optimization Filter (`WordNorm`) Benchmark Comparison

When Spectra8's text pre-filter pipeline (`Filter::WordNorm = 33`, `word_filter.hpp`) is applied:
- **Pre-transform intermediate**: The 100,000,000-byte corpus shrinks to **87,946,653 bytes** (**12.05 MB eliminated** pre-compression) in **1.63 s** (**16.3 ns/byte**).
- **Inverse transform**: Decompressing and restoring words/case runs at **> 600 MB/s** (**0.14 s / 1.4 ns/byte**).
- **Full verification**: 100% bitwise round-trip match against original uncompressed `enwik8`.

### Raw vs Filtered (`+ WordNorm`) Side-by-Side:

| Codec & Configuration | Filter | Compressed Size | Ratio | bpc | Enc Speed | Enc Time | Enc Latency | Dec Speed | Dec Time | Dec Latency | Verified |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **prism** (128 MiB, L11) | Raw | 20,742,407 B | 4.821x | 1.659 | 3.4 MB/s | 27.90 s | 279.0 ns/B | 4.7 MB/s | 20.12 s | 201.2 ns/B | OK |
| **prism** (128 MiB, L11) | **+ WordNorm** | **20,713,377 B** | **4.828x** | **1.657** | **5.2 MB/s** | **18.33 s** | **183.3 ns/B** | **19.6 MB/s** | **4.86 s** | **48.6 ns/B** | **OK** |
| | | *(-29,030 B)* | *(+0.007x)* | | *(+53% speed)* | *(-9.57 s)* | | *(+317% speed)* | *(-15.26 s)* | *(4.1x faster)* | |
| **bzip3** (128 MiB block) | Raw | 20,749,615 B | 4.819x | 1.660 | 1.4 MB/s | 69.53 s | 695.3 ns/B | 2.7 MB/s | 35.22 s | 352.2 ns/B | OK |
| **bzip3** (128 MiB block) | **+ WordNorm** | **20,721,289 B** | **4.826x** | **1.658** | **10.8 MB/s** | **8.81 s** | **88.1 ns/B** | **10.8 MB/s** | **8.83 s** | **88.3 ns/B** | **OK** |
| | | *(-28,326 B)* | *(+0.007x)* | | *(+671% speed)* | *(-60.72 s)* | *(7.9x faster)* | *(+300% speed)* | *(-26.39 s)* | *(4.0x faster)* | |
| **zstd** (level 5) | Raw | 33,699,396 B | 2.967x | 2.700 | 21.5 MB/s | 4.44 s | 44.4 ns/B | 216.2 MB/s | 0.44 s | 4.4 ns/B | OK |
| **zstd** (level 5) | **+ WordNorm** | **32,678,489 B** | **3.060x** | **2.614** | **11.1 MB/s** | **8.61 s** | **86.1 ns/B** | **108.4 MB/s** | **0.88 s** | **8.8 ns/B** | **OK** |
| | | *(-1,020,907 B)* | *(+0.093x)* | | | *(+4.17 s)* | | | *(+0.44 s)* | | |

*Note: Total Enc Time and Total Dec Time include both the compression engine time AND the WordNorm forward/inverse transformation overhead.*

---

## 3. Detailed Performance Breakdown

### 3.1 BWT on WordNorm: Why Speed Dramatically Increases
1. **Alphabet Reduction & Superlinear Suffix Sort**:
   Burrows-Wheeler suffix sorting algorithms (such as `divsufsort`) have superlinear complexity with respect to block length and string repetition depth. By transforming long repetitive XML entities (`&quot;`, `&#160;`), HTML tags (`<page>`, `</text>`), and common words (`"which"`, `"between"`, `"university"`) into single-byte tokens:
   - Suffix sort string comparisons terminate in fewer iterations.
   - For **bzip3**, compression time plummeted from **69.53 s down to 7.17 s** (+ 1.63 s WordNorm forward = **8.81 s total**), a **7.9x end-to-end speedup**.
   - For **prism**, compression time dropped from **27.90 s to 16.40 s** (+ 1.93 s WordNorm = **18.33 s total**), a **1.5x end-to-end speedup**.
2. **Decompression Acceleration**:
   The inverse BWT and arithmetic decoding stage runs on 87.95 million bytes rather than 100 million bytes. Because the WordNorm inverse filter streams at $>1.5\text{ GB/s}$ (requiring only 0.14–0.16 seconds on the whole corpus):
   - **Prism decode latency drops from 201.2 ns/B down to 48.6 ns/B** (a **4.1x faster** decode!).
   - **Bzip3 decode latency drops from 352.2 ns/B down to 88.3 ns/B** (a **4.0x faster** decode!).

---

### 3.2 Ultra LZ77 & Optimal Parsing: `zstd -22` vs `lzma -9`
- **Compression Density**:
  - `lzma -9`: **24.87 MB** (4.022x, 1.989 bpc)
  - `zstd -22`: **25.27 MB** (3.957x, 2.022 bpc)
  - *LZMA gains ~404 KB over Zstd-22 due to its bit-level Markov range coder and state machine.*
- **Encode Cost**:
  - `zstd -22`: **341.20 s** (~5.7 min) $\rightarrow$ **3,412.0 ns/byte** (~3.41 µs/byte)
  - `lzma -9`: **341.49 s** (~5.7 min) $\rightarrow$ **3,414.9 ns/byte** (~3.41 µs/byte)
  - *Both engines spend nearly identical compute budgets performing exhaustive binary tree search and optimal parse graph evaluation.*
- **Decode Cost (Asymmetry)**:
  - `zstd -22`: **0.34 s** (278.7 MB/s) $\rightarrow$ **3.4 ns/byte**
  - `lzma -9`: **3.14 s** (30.4 MB/s) $\rightarrow$ **31.4 ns/byte**
  - *Zstd decompresses **9.2x faster** than LZMA because FSE/Huffman token decoding vectorizes far better than bit-serial range decoding.*

---

### 3.3 Context Mixing & Multi-Stream Models: `crystal` (L5 and L11)
- **`crystal (level 5)` (Static / Lightweight Lattice CM)**:
  - Compressed Size: **29.39 MB** (3.403x, 2.351 bpc)
  - Gain vs `zstd -5`: Compresses **4.31 MB smaller** than raw Zstd-5.
  - Encode: **18.17 s** (5.2 MB/s) $\rightarrow$ **181.7 ns/byte**
  - Decode: **2.51 s** (38.0 MB/s) $\rightarrow$ **25.1 ns/byte**
- **`crystal (level 11)` (Deep Tier-2 CM Ensemble)**:
  - Compressed Size: **26.22 MB** (3.814x, 2.097 bpc)
  - Encode: **252.06 s** (0.4 MB/s) $\rightarrow$ **2,520.6 ns/byte** (~2.52 µs/byte)
  - Decode: **274.55 s** (0.3 MB/s) $\rightarrow$ **2,745.5 ns/byte** (~2.75 µs/byte)
  - *Processes independent 1 MiB chunk blocks (designed for columnar and micro-chunk data), avoiding global memory spikes and enabling random page access.*

---

## 4. Frontier Summary Winner Matrix

| Objective | Winner | Configuration | Value |
| :--- | :--- | :--- | :--- |
| **All-Time Smallest Compressed Size** | `prism` (128 MiB, L11) | **+ WordNorm** | **20,713,377 B** (4.828x, 1.657 bpc) |
| **Fastest High-Ratio BWT Encode** | `bzip3` (128 MiB) | **+ WordNorm** | **88.1 ns/byte** (8.81 s total, 7.9x speedup) |
| **Fastest High-Ratio BWT Decode** | `prism` (128 MiB, L11) | **+ WordNorm** | **48.6 ns/byte** (4.86 s total, 4.1x speedup) |
| **Fastest Line-Rate Decompression** | `zstd` (level 22) | Raw | **278.7 MB/s** (3.4 ns/byte) |
| **Best Balanced Intermediate** | `crystal` (level 5) | Raw | **3.403x** ratio, **25.1 ns/byte** decode |
