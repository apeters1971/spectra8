// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 CERN
// Author: Andreas Joachim Peters <andreas.joachim.peters@cern.ch>
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Experimental cmix-style LSTM byte predictor for the Crystal context mixer.
//
// cmix's distinguishing addition over the PAQ family is a recurrent neural
// network that predicts the next byte from the byte sequence and whose
// probability is fed to the logistic mixer as one more model. This is a
// minimal version of that idea: a single-layer LSTM with a one-hot byte input
// (optionally plus a one-hot record-lane input), a softmax over the 256 next
// bytes, trained online by truncated back-propagation through time every
// `horizon` bytes with Adam. The per-bit probability the mixer needs is read
// off the prefix sums of the byte distribution.
//
// The arithmetic is IEEE float, so the stream is only decodable by a build
// with identical floating-point behaviour (same as cmix). This is an
// experiment on the ratio ceiling, not a shipping format.

#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace spectra8::detail {

class LstmByteModel {
 public:
  // cells: hidden size H; horizon: BPTT length T; stride: record stride for
  // the lane input (1 disables it); lr: Adam step size.
  LstmByteModel(int cells, int horizon, int stride, float lr)
      : H_(cells), T_(horizon), S_(stride), lr_(lr) {
    const int G = 4 * H_;
    Wx_.assign(static_cast<size_t>(256) * G, 0.f);
    Wl_.assign(static_cast<size_t>(S_) * G, 0.f);
    Wh_.assign(static_cast<size_t>(G) * H_, 0.f);
    b_.assign(G, 0.f);
    Wo_.assign(static_cast<size_t>(256) * H_, 0.f);
    bo_.assign(256, 0.f);
    std::uint64_t seed = 0x9E3779B97F4A7C15ull;
    init_uniform(Wx_, 1.f / std::sqrt(static_cast<float>(H_ + 1)), seed);
    init_uniform(Wl_, 1.f / std::sqrt(static_cast<float>(H_ + 1)), seed);
    init_uniform(Wh_, 1.f / std::sqrt(static_cast<float>(H_)), seed);
    init_uniform(Wo_, 1.f / std::sqrt(static_cast<float>(H_)), seed);
    for (int i = H_; i < 2 * H_; ++i) b_[i] = 1.f;  // forget gate open
    const size_t np = Wx_.size() + Wl_.size() + Wh_.size() + b_.size() + Wo_.size() + bo_.size();
    grad_.assign(np, 0.f);
    m_.assign(np, 0.f);
    v_.assign(np, 0.f);
    h_.assign(H_, 0.f);
    c_.assign(H_, 0.f);
    h0_.assign(H_, 0.f);
    c0_.assign(H_, 0.f);
    hist_in_.assign(T_, 0);
    hist_lane_.assign(T_, 0);
    hist_tgt_.assign(T_, 0);
    hist_gate_.assign(static_cast<size_t>(T_) * G, 0.f);
    hist_c_.assign(static_cast<size_t>(T_) * H_, 0.f);
    hist_h_.assign(static_cast<size_t>(T_) * H_, 0.f);
    hist_p_.assign(static_cast<size_t>(T_) * 256, 0.f);
    dz_.assign(G, 0.f);
    dh_.assign(H_, 0.f);
    dc_.assign(H_, 0.f);
    dh_next_.assign(H_, 0.f);
    dlog_.assign(256, 0.f);
    // Before any input the output layer is untrained: uniform distribution.
    for (int i = 0; i <= 256; ++i) cum_[i] = static_cast<float>(i) / 256.f;
    forward(0, 0);
  }

  // Prefix sums of P(next byte): cum[b+1] - cum[b] = P(b); cum[0] = 0.
  const float* cum() const noexcept { return cum_; }

  // The byte just coded was `y`; the byte after it sits in record lane
  // `next_lane`. Learns from y, then predicts the next byte.
  void observe(int y, int next_lane) {
    hist_tgt_[t_] = static_cast<std::uint8_t>(y);
    if (++t_ == T_) {
      backward();
      t_ = 0;
      h0_ = h_;
      c0_ = c_;
    }
    forward(y, S_ > 1 ? next_lane : 0);
  }

 private:
  static inline float sigmoid(float x) noexcept { return 1.f / (1.f + std::exp(-x)); }

  static void init_uniform(std::vector<float>& w, float a, std::uint64_t& s) {
    for (float& x : w) {
      s ^= s << 13;
      s ^= s >> 7;
      s ^= s << 17;
      x = (static_cast<float>(s >> 40) / 16777216.f * 2.f - 1.f) * a;
    }
  }

  // One time step with input byte x and lane l; writes history slot t_.
  void forward(int x, int l) {
    const int H = H_, G = 4 * H;
    float* z = hist_gate_.data() + static_cast<size_t>(t_) * G;
    const float* wx = Wx_.data() + static_cast<size_t>(x) * G;
    const float* wl = Wl_.data() + static_cast<size_t>(l) * G;
    for (int r = 0; r < G; ++r) {
      const float* wh = Wh_.data() + static_cast<size_t>(r) * H;
      float acc = b_[r] + wx[r] + wl[r];
      for (int i = 0; i < H; ++i) acc += wh[i] * h_[i];
      z[r] = acc;
    }
    // Gate layout: [i | f | o | g].
    for (int i = 0; i < 3 * H; ++i) z[i] = sigmoid(z[i]);
    for (int i = 3 * H; i < G; ++i) z[i] = std::tanh(z[i]);
    float* hc = hist_c_.data() + static_cast<size_t>(t_) * H;
    float* hh = hist_h_.data() + static_cast<size_t>(t_) * H;
    for (int i = 0; i < H; ++i) {
      c_[i] = z[H + i] * c_[i] + z[i] * z[3 * H + i];
      hc[i] = c_[i];
      h_[i] = z[2 * H + i] * std::tanh(c_[i]);
      hh[i] = h_[i];
    }
    hist_in_[t_] = static_cast<std::uint8_t>(x);
    hist_lane_[t_] = static_cast<std::uint8_t>(l);
    // Output softmax.
    float* p = hist_p_.data() + static_cast<size_t>(t_) * 256;
    float mx = -1e30f;
    for (int j = 0; j < 256; ++j) {
      const float* wo = Wo_.data() + static_cast<size_t>(j) * H;
      float acc = bo_[j];
      for (int i = 0; i < H; ++i) acc += wo[i] * h_[i];
      p[j] = acc;
      if (acc > mx) mx = acc;
    }
    float sum = 0.f;
    for (int j = 0; j < 256; ++j) {
      p[j] = std::exp(p[j] - mx);
      sum += p[j];
    }
    const float inv = 1.f / sum;
    float run = 0.f;
    cum_[0] = 0.f;
    for (int j = 0; j < 256; ++j) {
      p[j] *= inv;
      run += p[j];
      cum_[j + 1] = run;
    }
    cum_[256] = 1.f;
  }

  // Truncated BPTT over the T_ stored steps, then one Adam update.
  void backward() {
    const int H = H_, G = 4 * H;
    float* gWx = grad_.data();
    float* gWl = gWx + Wx_.size();
    float* gWh = gWl + Wl_.size();
    float* gb = gWh + Wh_.size();
    float* gWo = gb + b_.size();
    float* gbo = gWo + Wo_.size();
    for (int i = 0; i < H; ++i) dh_next_[i] = 0.f, dc_[i] = 0.f;
    for (int t = T_ - 1; t >= 0; --t) {
      const float* z = hist_gate_.data() + static_cast<size_t>(t) * G;
      const float* ct = hist_c_.data() + static_cast<size_t>(t) * H;
      const float* ht = hist_h_.data() + static_cast<size_t>(t) * H;
      const float* hprev = t > 0 ? hist_h_.data() + static_cast<size_t>(t - 1) * H : h0_.data();
      const float* cprev = t > 0 ? hist_c_.data() + static_cast<size_t>(t - 1) * H : c0_.data();
      const float* p = hist_p_.data() + static_cast<size_t>(t) * 256;
      const int tgt = hist_tgt_[t];
      // Output layer: cross-entropy gradient.
      for (int j = 0; j < 256; ++j) dlog_[j] = p[j] - (j == tgt ? 1.f : 0.f);
      for (int i = 0; i < H; ++i) dh_[i] = dh_next_[i];
      for (int j = 0; j < 256; ++j) {
        const float d = dlog_[j];
        const float* wo = Wo_.data() + static_cast<size_t>(j) * H;
        float* g = gWo + static_cast<size_t>(j) * H;
        gbo[j] += d;
        for (int i = 0; i < H; ++i) {
          g[i] += d * ht[i];
          dh_[i] += d * wo[i];
        }
      }
      // Cell.
      for (int i = 0; i < H; ++i) {
        const float ig = z[i], fg = z[H + i], og = z[2 * H + i], gg = z[3 * H + i];
        const float tc = std::tanh(ct[i]);
        const float d_o = dh_[i] * tc;
        const float dc = dh_[i] * og * (1.f - tc * tc) + dc_[i];
        const float d_i = dc * gg, d_g = dc * ig, d_f = dc * cprev[i];
        dz_[i] = d_i * ig * (1.f - ig);
        dz_[H + i] = d_f * fg * (1.f - fg);
        dz_[2 * H + i] = d_o * og * (1.f - og);
        dz_[3 * H + i] = d_g * (1.f - gg * gg);
        dc_[i] = dc * fg;
      }
      const int x = hist_in_[t], l = hist_lane_[t];
      float* gx = gWx + static_cast<size_t>(x) * G;
      float* gl = gWl + static_cast<size_t>(l) * G;
      for (int i = 0; i < H; ++i) dh_next_[i] = 0.f;
      for (int r = 0; r < G; ++r) {
        const float d = dz_[r];
        gx[r] += d;
        gl[r] += d;
        gb[r] += d;
        const float* wh = Wh_.data() + static_cast<size_t>(r) * H;
        float* g = gWh + static_cast<size_t>(r) * H;
        for (int i = 0; i < H; ++i) {
          g[i] += d * hprev[i];
          dh_next_[i] += d * wh[i];
        }
      }
    }
    adam();
  }

  void adam() {
    ++step_;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-6f;
    const float c1 = 1.f / (1.f - std::pow(b1, static_cast<float>(step_)));
    const float c2 = 1.f / (1.f - std::pow(b2, static_cast<float>(step_)));
    const float lr = lr_;
    size_t off = 0;
    auto apply = [&](std::vector<float>& w) {
      float* g = grad_.data() + off;
      float* m = m_.data() + off;
      float* v = v_.data() + off;
      const size_t n = w.size();
      for (size_t i = 0; i < n; ++i) {
        float gi = g[i];
        if (gi > 2.f) gi = 2.f;
        if (gi < -2.f) gi = -2.f;
        m[i] = b1 * m[i] + (1.f - b1) * gi;
        v[i] = b2 * v[i] + (1.f - b2) * gi * gi;
        w[i] -= lr * (m[i] * c1) / (std::sqrt(v[i] * c2) + eps);
        g[i] = 0.f;
      }
      off += n;
    };
    apply(Wx_);
    apply(Wl_);
    apply(Wh_);
    apply(b_);
    apply(Wo_);
    apply(bo_);
  }

  int H_, T_, S_;
  float lr_;
  int t_ = 0;
  int step_ = 0;
  std::vector<float> Wx_, Wl_, Wh_, b_, Wo_, bo_;
  std::vector<float> grad_, m_, v_;
  std::vector<float> h_, c_, h0_, c0_;
  std::vector<std::uint8_t> hist_in_, hist_lane_, hist_tgt_;
  std::vector<float> hist_gate_, hist_c_, hist_h_, hist_p_;
  std::vector<float> dz_, dh_, dc_, dh_next_, dlog_;
  float cum_[257];
};

}  // namespace spectra8::detail
