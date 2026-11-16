#include "zt_calib.h"

#include <math.h>

namespace zt {

static void reset_monitor(Detector& d) {
    d.n_win = 0;
    d.win_sum = 0.0f;
    d.n_burn = 0;
    d.burn_mean = 0.0f;
    d.burn_m2 = 0.0f;
    for (int i = 0; i < VOTE_N; ++i) d.flags[i] = 0;
    d.pos = 0;
    d.count = 0;
}

void dequantize(const int8_t q[D], float scale, int zero, float x[D]) {
    for (int i = 0; i < D; ++i) x[i] = (float)((int)q[i] - zero) * scale;
}

float score(const float x[D], const Model& m) {
    float s = 0.0f;
    for (int i = 0; i < D; ++i) {
        const float e = x[i] - m.mu[i];
        s += e * e / m.var[i];
    }
    return s / (float)D;
}

float lookup(const Factory& f, const float table[TABLE], int n) {
    const float s = (float)n;
    if (s <= f.seconds[0]) return table[0];
    for (int i = 1; i < TABLE; ++i) {
        if (s <= f.seconds[i]) {
            const float t = (s - f.seconds[i - 1]) / (f.seconds[i] - f.seconds[i - 1]);
            return table[i - 1] + t * (table[i] - table[i - 1]);
        }
    }
    return table[TABLE - 1];
}

void begin(Detector& d, const Factory& f, bool use_vote) {
    d.factory = &f;
    d.phase = CALIBRATING;
    d.use_vote = use_vote;
    d.n_cal = 0;
    for (int i = 0; i < D; ++i) {
        d.model.mu[i] = 0.0f;
        d.model.var[i] = f.var[i];
        d.m2[i] = 0.0f;
    }
    d.model.thr = 0.0f;
    reset_monitor(d);
}

// Running mean and sum of squared deviations (Welford). Stays accurate in 32-bit floats,
// unlike sum-of-squares minus squared-sum.
void add_calibration_window(Detector& d, const float x[D]) {
    if (d.phase != CALIBRATING) return;
    d.n_cal += 1;
    for (int i = 0; i < D; ++i) {
        const float delta = x[i] - d.model.mu[i];
        d.model.mu[i] += delta / (float)d.n_cal;
        d.m2[i] += delta * (x[i] - d.model.mu[i]);
    }
}

bool finish_calibration(Detector& d) {
    if (d.phase != CALIBRATING || d.n_cal < 1) return false;
    const Factory& f = *d.factory;
    const float n = (float)d.n_cal;
    // Mix of factory and measured variances: with few windows the factory value dominates.
    for (int i = 0; i < D; ++i) {
        const float measured = d.m2[i] / n;
        d.model.var[i] = (f.n0 * f.var[i] + n * measured) / (f.n0 + n) + VAR_FLOOR;
    }
    d.model.thr = lookup(f, f.thr_table, d.n_cal);
    reset_monitor(d);
    d.phase = f.burn_clips > 0 ? BURN_IN : MONITORING;
    return true;
}

Result push_window(Detector& d, const float x[D]) {
    Result r = {false, 0.0f, false, false};
    if (d.phase == CALIBRATING) return r;

    d.win_sum += score(x, d.model);
    d.n_win += 1;
    if (d.n_win < WINS) return r;

    r.clip_done = true;
    r.clip_score = d.win_sum / (float)WINS;
    d.n_win = 0;
    d.win_sum = 0.0f;

    if (d.phase == BURN_IN) {
        d.n_burn += 1;
        const float delta = r.clip_score - d.burn_mean;
        d.burn_mean += delta / (float)d.n_burn;
        d.burn_m2 += delta * (r.clip_score - d.burn_mean);
        if (d.n_burn >= d.factory->burn_clips) {
            const Factory& f = *d.factory;
            const float k = lookup(f, d.use_vote ? f.k_vote : f.k_clip, d.n_cal);
            d.model.thr = d.burn_mean + k * sqrtf(d.burn_m2 / (float)d.n_burn);
            d.phase = MONITORING;
        }
        return r;
    }

    r.above = r.clip_score > d.model.thr;
    d.count += (r.above ? 1 : 0) - (int)d.flags[d.pos];
    d.flags[d.pos] = r.above ? 1 : 0;
    d.pos = (d.pos + 1) % VOTE_N;
    r.alarm = d.use_vote ? (d.count >= VOTE_M) : r.above;
    return r;
}

void resume(Detector& d, const Factory& f, bool use_vote, const Model& saved) {
    begin(d, f, use_vote);
    d.model = saved;
    d.phase = MONITORING;
}

}  // namespace zt
