// On-device calibration, burn-in and alarm vote. C++ port of src/calibration.py
// (method "burn_5min_vote", and "burn_5min" / "centroid_var_cal" as options).
//
// No dynamic memory, no library beyond <math.h>: every array has a fixed size and a whole
// Detector is under 500 bytes. One call per 1 s window:
//
//   zt::Detector det;
//   zt::begin(det, ZT_FACTORY, /*use_vote=*/true);
//   // user confirms the machine is healthy, then for 10-30 seconds:
//   zt::add_calibration_window(det, emb);
//   zt::finish_calibration(det);
//   // from then on, forever:
//   zt::Result r = zt::push_window(det, emb);
//   if (r.clip_done && r.alarm) ...
//
// Phases: CALIBRATING (verified normal audio) -> BURN_IN (listens, raises no alarm, sets
// the threshold from its own clip scores) -> MONITORING.
// A window score is the diagonal Mahalanobis distance mean_d (x[d]-mu[d])^2 / var[d];
// a clip is WINS windows and its score is the mean of their scores.

#ifndef ZT_CALIB_H
#define ZT_CALIB_H

#include <stdint.h>

namespace zt {

const int D = 32;        // numbers per embedding
const int WINS = 10;     // 1 s windows per clip
const int VOTE_N = 12;   // the alarm looks at the last 12 clips (2 minutes) ...
const int VOTE_M = 9;    // ... and needs 9 of them above the threshold
const int TABLE = 4;     // entries in the factory tables
const float VAR_FLOOR = 1e-6f;

// Constants from the PC (src/export_factory.py). Never changed on the device.
struct Factory {
    float mu[D];             // mean of the factory normal windows (starting point only)
    float var[D];            // variance of one healthy machine around its own mean
    float n0;                // how many windows the factory variances are "worth"
    int burn_clips;          // burn-in length in clips; 0 = no burn-in, use thr_table
    float seconds[TABLE];    // calibration lengths the tables were made for, ascending
    float thr_table[TABLE];  // clip threshold right after calibration (no burn-in)
    float k_clip[TABLE];     // burn-in threshold = mean + k * std, single-clip alarm
    float k_vote[TABLE];     // same, lower k for the VOTE_M-of-VOTE_N alarm
};

// What calibration produces. Plain data: can be stored in flash and loaded at boot.
struct Model {
    float mu[D];
    float var[D];
    float thr;
};

enum Phase { CALIBRATING, BURN_IN, MONITORING };

struct Detector {
    const Factory* factory;
    Phase phase;
    bool use_vote;
    Model model;
    // calibration: running mean (in model.mu) and sum of squared deviations per dimension
    int n_cal;
    float m2[D];
    // clip being collected
    int n_win;
    float win_sum;
    // burn-in: count, running mean and sum of squared deviations of the clip scores
    int n_burn;
    float burn_mean;
    float burn_m2;
    // vote: one flag per clip in a ring buffer, and how many are set
    uint8_t flags[VOTE_N];
    int pos;
    int count;
};

struct Result {
    bool clip_done;    // true on every WINS-th window; the fields below are valid only then
    float clip_score;
    bool above;        // clip score above the threshold (always false during burn-in)
    bool alarm;        // the decision: vote result, or `above` when use_vote is false
};

// Embedding as the int8 model returns it -> floats. scale and zero are the model's output
// quantization (models/<name>_int8.json: out_scale, out_zero).
void dequantize(const int8_t q[D], float scale, int zero, float x[D]);

// Anomaly score of one window.
float score(const float x[D], const Model& m);

// Table value for n calibration windows; linear between entries, clamped at both ends.
float lookup(const Factory& f, const float table[TABLE], int n);

// Start (or restart) a calibration. use_vote: alarm on 9 of 12 clips instead of one clip.
void begin(Detector& d, const Factory& f, bool use_vote);

// One window of verified normal audio. Call 5-30 times, then finish_calibration().
void add_calibration_window(Detector& d, const float x[D]);

// Fixes mean and variances and starts the burn-in (or monitoring when burn_clips is 0).
// Returns false, and changes nothing, when no window was added.
bool finish_calibration(Detector& d);

// One window after calibration. Does nothing while the detector is still calibrating.
Result push_window(Detector& d, const float x[D]);

// Continue monitoring with a model saved earlier (skips calibration and burn-in).
void resume(Detector& d, const Factory& f, bool use_vote, const Model& saved);

}  // namespace zt

#endif
