#include "zt_logmel.h"

#include <math.h>
#include <string.h>

#include "zt_logmel_tables.h"

namespace zt {

static_assert(ZT_TABLE_SR == SR && ZT_TABLE_N_FFT == N_FFT && ZT_TABLE_N_MELS == N_MELS,
              "zt_logmel_tables.h was generated for other audio settings");
static_assert(WIN_SAMPLES >= (FRAMES - 1) * HOP + N_FFT, "frames do not fit in one window");

void logmel_begin(LogMel& s, float in_scale, int in_zero, float offset_db) {
    s.in_scale = in_scale;
    s.in_zero = in_zero;
    s.offset_db = offset_db;
    s.n_buf = 0;
    s.frame = 0;
    s.skip = 0;
}

void logmel_frame(LogMel& s, const int16_t pcm[N_FFT], float db[N_MELS]) {
    float* re = s.re;
    float* im = s.im;

    // Scale to -1..1, apply the periodic Hann window 0.5 - 0.5*cos(2*pi*n/N), and store in
    // bit-reversed order for the in-place FFT. cos for the second half is minus the first.
    int j = 0;
    for (int n = 0; n < N_FFT; ++n) {
        const float c = n < N_FFT / 2 ? ZT_COS[n] : -ZT_COS[n - N_FFT / 2];
        re[j] = (float)pcm[n] * (1.0f / 32768.0f) * (0.5f - 0.5f * c);
        im[j] = 0.0f;
        int bit = N_FFT >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
    }

    // Radix-2 FFT. Twiddle factor exp(-2*pi*i*k/N) = ZT_COS[k] - i*ZT_SIN[k].
    for (int half = 1; half < N_FFT; half <<= 1) {
        const int step = (N_FFT / 2) / half;
        for (int start = 0; start < N_FFT; start += 2 * half) {
            for (int k = 0; k < half; ++k) {
                const float wr = ZT_COS[k * step];
                const float wi = -ZT_SIN[k * step];
                const int a = start + k;
                const int b = a + half;
                const float tr = re[b] * wr - im[b] * wi;
                const float ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
    }

    // Power spectrum of the bins 0..N/2, kept in re[].
    for (int k = 0; k < N_BINS; ++k) re[k] = re[k] * re[k] + im[k] * im[k];

    // Mel filter bank (triangles stored without their zeros), then dB.
    const float* w = ZT_MEL_W;
    for (int m = 0; m < N_MELS; ++m) {
        const float* p = re + ZT_MEL_START[m];
        const int len = ZT_MEL_LEN[m];
        float acc = 0.0f;
        for (int i = 0; i < len; ++i) acc += w[i] * p[i];
        w += len;
        if (acc < 1e-10f) acc = 1e-10f;
        db[m] = 10.0f * log10f(acc) + s.offset_db;
    }
}

int8_t quantize_db(float db, float in_scale, int in_zero) {
    float q = rintf(db / in_scale) + (float)in_zero;
    if (q < -128.0f) q = -128.0f;
    if (q > 127.0f) q = 127.0f;
    return (int8_t)q;
}

int logmel_push(LogMel& s, const int16_t* pcm, int n, int8_t window[WINDOW_SIZE],
                bool* window_done) {
    *window_done = false;
    int used = 0;
    while (used < n) {
        if (s.skip > 0) {   // unused tail of the previous second
            const int k = s.skip < n - used ? s.skip : n - used;
            s.skip -= k;
            used += k;
            continue;
        }
        const int k = N_FFT - s.n_buf < n - used ? N_FFT - s.n_buf : n - used;
        memcpy(s.buf + s.n_buf, pcm + used, (size_t)k * sizeof(int16_t));
        s.n_buf += k;
        used += k;
        if (s.n_buf < N_FFT) break;

        float db[N_MELS];
        logmel_frame(s, s.buf, db);
        for (int m = 0; m < N_MELS; ++m)
            window[m * FRAMES + s.frame] = quantize_db(db[m], s.in_scale, s.in_zero);
        s.frame += 1;
        if (s.frame == FRAMES) {
            s.frame = 0;
            s.n_buf = 0;
            s.skip = WIN_SAMPLES - ((FRAMES - 1) * HOP + N_FFT);
            *window_done = true;
            return used;
        }
        memmove(s.buf, s.buf + HOP, (size_t)(N_FFT - HOP) * sizeof(int16_t));
        s.n_buf = N_FFT - HOP;
    }
    return used;
}

}  // namespace zt
