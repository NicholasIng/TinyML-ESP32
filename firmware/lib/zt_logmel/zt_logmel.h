// Log-mel front end: 16 kHz int16 audio -> the int8 input window of the embedding CNN.
// C++ port of src/features.py with the settings of src/audio_config.py:
//   frames of 1024 samples every 512, periodic Hann window, power spectrum, 64 Slaney mel
//   bands 0-8000 Hz, dB = 10*log10(max(power, 1e-10)), no padding. One 1 s window is
//   16000 samples = 30 frames (the last 128 samples of each second are not used), stored
//   as window[mel * 30 + frame], which is the model's input layout (64, 30, 1).
//
// No dynamic memory. A LogMel is about 10 kB of RAM (one frame of samples plus the FFT
// work space); the tables in zt_logmel_tables.h are about 8 kB of flash.
//
//   zt::LogMel fe;                 // global or static: too big for a task stack
//   zt::logmel_begin(fe, ZT_IN_SCALE, ZT_IN_ZERO, 0.0f);
//   // whenever the microphone delivers n samples:
//   while (n > 0) {
//       bool done;
//       int used = zt::logmel_push(fe, pcm, n, window, &done);
//       pcm += used; n -= used;
//       if (done) { /* run the model on window */ }
//   }

#ifndef ZT_LOGMEL_H
#define ZT_LOGMEL_H

#include <stdint.h>

namespace zt {

const int SR = 16000;
const int N_FFT = 1024;
const int HOP = 512;
const int N_BINS = N_FFT / 2 + 1;
const int N_MELS = 64;
const int FRAMES = 30;               // frames per 1 s window
const int WIN_SAMPLES = 16000;       // samples per 1 s window
const int WINDOW_SIZE = N_MELS * FRAMES;

struct LogMel {
    float in_scale;      // int8 input quantization of the model (models/<name>_int8.json)
    int in_zero;
    float offset_db;     // added to every dB value: microphone level correction
    int16_t buf[N_FFT];  // samples of the frame being collected
    int n_buf;           // how many of them are valid
    int frame;           // next frame of the current window, 0..FRAMES-1
    int skip;            // samples still to drop before the next window starts
    float re[N_FFT];     // FFT work space
    float im[N_FFT];
};

// offset_db: 0 for audio at the level of the training data. For a microphone that is g
// times more sensitive, use -20*log10(g).
void logmel_begin(LogMel& s, float in_scale, int in_zero, float offset_db);

// One frame: N_FFT samples -> N_MELS values in dB (offset_db included).
void logmel_frame(LogMel& s, const int16_t pcm[N_FFT], float db[N_MELS]);

// dB -> int8 as the model expects it: clamp(round(db / in_scale) + in_zero, -128, 127).
int8_t quantize_db(float db, float in_scale, int in_zero);

// Feed n samples. Fills `window` frame by frame and stops as soon as a window is complete
// (*window_done = true). Returns how many samples were used; call again with the rest.
int logmel_push(LogMel& s, const int16_t* pcm, int n, int8_t window[WINDOW_SIZE],
                bool* window_done);

}  // namespace zt

#endif
