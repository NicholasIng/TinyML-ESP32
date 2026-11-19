// Week 6b: the whole chain on the ESP32-S3, on MIMII clips stored in flash.
//
// For every 10 s clip: 16-bit audio -> zt_logmel (in chunks of 512 samples, as from the
// microphone) -> int8 window -> TensorFlow Lite Micro -> int8 embedding -> zt_calib.
// The first clip is used as the calibration audio (10 s of "verified normal" sound).
// Output lines (read by src/test_device.py):
//   I ...                         information
//   Q <clip> <win> <hex>          int8 model input, 1920 values
//   E <clip> <win> <hex>          int8 embedding, 32 values
//   S <clip> <factory> <calib>    clip score with the factory model / the calibrated model
//   T ...                         time per 1 s window of each step
//   M ...                         memory
//   D done                        end
// Anything that fails prints "X <reason>".

#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"
#include "zt_calib.h"
#include "zt_factory.h"
#include "zt_logmel.h"
#include "zt_model_data.h"

extern const uint8_t clips_start[] asm("_binary_bench_clips_raw_start");
extern const uint8_t clips_end[] asm("_binary_bench_clips_raw_end");

namespace {

constexpr int kArenaSize = 128 * 1024;
constexpr int kChunk = 512;               // samples per logmel_push call
constexpr int kClipSamples = 10 * zt::SR;

alignas(16) uint8_t arena[kArenaSize];
zt::LogMel fe;                            // about 10 kB: static, not on the stack
zt::Detector det;
int8_t window[zt::WINDOW_SIZE];
float emb[zt::WINS][zt::D];
char hex[2 * zt::WINDOW_SIZE + 1];

void print_hex(char tag, int clip, int win, const int8_t* p, int n) {
    static const char digits[] = "0123456789abcdef";
    for (int i = 0; i < n; ++i) {
        hex[2 * i] = digits[(uint8_t)p[i] >> 4];
        hex[2 * i + 1] = digits[(uint8_t)p[i] & 15];
    }
    hex[2 * n] = 0;
    printf("%c %d %d %s\n", tag, clip, win, hex);
}

void run() {
    printf("I zt device_test 1\n");
    const tflite::Model* model = tflite::GetModel(ZT_MODEL);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("X model schema %lu, expected %d\n", (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    static tflite::MicroMutableOpResolver<6> ops;
    ops.AddAdd();
    ops.AddConv2D();
    ops.AddFullyConnected();
    ops.AddMaxPool2D();
    ops.AddMean();
    ops.AddMul();
    static tflite::MicroInterpreter interpreter(model, ops, arena, kArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("X AllocateTensors failed (arena %d bytes)\n", kArenaSize);
        return;
    }
    TfLiteTensor* in = interpreter.input(0);
    TfLiteTensor* out = interpreter.output(0);
    if (in->type != kTfLiteInt8 || out->type != kTfLiteInt8 || in->bytes != zt::WINDOW_SIZE ||
        out->bytes != zt::D) {
        printf("X unexpected model tensors: in %d bytes, out %d bytes\n", (int)in->bytes, (int)out->bytes);
        return;
    }
    printf("I input scale %.7g zero %d (header %.7g %d), output scale %.7g zero %d (header %.7g %d)\n",
           in->params.scale, (int)in->params.zero_point, ZT_IN_SCALE, ZT_IN_ZERO, out->params.scale,
           (int)out->params.zero_point, ZT_EMB_SCALE, ZT_EMB_ZERO);

    zt::Model factory;
    memcpy(factory.mu, ZT_FACTORY.mu, sizeof factory.mu);
    memcpy(factory.var, ZT_FACTORY.var, sizeof factory.var);
    factory.thr = 0;
    zt::begin(det, ZT_FACTORY, true);

    const int16_t* pcm = (const int16_t*)clips_start;
    const int n_clips = (int)((clips_end - clips_start) / 2 / kClipSamples);
    printf("I %d clips in flash\n", n_clips);
    int64_t t_mel = 0, t_model = 0, t_score = 0, t_worst = 0;
    int n_win = 0;

    for (int c = 0; c < n_clips; ++c) {
        zt::logmel_begin(fe, in->params.scale, in->params.zero_point, 0.0f);
        const int16_t* p = pcm + (size_t)c * kClipSamples;
        int left = kClipSamples;
        int w = 0;
        int64_t t_win = 0;
        while (left > 0 && w < zt::WINS) {
            bool done = false;
            const int n = left < kChunk ? left : kChunk;
            int64_t t0 = esp_timer_get_time();
            const int used = zt::logmel_push(fe, p, n, window, &done);
            const int64_t dt_mel = esp_timer_get_time() - t0;
            t_mel += dt_mel;
            t_win += dt_mel;
            p += used;
            left -= used;
            if (!done) continue;

            memcpy(in->data.int8, window, zt::WINDOW_SIZE);
            t0 = esp_timer_get_time();
            if (interpreter.Invoke() != kTfLiteOk) {
                printf("X Invoke failed\n");
                return;
            }
            const int64_t dt_model = esp_timer_get_time() - t0;
            t_model += dt_model;
            t_win += dt_model;
            if (t_win > t_worst) t_worst = t_win;
            t_win = 0;
            zt::dequantize(out->data.int8, out->params.scale, out->params.zero_point, emb[w]);
            print_hex('Q', c, w, window, zt::WINDOW_SIZE);
            print_hex('E', c, w, out->data.int8, zt::D);
            ++w;
            ++n_win;
        }
        if (w != zt::WINS) {
            printf("X clip %d gave %d windows\n", c, w);
            return;
        }
        if (c == 0) {   // calibration on the first clip
            for (int i = 0; i < zt::WINS; ++i) zt::add_calibration_window(det, emb[i]);
            if (!zt::finish_calibration(det)) {
                printf("X finish_calibration failed\n");
                return;
            }
        }
        const int64_t t0 = esp_timer_get_time();
        float s_factory = 0, s_calib = 0;
        for (int i = 0; i < zt::WINS; ++i) {
            s_factory += zt::score(emb[i], factory);
            s_calib += zt::score(emb[i], det.model);
        }
        t_score += esp_timer_get_time() - t0;
        printf("S %d %.7g %.7g\n", c, s_factory / zt::WINS, s_calib / zt::WINS);
        vTaskDelay(1);   // let the idle task run (watchdog)
    }

    printf("T per 1 s window: logmel %.1f ms, model %.1f ms, score %.3f ms, worst window %.1f ms "
           "(%d windows, CPU %d MHz)\n",
           t_mel / 1000.0 / n_win, t_model / 1000.0 / n_win, t_score / 1000.0 / n_win / 2,
           t_worst / 1000.0, n_win, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
    printf("M arena used %u of %d bytes; LogMel %u bytes; Detector %u bytes; model %u bytes in flash\n",
           (unsigned)interpreter.arena_used_bytes(), kArenaSize, (unsigned)sizeof(zt::LogMel),
           (unsigned)sizeof(zt::Detector), (unsigned)sizeof(ZT_MODEL));
    printf("M internal RAM free %u (lowest %u, largest block %u); main stack unused %u bytes\n",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
           (unsigned)uxTaskGetStackHighWaterMark(NULL));
}

}  // namespace

extern "C" void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(300));   // let the PC open the port after the reset
    run();
    printf("D done\n");
    for (;;) vTaskDelay(pdMS_TO_TICKS(1000));
}
