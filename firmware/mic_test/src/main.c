// Week 6, step 1: microphone test for the ESP32-S3 + INMP441.
//
// Reads the INMP441 over I2S at 16 kHz and sends two kinds of text lines on the console
// (works on either USB port of the board, no commands needed):
//   L ...                          once per second: level of both I2S slots, for wiring checks
//   A <seq> <checksum> <base64>    256 samples (16 ms) of 16-bit audio, little endian
// src/mic_record.py on the PC shows the levels and saves the audio as a wav file.
//
// Wiring (INMP441 -> ESP32-S3):
//   VDD -> 3V3     GND -> GND     L/R -> GND (left slot)
//   SCK -> GPIO 4  WS  -> GPIO 5  SD  -> GPIO 6
//
// The INMP441 sends 24-bit samples, left aligned in a 32-bit slot. We keep the top 16 bits,
// as the C++ front end expects (firmware/lib/zt_logmel). The level difference to the
// training data is corrected later with one number, offset_db.

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "driver/i2s_std.h"
#include "esp_attr.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/stream_buffer.h"
#include "freertos/task.h"

#define PIN_SCK GPIO_NUM_4
#define PIN_WS GPIO_NUM_5
#define PIN_SD GPIO_NUM_6

#define SR 16000
#define READ_FRAMES 256      // stereo frames per I2S read (16 ms)
#define LINE_SAMPLES 256     // samples per "A" line (16 ms)
#define BUFFER_BYTES 65536   // 2 s of audio between the reader and the console

typedef struct {
    float rms_db[2];    // AC level of the left and right slot, dB relative to full scale
    int peak[2];        // largest absolute 16-bit sample
    uint32_t zeros[2];  // samples that were exactly 0
    int slot;           // slot that is sent (0 = left, 1 = right)
    uint32_t ovf;       // I2S receive overflows so far (samples lost inside the chip)
    uint32_t drop;      // samples dropped because the console was too slow
} level_t;

static i2s_chan_handle_t rx;
static StreamBufferHandle_t audio;
static QueueHandle_t levels;
static volatile uint32_t n_ovf;
static volatile uint32_t n_drop;

static bool IRAM_ATTR on_overflow(i2s_chan_handle_t h, i2s_event_data_t *e, void *ctx) {
    (void)h;
    (void)e;
    (void)ctx;
    n_ovf++;
    return false;
}

static void mic_init(void) {
    i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan.dma_desc_num = 8;
    chan.dma_frame_num = 256;  // 8 x 16 ms of DMA buffer
    ESP_ERROR_CHECK(i2s_new_channel(&chan, NULL, &rx));

    i2s_std_config_t std = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SR),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = PIN_SCK,
            .ws = PIN_WS,
            .dout = I2S_GPIO_UNUSED,
            .din = PIN_SD,
            .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false},
        },
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx, &std));
    i2s_event_callbacks_t cbs = {.on_recv_q_ovf = on_overflow};
    ESP_ERROR_CHECK(i2s_channel_register_event_callback(rx, &cbs, NULL));
    ESP_ERROR_CHECK(i2s_channel_enable(rx));
}

// Reads the microphone, measures both slots, and passes the chosen slot on as 16-bit audio.
static void reader_task(void *arg) {
    (void)arg;
    static int32_t raw[READ_FRAMES * 2];
    static int16_t mono[READ_FRAMES];
    double sum[2] = {0, 0}, sum2[2] = {0, 0};
    int peak[2] = {0, 0};
    uint32_t zeros[2] = {0, 0};
    uint32_t n = 0;
    int slot = -1;  // chosen after the first second: the slot that carries the signal

    for (;;) {
        size_t got = 0;
        if (i2s_channel_read(rx, raw, sizeof raw, &got, portMAX_DELAY) != ESP_OK) continue;
        const int frames = (int)(got / (2 * sizeof(int32_t)));
        for (int i = 0; i < frames; ++i) {
            for (int c = 0; c < 2; ++c) {
                const int16_t s = (int16_t)(raw[2 * i + c] >> 16);
                sum[c] += s;
                sum2[c] += (double)s * s;
                const int a = s < 0 ? -s : s;
                if (a > peak[c]) peak[c] = a;
                if (s == 0) zeros[c]++;
            }
            if (slot >= 0) mono[i] = (int16_t)(raw[2 * i + slot] >> 16);
        }
        n += (uint32_t)frames;

        if (slot >= 0) {
            const size_t bytes = (size_t)frames * sizeof(int16_t);
            if (xStreamBufferSpacesAvailable(audio) < bytes)
                n_drop += (uint32_t)frames;
            else
                xStreamBufferSend(audio, mono, bytes, 0);
        }

        if (n >= SR) {
            level_t lv;
            double var[2];
            for (int c = 0; c < 2; ++c) {
                const double mean = sum[c] / n;
                var[c] = sum2[c] / n - mean * mean;
                lv.rms_db[c] = (float)(10.0 * log10(var[c] / (32768.0 * 32768.0) + 1e-20));
                lv.peak[c] = peak[c];
                lv.zeros[c] = zeros[c];
                sum[c] = sum2[c] = 0;
                peak[c] = 0;
                zeros[c] = 0;
            }
            if (slot < 0) slot = var[1] > var[0] ? 1 : 0;
            lv.slot = slot;
            lv.ovf = n_ovf;
            lv.drop = n_drop;
            n = 0;
            xQueueSend(levels, &lv, 0);
        }
    }
}

static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static int base64(const uint8_t *in, int n, char *out) {
    int o = 0;
    for (int i = 0; i < n; i += 3) {
        const uint32_t v = (uint32_t)in[i] << 16 | (i + 1 < n ? (uint32_t)in[i + 1] << 8 : 0) |
                           (i + 2 < n ? in[i + 2] : 0);
        out[o++] = B64[v >> 18 & 63];
        out[o++] = B64[v >> 12 & 63];
        out[o++] = i + 1 < n ? B64[v >> 6 & 63] : '=';
        out[o++] = i + 2 < n ? B64[v & 63] : '=';
    }
    out[o] = 0;
    return o;
}

void app_main(void) {
    audio = xStreamBufferCreate(BUFFER_BYTES, LINE_SAMPLES * sizeof(int16_t));
    levels = xQueueCreate(4, sizeof(level_t));
    mic_init();
    printf("\nI zt mic_test 1: INMP441 SCK=%d WS=%d SD=%d, %d Hz, %d samples per A line\n", PIN_SCK,
           PIN_WS, PIN_SD, SR, LINE_SAMPLES);
    esp_log_level_set("*", ESP_LOG_WARN);  // keep the console for our lines
    xTaskCreatePinnedToCore(reader_task, "mic", 4096, NULL, 10, NULL, 1);

    static int16_t pcm[LINE_SAMPLES];
    static char text[4 * (LINE_SAMPLES * 2 + 2) / 3 + 4];
    size_t have = 0;
    uint32_t seq = 0;
    for (;;) {
        level_t lv;
        while (xQueueReceive(levels, &lv, 0) == pdTRUE) {
            printf("L slot=%c rms_l=%.1f rms_r=%.1f peak_l=%d peak_r=%d zeros_l=%lu zeros_r=%lu ovf=%lu drop=%lu\n",
                   lv.slot ? 'R' : 'L', lv.rms_db[0], lv.rms_db[1], lv.peak[0], lv.peak[1],
                   (unsigned long)lv.zeros[0], (unsigned long)lv.zeros[1], (unsigned long)lv.ovf,
                   (unsigned long)lv.drop);
        }
        have += xStreamBufferReceive(audio, (uint8_t *)pcm + have, sizeof pcm - have, pdMS_TO_TICKS(50));
        if (have < sizeof pcm) continue;
        have = 0;
        const uint8_t *b = (const uint8_t *)pcm;
        uint32_t check = 0;
        for (size_t i = 0; i < sizeof pcm; ++i) check += b[i];
        base64(b, sizeof pcm, text);
        printf("A %lu %lu %s\n", (unsigned long)seq++, (unsigned long)(check & 0xFFFF), text);
    }
}
