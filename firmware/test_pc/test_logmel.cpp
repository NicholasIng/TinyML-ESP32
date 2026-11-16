// PC test harness for firmware/lib/zt_logmel, driven by src/test_cpp_logmel.py.
// Reads raw 16-bit mono audio and writes the log-mel windows two ways:
//   <out>.db  float32, (windows, 64, 30): logmel_frame() on every frame
//   <out>.q   int8,    (windows, 64, 30): logmel_push(), fed in chunks of <chunk> samples
//             the way a microphone driver delivers audio
// Only this harness uses files and std::vector; the library itself does not.
//
//   test_logmel <audio.raw> <out> <chunk> <in_scale> <in_zero> <offset_db>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "zt_logmel.h"

static zt::LogMel fe;   // about 10 kB: static, as in the firmware

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: test_logmel <audio.raw> <out> <chunk> <in_scale> <in_zero> <offset_db>\n");
        return 2;
    }
    const int chunk = std::atoi(argv[3]);
    const float in_scale = (float)std::atof(argv[4]);
    const int in_zero = std::atoi(argv[5]);
    const float offset_db = (float)std::atof(argv[6]);

    std::FILE* fi = std::fopen(argv[1], "rb");
    if (!fi) return 1;
    std::fseek(fi, 0, SEEK_END);
    std::vector<int16_t> pcm((size_t)std::ftell(fi) / sizeof(int16_t));
    std::fseek(fi, 0, SEEK_SET);
    if (std::fread(pcm.data(), sizeof(int16_t), pcm.size(), fi) != pcm.size()) return 1;
    std::fclose(fi);
    const int n_win = (int)(pcm.size() / zt::WIN_SAMPLES);

    // 1. every frame on its own, floats
    zt::logmel_begin(fe, in_scale, in_zero, offset_db);
    std::vector<float> db((size_t)n_win * zt::WINDOW_SIZE);
    float col[zt::N_MELS];
    for (int w = 0; w < n_win; ++w)
        for (int f = 0; f < zt::FRAMES; ++f) {
            zt::logmel_frame(fe, &pcm[(size_t)w * zt::WIN_SAMPLES + (size_t)f * zt::HOP], col);
            for (int m = 0; m < zt::N_MELS; ++m)
                db[(size_t)w * zt::WINDOW_SIZE + m * zt::FRAMES + f] = col[m];
        }

    // 2. streaming, int8
    zt::logmel_begin(fe, in_scale, in_zero, offset_db);
    std::vector<int8_t> q;
    int8_t window[zt::WINDOW_SIZE];
    for (size_t at = 0; at < pcm.size(); at += (size_t)chunk) {
        const int16_t* p = &pcm[at];
        int n = (int)(pcm.size() - at < (size_t)chunk ? pcm.size() - at : (size_t)chunk);
        while (n > 0) {
            bool done = false;
            const int used = zt::logmel_push(fe, p, n, window, &done);
            p += used;
            n -= used;
            if (done) q.insert(q.end(), window, window + zt::WINDOW_SIZE);
        }
    }
    if ((int)(q.size() / zt::WINDOW_SIZE) != n_win) {
        std::fprintf(stderr, "test_logmel: streaming produced %d windows, expected %d\n",
                     (int)(q.size() / zt::WINDOW_SIZE), n_win);
        return 1;
    }

    const std::string out(argv[2]);
    std::FILE* fo = std::fopen((out + ".db").c_str(), "wb");
    if (!fo) return 1;
    std::fwrite(db.data(), sizeof(float), db.size(), fo);
    std::fclose(fo);
    fo = std::fopen((out + ".q").c_str(), "wb");
    if (!fo) return 1;
    std::fwrite(q.data(), sizeof(int8_t), q.size(), fo);
    std::fclose(fo);
    return 0;
}
