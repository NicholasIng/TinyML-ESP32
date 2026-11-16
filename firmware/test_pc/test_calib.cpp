// PC test harness for firmware/lib/zt_calib, driven by src/test_cpp_calib.py.
// Reads one test case (float32 numbers), runs the detector exactly as the firmware will
// (one window at a time), and writes what it computed for comparison with calibration.py.
// Only this harness uses files and std::vector; the library itself does not.
//
//   test_calib <case.bin> <out.bin>
//
// case.bin: use_vote, n_cal, n_stream, Factory fields in struct order (ints as floats),
//           n_cal x D calibration windows, n_stream x D windows heard afterwards.
// out.bin : mu[D], var[D], thr after calibration, final thr, sizeof(Detector),
//           then score, above, alarm for every clip in the monitoring phase.
// Built with -DZT_USE_HEADER it ignores the Factory in the file and uses the generated
// firmware/generated/zt_factory.h instead (checks the header writer).

#include <cstdio>
#include <vector>

#include "zt_calib.h"
#ifdef ZT_USE_HEADER
#include "zt_factory.h"
#endif

static bool fail(const char* what) {
    std::fprintf(stderr, "test_calib: %s\n", what);
    return false;
}

static bool run(const char* in_path, const char* out_path) {
    std::FILE* fi = std::fopen(in_path, "rb");
    if (!fi) return fail("cannot open case file");
    std::fseek(fi, 0, SEEK_END);
    const long bytes = std::ftell(fi);
    std::fseek(fi, 0, SEEK_SET);
    std::vector<float> in((size_t)bytes / sizeof(float));
    const size_t got = std::fread(in.data(), sizeof(float), in.size(), fi);
    std::fclose(fi);
    if (got != in.size() || in.size() < 3) return fail("short case file");

    size_t p = 0;
    const bool use_vote = in[p++] != 0.0f;
    const int n_cal = (int)in[p++];
    const int n_stream = (int)in[p++];
    const size_t n_factory = 2 * zt::D + 2 + 4 * zt::TABLE;
    if (in.size() != 3 + n_factory + (size_t)(n_cal + n_stream) * zt::D)
        return fail("case file has the wrong size");

    zt::Factory from_file;
    for (int i = 0; i < zt::D; ++i) from_file.mu[i] = in[p++];
    for (int i = 0; i < zt::D; ++i) from_file.var[i] = in[p++];
    from_file.n0 = in[p++];
    from_file.burn_clips = (int)in[p++];
    for (int i = 0; i < zt::TABLE; ++i) from_file.seconds[i] = in[p++];
    for (int i = 0; i < zt::TABLE; ++i) from_file.thr_table[i] = in[p++];
    for (int i = 0; i < zt::TABLE; ++i) from_file.k_clip[i] = in[p++];
    for (int i = 0; i < zt::TABLE; ++i) from_file.k_vote[i] = in[p++];
#ifdef ZT_USE_HEADER
    (void)from_file;
    const zt::Factory& factory = ZT_FACTORY;
#else
    const zt::Factory& factory = from_file;
#endif
    const float* cal = &in[p];
    const float* stream = cal + (size_t)n_cal * zt::D;

    std::vector<float> out;
    zt::Detector det;
    zt::begin(det, factory, use_vote);
    if (zt::finish_calibration(det)) return fail("calibration finished without any window");
    if (zt::push_window(det, stream).clip_done || det.n_win != 0)
        return fail("push_window must do nothing while calibrating");
    for (int i = 0; i < n_cal; ++i) zt::add_calibration_window(det, cal + (size_t)i * zt::D);
    if (!zt::finish_calibration(det)) return fail("finish_calibration failed");
    zt::add_calibration_window(det, cal);   // must be ignored now
    if (det.n_cal != n_cal) return fail("calibration window accepted after finishing");

    for (int i = 0; i < zt::D; ++i) out.push_back(det.model.mu[i]);
    for (int i = 0; i < zt::D; ++i) out.push_back(det.model.var[i]);
    out.push_back(det.model.thr);
    const size_t thr_slot = out.size();
    out.push_back(0.0f);
    out.push_back((float)sizeof(zt::Detector));

    int first_monitor = -1;   // index of the first window of the monitoring phase
    for (int i = 0; i < n_stream; ++i) {
        if (det.phase == zt::MONITORING && first_monitor < 0) first_monitor = i;
        const bool burn_in = det.phase == zt::BURN_IN;
        const zt::Result r = zt::push_window(det, stream + (size_t)i * zt::D);
        if (burn_in && (r.above || r.alarm)) return fail("alarm during burn-in");
        if (r.clip_done && !burn_in) {
            out.push_back(r.clip_score);
            out.push_back(r.above ? 1.0f : 0.0f);
            out.push_back(r.alarm ? 1.0f : 0.0f);
        }
    }
    if (det.phase != zt::MONITORING) return fail("stream ended before monitoring started");
    out[thr_slot] = det.model.thr;

    // A model saved after burn-in and loaded again (resume) must give the same decisions.
    if (first_monitor >= 0) {
        zt::Detector again;
        zt::resume(again, factory, use_vote, det.model);
        size_t q = thr_slot + 2;
        for (int i = first_monitor; i < n_stream; ++i) {
            const zt::Result r = zt::push_window(again, stream + (size_t)i * zt::D);
            if (!r.clip_done) continue;
            if (r.clip_score != out[q] || r.above != (out[q + 1] != 0.0f) ||
                r.alarm != (out[q + 2] != 0.0f))
                return fail("resume() gives different results");
            q += 3;
        }
    }

    // dequantize: int8 -> float with the model's output scale and zero point
    int8_t q8[zt::D];
    float x[zt::D];
    for (int i = 0; i < zt::D; ++i) q8[i] = (int8_t)(i * 8 - 128);
    zt::dequantize(q8, 0.05f, 4, x);
    for (int i = 0; i < zt::D; ++i)
        if (x[i] != (float)(i * 8 - 128 - 4) * 0.05f) return fail("dequantize is wrong");

    // lookup: exact at the table entries, clamped outside, halfway in between
    const float* t = factory.k_vote;
    for (int i = 0; i < zt::TABLE; ++i)
        if (zt::lookup(factory, t, (int)factory.seconds[i]) != t[i]) return fail("lookup at an entry");
    if (zt::lookup(factory, t, 1) != t[0] || zt::lookup(factory, t, 1000) != t[zt::TABLE - 1])
        return fail("lookup outside the table");
    const int mid = (int)(factory.seconds[1] + factory.seconds[2]) / 2;
    const float want = t[1] + ((float)mid - factory.seconds[1]) /
                                  (factory.seconds[2] - factory.seconds[1]) * (t[2] - t[1]);
    if (zt::lookup(factory, t, mid) != want) return fail("lookup between entries");

    std::FILE* fo = std::fopen(out_path, "wb");
    if (!fo) return fail("cannot write output file");
    std::fwrite(out.data(), sizeof(float), out.size(), fo);
    std::fclose(fo);
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: test_calib <case.bin> <out.bin>\n");
        return 2;
    }
    return run(argv[1], argv[2]) ? 0 : 1;
}
