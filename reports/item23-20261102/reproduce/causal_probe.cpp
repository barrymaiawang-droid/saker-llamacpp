#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static std::string dump_dir;
static int dump_index = 0;
static bool dump_tensor(ggml_tensor * t, bool ask, void *) {
    const std::string name = t->name;
    if (name.rfind("l_out-", 0) != 0 || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) { return false; }
    if (ask) { return true; }
    std::vector<char> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    const std::string file = std::to_string(dump_index++) + "-" + name + ".bin";
    std::ofstream out(dump_dir + "/" + file, std::ios::binary);
    out.write(bytes.data(), bytes.size());
    printf("{\"file\":\"%s\",\"name\":\"%s\",\"ne\":[%lld,%lld,%lld,%lld]}\n", file.c_str(), name.c_str(),
           (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
    fflush(stdout);
    return true;
}

static std::vector<float> evaluate(llama_context * ctx, const std::vector<llama_token> & tokens, int n_vocab, int score_first) {
    llama_memory_clear(llama_get_memory(ctx), true);
    auto batch = llama_batch_init(512, 0, 1);
    std::vector<float> result;
    for (int start = 0; start < int(tokens.size()); start += 512) {
    batch.n_tokens = std::min(512, int(tokens.size()) - start);
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[start + i];
        batch.pos[i] = start + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    if (llama_decode(ctx, batch)) { throw std::runtime_error("decode failed"); }
    const auto * logits = llama_get_logits(ctx);
    const int first = std::max(score_first, start) - start;
    const int end = std::min(score_first + 128, start + batch.n_tokens) - start;
    if (end > first) { result.insert(result.end(), logits + size_t(first) * n_vocab, logits + size_t(end) * n_vocab); }
    }
    llama_batch_free(batch);
    if (result.size() != size_t(128) * n_vocab) { throw std::runtime_error("missing output rows"); }
    return result;
}

static void compare(const char * name, const std::vector<float> & a, const std::vector<float> & b, int n_vocab, int rows) {
    double max_abs = 0, sum_kl = 0, max_kl = 0;
    int changes = 0;
    for (int i = 0; i < rows; ++i) {
        const float * x = a.data() + size_t(i) * n_vocab;
        const float * y = b.data() + size_t(i) * n_vocab;
        const int ix = std::max_element(x, x + n_vocab) - x;
        const int iy = std::max_element(y, y + n_vocab) - y;
        changes += ix != iy;
        double zx = 0, zy = 0;
        for (int t = 0; t < n_vocab; ++t) {
            if (!std::isfinite(x[t]) || !std::isfinite(y[t])) { throw std::runtime_error("nonfinite logits"); }
            zx += std::exp(double(x[t] - x[ix]));
            zy += std::exp(double(y[t] - y[iy]));
            max_abs = std::max(max_abs, std::abs(double(x[t]) - y[t]));
        }
        double kl = 0;
        const double lx = std::log(zx) + x[ix], ly = std::log(zy) + y[iy];
        for (int t = 0; t < n_vocab; ++t) {
            const double lp = x[t] - lx, lq = y[t] - ly;
            kl += std::exp(lp) * (lp - lq);
        }
        sum_kl += kl;
        max_kl = std::max(max_kl, kl);
    }
    printf("{\"comparison\":\"%s\",\"rows\":%d,\"top_changes\":%d,\"max_abs_logit_delta\":%.9g,\"mean_kl\":%.9g,\"max_kl\":%.9g}\n",
           name, rows, changes, max_abs, sum_kl / rows, max_kl);
    fflush(stdout);
}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) { fprintf(stderr, "usage: causal_probe model.gguf code.txt gpu_layers [full_tokens]\n"); return 2; }
    const int n_full = argc == 5 ? std::atoi(argv[4]) : 256;
    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = std::atoi(argv[3]);
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { return 1; }
    const auto * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    std::ifstream f(argv[2]);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), n, true, true) != n || n < n_full + 128 || n_full < 256) { return 3; }
    auto cp = llama_context_default_params();
    cp.n_ctx = n_full + 256;
    cp.n_batch = 512;
    cp.n_ubatch = 256;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (const char * dir = std::getenv("CAUSAL_TENSOR_DUMP")) {
        dump_dir = dir;
        std::filesystem::create_directories(dump_dir);
        cp.cb_eval = dump_tensor;
    }
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) { return 1; }
    std::vector<llama_token> full(tokens.begin(), tokens.begin() + n_full);
    const int score_first = n_full - 256;
    auto a = evaluate(ctx, full, nv, score_first);
    if (!dump_dir.empty()) {
        llama_free(ctx); llama_model_free(model); llama_backend_free(); return 0;
    }
    auto repeat = evaluate(ctx, full, nv, score_first);
    compare("repeat", a, repeat, nv, 128);
    auto changed = full;
    std::copy(tokens.begin() + n_full, tokens.begin() + n_full + 128, changed.begin() + n_full - 128);
    auto b = evaluate(ctx, changed, nv, score_first);
    compare("changed_future", a, b, nv, 128);
    std::vector<llama_token> prefix(full.begin(), full.begin() + n_full - 128);
    auto c = evaluate(ctx, prefix, nv, score_first);
    compare("prefix_only", a, c, nv, 128);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
