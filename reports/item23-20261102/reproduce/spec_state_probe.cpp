#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static void decode(llama_context * ctx, const std::vector<llama_token> & tokens,
                   int pos, int count, bool logits, int keep = -1) {
    auto b = llama_batch_init(count, 0, 1);
    for (int i = 0; i < count; ++i) {
        b.token[i] = tokens[pos + i];
        if (keep >= 0 && i >= keep) { b.token[i] = (b.token[i] + 17) % 248320; }
        b.pos[i] = pos + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = logits;
    }
    b.n_tokens = count;
    if (llama_decode(ctx, b)) { throw std::runtime_error("decode failed"); }
    llama_batch_free(b);
}

static std::vector<float> run(llama_model * model, const std::vector<llama_token> & tokens,
                              int prefix, int count, int nrs, int step, bool reject) {
    auto cp = llama_context_default_params();
    cp.n_ctx = prefix + count + 256; cp.n_batch = 512; cp.n_ubatch = 256;
    cp.n_threads = cp.n_threads_batch = 4; cp.n_seq_max = 1; cp.n_rs_seq = nrs;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) { throw std::runtime_error("context failed"); }
    if (nrs) { llama_set_embeddings_nextn(ctx, true, false); }
    for (int p = 0; p < prefix; p += 512) { decode(ctx, tokens, p, std::min(512, prefix-p), false); }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> result;
    int round = 0;
    for (int p = prefix; p < prefix + count;) {
        const int n = std::min(step, prefix + count - p);
        const int keep = reject ? std::min(n, 1 + round % step) : n;
        decode(ctx, tokens, p, n, true, reject ? keep : -1);
        for (int i = 0; i < keep; ++i) {
            auto * lg = llama_get_logits_ith(ctx, i);
            if (!lg) { throw std::runtime_error("missing logits"); }
            result.insert(result.end(), lg, lg + nv);
        }
        if (keep < n && !llama_memory_seq_rm(llama_get_memory(ctx), 0, p+keep, -1)) {
            throw std::runtime_error("rollback rejected");
        }
        p += keep; ++round;
    }
    llama_free(ctx);
    return result;
}

static void compare(const char * mode, const std::vector<float> & a, const std::vector<float> & b, int nv) {
    if (a.size() != b.size()) { throw std::runtime_error("row count mismatch"); }
    const int rows = a.size() / nv;
    double total = 0, maximum = 0; int changes = 0;
    for (int i = 0; i < rows; ++i) {
        const float * x = a.data() + size_t(i)*nv, * y = b.data() + size_t(i)*nv;
        int ix = std::max_element(x, x+nv)-x, iy = std::max_element(y, y+nv)-y;
        double zx = 0, zy = 0;
        for (int t = 0; t < nv; ++t) {
            if (!std::isfinite(x[t]) || !std::isfinite(y[t])) { throw std::runtime_error("nonfinite logits"); }
            zx += std::exp(double(x[t]-x[ix])); zy += std::exp(double(y[t]-y[iy]));
        }
        double lx = std::log(zx)+x[ix], ly = std::log(zy)+y[iy], kl = 0;
        for (int t = 0; t < nv; ++t) { kl += std::exp(x[t]-lx)*((x[t]-lx)-(y[t]-ly)); }
        total += kl; maximum = std::max(maximum, kl); changes += ix != iy;
    }
    printf("{\"mode\":\"%s\",\"rows\":%d,\"top_changes\":%d,\"mean_kl\":%.9g,\"max_kl\":%.9g}\n",mode,rows,changes,total/rows,maximum);
    fflush(stdout);
}

int main(int argc, char ** argv) {
    if (argc != 5) { return 2; }
    ggml_backend_load_all();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { return 1; }
    std::ifstream input(argv[2]);
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    auto * vocab = llama_model_get_vocab(model);
    int nt = -llama_tokenize(vocab,text.c_str(),text.size(),nullptr,0,true,true);
    std::vector<llama_token> tokens(nt);
    if (llama_tokenize(vocab,text.c_str(),text.size(),tokens.data(),nt,true,true) != nt) { return 3; }
    int prefix = std::atoi(argv[3]), count = std::atoi(argv[4]), nv = llama_vocab_n_tokens(vocab);
    if (prefix + count > nt) { return 4; }
    auto ar = run(model,tokens,prefix,count,0,1,false);
    puts("{\"reference\":\"single-token AR\"}"); fflush(stdout);
    auto batch = run(model,tokens,prefix,count,0,3,false);
    compare("batch3_vs_ar",ar,batch,nv);
    auto mtp = run(model,tokens,prefix,count,2,3,false);
    compare("mtp_accept_all_vs_ar",ar,mtp,nv);
    compare("mtp_accept_all_vs_batch3",batch,mtp,nv);
    auto rolled = run(model,tokens,prefix,count,2,3,true);
    compare("mtp_reject_mix_vs_ar",ar,rolled,nv);
    compare("mtp_reject_mix_vs_accept_all",mtp,rolled,nv);
    llama_model_free(model); llama_backend_free();
}
