// #ifdef USE_QNN
#include <iostream>
#include <csignal>
#include <memory>
#include <vector>
#include "Executor.hpp"
#include "Types.hpp"
#include "backends/qnn/QNNNet.hpp"
#include "cmdline.h"
#include "Net.hpp"
#include "backends/qnn/QNNExecutor.hpp"

#include "models/qwen/tokenization_qwen.hpp"
#include "main_qwen_npu.hpp"

using namespace mllm;

unsigned int argmax(const std::vector<float> &scores) {
    return std::max_element(scores.begin(), scores.end()) - scores.begin();
}

unsigned int postProcessing(shared_ptr<Tensor> result, shared_ptr<Tensor> &out_result) {
    assert(result->batch() == 1);
    assert(result->head() == 1);
    out_result->reshape(1, 1, 1, 1);
    out_result->alloc();
    vector<float> scores;
    for (int i = 0; i < result->dimension(); ++i) {
        auto value = result->dataAt<float>(0, 0, result->sequence() - 1, i);
        scores.push_back(value);
    }
    auto token_idx = argmax(scores);
    out_result->setDataAt<float>(0, 0, 0, 0, token_idx);
    return token_idx;
}

unsigned int postProcessing_prefill(shared_ptr<Tensor> result, shared_ptr<Tensor> &out_result, int seq) {
    assert(result->batch() == 1);
    assert(result->head() == 1);
    out_result->reshape(1, 1, 1, 1);
    out_result->alloc();
    vector<float> scores;
    for (int i = 0; i < result->dimension(); ++i) {
        auto value = result->dataAt<float>(0, 0, seq - 1, i);
        scores.push_back(value);
    }
    auto token_idx = argmax(scores);
    out_result->setDataAt<float>(0, 0, 0, 0, token_idx);
    return token_idx;
}

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/qwen_vocab.mllm");

    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 600);
    cmdParser.add<int>("ntokens", 'n', "new tokens", false, 50);

    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.add<int>("seq", 's', "seqenth length", false, 64);
    cmdParser.add<bool>("chunk", 'c', "use chunk execute", false, true);
    cmdParser.add<int>("head", 'h', "num of heads", false, 16);

    cmdParser.add<int>("ffn", 'f', "size of ffn hidden size", false, 5504);
    cmdParser.add<int>("hds", 'd', "size of hidden size", false, 2048);

    cmdParser.add<bool>("readfile", 'r', "read prompt from file", false, false);
    cmdParser.add<bool>("warmiter", 'w', "duplicate prompt from file for warmup", false, false);
    cmdParser.add<int>("per_chunk_size", 'p', "per chunk size", false, 128);

    cmdParser.parse_check(argc, argv);

    const string npu_model_path = "../models/qwen-1.5-1.8b-chat-int8.mllm";
    const string cpu_model_path = "../models/qwen-1.5-1.8b-chat-q4k.mllm";
    const string merge_file_path = "../vocab/qwen_merges.txt";

    string vocab_path = cmdParser.get<string>("vocab");
    int tokens_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");
    int seqLength = cmdParser.get<int>("seq");
    bool isChunkExecute = cmdParser.get<bool>("chunk");
    int head_num = cmdParser.get<int>("head");
    int ntokens = cmdParser.get<int>("ntokens");

    bool read_file = cmdParser.get<bool>("readfile");
    bool warmiter = cmdParser.get<bool>("warmiter");
    int chunk_size = cmdParser.get<int>("per_chunk_size");

    int chunk = 1;
    if (isChunkExecute)
        chunk = seqLength / chunk_size;

    int vocab_size = 151936;
    int hidden_dim = cmdParser.get<int>("hds");
    int ffn_hidden_dim = cmdParser.get<int>("ffn");

    vector<string> in_strs;
    if (read_file) {
        std::ifstream file("./func_prompt.txt");
        if (!file) {
            std::cerr << "无法打开文件！" << std::endl;
            return 1;
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        if (warmiter) {
            in_strs = {
                buffer.str(),
                buffer.str()
            };
        }
        else {
            in_strs = {
                buffer.str()
            };        
        }
        file.close(); // 关闭文件
    }
    else {
        in_strs = {
            "Large Language Models (LLMs) are advanced artificial intelligence systems designed to understand and generate human-like text. These models are trained on vast amounts of data, enabling them to perform a wide range of tasks, from answering questions and summarizing text to generating creative content and engaging in conversational dialogue. LLMs like GPT-3 and GPT-4, developed by OpenAI, have set new benchmarks in natural language processing by leveraging deep learning architectures, particularly transformer models, which excel at capturing context and relationships within text. The scalability and versatility of LLMs make them invaluable tools for applications in education, customer service, content creation, and more. However, their deployment also raises ethical considerations, including issues of bias, misinformation, and the potential for misuse. As the field continues to evolve, ongoing research and responsible deployment strategies are essential to harnessing the full potential of these powerful AI systems while mitigating their risks. As LLMs continue to evolve, their applications expand across various fields. In education, they can personalize learning experiences by analyzing students' needs and progress, offering tailored recommendations and materials. In customer service, LLMs handle large volumes of inquiries, providing timely and accurate responses, thus improving customer satisfaction and operational efficiency. In content creation, LLMs assist in generating creative copy, writing articles and scripts, and even contributing to music and art production. Their powerful generative capabilities open new possibilities for the creative industries. However, this also raises discussions about copyright and originality, highlighting the need to ensure the legality and ethics of generated content. At the same time, the use of LLMs presents privacy and security challenges. Since these models rely on vast datasets, there is a risk of inadvertently exposing sensitive information or reinforcing existing biases. Ensuring data privacy and implementing robust security measures are crucial to addressing these concerns. Ethical considerations also include the potential for misuse, such as generating misinformation or deepfakes. It is essential to develop guidelines and policies that promote responsible use. As the field progresses, ongoing research and collaboration among stakeholders will be vital in balancing innovation with ethical responsibility, ensuring that LLMs are deployed in ways that benefit society while minimizing risks. The advancement of LLMs is a double-edged sword, offering immense potential for technological progress while also presenting unprecedented challenges. Generate a title for these content."
            // " What can you do?",
            // "Please introduce Beijing University of Posts and Telecommunications."};
        };
    }

    auto tokenizer = QWenTokenizer(vocab_path, merge_file_path);

    std::unique_ptr<Context> npu_ctx_ptr(new Context());
    auto *npu_ctx = npu_ctx_ptr.get();
    std::unique_ptr<Context> cpu_ctx_ptr(new Context());
    auto *cpu_ctx = cpu_ctx_ptr.get();
    std::unique_ptr<Context> inter_ctx_ptr(new Context());
    auto *inter_ctx = inter_ctx_ptr.get();

    // cache_max should be longer than seqLength
    modeling::qwen_npu(npu_ctx, vocab_size, hidden_dim, ffn_hidden_dim, head_num, tokens_limit, seqLength, chunk);
    modeling::qwen_npu_cpu_inter(inter_ctx, vocab_size, hidden_dim, ffn_hidden_dim, head_num, tokens_limit, seqLength, chunk);
    modeling::qwen_cpu_q4k(cpu_ctx, vocab_size, hidden_dim, ffn_hidden_dim, head_num, tokens_limit);

    BackendConfig bn;
    QNNNet npuNet(bn, npu_ctx);
    npuNet.convert(npu_ctx, BackendType::MLLM_QNN, thread_num);
    Net interNet(bn);
    interNet.convert(inter_ctx->sub_param_, BackendType::MLLM_CPU, thread_num);
    Net cpuNet(bn);
    cpuNet.convert(cpu_ctx->sub_param_, BackendType::MLLM_CPU, thread_num);

    ParamLoader npu_prefill_param_loader(npu_model_path);
    ParamLoader cpu_decoding_param_loader(cpu_model_path);
    ParamLoader inter_param_loader(npu_model_path);

    QNNExecutor *npuExePtr;
    if (isChunkExecute) {
        npuExePtr = new QNNPipelineExecutor(&npu_prefill_param_loader, chunk_size = chunk_size);
    } else {
        npuExePtr = new QNNExecutor(&npu_prefill_param_loader);
    }
    auto &npuExe = *npuExePtr;
    npuExe.setup(&npuNet);
    Executor interExe(&inter_param_loader);
    interExe.setup(&interNet);
    Executor cpuExe(&cpu_decoding_param_loader);
    cpuExe.setup(&cpuNet);

    shared_ptr<Tensor> input = std::make_shared<Tensor>();

    for (int str_i = 0; str_i < in_strs.size(); ++str_i) {
       auto input_string = in_strs[str_i];
        auto input_str = tokenizer.apply_chat_template(input_string);
        auto [real_seq_length, input_tensor] = tokenizer.tokenizeWithPadding(input_str, seqLength, vocab_size);
        auto input = std::make_shared<Tensor>(input_tensor);

        if (chunk != 1)
            npuExe.warmup(npu_ctx, &npuNet, {input});

        std::cout << "real_seq_length: " << real_seq_length << std::endl;
        std::cout << "[Q] " << input_string << std::endl;
        std::cout << "[A] " << std::flush;

        auto start_time = std::chrono::high_resolution_clock::now();

        do {
            // 1: Prefill stage using NPU chunk execute
            npuExe.run(npu_ctx, &npuNet, {input});
            auto result = npuExe.result();

            auto pend_time = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> duration = pend_time - start_time;
            double seconds = duration.count();

            double tokens_per_second = real_seq_length / seconds;

            std::cout << "Prefill Tokens: " << real_seq_length << std::endl;
            std::cout << "Prefill Time taken: " << seconds << " seconds" << std::endl;
            std::cout << "Prefill Speed: " << tokens_per_second << " tokens/s" << std::endl;

            start_time = std::chrono::high_resolution_clock::now();

            // inter model for prefill-decode
            interExe.run(&interNet, {result[0]});
            result = interExe.result();

            auto token_idx = postProcessing_prefill(result[0], input, real_seq_length);
            if (token_idx == 2) { // "</s>"
                break;
            }

            auto out_token = tokenizer.detokenize({token_idx});
            std::cout << out_token << std::flush;

            auto prefill_cpu_backend = dynamic_cast<CPUBackend *>(npuNet.backends()[MLLM_CPU].get());
            auto inter_cpu_backend = dynamic_cast<CPUBackend *>(interNet.backends()[MLLM_CPU].get());
            auto decode_cpu_backend = dynamic_cast<CPUBackend *>(cpuNet.backends()[MLLM_CPU].get());
            prefill_cpu_backend->setCurSequenceLength(real_seq_length);
            prefill_cpu_backend->setExecutionType(AUTOREGRESSIVE);
            prefill_cpu_backend->toggleSwitching();
            inter_cpu_backend->setCurSequenceLength(real_seq_length);
            inter_cpu_backend->setExecutionType(AUTOREGRESSIVE);
            inter_cpu_backend->toggleSwitching();
            decode_cpu_backend->setCurSequenceLength(real_seq_length);
            decode_cpu_backend->setExecutionType(AUTOREGRESSIVE);
            decode_cpu_backend->toggleSwitching();

            auto mid_time = std::chrono::high_resolution_clock::now();

            duration = mid_time - start_time;
            double inter_seconds = duration.count();

            int step_cnt = 0;
            // // 2: Decoding stage using CPU execute
            for (int step = real_seq_length; step < real_seq_length + ntokens; step++) {
                cpuExe.run(&cpuNet, {input});
                auto result = cpuExe.result();

                auto token_idx = postProcessing(result[0], input);
                auto out_token = tokenizer.detokenize({token_idx});

                auto [isOk, print_string] = tokenizer.postprocess(out_token);
                if (isOk) {
                    std::cout << print_string << std::flush;
                    step_cnt++;
                } else {
                    break;
                }

                if (step == real_seq_length) {
                    prefill_cpu_backend->toggleSwitching();
                    inter_cpu_backend->toggleSwitching();
                    decode_cpu_backend->toggleSwitching();
                }
            }

            auto end_time = std::chrono::high_resolution_clock::now();

            duration = end_time - mid_time;
            seconds = duration.count();

            tokens_per_second = (step_cnt) / seconds;

            std::cout << "\nIntermediate Time taken: " << inter_seconds << " seconds" << std::endl;
            std::cout << "Decode Tokens: " << (step_cnt) << std::endl;
            std::cout << "Decode Time taken: " << seconds << " seconds" << std::endl;
            std::cout << "Decode Speed: " << tokens_per_second << " tokens/s" << std::endl;

        } while (false);
        printf("\n");
    }

    std::cout << "====================" << std::endl;
    npuExe.perf();
    cpuExe.perf();

    // // free memory
    // for (auto *op : npu_ctx->net_ops) {
    //     delete op;
    // }
    // for (auto *tensor : npu_ctx->net_tensors) {
    //     delete tensor;
    // }

    return 0;
}
// #endif