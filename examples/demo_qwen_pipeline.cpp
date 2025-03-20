#include "Backend.hpp"
#include "Trace.hpp"
#include "Types.hpp"
#include "backends/cpu/CPUBackend.hpp"
#include "cmdline.h"
#include "models/qwen/configuration_qwen.hpp"
#include "models/qwen/modeling_qwen_npu.hpp"
#include "models/qwen/modeling_qwen.hpp"
#include "models/qwen/tokenization_qwen.hpp"
#include "processor/PostProcess.hpp"
#include "Parallel.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/qwen_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/qwen_merges.txt");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/qwen-1.5-1.8b-chat-int8.mllm");
    cmdParser.add<string>("billion", 'b', "[0.5B | 1.8B]", false, "1.8B");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 512);
    cmdParser.add<int>("ntokens", 'n', "new tokens", false, 20);
    cmdParser.add<int>("chunk_size", 'c', "chunk size", false, 128);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("model");
    string model_billion = cmdParser.get<string>("billion");
    int tokens_limit = cmdParser.get<int>("limits");
    int ntokens = cmdParser.get<int>("ntokens");
    int chunk_size = cmdParser.get<int>("chunk_size");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = QWenTokenizer(vocab_path, merge_path);
    QWenConfig config(tokens_limit, model_billion, RoPEType::HFHUBROPE);
    auto model = QWenForCausalLM_NPU(config, chunk_size);
    model.load(model_path);
    auto decoding_model = QWenForCausalLM(config);
    decoding_model.load("../models/qwen-1.5-1.8b-chat-q4k.mllm");

    string trace_string = " ";
    auto [_, input_tensor] = tokenizer.tokenizePaddingByChunk(trace_string, chunk_size, config.vocab_size);
    Tracer::trace(&model, {input_tensor});
    std::cout << "Trace and Warmup finished" << std::endl;

    vector<string> in_strs = {
        // " Give me a short introduction to large language model.",
        "Large Language Models (LLMs) are advanced artificial intelligence systems designed to understand and generate human-like text. These models are trained on vast amounts of data, enabling them to perform a wide range of tasks, from answering questions and summarizing text to generating creative content and engaging in conversational dialogue. LLMs like GPT-3 and GPT-4, developed by OpenAI, have set new benchmarks in natural language processing by leveraging deep learning architectures, particularly transformer models, which excel at capturing context and relationships within text. The scalability and versatility of LLMs make them invaluable tools for applications in education, customer service, content creation, and more. However, their deployment also raises ethical considerations, including issues of bias, misinformation, and the potential for misuse. As the field continues to evolve, ongoing research and responsible deployment strategies are essential to harnessing the full potential of these powerful AI systems while mitigating their risks. As LLMs continue to evolve, their applications expand across various fields. In education, they can personalize learning experiences by analyzing students' needs and progress, offering tailored recommendations and materials. In customer service, LLMs handle large volumes of inquiries, providing timely and accurate responses, thus improving customer satisfaction and operational efficiency. In content creation, LLMs assist in generating creative copy, writing articles and scripts, and even contributing to music and art production. Their powerful generative capabilities open new possibilities for the creative industries. However, this also raises discussions about copyright and originality, highlighting the need to ensure the legality and ethics of generated content. At the same time, the use of LLMs presents privacy and security challenges. Since these models rely on vast datasets, there is a risk of inadvertently exposing sensitive information or reinforcing existing biases. Ensuring data privacy and implementing robust security measures are crucial to addressing these concerns. Ethical considerations also include the potential for misuse, such as generating misinformation or deepfakes. It is essential to develop guidelines and policies that promote responsible use. As the field progresses, ongoing research and collaboration among stakeholders will be vital in balancing innovation with ethical responsibility, ensuring that LLMs are deployed in ways that benefit society while minimizing risks. The advancement of LLMs is a double-edged sword, offering immense potential for technological progress while also presenting unprecedented challenges. Generate a title for these content."
    };

    for (int i = 0; i < in_strs.size(); ++i) {
        auto input_str = tokenizer.apply_chat_template(in_strs[i]);
        auto [real_seq_length, input_tensor] = tokenizer.tokenizePaddingByChunk(input_str, chunk_size, config.vocab_size);

        // set total seq length for HeadLinear execute, which can not get the real seq length from Opts
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setTotalSequenceLength(real_seq_length);
        // set chunk size for the HeadLinear execute, which can not get the chunk size from Opts
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setChunkSize(chunk_size);

        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;
        std::cout << "real_seq_length: " << real_seq_length << std::endl;

        LlmTextGeneratorOpts opt{
            .max_new_tokens = 1,
            .do_sample = false,
            .is_padding = true,
            .seq_before_padding = real_seq_length,
            .chunk_size = chunk_size,
        };

        // tensor vectors to save the chunked tensors of the QNN prefilling input
        bool isSwitched = false;

        ChunkPipeline pipeline(real_seq_length, chunk_size);
        auto prefill_result = pipeline.run(input_tensor, opt, tokenizer, model, isSwitched);

        Module::isMultiChunkPrefilling = true;
        Module::isFirstChunk = false;

        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setCurSequenceLength(real_seq_length);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(AUTOREGRESSIVE);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();

        LlmTextGeneratorOpts decoding_opt{
            .max_new_tokens = static_cast<size_t>(ntokens),
            .do_sample = false,
            .temperature = 0.3f,
            .top_k = 50,
            .top_p = 0.f,
            .is_padding = false,
        };
        isSwitched = false;

        Tensor decoding_input;
        decoding_input.setBackend(Backend::global_backends[MLLM_CPU]);
        decoding_input.setTtype(INPUT_TENSOR);
        decoding_input.reshape(1, 1, 1, 1);
        decoding_input.setName("input0");
        decoding_input.alloc();
        decoding_input.setDataAt(0, 0, 0, 0, prefill_result->dataAt<float>(0, 0, 0, 0));
        decoding_model.generate(decoding_input, decoding_opt, [&](unsigned int out_token) -> bool {
            // call only once of switchDecodeTag
            if (!isSwitched) {
                static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
                isSwitched = true;
            }
            auto out_string = tokenizer.detokenize({out_token});
            auto [isOk, print_string] = tokenizer.postprocess(out_string);
            if (isOk) {
                std::cout << print_string << std::flush;
            } else {
                return false;
            }
            return true;
        });

        // turn on switching, set sequence length and execution type
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setCurSequenceLength(0);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->setExecutionType(PROMPT);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU])->toggleSwitching();
        std::cout << "\n";
    }
}