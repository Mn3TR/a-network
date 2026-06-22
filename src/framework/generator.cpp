#include "generator.h"
#include <iostream>
#include <algorithm>

Generator::Generator(Network& net, const Config& cfg)
    : m_net(net), m_cfg(cfg)
{
}

void Generator::setup()
{
    load_tokenizer(m_cfg.tokenizer_path);
    // 注意：不在此处调用 load()，调用者需提前 net.load() 或 net.init_weights()
}

std::string Generator::token_str(size_t id)
{
    std::string s = global_tokenizer().id_to_token(id);
    for (auto& c : s) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc < 32 && uc != '\n') c = '.';
    }
    return s;
}

std::string Generator::generate(const std::string& seed, int max_tokens)
{
    auto seed_tokens = sentencepiece_to_tokens(seed);
    std::cout << "Seed: \"" << seed << "\"" << std::endl;

    std::vector<size_t> generated = seed_tokens;

    // 用前 N-1 个种子 token 构建场状态（最后一个留给自回归循环处理）
    m_net.reset_state();
    for (size_t t = 0; t + 1 < seed_tokens.size(); ++t)
        m_net.generate_step(seed_tokens[t]);

    // 自回归生成（第一个循环迭代处理最后一个种子 token）
    for (int i = 0; i < max_tokens; ++i) {
        size_t pred = m_net.generate_step(generated.back());

        if (pred == 1) { // EOS
            std::cout << "[EOS]";
            break;
        }

        generated.push_back(pred);

        std::string s = token_str(pred);
        if (pred >= 3 && s.find("<|") == std::string::npos)
            std::cout << s;
    }
    std::cout << "\"" << std::endl;

    return tokens_to_sentence(generated);
}
