#pragma once
#include "framework/config.h"
#include "framework/network.h"
#include "framework/tokenizer.h"
#include <string>
#include <vector>

// ============ Generator — 自回归生成 ============

class Generator {
public:
    Generator(Network& net, const Config& cfg);

    // 初始化：加载 tokenizer 和权重
    void setup();

    // 自回归生成
    // seed: 种子文本
    // max_tokens: 最多生成多少 token
    // 返回完整文本
    std::string generate(const std::string& seed, int max_tokens = 100);

private:
    Network& m_net;
    Config m_cfg;

    // token ID → 可打印字符串
    static std::string token_str(size_t id);
};
