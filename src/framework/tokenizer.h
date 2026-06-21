#pragma once
#include <vector>
#include <string>
#include <unordered_map>

// ============ 词表大小（硬上限，与 tokenizer.json 一致） ============
constexpr size_t g_vocab_size = 128256;

// ============ BPE Tokenizer ============
// 封装编码/解码和词表管理。可独立实例化，也提供默认全局实例。

class Tokenizer {
public:
    Tokenizer() = default;
    explicit Tokenizer(const std::string& path);

    // 加载 tokenizer.json
    void load(const std::string& path);

    // 文本 → token ID 序列
    std::vector<size_t> encode(const std::string& text) const;

    // token ID 序列 → 文本
    std::string decode(const std::vector<size_t>& ids) const;

    // token ID → 字符串（日志用）
    const std::string& id_to_token(size_t id) const;
    size_t token_to_id(const std::string& token) const;

    // 词表大小
    size_t vocab_size() const { return m_id_to_token.size(); }

private:
    std::vector<std::string> m_id_to_token;
    std::unordered_map<std::string, size_t> m_token_to_id;
    std::vector<std::pair<std::string, std::string>> m_merges;
    std::unordered_map<std::string, size_t> m_pair_priority;

    // BPE 合并（按 g_pair_priority 优先级）
    std::vector<std::string> bpe_merge(std::vector<std::string> pieces) const;
};

// 全局默认 Tokenizer 实例
Tokenizer& global_tokenizer();

// ============ 向后兼容的全局函数 ============
inline void load_tokenizer(const std::string& path) { global_tokenizer().load(path); }
inline std::vector<size_t> sentencepiece_to_tokens(const std::string& text) { return global_tokenizer().encode(text); }
inline std::string tokens_to_sentence(const std::vector<size_t>& ids) { return global_tokenizer().decode(ids); }
