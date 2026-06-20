#pragma once
#include "core/types.h"
#include <vector>
#include <string>
#include <unordered_map>

// ============ 词表 ============
extern std::vector<std::string> g_id_to_token;
extern std::unordered_map<std::string, size_t> g_token_to_id;
extern std::vector<std::pair<std::string, std::string>> g_merges;
extern std::unordered_map<std::string, size_t> g_pair_priority;

// ============ 加载 ============
void load_tokenizer(const std::string& path);

// ============ 编码/解码 ============
std::vector<size_t> sentencepiece_to_tokens(const std::string& text);
std::string tokens_to_sentence(const std::vector<size_t>& ids);
