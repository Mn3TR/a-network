#include "framework/tokenizer.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdint>

// ============ GPT-2 ByteLevel 映射表 ============
static int g_byte_to_cp[256];
static int g_cp_to_byte[512];

static void cp_to_utf8(int cp, std::string& out)
{
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

static void build_bytelevel_tables()
{
    static bool built = false;
    if (built) return;
    built = true;

    for (int i = 0; i < 512; ++i) g_cp_to_byte[i] = -1;

    bool nice[256] = {};
    for (int b = 33; b <= 126; ++b) nice[b] = true;
    for (int b = 161; b <= 172; ++b) nice[b] = true;
    for (int b = 174; b <= 255; ++b) nice[b] = true;

    int next_cp = 256;
    for (int b = 0; b < 256; ++b) {
        if (nice[b]) {
            g_byte_to_cp[b] = b;
            g_cp_to_byte[b] = b;
        } else {
            g_byte_to_cp[b] = next_cp;
            g_cp_to_byte[next_cp] = b;
            ++next_cp;
        }
    }
}

static std::string text_to_bytelevel(const std::string& text)
{
    build_bytelevel_tables();
    std::string result;
    for (unsigned char c : text) {
        cp_to_utf8(g_byte_to_cp[c], result);
    }
    return result;
}

static std::string bytelevel_to_text(const std::string& str)
{
    build_bytelevel_tables();
    std::string result;
    const char* p = str.data();
    const char* end = p + str.size();
    while (p < end) {
        unsigned char c = *p;
        int cp;
        if (c < 0x80) { cp = c; ++p; }
        else if ((c & 0xE0) == 0xC0) {
            cp = (c & 0x1F) << 6; ++p;
            if (p < end) cp |= (*p & 0x3F); ++p;
        } else if ((c & 0xF0) == 0xE0) {
            cp = (c & 0x0F) << 12; ++p;
            if (p < end) cp |= ((*p & 0x3F) << 6); ++p;
            if (p < end) cp |= (*p & 0x3F); ++p;
        } else { ++p; continue; }
        int b = (cp >= 0 && cp < 512) ? g_cp_to_byte[cp] : -1;
        result += (b >= 0) ? static_cast<char>(b) : '?';
    }
    return result;
}

// ============ 简易 JSON 解析器 ============

static void json_skip_ws(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
}

static std::string json_parse_string(const char*& p, const char* end) {
    if (p >= end || *p != '"') return "";
    ++p;
    std::string s;
    while (p < end && *p != '"') {
        if (*p == '\\') {
            ++p;
            if (p >= end) break;
            switch (*p) {
                case '"': s += '"'; break;
                case '\\': s += '\\'; break;
                case 'n': s += '\n'; break;
                case 'r': s += '\r'; break;
                case 't': s += '\t'; break;
                default: s += *p; break;
            }
            ++p;
        } else { s += *p; ++p; }
    }
    if (p < end && *p == '"') ++p;
    return s;
}

static void json_skip_value(const char*& p, const char* end) {
    json_skip_ws(p, end);
    if (p >= end) return;
    if (*p == '{') {
        int d = 0;
        while (p < end) {
            if (*p == '{') ++d;
            else if (*p == '}') { --d; if (d == 0) { ++p; return; } }
            else if (*p == '"') json_parse_string(p, end);
            ++p;
        }
    } else if (*p == '[') {
        int d = 0;
        while (p < end) {
            if (*p == '[') ++d;
            else if (*p == ']') { --d; if (d == 0) { ++p; return; } }
            else if (*p == '"') json_parse_string(p, end);
            ++p;
        }
    } else if (*p == '"') { json_parse_string(p, end); }
    else {
        while (p < end && *p != ',' && *p != '}' && *p != ']' && *p != ':'
               && !(*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    }
}

static bool json_find_key_here(const char*& p, const char* end, const std::string& key) {
    while (p < end) {
        json_skip_ws(p, end);
        if (*p == '}') { ++p; return false; }
        auto k = json_parse_string(p, end);
        json_skip_ws(p, end);
        if (p >= end || *p != ':') return false;
        ++p;
        if (k == key) return true;
        json_skip_value(p, end);
        json_skip_ws(p, end);
        if (*p == ',') ++p;
    }
    return false;
}

static bool json_enter_key(const char*& p, const char* end, const std::string& key) {
    json_skip_ws(p, end);
    if (p >= end || *p != '{') return false;
    ++p;
    if (!json_find_key_here(p, end, key)) return false;
    json_skip_ws(p, end);
    if (p < end && (*p == '{' || *p == '[')) ++p;
    return true;
}

// ============ 构造 ============

Tokenizer::Tokenizer(const std::string& path)
{
    load(path);
}

// ============ 加载 tokenizer.json ============

void Tokenizer::load(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { std::cerr << "ERROR: Cannot open " << path << std::endl; return; }
    auto size = file.tellg();
    std::string content(size, '\0');
    file.seekg(0);
    file.read(content.data(), size);
    file.close();

    const char* p = content.data();
    const char* end = p + content.size();

    if (!json_enter_key(p, end, "model")) {
        std::cerr << "ERROR: Cannot find 'model'" << std::endl; return;
    }
    const char* model_start = p;

    // merges
    p = model_start;
    if (!json_find_key_here(p, end, "merges")) {
        std::cerr << "ERROR: Cannot find 'merges'" << std::endl; return;
    }
    json_skip_ws(p, end);
    if (p >= end || *p != '[') { std::cerr << "ERROR: merges not array" << std::endl; return; }
    ++p;

    m_merges.clear();
    m_pair_priority.clear();
    size_t merge_count = 0;
    while (p < end) {
        json_skip_ws(p, end);
        if (*p == ']') { ++p; break; }
        auto pair_str = json_parse_string(p, end);
        if (!pair_str.empty()) {
            auto sp = pair_str.find(' ');
            if (sp != std::string::npos) {
                std::string left = pair_str.substr(0, sp);
                std::string right = pair_str.substr(sp + 1);
                m_merges.emplace_back(left, right);
                m_pair_priority[pair_str] = merge_count;
                ++merge_count;
            }
        }
        json_skip_ws(p, end);
        if (*p == ',') ++p;
    }

    // vocab
    p = model_start;
    if (!json_find_key_here(p, end, "vocab")) {
        std::cerr << "ERROR: Cannot find 'vocab'" << std::endl; return;
    }
    json_skip_ws(p, end);
    if (p >= end || *p != '{') { std::cerr << "ERROR: vocab not object" << std::endl; return; }
    ++p;

    m_id_to_token.clear();
    m_token_to_id.clear();
    m_id_to_token.resize(g_vocab_size);

    size_t loaded = 0;
    while (p < end) {
        json_skip_ws(p, end);
        if (*p == '}') { ++p; break; }
        auto token_str = json_parse_string(p, end);
        if (token_str.empty()) { json_skip_value(p, end); continue; }
        json_skip_ws(p, end);
        if (*p != ':') break;
        ++p;
        json_skip_ws(p, end);
        size_t id = 0;
        while (p < end && *p >= '0' && *p <= '9') {
            id = id * 10 + static_cast<size_t>(*p - '0');
            ++p;
        }
        if (id < g_vocab_size) {
            m_id_to_token[id] = token_str;
            m_token_to_id[token_str] = id;
            ++loaded;
        }
        json_skip_ws(p, end);
        if (*p == ',') ++p;
    }

    for (size_t i = 0; i < g_vocab_size; ++i) {
        if (m_id_to_token[i].empty()) {
            m_id_to_token[i] = "<UNK>";
            m_token_to_id["<UNK>"] = i;
        }
    }

    std::cout << "Loaded tokenizer: " << loaded << " tokens, "
              << merge_count << " merges" << std::endl;
}

// ============ BPE 合并 ============

std::vector<std::string> Tokenizer::bpe_merge(std::vector<std::string> pieces) const
{
    while (pieces.size() > 1) {
        size_t best_pos = SIZE_MAX;
        size_t best_pri = SIZE_MAX;
        for (size_t i = 0; i < pieces.size() - 1; ++i) {
            auto key = pieces[i] + " " + pieces[i + 1];
            auto it = m_pair_priority.find(key);
            if (it != m_pair_priority.end() && it->second < best_pri) {
                best_pri = it->second;
                best_pos = i;
            }
        }
        if (best_pos == SIZE_MAX) break;
        pieces[best_pos] = pieces[best_pos] + pieces[best_pos + 1];
        pieces.erase(pieces.begin() + static_cast<ptrdiff_t>(best_pos) + 1);
    }
    return pieces;
}

// ============ 文本 → token ID ============

std::vector<size_t> Tokenizer::encode(const std::string& text) const
{
    build_bytelevel_tables();
    std::string normalized = " " + text;
    std::string bytelevel = text_to_bytelevel(normalized);

    std::vector<std::string> pieces;
    const char* p = bytelevel.data();
    const char* end = p + bytelevel.size();
    while (p < end) {
        unsigned char c = *p;
        if (c < 0x80) { pieces.emplace_back(1, static_cast<char>(c)); ++p; }
        else if ((c & 0xE0) == 0xC0) { pieces.emplace_back(p, 2); p += 2; }
        else if ((c & 0xF0) == 0xE0) { pieces.emplace_back(p, 3); p += 3; }
        else { ++p; }
    }

    auto merged = bpe_merge(pieces);

    std::vector<size_t> ids;
    ids.reserve(merged.size());
    for (const auto& tok : merged) {
        auto it = m_token_to_id.find(tok);
        if (it != m_token_to_id.end()) ids.push_back(it->second);
    }
    return ids;
}

// ============ token ID → 文本 ============

std::string Tokenizer::decode(const std::vector<size_t>& ids) const
{
    build_bytelevel_tables();
    std::string bytelevel_str;
    for (auto id : ids) {
        if (id < m_id_to_token.size())
            bytelevel_str += m_id_to_token[id];
    }
    std::string result = bytelevel_to_text(bytelevel_str);
    if (!result.empty() && result[0] == ' ')
        result = result.substr(1);
    return result;
}

// ============ 查询 ============

const std::string& Tokenizer::id_to_token(size_t id) const
{
    static std::string s_unknown = "<?>";
    if (id >= m_id_to_token.size()) return s_unknown;
    return m_id_to_token[id];
}

size_t Tokenizer::token_to_id(const std::string& token) const
{
    auto it = m_token_to_id.find(token);
    return (it != m_token_to_id.end()) ? it->second : 0;
}

// ============ 全局默认实例 ============

static Tokenizer s_default_tokenizer;

Tokenizer& global_tokenizer()
{
    return s_default_tokenizer;
}
