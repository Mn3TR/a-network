#pragma once
#include "framework/tokenizer.h"
#include <string>
#include <vector>

struct DataLoader {
    std::vector<size_t> tokens;  // 完整的 token 序列

    // 从 .txt 文件读取并 tokenize
    void load_txt(const std::string& path);
    // 从文件夹读取所有 .txt 文件并 tokenize
    void load_dir(const std::string& dir_path);
};
