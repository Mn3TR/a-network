#include "data.h"
#include "tokenizer/bpe.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
namespace fs = std::filesystem;

void DataLoader::load_txt(const std::string& path)
{
    std::ifstream file(path);
    if (!file) {
        std::cerr << "ERROR: Cannot open " << path << std::endl;
        return;
    }

    std::stringstream ss;
    ss << file.rdbuf();
    std::string text = ss.str();
    file.close();

    if (text.empty()) {
        std::cerr << "WARN: Empty file " << path << std::endl;
        return;
    }

    auto new_tokens = sentencepiece_to_tokens(text);

    // 每篇文本加 BOS(0) 和 EOS(1)，让模型学会句子边界
    tokens.push_back(0);  // <|begin_of_sentence|>
    tokens.insert(tokens.end(), new_tokens.begin(), new_tokens.end());
    tokens.push_back(1);  // <|end_of_sentence|>

    std::cout << "  " << path << ": " << text.size() << " chars -> "
              << (new_tokens.size() + 2) << " tokens (with BOS+EOS)" << std::endl;
}

void DataLoader::load_dir(const std::string& dir_path)
{
    tokens.clear();

    if (!fs::is_directory(dir_path)) {
        std::cerr << "ERROR: " << dir_path << " is not a directory" << std::endl;
        return;
    }

    int count = 0;
    for (auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.path().extension() == ".txt") {
            load_txt(entry.path().string());
            ++count;
        }
    }

    std::cout << "Loaded " << count << " files, total " << tokens.size() << " tokens" << std::endl;
}
