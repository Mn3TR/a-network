#pragma once
#include "framework/config.h"
#include "framework/network.h"
#include "framework/optimizer.h"
#include "framework/tokenizer.h"
#include "framework/data.h"
#include "framework/progress.h"
#include "framework/logger.h"
#include <string>

// ============ Trainer — 训练循环编排 ============

class Trainer {
public:
    Trainer(Network& net, const Config& cfg);

    // 初始化：加载 tokenizer、数据、创建日志目录、初始化优化器
    // 注意：不负责 init_weights() 或 load()，调用者需提前准备好网络状态
    void setup();

    // 运行完整训练
    void train();

private:
    Network& m_net;
    Config m_cfg;
    Adam m_optim;
    DataLoader m_data;
    ProgressBar m_pb;
    RunLogger m_logger;
};
