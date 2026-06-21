#pragma once
#include "framework/config.h"
#include "framework/network.h"
#include <string>
#include <fstream>

// ============ 运行日志器 ============
// 管理时间戳目录、CSV 训练日志、场快照。

struct RunLogger {
    std::string log_dir;
    std::string out_dir;
    std::string weights_path;

    // 创建时间戳目录（基于 Config.log_dir）
    void create_timestamp_dirs(const Config& cfg);

    // 写入 epoch 日志行（首次调用自动写 CSV 表头）
    void log_epoch(int epoch, float avg_loss, float epoch_s,
                   float avg_step_ms, float lr);

    // 场快照：调用 net.dump_field()
    void snapshot_field(Network& net, int epoch, int step);

private:
    std::ofstream m_log_file;
    bool m_header_written = false;
};
