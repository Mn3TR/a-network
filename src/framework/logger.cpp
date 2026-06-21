#include "framework/logger.h"
#include <iostream>
#include <chrono>
#include <filesystem>

static std::string make_timestamp_dir()
{
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char buf[24];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d-%H%M", &tm);
    return std::string(buf);
}

void RunLogger::create_timestamp_dirs(const Config& cfg)
{
    std::string stamp = make_timestamp_dir();
    log_dir = cfg.log_dir + stamp + "/";
    out_dir = "output/output-" + stamp + "/";
    weights_path = out_dir + "weights.bin";

    std::filesystem::create_directories(log_dir);
    std::filesystem::create_directories(out_dir);

    std::cout << "Log dir: " << log_dir << std::endl;
    std::cout << "Output dir: " << out_dir << std::endl;

    // 打开 CSV 日志（覆写）
    m_log_file.open(log_dir + "train_log.csv", std::ios::trunc);
    m_header_written = false;
}

void RunLogger::log_epoch(int epoch, float avg_loss, float epoch_s,
                           float avg_step_ms, float lr)
{
    if (!m_log_file.is_open()) return;

    if (!m_header_written) {
        m_log_file << "epoch,avg_loss,avg_step_ms,epoch_s,lr\n";
        m_header_written = true;
    }

    m_log_file << epoch << ","
               << avg_loss << ","
               << avg_step_ms << ","
               << epoch_s << ","
               << lr << "\n";
    m_log_file.flush();
}

void RunLogger::snapshot_field(Network& net, int epoch, int step)
{
    std::string fname = log_dir + "field_e"
                      + std::to_string(epoch) + "_t"
                      + std::to_string(step) + ".bin";
    net.dump_field(fname);
}
