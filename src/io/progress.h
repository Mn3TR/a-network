#pragma once
#include <chrono>
#include <string>

class ProgressBar {
    int m_epoch = 0;
    int m_total_epochs = 0;
    int m_total_steps = 0;
    int m_current_step = 0;
    float m_current_loss = 0.0f;
    float m_avg_step_ms = 0.0f;
    std::chrono::steady_clock::time_point m_step_start;
    bool m_first_step = true;

public:
    void start_epoch(int epoch, int total_epochs, int total_steps);
    void step(float loss);
    void end_epoch(float avg_loss);

private:
    static std::string bar(int filled, int total, int width);
};
