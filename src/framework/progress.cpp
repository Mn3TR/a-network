#include "progress.h"
#include <iostream>
#include <iomanip>
#include <sstream>

void ProgressBar::start_epoch(int epoch, int total_epochs, int total_steps)
{
    m_epoch = epoch;
    m_total_epochs = total_epochs;
    m_total_steps = total_steps;
    m_current_step = 0;
    m_first_step = true;
}

void ProgressBar::step(float loss)
{
    auto now = std::chrono::steady_clock::now();

    if (!m_first_step) {
        float ms = std::chrono::duration<float, std::milli>(now - m_step_start).count();
        if (m_current_step == 1)
            m_avg_step_ms = ms;
        else
            m_avg_step_ms = m_avg_step_ms * 0.95f + ms * 0.05f;
    }
    m_first_step = false;
    m_step_start = now;

    ++m_current_step;
    m_current_loss = loss;

    // 估算总剩余时间（所有 epoch）
    std::string eta_text;
    if (m_avg_step_ms > 0 && m_current_step > 1) {
        int steps_left_this = m_total_steps - m_current_step;
        int epochs_left = m_total_epochs - (m_epoch + 1);
        int total_steps_left = steps_left_this + epochs_left * m_total_steps;
        int remaining_ms = static_cast<int>(total_steps_left * m_avg_step_ms);

        if (remaining_ms >= 60000)
            eta_text = "  ETA:" + std::to_string(remaining_ms / 60000) + "min";
        else if (remaining_ms >= 1000)
            eta_text = "  ETA:" + std::to_string(remaining_ms / 1000) + "s";
    }

    std::ostringstream os;
    os << "\r  E " << (m_epoch + 1) << "/" << m_total_epochs
       << " [" << bar(m_current_step, m_total_steps, 22) << "]"
       << " " << (m_current_step * 100 / m_total_steps) << "%"
       << "  loss:" << std::fixed << std::setprecision(4) << m_current_loss
       << "  step:" << std::setprecision(1) << m_avg_step_ms << "ms"
       << eta_text
       << std::flush;
    std::cout << os.str();
}

void ProgressBar::end_epoch(float avg_loss)
{
    std::ostringstream os;
    os << "\r  E " << (m_epoch + 1) << "/" << m_total_epochs
       << " [" << bar(m_total_steps, m_total_steps, 22) << "]"
       << " 100%"
       << "  avg:" << std::fixed << std::setprecision(4) << avg_loss
       << "  step:" << std::setprecision(1) << m_avg_step_ms << "ms"
       << "  done  ";
    std::cout << os.str() << std::endl;
}

std::string ProgressBar::bar(int filled, int total, int width)
{
    int n = (total > 0) ? (filled * width / total) : 0;
    std::string s;
    for (int i = 0; i < width; ++i)
        s += (i < n) ? '=' : ((i == n) ? '>' : ' ');
    return s;
}
