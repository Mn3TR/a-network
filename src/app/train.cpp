#include "framework/config.h"
#include "framework/trainer.h"
#include "framework/generator.h"
#include "framework/a-network/a_network.h"
#include <iostream>
#include <string>

// ============ 入口点 ============
// 用法:
//   train.exe              — 全新训练
//   train.exe load         — 从 checkpoint 继续训练
//   train.exe gen [seed]   — 生成模式

int main(int argc, char* argv[])
{
    bool gen_mode  = (argc > 1 && std::string(argv[1]) == "gen");
    bool load_mode = (argc > 1 && std::string(argv[1]) == "load");

    Config cfg;
    ANetworkConfig a_cfg;

    if (gen_mode) {
        // ======== 生成模式 ========
        ANetwork net(a_cfg);
        net.load(cfg.weights_path);
        Generator gen(net, cfg);
        gen.setup();

        std::string seed = (argc > 2) ? argv[2] : "Time";
        std::string result = gen.generate(seed, 100);
        std::cout << "\nFull text:\n" << result << std::endl;
        return 0;
    }

    // ======== 训练模式 ========
    ANetwork net(a_cfg);

    if (load_mode) {
        net.load(cfg.weights_path);
        std::cout << "Loaded weights from " << cfg.weights_path << std::endl;
    } else {
        net.init_weights();
    }

    Trainer trainer(net, cfg);
    trainer.setup();
    trainer.train();

    return 0;
}
