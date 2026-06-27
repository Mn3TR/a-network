"""Generate Colab notebook with proper import handling"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "colab": {
        "provenance": [],
        "name": "A-Network Training",
        "include_colab_link": True
    },
    "kernelspec": {
        "name": "python3",
        "display_name": "Python 3"
    },
    "accelerator": "GPU"
}

cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# A-Network — 物理神经网络训练\n"
    "\n"
    "基于 PyTorch 的 3D 张量场（80×80×80）神经网络。\n"
    "Token 注入场中，经 20 步 26-邻域扩散后读出为 logits。\n"
    "\n"
    "**仓库**: https://github.com/Mn3TR/a-network\n"
    "\n"
    "权重和日志自动保存到 Google Drive（`a-network/` 目录）。"
))

cells.append(nbf.v4.new_markdown_cell("## 1. 环境"))

cells.append(nbf.v4.new_code_cell(
    "!rm -rf /content/a-network\n"
    "!git clone https://github.com/Mn3TR/a-network.git /content/a-network\n"
    "%cd /content/a-network\n"
    "!pip install -q torch tokenizers numpy matplotlib datasets\n"
    "import torch\n"
    "print(f'PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}')\n"
    "if torch.cuda.is_available():\n"
    "    print(f'  GPU: {torch.cuda.get_device_name(0)}  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
))

cells.append(nbf.v4.new_code_cell(
    "from google.colab import drive\n"
    "drive.mount('/content/drive')\n"
    "OUTPUT_DIR = '/content/drive/MyDrive/a-network'\n"
    "!mkdir -p {OUTPUT_DIR}/output {OUTPUT_DIR}/log\n"
    "print(f'输出目录: {OUTPUT_DIR}')"
))

cells.append(nbf.v4.new_markdown_cell("## 2. 配置"))

cells.append(nbf.v4.new_code_cell(
    "import sys, importlib\n"
    "\n"
    "# 清除旧缓存，确保用最新代码\n"
    "for mod in list(sys.modules.keys()):\n"
    "    if mod.startswith('src.'):\n"
    "        del sys.modules[mod]\n"
    "!rm -rf /content/a-network/src/anetwork/__pycache__\n"
    "sys.path.insert(0, '.')\n"
    "\n"
    "from src.anetwork.config import ANetworkConfig\n"
    "from src.anetwork.model import ANetwork\n"
    "from src.anetwork.tokenizer import TokenizerWrapper\n"
    "\n"
    "TRAIN_EPOCHS = 5\n"
    "LEARNING_RATE = 1e-4\n"
    "GRAD_ACCUM = 4\n"
    "BATCH_SIZE = 8\n"
    "DATASET = 'roneneldan/TinyStories'\n"
    "\n"
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    "tokenizer = TokenizerWrapper('tokenizer/tokenizer.json')\n"
    "print(f'Vocab: {tokenizer.vocab_size}')\n"
    "\n"
    "a_cfg = ANetworkConfig(vocab_size=tokenizer.vocab_size)\n"
    "net = ANetwork(a_cfg, device=DEVICE).to(DEVICE)\n"
    "print(f'Params: {sum(p.numel() for p in net.parameters()):,}')"
))

cells.append(nbf.v4.new_markdown_cell("## 3. 加载数据"))

cells.append(nbf.v4.new_code_cell(
    "from src.anetwork.data import load_data, pack_batch, count_batch_steps\n"
    "tokens = load_data(DATASET, tokenizer)\n"
    "total_steps = count_batch_steps(tokens, BATCH_SIZE)\n"
    "print(f'Tokens: {len(tokens):,}  Steps/epoch: {total_steps:,}  B={BATCH_SIZE}')"
))

cells.append(nbf.v4.new_markdown_cell("## 4. 训练"))

cells.append(nbf.v4.new_code_cell(
    "import math, time\n"
    "B, N = BATCH_SIZE, a_cfg.N\n"
    "optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)\n"
    "for epoch in range(TRAIN_EPOCHS):\n"
    "    if TRAIN_EPOCHS > 1:\n"
    "        p = epoch / (TRAIN_EPOCHS - 1)\n"
    "        lr = float(1e-6 + (LEARNING_RATE - 1e-6) * (1 + math.cos(math.pi * p)) * 0.5)\n"
    "        for pg in optimizer.param_groups:\n"
    "            pg['lr'] = lr\n"
    "    net.train(); epoch_loss = 0.0; epoch_start = time.time(); optimizer.zero_grad()\n"
    "    fields = torch.zeros(B, N, device=DEVICE)\n"
    "    incomings = torch.zeros(B, N, device=DEVICE)\n"
    "    for step, (inp, tgt) in enumerate(pack_batch(tokens, B)):\n"
    "        loss, fields, incomings = net.train_step_batch(\n"
    "            fields, incomings,\n"
    "            torch.tensor(inp, device=DEVICE),\n"
    "            torch.tensor(tgt, device=DEVICE))\n"
    "        loss.backward(); epoch_loss += loss.item()\n"
    "        if (step + 1) % GRAD_ACCUM == 0:\n"
    "            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)\n"
    "            optimizer.step(); optimizer.zero_grad()\n"
    "        if (step + 1) % max(1, total_steps // 10) == 0:\n"
    "            print(f'  e{epoch} [{step+1}/{total_steps}] loss={loss.item():.4f}')\n"
    "    if total_steps % GRAD_ACCUM:\n"
    "        optimizer.step(); optimizer.zero_grad()\n"
    "    avg_loss = epoch_loss / total_steps\n"
    "    print(f'Epoch {epoch}: loss={avg_loss:.4f}  time={time.time()-epoch_start:.0f}s  lr={lr:.8f}')\n"
    "    net.save(f'{OUTPUT_DIR}/output/weights_e{epoch}.pt')\n"
    "print('Done!')"
))

cells.append(nbf.v4.new_markdown_cell("## 5. 生成"))

cells.append(nbf.v4.new_code_cell(
    "net.eval()\n"
    "seed = tokenizer.encode('Time')[:3]\n"
    "gen = net.generate(seed, max_tokens=50)\n"
    "print(tokenizer.decode(gen))"
))

cells.append(nbf.v4.new_markdown_cell("## 6. 场可视化"))

cells.append(nbf.v4.new_code_cell(
    "import numpy as np, matplotlib.pyplot as plt\n"
    "f = net.field.detach().cpu().numpy().reshape(80, 80, 80)\n"
    "fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))\n"
    "for i, z in enumerate([20, 40, 60]):\n"
    "    im = ax[i].imshow(f[:, :, z], cmap='viridis')\n"
    "    ax[i].set_title(f'z={z}'); ax[i].axis('off')\n"
    "fig.colorbar(im, ax=ax, shrink=0.6); plt.show()"
))

nb.cells = cells
for c in nb.cells:
    cid = c.pop('id', None)
    if cid:
        c['metadata']['id'] = cid
    elif 'id' not in c['metadata']:
        c['metadata']['id'] = c['cell_type'] + '_' + str(hash(c['source'][:30]))

import json
with open('A_Network_Training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)
print(f'OK {len(cells)} cells')
