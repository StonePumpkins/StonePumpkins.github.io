---
title: "MiniMind-O 复现：0.1B Omni 模型的踩坑记录"
date: "2026-07-18"
tag: "复现笔记"
summary: "从 SigLIP2 + SenseVoice 编码器到 Mimi 音频解码，单卡 3090 跑通全模态链路。"
---

## 代码结构

```
minimind-o/
├── model/model_omni.py      # Thinker + Talker + Bridge
├── model/projector.py       # 2层 MLP 投影器
└── train_sft_omni.py        # 三阶段训练
```

## 关键发现

**Bridge 层**取中间层而非最后一层，避免被 LM head 过度塑形。

Talker 用 MTP 同时预测 8 层 Mimi codebook，维度别太小，384/512 维中长句容易漏词。

### 训练三阶段

| 阶段 | 冻结部分 | 学习率 | 数据 |
|------|----------|--------|------|
| 1. Alignment | Thinker + Talker | 1e-3 | 图文/音文对齐 |
| 2. SFT | Thinker | 5e-5 | 指令微调 |
| 3. DPO | 全量 | 1e-6 | 偏好对齐 |

## 显存优化技巧

```python
# 梯度检查点 + 梯度累积
model.gradient_checkpointing_enable()
accumulation_steps = 4

# 量化 LoRA
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none"
)
model = get_peft_model(model, lora_config)
```

## 坑点总结

1. **音频 tokenizer 对齐**：Mimi 的 codebook 维度要与 Talker 输出头匹配
2. **多模态位置编码**：图像 patch 和音频 frame 需要统一的位置编码空间
3. **损失平衡**：`L_total = L_text + 0.5 * L_audio + 0.3 * L_vision` 效果最好

---

## 相关资源

- [MiniMind-O 仓库](https://github.com/StonePumpkins/MiniMind-O)
- [Mimi 论文](https://arxiv.org/abs/2311.17131)
- [SenseVoice 技术报告](https://github.com/FunAudioLLM/SenseVoice)