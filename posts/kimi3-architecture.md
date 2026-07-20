---
title: "Kimi-3 架构思考：2.8T MoE 的设计哲学"
date: "2026-07-19"
tag: "架构分析"
summary: "从 K2 到 K3，Moonshot 如何在一年内将参数规模从 1T 推至 2.8T，同时保持 32B 激活？"
---

## 从 K2 到 K3：规格跃迁

Kimi 系列在过去一年内经历了极为密集的迭代。K2（2025.07）以 1T 总参数出道；K2.5 加入原生多模态；K3 则是集大成者。

## MoE 路由：从 384 到 896 专家

K3 将专家数量扩展至 896 个，每 token 仅激活 16 个专家。这种设计带来了几个关键影响：

- **稀疏度提升**：激活比例进一步降低
- **专家特化**：每个专家覆盖更细粒度的知识领域
- **负载均衡挑战**：需要更精细的路由设计

### 路由机制细节

```python
# 简化的 Top-K 路由逻辑
def moe_router(x, num_experts=896, top_k=16):
    # x: [batch, seq_len, hidden_dim]
    logits = router_proj(x)  # [batch, seq_len, num_experts]
    topk_logits, topk_indices = logits.topk(top_k, dim=-1)
    gates = F.softmax(topk_logits, dim=-1)
    return gates, topk_indices
```

## 对小型 Omni 模型的启示

1. **解耦思维**：Thinker-Talker 解耦让语言理解和语音生成各司其职
2. **稀疏即正义**：小模型可以在局部做专家混合
3. **上下文是护城河**：1M 上下文让模型真正具备“读完再答”的能力

> 核心观点：规模不是目的，稀疏架构才是让大模型可扩展的关键。

---

## 参考链接

- [Kimi Technical Report](https://arxiv.org/abs/xxxx.xxxxx)
- [MoE 论文综述](https://github.com/StonePumpkins/moe-survey)