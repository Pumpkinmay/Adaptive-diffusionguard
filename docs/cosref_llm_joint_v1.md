# 受限 COSREF + LLM 联合实验框架 v1

## 目的与非目标

该框架固定结构化动作模型、提示模板、五个合成 Agent profile、动作接口、
初始帖子、网络和活动顺序，只改变 OASIS 网络层曝光策略：

1. `no_intervention`；
2. `static_cosref`；
3. `theory_informed_cosref`。

主要比较为 theory-informed 与 static。无干预只作为参考。该 pilot 不生成训练
数据，不评估真实人类代表性，也不把 Fake Backend 数值解释为真实模型效果。

## 决策链与角色分离

```text
固定初始网络和平台状态
  -> COSREF exposure keep probability
  -> OASIS refresh + impression log
  -> 已验证的动态合法动作掩码和 strict response_format
  -> 同一结构化模型接口（Fake 或明确授权后的 Groq）
  -> 已验证的本地 ActionDecisionGateway
  -> OASIS action trace
```

联合路径不向模型提供 `mu`、策略名称或理论分配信息。输入仍由既有
`build_decision_messages()` 生成。`paper_omega` 不作为曝光参数：生产路径只使用
`oasis_keep_intra/inter`。

threshold-response 仅在冻结 DecisionSnapshot 之后做只读 shadow 计算，输出
传播压力、潜在采纳和曝光阻断诊断。它不调用 refresh、不 dispatch、不写动作
trace、不生成 repost、不改变 prompt，也不消耗远程请求。其 provenance 为
`counterfactual_shadow_diagnostic_project_adaptation`。

## 初始条件配对

每个社区条件下，三个策略共享同一网络边、社区标签、profile、初始帖子、风险
标签、空的初始动作历史、活动顺序、OASIS 随机种子和 prompt/schema 代码。
每个单元开始前重置 Python、OASIS 所用模块级随机状态、项目 policy RNG，若
NumPy 可用也重置其 RNG。

处理开始后，不同 Feed 和动作会使平台状态自然分化。因此这是
**初始条件配对**，不是逐事件完全配对。

## 网络与治理参数

每个网络由固定 seed 生成并按实际边重新测量 `mu`。理论 keep 对由实际 `mu`
调用严格分配器选择：强社区为 intra 优先，中等容差带为 balanced，弱社区为
inter 优先。实际测量不满足预注册区间或没有合法候选时直接失败。

曝光治理使用已知合成风险真值，配置和所有 summary 均标记：

```text
oracle_synthetic_risk_labels=true
```

因此 benign exposure loss 为零或较低不能解释为真实部署性能；本实验没有加入
风险分类器误差。

## 规模和安全上限

- 3 个网络条件 × 3 个策略 = 9 个独立单元；
- 每单元 5 Agent × 2 时间步 = 10 个逻辑决策；
- 全部 90 个逻辑决策；
- 每个决策最多一次既有 `json_validate_failed` 纠正重试；
- 每单元最多 20 次物理请求，全局最多 180 次；
- concurrency=1、temperature=0、max_tokens=512；
- 每个单元独立数据库、cache、run_id 和 complete marker。

`--plan` 只读取配置、生成确定性网络并计算实际 `mu`，不会导入 OASIS 平台、
读取 `.env`、创建模型、数据库、cache 或输出目录。Groq 非 plan 执行必须显式
提供 `--confirm-remote-run`，且该检查发生在环境读取之前。

基于 12 秒请求间隔：90 个无重试请求的 interval-only 等待约 972 秒；180 次
最坏请求的 interval-only 等待约 2052 秒。若每次请求都达到60秒 timeout，
保守上界为12852秒。这些是计划上限，不是服务延迟承诺。

## Fake Backend

Fake Backend 是确定性工程夹具：根据合成 profile 和 Feed 文本覆盖低风险 repost、
高风险 report、混合 Feed quote、空 Feed/仅 ignore，并在每单元首个决策注入一次
`json_validate_failed` 后由既有纠正重试恢复。最终失败路径只在测试中注入，
用于确认 failed snapshot 不绑定 trace、不写训练样本。

Fake 响应经过与远程路径相同的 Pydantic schema、原始合法动作重新校验、
dispatcher、DecisionSnapshot 和成功后缓存边界。Fake 结果不得称为真实 LLM
行为、治理效果或因果效应。

## Resume

resume 仅以完整单元为边界。complete marker 校验配置哈希、实现哈希、SQLite
integrity、快照数、成功数、唯一 decision ID 及单元文件哈希。完整单元不重跑；
不完整单元整体移入 `quarantine/` 后从单元起点重建。配置或 backend 改变时拒绝
resume，不从单个逻辑决策中间恢复。

## 命令

只读真实计划：

```bash
uv run diffusionguard-cosref-llm-joint \
  --config configs/cosref_llm_joint_v1.json \
  --backend groq \
  --output runs/cosref-llm-joint-v1-groq \
  --plan
```

本地 Fake 验证：

```bash
uv run diffusionguard-cosref-llm-joint \
  --config configs/cosref_llm_joint_v1.json \
  --backend fake \
  --output runs/cosref-llm-joint-v1-fake
```

本轮没有授权、也不会执行 Groq 非 plan 命令。
