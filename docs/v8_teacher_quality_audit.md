# v8 教师决策质量审计

本报告由离线只读审计生成；它不是模型准确率、治理有效性或因果效果证明。

## 数据完整性

- summary teacher_examples：15
- JSONL 有效行：15
- 唯一 sample ID：15
- 与成功 trace 一一对应：True
- 重复动作 trace：0
- 失败或标签不匹配样本：0
- 凭据模式扫描：True（未扫描 .env）
- JSONL feed_post 与真实 Feed 不匹配：15 条

## 动作与风险交叉表

| 动作 | 数量 | 低风险目标 | 高风险目标 | 无目标 |
|---|---:|---:|---:|---:|
| ignore | 10 | 0 | 0 | 10 |
| quote | 0 | 0 | 0 | 0 |
| report | 2 | 0 | 2 | 0 |
| repost | 3 | 3 | 0 | 0 |

- 可明确判断的 rule-aligned：5/5 (1.000)
- 明确反向动作：无
- 需人工复核：['trace-25']
- missed_intervention_candidate：0
- rationale 事实性作者错误：['trace-25']

## 曝光与治理

| 分组 | 候选 | 展示 | 抑制 | 抑制率 |
|---|---:|---:|---:|---:|
| 总计 | 30 | 23 | 7 | 0.233 |
| 低风险 | 20 | 20 | 0 | 0.000 |
| 高风险 | 10 | 3 | 7 | 0.700 |
| intra-community | 14 | 14 | 0 | 0.000 |
| inter-community | 16 | 9 | 7 | 0.438 |

每时间步：

- t=0: 候选 10，展示 7，抑制 3，抑制率 0.300
- t=1: 候选 10，展示 8，抑制 2，抑制率 0.200
- t=2: 候选 10，展示 8，抑制 2，抑制率 0.200

高风险抑制率高于低风险抑制率；这只是该合成运行中的分层观察，不能解释为因果效果。

## 样本明细

| sample ID | t | user | action | target | root | risk | relation | sanity |
|---|---:|---:|---|---:|---:|---|---|---|
| trace-9 | 0 | 0 | repost | 1 | 1 | low | intra | aligned_repost_low |
| trace-11 | 0 | 1 | repost | 1 | 1 | low | inter | aligned_repost_low |
| trace-13 | 0 | 2 | ignore | None | None | unknown | not_applicable | ignore_low_risk_only_usually_acceptable |
| trace-15 | 0 | 3 | report | 2 | 2 | high | intra | aligned_report_high |
| trace-17 | 0 | 4 | ignore | None | None | unknown | not_applicable | ignore_low_risk_only_usually_acceptable |
| trace-19 | 1 | 0 | ignore | None | None | unknown | not_applicable | ignore_low_risk_only_usually_acceptable |
| trace-21 | 1 | 1 | report | 2 | 2 | high | intra | aligned_report_high |
| trace-23 | 1 | 2 | ignore | None | None | unknown | not_applicable | ignore_low_risk_only_usually_acceptable |
| trace-25 | 1 | 3 | repost | 1 | 1 | low | inter | aligned_repost_low |
| trace-27 | 1 | 4 | ignore | None | None | unknown | not_applicable | ignore_low_risk_only_usually_acceptable |
| trace-29 | 2 | 0 | ignore | None | None | unknown | not_applicable | ignore_low_risk_only_usually_acceptable |
| trace-31 | 2 | 1 | ignore | None | None | unknown | not_applicable | ignore_low_risk_only_usually_acceptable |
| trace-33 | 2 | 2 | ignore | None | None | unknown | not_applicable | ignore_low_risk_only_usually_acceptable |
| trace-35 | 2 | 3 | ignore | None | None | unknown | not_applicable | ignore_low_risk_only_usually_acceptable |
| trace-37 | 2 | 4 | ignore | None | None | unknown | not_applicable | ignore_low_risk_only_usually_acceptable |

## 限制

- Only 5 synthetic agents, 3 timesteps, and 15 actions were observed.
- Only two synthetic seed posts were used.
- Samples are temporally and socially dependent, not independent and identically distributed.
- Risk labels are experiment presets, not human annotations.
- No claim about representative human behavior has been tested.
- These data cannot establish model accuracy, governance effectiveness, or causal effects.
- The audit does not justify starting LoRA training.
- Quotes and ignores with high-risk or mixed feeds require human review.
- The teacher JSONL feed_post field contains action payloads rather than the reconstructed visible feed.

## 结论

**1. 可以进入100决策评估，但不能开始LoRA。**

动作 sanity check 支持扩大到 100 决策做稳定性评估；但在任何 LoRA 使用前，必须修复 JSONL 的 Feed 序列化，并继续审计 rationale 事实错误。
