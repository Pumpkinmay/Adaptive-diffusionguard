# Adaptive DiffusionGuard — Portfolio Summary

## 项目问题

信息扩散治理不仅取决于内容判断，还取决于平台让谁看到什么。Adaptive
DiffusionGuard 基于 OASIS 构建一个可复现研究原型，把网络层曝光治理、LLM Agent
行为和传播指标分开记录，研究仅改变社区内外曝光策略时，合成 Agent 的可见内容、
动作和后续扩散如何变化。

项目使用合成网络、合成帖子和 oracle 风险标签。它不是现实部署，也不把 LLM Agent
当作真实人类。

## 技术架构

```text
COSREF论文方向 ──> 项目校准层 ──> OASIS keep probability
                                      │
OASIS网络/帖子 ──> 自定义refresh ──> 冻结Feed + DecisionSnapshot
                                      │
合成profile ────────────────────────> LLM合法动作选择
                                      │
                         本地校验与确定性dispatcher
                                      │
                 impression / trace / cascade / 成本审计

threshold-response ── shadow diagnostic（不执行LLM动作）
```

职责边界明确：COSREF只控制曝光；LLM只在当前合法动作中选择；threshold-response仅做
离线或shadow机制诊断。论文 `paper_omega` 与项目 `oasis_keep` 始终分离。

## 主要新增工作

- 在不修改 OASIS 源码的前提下实现 `AdaptiveDiffusionPlatform`，同时治理推荐内容与
  follow内容，并追踪quote/repost到canonical root。
- 设计规范化 impression、DecisionSnapshot、动作trace绑定和失败审计。
- 实现静态 COSREF、共享预算的规则动态控制器及理论引导分配器。
- 建立多provider LLM工厂、请求预算、缓存隔离、脱敏、token统计和有边界重试。
- 迭代三代结构化动作协议，并保留所有失败：dynamic enum v1、fixed index v2、
  semantic redundancy v2.1。
- 实现同步、不可逆的 threshold-response bridge，验证多邻居阈值机制和曝光门控。
- 建立离线评估、Fake Backend、root级去重、恢复检查点和固定种子测试。

## COSREF 如何进入系统

理论来源是 Chen et al. (2026) 的社区结构—调控耦合模型及官方 v1.0.0 代码。项目先
核实 mixing parameter、社区内外影响权重、阈值不等式和成本，再把“控制方向”作为
OASIS曝光候选的先验。keep参数由本地固定网络扫描校准，而不是直接令论文omega等于
曝光概率。这一层明确标记为 project adaptation。

## 关键工程结果

- 完整测试集与Ruff通过；CPU-safe smoke不调用LLM。
- Fake联合实验完成90/90，9/9单元完整；Snapshot/trace绑定率1.0，root重复动作0。
- v2.1 Fake故意为每个决策注入一次target错绑，90次均在dispatch前拦截并纠正；
  纠正不新增refresh或impression。
- 真实运行没有生成教师或训练样本，rationale均不标记为训练可用。

## 理论实验结果

官方reference slice在强、弱社区条件下观察到与已核实材料一致的控制方向，但不是
完整论文复现。robustness pilot显示风险曝光相对无干预下降较稳定，不过控制方向和
精确keep选择并非三种条件都稳定，与对称static COSREF相比也没有普遍传播优势。

threshold-response预注册实验的paper-native方向对照通过。60节点、8时间步的合成
实验中，strong/weak条件出现较小的风险repost和cascade差异，moderate条件的balanced
策略与static相同。实验接近传播饱和，且部分realized-cost匹配未达到5%目标。

## 真实 Groq 结果

使用的是Groq托管的现成 `openai/gpt-oss-20b`，没有微调。

| 协议 | 真实规模 | 完成 | 主要发现 |
|---|---:|---:|---|
| v1 dynamic enum | 90 | 78/90 | 12个最终provider失败 |
| v2 fixed index | 20 | 20/20 | 发现1条数字索引与rationale目标错绑 |
| v2.1 semantic redundancy | 12 | 11/12 | 11条成功响应字段一致；1个最终HTTP 400 `json_validate_failed` |

另一个独立的 real-llm 入口使用动态 `choice_id` 枚举协议，最近一次真实端到端运行完成
15/15；首次成功14/15（93.33%），一次 provider `json_validate_failed` 经结构化纠正
成功恢复。它与v2.1冗余语义协议的11/12验证用途和协议不同，不能合并为同一成功率，
也不表示协议问题已最终解决或治理有效。

v2.1共19/24次物理请求，首次成功10/12；choice/action/target本地错绑为0，dispatcher
失败为0。结果是`degraded`，因此协议链路被冻结，而不是宣称最终解决。

## 失败保护

- 远程调用必须显式启用，provider密钥不互相映射。
- 每单元独立缓存，key包含协议、完整有序动作列表、messages和response format。
- 非法或过期动作不转换成ignore；失败snapshot不进入trace或训练数据。
- report/repost按user与canonical root去重；dispatch前再次检查当前数据库合法性。
- 请求上限、并发、间隔、重试、退出码和未恢复错误分别审计。
- 不覆盖完成结果；配置哈希变化时拒绝恢复。

## 局限与未来工作

结果来自有限合成网络和单次小规模真实provider pilot，不具有人类外部效度，也不能
证明COSREF的现实治理效果。oracle风险标签高估了风险定向策略在真实分类误差下的
表现。v2.1仍有provider严格结构化输出失败，项目按预注册规则停止协议开发。

若未来重新立项，应首先获得新的实验授权，使用独立种子、替代provider/model和明确
风险分类误差；不应继续在本项目结果上事后调参。当前不应启动LoRA。
