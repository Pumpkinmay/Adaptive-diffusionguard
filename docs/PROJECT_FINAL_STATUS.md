# Adaptive DiffusionGuard：项目最终状态

更新日期：2026-10-02。项目在 v2.1 真实微型验证后冻结。本文件汇总已有产物，
不改变任何既有实验结果，也不把合成实验解释为真实人类行为或治理效果。

## LLM 工程链路

项目已实现：多提供商模型配置、密钥脱敏、并发与物理请求预算、退避和缓存、
DecisionSnapshot 持久化、动态合法动作掩码、canonical-root 合法性与 report/repost
去重、本地确定性 OASIS dispatcher、结构化输出审计，以及互斥的 provider/local/
dispatcher 失败统计。

协议按版本保留：

- v1：动态 `choice_id` enum。真实预注册运行完成 78/90，物理请求 116，退出码
  2；离线纠正统计为 provider 最终失败 12、dispatcher 失败 0、未完成 12。
- v2：固定整数 `choice_index`。真实 micro-pilot 完成 20/20，物理请求 27，首次
  成功率 95%，退出码 0；但记录
  `joint-v2-strong-community-static_cosref-61001:t000001:u000031:d000002`
  暴露了索引与 rationale 语义错绑：执行 `report:2`，理由却明确讨论 post 3。
- v2.1：固定 `choice_index + action_type + target_post_id + quote_text` 冗余语义
  校验。真实 micro-pilot 完成 11/12，物理请求 19/24，首次成功 10/12
  （83.33%），退出码 2。11 条成功响应的 index/action/target 均一致，语义错绑
  和 dispatcher 失败均为 0；一个弱社区无干预决策最终发生 provider
  `json_validate_failed`（HTTP 400），因此整体状态为 `degraded`。没有重跑。

v2.1 本轮还有 6 次已恢复限流、7 次底层重试、2 次 provider
`json_validate_failed` 尝试，其中一次结构化纠正成功、一次成为最终 provider
失败。DecisionSnapshot 共 12 条（11 succeeded、1 failed），成功 trace 11 条，
绑定率 1.0；root 重复 report/repost 为 0；训练或教师样本为 0。

因此，固定冗余字段能够拦截本地可观察的语义绑定错误，但当前 Groq 模型/严格
结构化输出组合没有在最后一次小样本验证中达到 12/12 provider 可靠性。根据预先
约定，协议开发停止，不新增 v2.2，不扩大真实运行。

## COSREF 理论桥

`docs/cosref_bridge_spec.md` 区分了论文定义与项目适配：论文的
`paper_omega_intra/inter` 不被视为 OASIS 的 `oasis_keep_intra/inter`。项目根据实际
网络边与社区标签计算 mixing parameter，并通过离线曝光扫描将论文支持的控制方向
映射为候选 keep probabilities。

官方 v1.0.0 代码的小型 reference slice 在 `mu=0.2` 和 `mu=0.8` 下分别观察到更强
社区内和社区间控制方向，与核实的补充材料方向一致；离散小网格最优点并不等同于
论文连续网格结果，也不是整篇论文复现。

后续 robustness pilot 使用 5 个校准种子和每条件/策略 10 个评估种子。风险曝光相对
无干预下降较稳定，但精确参数和理论方向选择并非三种网络条件都稳定；与对称 static
COSREF 相比没有普遍稳定优势。该结果支持“可运行的理论桥和可测量的通道分配”，
不支持 COSREF 在真实平台或一般网络中更优。

## 阈值传播验证

独立 threshold-response bridge 实现了同步更新、不可逆采纳、社区内外已采纳邻居
计数、单 user/root 一次采纳，以及曝光门控。paper omega 与 OASIS keep 参数保持
分离，threshold shadow 不执行 LLM 动作。

小型 paper-native 方向对照通过。预注册的 60 节点、8 时间步实验显示阈值层可产生
跨时间步传播；在该合成配置中，strong/weak 条件下曝光差异传导到较小的 repost 和
cascade 差异，moderate 条件的 balanced 参数与 static 相同。网络接近饱和、weak
realized-cost 匹配超出 5%目标、使用 oracle 合成风险标签，因而不能外推为现实治理
效果或完整论文复现。

## 能够支持的结论

- 已构建一个可复现的 OASIS 网络曝光治理研究原型，并实现版本化、可审计的 LLM
  动作链路。
- 本地 Fake、单元测试和合成网络实验能够验证快照、合法动作、root 去重、缓存隔离、
  请求预算、dispatcher 与阈值机制等工程性质。
- 小型真实 Groq pilot 揭示了动态 enum、裸索引语义绑定和 provider 严格结构化输出
  的不同失败模式；v2.1 成功消除了已完成响应中的可观察 index/action/target 错绑，
  但未达到全量 provider 可靠性。
- COSREF 的社区内外控制方向可以作为 OASIS 曝光策略的项目适配先验，并可用本地
  校准与传播指标检验。

## 不能支持的结论

- 不能宣称 LLM Agent 代表真实人类，或报告真实人类行为准确率。
- 不能宣称 COSREF、任一曝光策略或 LLM 决策在现实中具有因果治理效果或普遍优势。
- 不能宣称论文 omega 与曝光保留概率数学等价，或已经复现整篇论文。
- 不能把 oracle 风险标签下的零正常内容损失外推到真实部署。
- 不能把单次、有限合成 pilot 当作独立重复、统计显著性或外部有效性证据。
- rationale 尚未被批准为训练标签；项目没有完成 LoRA/QLoRA 训练，也不应依据这些
  pilot 结果启动训练。

## 各入口结果对照

- `diffusionguard-real-llm` 使用动态 `choice_id` 枚举协议。最近一次 real-llm 运行完成
  15/15 个决策，状态为 `success`、退出码 0；首次成功 14/15（93.33%），一次
  provider `json_validate_failed` 经一次结构化纠正重试成功恢复，共产生 16 次物理
  远程请求。
- `cosref-llm-joint-v2.1` 使用 `fixed-choice-semantic-v2.1` 冗余语义协议。真实
  micro-pilot 完成 11/12，状态为 `degraded`、退出码 2，共产生 19 次物理远程请求；
  11 条成功响应的 choice/action/target 语义字段一致，另有 1 个最终 provider 失败。
- 100 决策评估框架的 Groq 运行只执行了 `batch-01`，该批次 20/20 完成，状态为
  `success`。其动作质量评分来自 synthetic rule-based rubric，不是人类行为金标准。

三个入口的协议、实验目的和规模不同，结果互相不可替代。协议开发仍按预注册停止
规则冻结在 v2.1；本对照仅澄清入口差异，不改变任何既有工程、行为或治理结论。

## 冻结状态

- LLM 协议冻结在 `fixed-choice-semantic-v2.1`，最终真实结果为 11/12、degraded。
- 不开发 v2.2，不重跑 v2.1，不执行 90 决策真实扩展。
- 不运行 LoRA，不把 v1/v2/v2.1 rationale 或成功动作自动转成训练样本。
- 既有实验目录保留为不可改写的审计证据。

## 简历可使用的准确表述

> 构建了基于 OASIS 的可复现网络扩散治理研究原型，实现 COSREF 社区混合统计与
> 曝光校准、同步阈值传播模拟、DecisionSnapshot 数据审计，以及带请求预算、缓存
> 隔离、root 级合法性检查和固定结构化输出的多提供商 LLM 动作链路；通过本地
> Fake/单元测试和小规模 Groq pilot 识别并记录 provider 与语义绑定失效边界。

不应写成“证明治理有效”“复现论文全部结果”“训练出人类行为模型”或“LLM 协议
达到生产可靠性”。
