# COSREF + LLM 联合实验 v1 失败复盘

## 范围与证据边界

本复盘只读取冻结的 `runs/cosref-llm-joint-v1-groq/` 产物及对应 SQLite
数据库。没有读取 `.env`、请求头、完整 provider 异常正文或
`failed_generation`，也没有调用远程 API。原始文件的 SHA-256 在复盘前后均与
已记录基线一致。机器可读结果见 `corrected_accounting_audit.json` 和
`v1_role_quality_audit.json`；它们是新增派生文件，不修改 v1 的 summary、数据库、
日志或 complete marker。

## 失败清单

90 个逻辑决策中 78 个进入 OASIS action trace，12 个最终未完成。12 个失败均为
HTTP 400 `json_validate_failed` 的稳定脱敏类别，发生在 provider 严格结构化输出
阶段；没有证据显示这些失败到达 Pydantic、本地合法性检查或 dispatcher。

| 条件 / 策略 | timestep | user | Feed | 合法动作数 |
|---|---:|---:|---:|---:|
| moderate / no_intervention | 1 | 2 | 7 | 22 |
| moderate / no_intervention | 1 | 31 | 7 | 22 |
| moderate / no_intervention | 2 | 1 | 7 | 21 |
| moderate / static | 2 | 1 | 6 | 18 |
| strong / no_intervention | 2 | 0 | 10 | 30 |
| strong / static | 1 | 31 | 6 | 19 |
| strong / static | 2 | 0 | 5 | 15 |
| strong / static | 2 | 31 | 7 | 22 |
| strong / theory-informed | 2 | 1 | 5 | 16 |
| weak / static | 2 | 31 | 5 | 15 |
| weak / theory-informed | 2 | 1 | 5 | 15 |
| weak / theory-informed | 2 | 2 | 6 | 18 |

完整稳定 decision ID 保存在纠正审计的 `failed_decisions` 中。这里不复制 provider
失败生成内容。

## 为什么 27 次纠正只有 15 次成功

v1 的 provider schema 在
`adaptive_diffusionguard/llm/structured_actions.py::build_response_format` 中，把每次
实时动作掩码直接编码成动态字符串 `choice_id.enum`。90 次首次请求中 63 次成功，
27 次首次发生 `json_validate_failed`，于是按设计各进行一次同快照纠正；其中 15 次
成功、12 次再次失败。因此共有 39 次 provider schema 拒绝（27 次首次 + 12 次
重试），但只有 12 个最终未恢复的逻辑错误。

失败组平均 Feed 大小为 6.33、合法动作数为 19.42；成功组分别为 5.56 和
17.32。失败组覆盖 5--10 条 Feed、15--30 个合法动作，成功组也覆盖 3--10 条
Feed、9--31 个合法动作，分布明显重叠。该单次小样本只能说明较大动作集合与失败
同时出现过，不能证明 Feed 大小或合法动作数造成失败。v2 因此移除 provider 侧动态
enum，而不是依据这些描述性均值设阈值。

## 角色语义

v1 prompt 使用“behavior teacher”，但离线质量汇总曾套用安全治理规则，混淆了两种
任务。v2 明确区分：

- `behavior_simulator`：模拟具有资料、分享倾向与风险偏好的合成普通用户。高风险
  repost 是安全不利的传播结果，但不自动等于推理错误。
- `safety_policy_agent`：假设 Agent 承担治理职责，才用低风险 report、高风险
  repost 和漏干预等反向规则评价。

v1 的 78 条成功动作只可作为本次合成行为运行记录，不能转成训练数据。行为视图与
安全视图分别保存在 `v1_role_quality_audit.json`，没有合并为单一准确率。行为视图
还保留事实/理由错误与 profile 不一致；profile 证据不足的样本标为 `unscored`。
启发式文本审计本身不等于人工事实核验，所列 rationale 问题仍需人工复核。

## root 级重复 report 的形成路径

v1 `ActionMaskBuilder` 仅按直接 `post_id` 查询 report 历史。用户先 report root 或某个
quote/repost 衍生帖后，只会隐藏同一个直接帖子，另一个解析到相同 root 的可见衍生
帖仍可生成 report 选项。冻结数据中有三组这种情况：

1. moderate/static，user 0：post 1 与 post 6，同属 root 1；
2. strong/static，user 2：post 6 与 post 1，同属 root 1；
3. strong/theory-informed，user 31：post 6 与 post 1，同属 root 1。

v2 在生成掩码和 dispatch 前都沿 `post.original_post_id` 解析 canonical root，并按
`(user_id, root_post_id)` 去重 report 和 repost。quote 仍遵循 OASIS 现有语义，允许
对同 root 多次引用并提供不同文本。

## 统计重复计数根因

`adaptive_diffusionguard/experiments/cosref_llm_joint.py` 的 v1 汇总在约第 1575 行把
`logical - completed` 全部写成 `dispatcher_failure_count`；约第 1590 行又把同一个
差值加到 runtime 已记录的 `unrecovered_error_count`。所以 12 个 provider 最终失败
既被误称为 dispatcher 失败，又第二次加入未恢复错误，得到 24。

只读重算为：provider 最终失败 12、dispatcher 成功 78、有证据的 dispatcher 失败
0、未完成 12、未恢复逻辑错误 12。v2 使用互斥 failure category，每个失败逻辑决策
只增加一次 `incomplete_decision_count` 和 `unrecovered_logical_error_count`。旧字段若
保留，仅作为带 `deprecated` 映射的兼容字段。

## 哪些 v1 指标仍可信

可继续使用的工程事实包括：逻辑决策 90、成功 action trace 78、物理请求 116、首次
成功 63、结构化纠正 27/成功 15/失败 12、各 DecisionSnapshot 与成功 trace 的绑定、
已记录 impression，以及冻结数据库中的实际动作和帖子关系。

不能直接用于策略效果比较的内容包括：

- 把 12 个 provider 失败称为 dispatcher 失败或把未恢复错误报成 24；
- 将完成数不同的实验单元的行为均值直接横向比较；
- 把高风险 repost 或 ignore 一律解释为普通用户模拟错误；
- 把 oracle 合成风险标签下的结果解释为真实治理效果；
- 将启发式 rationale 审计或单次预注册 pilot 称为真实人类准确率。

因此 v1 可用于定位工程失败与验证部分记录完整性，但不能据此宣称三种曝光策略的
LLM 行为优劣。v1 的 78 条动作不会进入训练集。
