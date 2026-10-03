# COSREF + LLM 联合实验 v2：离线协议与角色契约

## 角色契约

v2 配置固定 `agent_role=behavior_simulator`。模型接收合成 profile、冻结 Feed 和有序
合法动作列表；oracle 风险标签、网络条件、策略名、混合参数与控制参数不会进入
prompt。安全治理评分是独立的反事实审计视图，不能与普通用户行为模拟合并为准确率。

`behavior_simulator` 的核心检查是动作合法、profile 一致、内容理解和理由事实一致。
高风险 repost 计入传播上的 `legal_but_safety_adverse`，并不自动成为模型推理错误。
只有证据充分时才标 profile 不一致；否则为 `unscored`。

`safety_policy_agent` 视图单独统计低风险 report、高风险 repost、明显高风险内容仍被
ignore 的 missed intervention，以及 quote stance。这一视图假设 Agent 承担内容
安全职责，不用于评价普通用户行为真实性。

## 固定索引协议

provider 只看到固定 strict JSON Schema：

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "adaptive_diffusionguard_action_index_v2",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {
        "choice_index": {"type": "integer"},
        "rationale": {"type": "string"}
      },
      "required": ["choice_index", "rationale"],
      "additionalProperties": false
    }
  }
}
```

它不包含动态 enum，也不发送 tools 或 tool_choice。每次决策把 `ignore` 和当前合法的
report/repost/quote 按稳定顺序编号，完整有序列表只放在 messages 中。客户端用严格
Pydantic 模型拒绝非整数，再用冻结 DecisionSnapshot 验证索引范围并确定性映射到
choice ID；dispatch 前按当前数据库状态再次验证。

若第一次 provider `json_validate_failed`、本地 schema 失败或索引越界，最多追加一次
固定纠错提示。纠正请求复用同一个 Snapshot、Feed、有序动作列表与 response format，
不 refresh、不增加 impression、不改变动作顺序。第二次失败不会转换为 ignore，也
不会进入 trace 或任何训练数据。

缓存目录使用 `response_cache_v2.sqlite3`；key material 包含协议版本、完整有序动作
列表、messages 和固定 response format。每个策略单元仍使用独立缓存，因而 v1/v2
和跨策略响应不能互相命中。

## canonical root 规则

每个可见 post 沿 `original_post_id` 解析到稳定 root。用户 report 过 root 或任一衍生
帖后，同 root 的所有 report 选项消失；repost 同理。dispatch 前再构建一次 root-aware
掩码，避免 Snapshot 后状态变化。quote 按现有 OASIS 实现保留重复引用能力，其文本可
不同，不与 report/repost 一并禁用。

每条成功记录同时保存 visible post IDs、selected post ID 和 canonical root post ID。
历史 v1 数据不回写。

## 互斥失败统计

最终失败只归入 provider、local schema、invalid index 或 dispatcher 之一，同时只增加
一次 incomplete 与 unrecovered logical error。provider 失败不再计作 dispatcher
failure。v2 主要字段为：

- `provider_failure_count`
- `provider_json_validate_failed_count`
- `local_schema_failure_count`
- `invalid_choice_index_count`
- `dispatcher_failure_count`
- `completed_decision_count`
- `incomplete_decision_count`
- `unrecovered_logical_error_count`

旧字段只在 `deprecated_fields` 中给出明确映射。

## 真实微型验证计划

`--plan --micro-pilot` 只读取 v2 配置，不读取 `.env`，不创建模型、数据库、缓存或
输出目录。预注册选择覆盖强/中/弱三类网络、三种曝光策略、五种 behavior profile、
两个 timestep、root 与衍生帖上下文以及有/无高风险内容可能出现的决策上下文。模型
动作本身仍由真实响应决定，不能在计划阶段保证动作比例。

规模固定为 20 个逻辑决策、每个最多一次纠正、最多 40 次物理请求。真实 backend
同时需要 `--micro-pilot` 和 `--confirm-remote-run`，并在读取 env-file 或创建模型前
检查这两个参数。本次离线任务不执行真实微型验证。

示例只读计划：

```bash
uv run diffusionguard-cosref-llm-joint-v2 \
  --config configs/cosref_llm_joint_v2.json \
  --backend groq \
  --output runs/cosref-llm-joint-v2-groq-micro \
  --plan --micro-pilot
```

未来得到明确授权后，真实命令还必须显式添加 `--confirm-remote-run` 和经批准的
env-file；本文件不包含密钥。
