# Resume and Interview Wording

## 中文项目描述

Adaptive DiffusionGuard 是一个基于 OASIS 的可复现网络扩散治理研究原型。我在不修改
上游源码的前提下，实现了社区内外曝光控制、root级动作合法性、DecisionSnapshot
审计、多provider结构化LLM动作链路和同步阈值传播验证，并通过本地Fake实验与小规模
Groq pilot记录了模型输出协议的可靠性边界。

## English project description

Adaptive DiffusionGuard is a reproducible OASIS-based research prototype for
studying how community-aware exposure controls interact with LLM-agent actions
and synthetic diffusion. I implemented exposure governance, canonical-root
legality checks, DecisionSnapshot auditing, a bounded multi-provider structured
LLM runtime, and a synchronous threshold-response bridge, then evaluated their
engineering limits with deterministic backends and small Groq pilots.

## 中文简历 bullets

- 在OASIS之上实现独立曝光治理层，统一处理推荐与follow内容、quote/repost root映射、
  impression审计和共享预算控制，未修改上游源码。
- 构建带缓存隔离、调用预算、脱敏、严格schema、DecisionSnapshot和确定性dispatcher的
  LLM动作链路；90决策Fake验证完成90/90，并保留真实v2.1 pilot的11/12结果。
- 将COSREF论文中的社区混合与调控方向转化为显式project adaptation，并用固定种子
  曝光扫描和同步阈值传播实验评估曝光、repost、cascade与成本权衡。

## English résumé bullets

- Built a standalone exposure-governance layer on OASIS covering recommended
  and followed content, quote/repost root resolution, impression auditing, and
  shared-budget controls without patching upstream source.
- Implemented a bounded structured-LLM action pipeline with isolated caches,
  redaction, strict local validation, DecisionSnapshots, and deterministic
  dispatch; completed a 90/90 fake validation and preserved an 11/12 real v2.1
  pilot result.
- Bridged COSREF community-mixing theory into an explicitly labeled project
  adaptation and evaluated exposure, repost, cascade, and cost trade-offs with
  fixed-seed calibration and synchronous threshold simulations.

## 30秒面试介绍

这个项目研究平台曝光策略和LLM Agent行为之间的关系。我基于OASIS实现了一个不侵入
上游的治理层：平台先根据社区关系和风险分数控制Feed曝光，再冻结DecisionSnapshot，
让LLM从实时合法动作中选择，最后由本地dispatcher执行并审计。我还把COSREF的社区
内外控制方向接到本地校准和阈值传播实验中。项目最重要的结果不是“效果最好”，而是
建立了可复现边界，并真实记录了最终Groq验证只有11/12完成。

## 2分钟技术介绍

系统分三层。第一层是OASIS平台适配。我覆盖recommendation更新和refresh，使推荐与
关注内容都经过同一个曝光策略；每个候选记录root、作者社区、风险、keep probability
和shown。report/repost合法性按user和canonical root去重。

第二层是LLM动作链路。模型只看到合成profile、决策前Feed、历史和有序合法动作，不
看到策略名、mixing parameter或oracle风险标签。每次调用前持久化DecisionSnapshot；
响应经过Pydantic、索引—动作—目标一致性和当前状态复检，之后才进入确定性dispatcher。
缓存按协议、完整动作列表和schema隔离，并有调用上限、退避和脱敏。

第三层是COSREF理论桥。我没有把论文omega直接当成曝光概率，而是先核实mixing和阈值
机制，再把社区内外控制方向映射为待校准的OASIS keep候选。独立threshold-response
模块按同步、不可逆规则验证多邻居信号能否传导到cascade；联合实验中它只做shadow，
不会与LLM同时决策。

工程上Fake验证完成90/90。真实协议经历了v1动态enum、v2固定索引和v2.1冗余语义
字段：v2.1的11条成功响应都通过一致性检查，但第12条遭遇provider HTTP 400，因此
最终是11/12、degraded。项目据此冻结，不把小型合成结果包装成真实治理结论。

## “为什么只有11/12？”

这是预注册停止规则下保留的真实结果。12个决策中，11个获得合法结构化响应并成功
dispatch；另一个在弱社区无干预单元最终收到Groq `json_validate_failed` HTTP 400。
它没有被转换成ignore，也没有进入动作trace或训练数据。11条成功响应中没有检测到
choice index、action type和target post错绑，dispatcher失败也是0，但provider最终
可靠性仍未达到12/12。因此我没有重跑、换模型或继续开发v2.2，而是把项目冻结并明确
记录：本地语义防线有效，不等于远程严格结构化输出已经完全可靠。
