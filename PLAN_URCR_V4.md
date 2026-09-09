---
title: "URCR-V4：统一动作效用、思考责任与局部残差信用路由"
date: "2026-09-08"
method_revision: "urcr_v4_unified_typed_r1_final"
status: "方法框架已冻结；仅待本地数值校准、实现核验与正式执行"
primary_model: "Qwen2.5-3B-Instruct"
primary_codebase: "现有 SDAR/EviSD-derived veRL 工程"
diagnostic_checkpoint: "/data1/kongmu/model_checkpoints/base_grpo_3b_s150_hf；已由导出 manifest 核验为纯 GRPO S150，使用前复核 checksums"
formal_initialization: "原始 Qwen2.5-3B-Instruct，禁止从诊断 S150 续训"
formal_budget: "300 outer/global steps；128 questions × 8 rollouts"
---

# URCR-V4 最终方法与 Codex 执行计划

## 0. 任务、授权与完成定义

本文件是自包含的方法规格和执行计划，不是继续征集架构的讨论稿。执行者不需要另读会话才能确定主方法。

执行顺序固定为：

```text
定位真实代码、纯 GRPO S150、原始 Base 和可用资源
  → 实现最小 scorer / credit reference，复用原有 rollout 与 span 工具
  → 纯 GRPO S150 小样本诊断，确定数值尺度并测量完整评分成本
  → 冻结 resolved_method.json
  → 接入原始 actor update，完成必要数值与分布式检查
  → 从原始 Base 做 5-step disposable smoke
  → 清空 smoke 科学状态，从原始 Base 开始唯一一条 300-step 正式训练
  → S150 / S300 保存并完成七集评估，保留原始题级输出
```

用户已授权完成上述实现、必要验证和条件满足后的真实长训练。不要在检查通过后停在“launcher 已写好”或再次询问是否启动。也不要在核心数据流错误、权限/资源不存在时假装已经执行。

本轮只做一条正式主线；不自动增加 query-only、AGAM-only、多个 seed、多个 utility 架构或多个 300-step run。已有 GRPO 运行不得被打断。不能杀死其他用户进程、清空共享 Ray 集群、关闭共享 retriever、改动共享代理或系统配置来腾资源。

若资源暂不可用，可以完成本地代码和 CPU 工作，再报告明确的资源阻塞；不能把未运行的任务描述成后台运行。允许使用用户环境中已有的调度器提交任务，但必须回报真实 job ID 和排队状态。

### 0.1 三种“确定”分开

**已经确定：**本文件的方法公式、监督边界、启用模块、归约目标、正式训练起点和预算。

**需要小数据确定：**utility 的死区与尺度、Q/A 各自的责任数值尺度、一个总体 local-loss 系数，以及等价评分后端的选择。这些不是新架构搜索。

**尚未被实验确定：**V4 的准确率、调用效率和 responsibility 的独立增益。本轮不把最小检查当成性能证明。

### 0.2 证据优先级

科学设计以当前用户授权和本文件为准；现场“实际运行了什么、数据与接口是什么”以真实源码、配置和原始数据为准。历史审计和旧建议只作背景。若实际实现与规格不一致，应报告并修正，不得用本文覆盖现场事实或伪造完成状态。

源材料包括合作者 C 的《URCR-V4：从“固定证据奖励”重构为“动作效用—思考责任—残差信用路由”》。本文件采用其中的 Unified Typed URCR + additive residual 路线，并明确记录以下收敛修改。正式方法统一称为 URCR-V4；不再使用不同会话中含义不一致的 V4-A/B/C 字母区分执行线。本文自身足以实施，不要求 Codex 取得 C 原文。

---

## 1. 最终架构及审查决定

### 1.1 主张与准确边界

URCR-V4 区分两个问题：

1. 当前 action 的可观测后果，是否提供值得学习的局部质量信号？
2. preceding think 的哪些内容，促进了这个实际 sampled action 的生成？

二者相乘构造局部 routed residual；全局 GRPO 继续负责最终任务目标。

本版本采用**促进型功能依赖**，即责任权重非负，而动作 utility 可正可负。它没有估计完整动作分布上的反事实期望效用差，也没有处理“think 抑制了坏动作但坏动作仍被采样”这一反向责任情形。不能把它写成严格因果责任、无偏 Q-value 或完整的新 advantage estimator。

### 1.2 冻结决定表

| 项目 | 本次正式决定 | 理由/范围 |
|---|---|---|
| Global GRPO | 保持当前纯 GRPO baseline 原样 | 不把 V4 分数写入 terminal reward 后重新组归一化 |
| Search utility | real-vs-control 与 real-vs-null 双参照，同号才定向 | 同时检查新增信息与无关内容对照 |
| Gold probe | 固定 alias 集合上的平均 length-normalized log-prob | 不在每个反事实里重新选一个最有利 alias |
| 随机对照 | K=3，同批其他题；优先匹配来源/阶段/长度 | 不为每种 metadata 或题型写不同 reward |
| Search 后处理 | 连续死区 + 有界 tanh；负侧系数 0.1 | 直接给出有界 utility，避免又一层无限尺度 |
| Hotpot bridge rescue | **本次训练关闭** | metadata 命中不充分证明桥接效用；不把估值盲区再次自动换成固定正奖 |
| Answer utility | 同题、同 EM stratum 内的 action-level F0.5 残差 | 只补充终端二值结果未区分的词级质量差异 |
| Answer 正确 stratum | 局部残差恒为零 | 不额外重复强化正确答案；正确性由全局 GRPO 处理 |
| Responsibility | actual sampled query/answer，full-vs-think-mask；Q/A 单独校准 | 不把 gold answer likelihood 当作此估计器的 sampled-action dependency |
| Chunk routing | positive soft-sparse，指数 p=2，最多 6 chunks | 不用硬 Mass50；无可信 mass 时零路由 |
| Query/think 关系 | additive，非严格零和 split | 可以共同对同一结果负责，但总预算受限 |
| Local 系数 | alpha_Q=1，alpha_A=0.25，lambda_TQ=lambda_TA=1 | 答案词重合代理较弱，先给予较小预算；不搜索这组常数 |
| Search trajectory cap | B_Q=2，包含 query 与 routed think 的绝对质量 | 不按 calls 机械均分，但限制局部质量累积 |
| Loss | signed span-mean PPO + 原始 trajectory denominator | 和全局 token-mean 分开，复用一个 actor/optimizer |
| 路由范围 | Q_t→T_t；A→final T | 不将 terminal residual 再广播到所有历史 think |
| 训练 metadata | 完全退出在线 rollout/scorer/routing/update；仅允许离线 sidecar 对比 | V4 信用只依赖可见 H/T/Q/O、训练标签和终端结果 |
| 关闭 | V1/V2 fixed reward、旧 AGAM、V3 focus KL、蒸馏、critic、额外训练期分叉 | 避免重建新的多组件研究项目 |

alpha_A=0.25、p=2、K=3、B_Q=2、lambda_T=1 是本次预先选定的工程设计，不声称是文献最优值。不能根据七集测试结果调这些量。

### 1.3 对 C 版的必要修补

**桥接补偿：**C 的 agreement operator 会把异号对照映射为零；这个零可能来自一正一负的强冲突，而非两个对照都接近零。因此不能仅用 `abs(g_dir) <= delta` 开启 meta rescue。本次直接关闭可选 rescue。new fact/doc 等旧 metadata 对比只允许在保存的原始轨迹上离线计算，不进入在线 rollout，也不实现新的 rescue 模块。

**答案残差：**stratum 必须在同一题的 8 个原始 rollout 内建立，不得跨问题或跨 minibatch 混合。正确答案全部 q=1，因此本版本中正确答案的 U_A=0；“正确答案必然得到正 answer-local credit”不适用于本公式。一个 EM 错误但高于同题错误答案平均 F0.5 的答案可以得到正残差，这只是词级相对质量，不是语义正确性。

**功能责任：**C 的 same-action 责任和本文件采用的促进型路由是一套合理的低成本估计器。但它不等于“改变 think 后重新生成动作”的总效应；也不能保证一次偶然错误答案完全不影响有帮助的 think。此次不叠加 gold-readiness veto、signed dependency 或候选动作分布来解决所有情况。

**预算：**trajectory cap 只限制局部系数总量，不证明 tool-call 一定下降；span mean 保持系数质量不随长度增长，不证明参数梯度范数严格长度无关。

---

## 2. 现场身份与正式训练协议

### 2.1 先核验诊断 checkpoint

本次诊断 checkpoint 选择：

```text
/data1/kongmu/model_checkpoints/base_grpo_3b_s150_hf
```

其 `export_manifest.json` 已记录：源 step=150、入口为 `verl.trainer.main_ppo`、`adv_estimator=grpo`，URCR/EviSD/SDAR/SDL 均关闭，且 FSDP→HF 参数比较通过。使用前仍须复核 manifest、`SHA256SUMS`、模型文件与当前诊断命令，并确认：

- 初始模型是原始 Qwen2.5-3B-Instruct；global step=150。
- 无 V1/V2 local credit、无 AGAM、无 EviSD search/answer teacher、无 V3 focus。
- 不是 `g2_real_no_answer_pi`、V2.1 S150 或旧 V3 checkpoint；这些虽然也可能无 answer PI，但不等于纯 GRPO。
- tokenizer、chat template、retriever、数据和 action schema 与现有主协议一致。

路径、hash、model revision、配置摘要写入 `identity.json`。不得因为目录名正确就跳过文件和配置核验；若该导出损坏或身份复核失败，先报告具体结果，不擅自用旧方法模型冒充。

### 2.2 正式起点

诊断 S150 只用于 frozen rollout、scorer/scale 校准。正式训练从原始 Qwen2.5-3B-Instruct 重新开始，optimizer、LR scheduler、dataloader、训练 RNG 和 global_step 均从头初始化。

禁用自动“搜索最新 checkpoint 并 resume”行为。只有恢复**同一条正式 V4 run**时才允许完整 resume。

### 2.3 正式协议

| 项 | 冻结值或选择原则 |
|---|---|
| 模型 | 原始 Qwen2.5-3B-Instruct |
| 预算 | 300 outer/global steps，不是 300 minibatch optimizer steps |
| 数据 | 当前 NQ+HotpotQA 合并训练集与原 shuffle；正式训练不固定 1:1 配额 |
| Batch/group | 128 questions × 8 independent rollouts |
| Retriever | 当前 E5/FAISS/wiki 检索配置；沿用 top-3，不改 corpus/index |
| 环境 horizon | 当前 4-step/history 协议；明确 budget exhausted 的语义 |
| Prompt/response | 沿用 4096/512 和原有模板/截断规则 |
| Rollout temperature | 1.0；不引入将同题 8 条复制成同轨迹的 seed override |
| PPO | 当前 matched GRPO 配置，1 epoch、mini-batch 256、clip 0.2；现场核验语义 |
| Optimizer | AdamW，LR=1e-6，30-step warm-up；其余沿用 matched GRPO |
| Reference KL / entropy | 沿用当前各 0.001 的基线设置；不能多算一份 KL |
| 并行 | 使用实际获授权空闲设备；优先保持当前全局 batch、group、update 语义 |
| Microbatch / offload | 允许为显存调整；不能改有效 batch 或混淆 per-GPU 与 global microbatch |
| Eval | S150、S300，greedy、每题一条；保留逐题轨迹和标准 EM/F1 |

现场配置若与表格存在实质科学差异，先明确记录并对齐当前用户已确定的 GRPO 主协议；不能同时借机更换训练配方。吞吐参数可调整并记录，不需要因为每个工程参数请求确认。

---

## 3. 记号、信息边界与样本身份

对题 x 的第 i 条原始 trajectory，第 t 个实际决策：

\[
H_{it}\rightarrow T_{it}\rightarrow a_{it},\qquad a_{it}\in\{Q_{it},A_i\}.
\]

H 是当前生成之前**实际可见、已经裁剪后的 prompt**；T 是本轮 sampled think；Q/A 是本轮 sampled action content；search 的后果是 O；最终答案的二值正确性为 R_i∈{0,1}。

R_i 专指 evaluator 的 binary EM acceptance，不包含 invalid penalty、KL penalty 或任何连续 shaping。全局 GRPO 继续用它原来应使用的环境 reward；V4 不能篡改该主链。

### 3.1 三种 UID 不得混淆

```text
question_id          数据集内稳定题目身份
rollout_group_id     当前 outer batch 的某次题目抽样实例，包含 8 个 siblings
trajectory_id        其中一条实际 rollout
turn_id              该 trajectory 内的实际 action row
```

答案 residual 按 `rollout_group_id + R_i` 分组。即使同一文本问题在同一个 batch 中被抽到两次，也不能把两个独立 group 合并为 16 条。

N 表示 outer batch 原始 trajectory 数，通常 N=1024；不包含 padding、审计复制、分支探针或为了 shape 对齐增加的 rows。

### 3.2 原始训练 tokens 不变

对 actor update，使用原始 sampled response IDs、原始 old log-prob、原始 response mask。probe 的 gold tokens、控制文档、屏蔽版本只存在于 no-grad 评分输入中，不成为 PPO training tokens。

本文所说“metadata 退出在线训练”专指数据集中的 Hotpot context、supporting facts、new fact/new doc/doc-only 等证据标注。终端 reward 原本就需要的 `ground_truth`/accepted aliases 属于训练标签，不属于该证据 metadata；它们只供原 reward、gold probe 和 answer F0.5 使用，不进入 sampled actor tokens。

Action 合法与 think 可路由是两个条件。合法的纯 search/answer 可以有 action-local credit；无 think、空 think、无法可靠分块时仅令 think-local=0。不能因此丢掉整个 action，也不能用伪造 think 补齐。

Action 合法性沿用环境对完整生成文本的语义判断，不能再增加依赖 tokenizer 标签边界的严格 gate。若 action content 与 `</search>`/`</answer>` 因子词切分落在同一个 sampled token，责任评分仍保留该 action-overlap token 作为实际动作目标；local actor update 只写入可明确归属的纯 content token。若短 action 没有纯 content token，则 action-local 系数为零，但已有的可路由 think-local 信用仍保留，该 turn 不因标签边界被整条过滤。

### 3.3 截断与剩余预算

评分只能使用 H 已可见的内容、当前 sampled T/Q 和该 search 实际返回的 O。不得恢复 H 中已经被截断的旧证据，也不得读取 O 之后的 sampled think、下一条 query 或最终答案来构造当前 utility。

O 使用后续决策真正能看到的 serialization/body，而非 retriever 的隐藏完整文档。若环境已终止、本次 O 没有机会被任何后续决策使用，则当前 search-local utility 置零，另记 `unconsumable_observation`；全局 invalid/budget 处理不变。

---

## 4. Gold-answer readiness scorer

### 4.1 固定目标集合

从训练题中原 evaluator 已使用的 accepted answer labels 构造固定 probe targets \(\mathcal Y_x\)：去掉 evaluator-normalized 为空的项，按 normalized string 去重。若有显式 primary/canonical answer，保留它；其余用稳定字典序选择，最多 3 个。无 primary 字段时使用稳定排序的前 3 个。保留选中项的原文用于 tokenizer，不把所有目标重新改成另一种表面形式。

集合只依赖该题的训练标签与 tokenizer 规则，不能依赖 sampled answer、real/control 得分、success stratum 或 optimizer step。不同反事实必须使用完全相同的 target IDs 和相同权重。

Dense answer quality（第 7 节）仍可使用全部 accepted aliases；最多 3 个的限制仅用于控制 teacher-forcing 成本。

### 4.2 分数

\[
\ell_x(C;y)=\frac{1}{|y|}\sum_{j=1}^{|y|}
\log\pi_{\theta^-}(y_j\mid \operatorname{Probe}(C),y_{<j}),
\]

\[
\boxed{\Phi_x(C)=\frac1{|\mathcal Y_x|}\sum_{y\in\mathcal Y_x}\ell_x(C;y).}
\]

只归约 answer **content** tokens；不把 `<answer>`、结束标签或强制提示词算入 |y|。score 单位是 nats per answer token，不是整题答对概率。本文不计算 `R - Phi`。

当前 outer step 任何 optimizer update 之前的 actor 为 \(\theta^-\)。所有 scorer 分数 no-grad/detached，eval 模式、temperature=1。后续 PPO minibatch 不能用已经更新过的 actor 重算同一批 targets。

### 4.3 Probe 模板

使用当前模型的原生 chat/action serialization，在需要时闭合 assistant/tool 消息，进入固定的 `<answer>` 评分位置。实现者可为合法序列做必要结构闭合，但模板必须在所有视角一致，不加入“证据已充分”“现在必须停止搜索”等有结论的自然语言提示。

IGPO 中的现成评分模板含有“enough information”式提示；这里只借用其 teacher-forcing、批处理、对齐方法，不原样搬入该语义。

模板版本、字符串、token IDs 示例、目标 offset mapping 写入 `probe_template.json`。首 token 使用正确的 predecessor logit；不能把标签所在位置错当成 content 第一个 token 的 predictor。

### 4.4 长度与缓存

评分可使用独立于 rollout 的 microbatch/token budget。默认允许的 scorer 总长度上限为 `min(model_context_limit, 8192)`；这是评分资源上限，不改变正式生成的 4096/512 设置。

尽量完整保留当前 H、T、Q 和实际可用 O，不额外裁掉 H 来迁就随机对照。超限样本标记 `score_context_overflow` 并不产生该 action 的 local credit；不能无声删除旧知识后将重见旧事实算作增益。根据真实长度降低 scorer microbatch，而不是首先丢信息。

缓存键必须包含 actor snapshot ID、prefix IDs、mask/position IDs、target IDs、probe version。不得跨 optimizer snapshot 复用 likelihood 数值。

---

## 5. Dynamic Search Utility

### 5.1 三类评分上下文

固定 H、sampled T、sampled Q，按同一 renderer 构造：

\[
C^0=(H,T,Q,\operatorname{EmptyObservationSlot}),
\]
\[
C^{\mathrm{real}}=(H,T,Q,O),
\qquad
C^{\mathrm{ctrl},k}=(H,T,Q,\widetilde O^{k}),\ k=1,2,3.
\]

EmptyObservationSlot 保留文档/工具消息的必要外壳，body 为空，不引入“没有相关结果”的语义判断。与 real 的长度差仍然存在，因此它只提供新增信息视角，不被宣称为完美 no-search intervention。

这些是固定本次 sampled action 的评分上下文，不必声称与环境生成下一 prompt 的内部历史摘要完全相同；H 之外的评分内容必须逐项有当前 T/Q/O 的原始来源。

### 5.2 随机对照选择

从完整 outer batch 的全局 observation pool 构造对照，再将请求分发给各 worker；不在每个 rank 的局部小池中独立改变选择规则。优先从其他题取 3 个不同的真实 observation。按以下层级选择，缺候选时逐步放宽，不丢弃整个数据源：

```text
其他题 + 相同数据源 + 相同 call bucket(1 / 2+) + 相近长度
→ 其他题 + 相同数据源 + 相近长度
→ 其他题 + 相近长度
→ 其他题的有效 observation
```

保留相同文档数量与 serialization；长度优先落在 real 的约 0.75–1.25 倍范围。过长 control 可只截其文档 body 到相同资源上限，不能改变 H/T/Q。排除同题、完全相同 observation，以及能确定与当前检索结果完全相同的文档集合。

**不依据当前 gold answer 是否出现来系统性挑选更差的 control。** 对短答案、yes/no、常见年份进行 alias 排除容易产生条件选择偏差。control 偶然含答案是允许的，它会使估值更保守；记录即可。不调用额外 retriever 专门找负例。

使用独立的、可恢复的 control RNG，不推进 dataloader、rollout 或模型训练 RNG。全局池中只有 1–2 个合格 control 时可有放回补齐 3 个，记录 unique-control count；这些退化对照不用于第 6 节背景变动带校准。若整个 batch 确实找不到任何不同题 control，当前 action utility=0 并记录，不从隐藏 gold evidence 构造负例。

### 5.3 两个 contrast 和 agreement

\[
g^{\mathrm{inc}}=\Phi(C^{\mathrm{real}})-\Phi(C^0),
\qquad
g^{\mathrm{cf}}=\Phi(C^{\mathrm{real}})-\frac13\sum_{k=1}^{3}\Phi(C^{\mathrm{ctrl},k}).
\]

\[
\boxed{
g^{\mathrm{dir}}=
\begin{cases}
\min(g^{\mathrm{inc}},g^{\mathrm{cf}}),&g^{\mathrm{inc}}>0,\ g^{\mathrm{cf}}>0,\\
-\min(|g^{\mathrm{inc}}|,|g^{\mathrm{cf}}|),&g^{\mathrm{inc}}<0,\ g^{\mathrm{cf}}<0,\\
0,&\text{otherwise}.
\end{cases}}
\]

另存 `contrast_sign_conflict`、两个原始 contrast。不能只保存 g_dir，因为“冲突归零”和“二者都微小”是不同情况。

### 5.4 连续死区与有界映射

\[
z=\frac{(|g^{\mathrm{dir}}|-\delta_U)_+}{s_U},
\]
\[
\boxed{
U_{it}^{Q,\mathrm{dyn}}=
\begin{cases}
\tanh(z),&g^{\mathrm{dir}}>\delta_U,\\
-0.1\tanh(z),&g^{\mathrm{dir}}<-\delta_U,\\
0,&\text{otherwise}.
\end{cases}}
\]

因此 \(U^{Q,\mathrm{dyn}}\in[-0.1,1]\)。不同正增益仍有不同幅度，不重新退化成固定 hit reward。

采用 tanh 而非不封顶对数 clipping，是本次为减少尺度自由度作出的选择。负侧 0.1 是探索保护系数，不是“坏 query 只有十分之一危害”的语义判断。它可能引入乐观偏置；本文不主张潜势保持或无偏策略目标。

### 5.5 重复和未知事件

若 O 的全部返回 passage 已经逐段存在于 H 当前可见 documents 中，并且没有新可见内容，则禁止正的重复奖励：

\[
U^Q=\min(U^{Q,\mathrm{dyn}},0).
\]

其余可评分 search 取 \(U^Q=U^{Q,\mathrm{dyn}}\)。只有 ever-seen、但 H 已看不到旧材料，不构成此精确重复判据；重新检索可能有用。不同段落来自同一 title 也不自动算重复。

不确定是否重复时不要强判重复。no-meta、no-support-hit 不等于负 reward；retriever/parse/score 错误不是负效用样本，local=0，保留原全局环境处理。

### 5.6 Metadata 的严格边界

数据集 `metadata` 及由它派生的 new fact、new doc、doc-only、support hit 等字段完全退出 V4 在线训练数据流：不复制进逐 turn rollout row，不进入 scorer request，不参与 utility、eligibility、responsibility、routing、trajectory cap 或 actor update，也不作为在线必须计算的指标。`visible-exact-repeat` 直接比较当前可见 H 与实际 O，不需要 gold supporting metadata。

如需与 V1/V2 做解释性对比，只在诊断或 S1–5/S30/S150/S300 等 milestone 保存的原始轨迹上，按稳定 UID 与原训练数据离线 join，生成独立 sidecar 日志。离线结果不得回填当前 batch 或改变训练。V4 在线链路即使完全没有该 metadata，也必须得到相同的 credit tensors 和 actor update。

不实施 Bridge Rescue，也不为它运行在线验证。将来若另行授权开启，至少必须同时验证两个 raw contrast 都处于弱区、无强符号冲突、当前未掌握且确实出现桥接相关信息。仅有 `g_dir=0`、低 likelihood 和离线 meta-hit 三项不够。

这一选择接受即时 gold-answer proxy 可能低估桥接步骤的边界，而不是用一个尚未验证的补偿把原 V2 固定奖励带回来。

---

## 6. Utility 数值校准——只确定尺度，不选架构

在第 12 节的小诊断集上一次性计算并冻结：

\[
e_b^{\mathrm{ctrl}}=|\Phi(C_b^{\mathrm{ctrl},1})-\Phi(C_b^{\mathrm{ctrl},2})|.
\]

令 e_num,U 为同一 frozen 请求在正式 backend、精度与固定 production batch contract 下重复评分的 absolute score difference 的 P95。同一 utility bundle 的 real/null/control/aliases 必须作为不可拆分 comparison group 进入同一个 padded forward；逐条或不同 batch shape 的 BF16 漂移另行报告，不混入这个实现误差项。

\[
\delta_U=\max\left(3e_{\mathrm{num},U},\ \operatorname{clip}(P75(e^{\mathrm{ctrl}}),0.05,0.5)\right).
\]

\[
s_U=\operatorname{clip}\left(P75\{\,|g_b^{\mathrm{dir}}|-\delta_U:|g_b^{\mathrm{dir}}|>\delta_U\,\},0.25,2.0\right).
\]

若没有足够非零样本用于分位数（少于 8 个），s_U=1.0，并报告校准覆盖有限；不通过放大到任意值“制造可见奖励”。

Q75(control-control) 是经验背景变动带，不是严格置信区间，也不能把所有差异都称作纯数值噪声。上下限是为了避免小样本把尺度拖到极端，不是经过测试集选择的参数。

两类训练题共享同一 delta_U 与 s_U。训练中不按 batch 自适应归一化，不因 active rate 低自动降低阈值；S30/S150 只监测漂移，不静默改科学配置。

---

## 7. Answer Action Utility：同题、同终端奖励类别的质量残差

### 7.1 合法终端答案与 F0.5

只对 canonical、完整、可解析的实际终端 answer content 计算局部信用。复用 evaluator 的 normalization 和 accepted aliases；不从整个 think/history 中摘一个看起来更接近答案的字符串。

对 normalized word multiset，令重合计数为 o，生成词数为 n_a，gold 词数为 n_y：

\[
P=o/n_a,\quad R_c=o/n_y,
\quad
F_{0.5}=\frac{1.25 P R_c}{0.25P+R_c}.
\]

无重合、任一有效词集为空或分母为零时 F=0。重复词用 multiset min-count，不能把重复写出 gold token 计成无限重合。这里 R_c 为词召回率，与 terminal R_i 区分。

\[
q_i=
\begin{cases}
1,&R_i=1,\\
\max_{y\in\mathrm{AllAliases}_x}F_{0.5}(A_i,y),&R_i=0.
\end{cases}
\]

只使用词级 normalization，不用 tokenizer subword overlap，也不引入语义 embedding/LLM judge。F0.5 是 partial-answer proxy；它可能无法区分词序、否定或共享词的不同实体，不宣称语义正确。

### 7.2 分组与残差

\[
G_{x,r}=\{j:\mathrm{rollout\_group\_id}_j=x,\ R_j=r,\ \mathrm{valid\_terminal\_answer}_j=1\}.
\]

\[
\boxed{
U_i^A=
\begin{cases}
q_i-|G_{x,R_i}|^{-1}\sum_{j\in G_{x,R_i}}q_j,&|G_{x,R_i}|\ge2,\\
0,&\text{otherwise}.
\end{cases}}
\]

不再除 group std，不施加正负不对称，不加每题难度权重，不乘 |A_out|。同 stratum 的原始 U_A 和为零；这不代表经过 responsibility、PPO ratio、长度归约后的参数梯度与 GRPO 正交。

整个 outer group 的 U_A 必须在 shuffle、DP partition 和 PPO minibatch 切分之前计算。结果以原始 trajectory ID 回填。

### 7.3 重要情形

| 情形 | Answer-local 结果 |
|---|---|
| 全部成功 | q 都为 1，U_A 全零 |
| mixed 中的成功 stratum | U_A 全零；成功与失败之差交由 global GRPO |
| 多条失败且 partial quality 不同 | 较高者正残差、较低者负残差 |
| 全部失败且 q 都为 0 | 全零，不虚构新信号 |
| 失败 stratum 仅 1 条合法答案 | 零 |
| invalid/no answer | 本分支不更新；不拿无答案样本压低其他合法答案的 stratum mean |

同题错误答案也处于不同 terminal histories，本残差是题内答案质量比较，不是相同状态下的精确 action advantage。此点是主动接受的近似。

### 7.4 为什么不使用旧 AGAM 或 readiness baseline

旧 AGAM 完全关闭。本文不额外使用 `R_i - V_gold(Z_i)`，也不把 gold-readiness chunk difference 与 sampled-answer dependency 混合。它们是不同估计器，不在这一条正式线里并行开启。

本版正的 answer residual 可以强化一条 EM 错误但相对更接近标签的 sampled answer，连同促进它的 think；使用 alpha_A=0.25 限制该弱代理的预算。这并不保证没有错误实体自强化风险，最终是否有益由正式结果决定。

---

## 8. Action-specific Responsibility 与 Soft Sparse Routing

### 8.1 固定 actual sampled action

对 a=Q 或 A：

\[
\ell^a_{\mathrm{full}}=\frac1{|a|}\sum_{p\in a}\log\pi_{\theta^-}(a_p\mid H,T,a_{<p}),
\]
\[
d^a=\ell^a_{\mathrm{full}}-\ell^a_{\mathrm{whole-mask}},
\qquad
\boxed{\rho^a=1-\exp\left(-\frac{[d^a-\delta_R^a]_+}{s_R^a}\right).}
\]

评分 target 是实际 sampled action content，full/mask 的 target IDs、位置和 target mask 相同。Q/A 各自校准 s_R，不能把旧 search 的 0.6201 不加区分套到短 answer 上。

只有 |U_a|>0、有合法 preceding think 的 anchor 需要 online responsibility。在线 whole-think rho=0 时可直接跳过 chunk 评分，所有 routed credit=0；校准时允许少量 shadow chunk 评分。shadow 评分不成为额外训练数据，所有 rank 仍须遵守现有 FSDP collective 调用对齐。

### 8.2 Mask 的方法契约

首选位置保持的 attention intervention，复用现有 scorer 基础设施。被屏蔽的 think content tokens 的 IDs 与 position IDs 保留，禁止它们作为 key/value 被所有后续未屏蔽位置读取；包括后续 think、结构标签、action prefix 和 action token 的 predictor rows。每次 mask 使用对应 attention 图重新前向。

这比仅屏蔽标成“query content”的行更明确：后者可能漏掉第一个 action token 的 predecessor、闭合标签或经后续表示的传递路径。必须用真实 causal shift 确定 predictor；不能让 mask 只作用于 target token 行而漏掉预测它的上一行。

一个规范参考：以正常 causal/padding mask 为底图，对 key∈S（被屏蔽 chunk）且 query 是其后未屏蔽 token 的边设为不可读；其他边不变。whole-think 与 chunk-mask 使用同一操作。单 chunk 时两者结果应一致。

后续保留下来的文本可能已经字面复述了被屏蔽信息，这仍是固定 sampled text 下的功能干预，不是重新采样的因果删除。不能声称阻断了所有历史语义影响。

评分后端若不能正确执行该 mask，允许采用**等价的隔离 attention 实现**或 scoring-only eager/SDPA 路径；不能因 FlashAttention 不接受 arbitrary mask 就无声忽略 mask。也不能把 text deletion 当作数值等价替换。真正无法支持时报告实现阻塞，不自动变成另一种方法。

### 8.3 Chunk 规则

尽量复用已验证 tokenizer/span parser：按标点/换行分块，目标最小 8 tokens、长块目标 64 tokens、最多 6 chunks。若 split 后超过 6，确定性合并相邻短块；max 6 和完整覆盖优先于 64 的目标长度。

不丢弃尾部 think，不包含 think tags，不重分词覆盖原 sampled IDs。单 chunk 可以合法路由；完全没有可信 chunk mass 才是零路由。

\[
d_k^a=\ell^a_{\mathrm{full}}-\ell^a_{\mathrm{chunk-mask},k},
\qquad
v_k^a=[d_k^a-\delta_{\mathrm{chunk}}^a]_+^2.
\]

\[
\boxed{
w_k^a=
\begin{cases}
v_k^a/\sum_j v_j^a,&\sum_jv_j^a>0,\\
0,&\text{otherwise}.
\end{cases}}
\]

归一化用 FP32。仅在 mass>0 时做除法，避免 `sum+epsilon` 使声称的和为 1 与实现不一致。无 mass 时 w 全零，不 fallback 到 whole think。

### 8.4 责任数值校准

对 Q/A 各自取同一输入、同一 mask 在正式 backend、精度与固定 production batch contract 下重复评分的差异 P95。若正式实现复用 rollout old-policy log-prob 作为 full，而 masked view 使用 scoring-only backend，则另测 old full 与同 comparison group 的 no-mask scorer 差异 P95；两者取较大值作为 e_num,R^a。一个 action 的 no-mask/whole/chunk views 必须作为不可拆分 comparison group；任意改变 microbatch 的诊断漂移不直接抬高责任死区。

\[
\delta_R^a=\delta_{\mathrm{chunk}}^a=\max(0.01,3e_{\mathrm{num},R}^a),
\]
\[
s_R^a=\operatorname{clip}\left(\operatorname{median}\{d_b^a-\delta_R^a:d_b^a>\delta_R^a\},0.1,2.0\right).
\]

少于 8 个正 whole dependency 时，Q 使用 0.6201、A 使用 0.2 作为**明确的尺度 fallback**，标记 limited calibration；不是把 rho 固定成这些数。零/负 d 仍映射为零。

只做一次尺度确定。不得为了让 rho 更大而挑不同 intervention；不得按 reward 符号分别调整 rho。初始化、强制 no-mask 应有 d≈0。

---

## 9. 局部信用、trajectory cap 与实际 loss

### 9.1 未乘总体 loss 系数的信用

\[
C_{it}^Q=U_{it}^Q,\qquad
C_{itk}^{TQ}=\rho_{it}^Q w_{itk}^Q U_{it}^Q,
\]
\[
C_i^A=0.25U_i^A,\qquad
C_{ik}^{TA}=0.25\rho_i^A w_{ik}^A U_i^A.
\]

这是 additive local residual，不扣掉 query 自己的信用。没有 think 时 action 保留 C，routed C 为零。rho 不是 base advantage，也不把 `A_out + C_action` 整体复制给 think。

### 9.2 Search trajectory cap

\[
B_i^{\mathrm{raw}}=\sum_{t\in\mathcal Q_i}\left(|C_{it}^Q|+\sum_k|C_{itk}^{TQ}|\right),
\]
\[
\boxed{s_i=\begin{cases}\min(1,2/B_i^{\mathrm{raw}}),&B_i^{\mathrm{raw}}>0,\\1,&B_i^{\mathrm{raw}}=0.\end{cases}}
\]

同一 s_i 同时乘该 trajectory 的 query 和 TQ 项。先计算完整原始 trajectory 的全部 search credits 再定 cap，不能在不同 minibatch 中各用一份新预算。

每条 trajectory 的 search absolute coefficient mass≤2；terminal mass≤0.5，因此四个 local channels 的总 absolute coefficient mass≤2.5（尚未乘 lambda_local）。这些是 coefficient budget，不是参数梯度上界。

不除以 calls；增加一个零 utility action 不会稀释旧信用，但高价值搜索累计到 cap 后会共同缩放。不能因此宣称已显式学会最优停止或搜索越少越好。

### 9.3 Signed span-local PPO

对 original sampled content span S，ratio 与全局 PPO 共用：

\[
r_p=\exp(\log\pi_\theta(y_p\mid prefix_p)-\log\pi_{\mathrm{old}}(y_p\mid prefix_p)).
\]

\[
\boxed{
\mathcal P(S,C)=-\frac1{|S|}\sum_{p\in S}
\min\{r_p C,\ \operatorname{clip}(r_p,1-\epsilon,1+\epsilon)C\}.
}
\]

C 可正可负。不能复用只对 C≥0 正确的 `min(ratio,clipped_ratio)*C` 简写。不能先把 C 除 |S|，随后又在 P 中除第二次长度。

\[
L_Q=\frac1N\sum_i s_i\sum_t\mathcal P(Q_{it},C_{it}^Q),
\]
\[
L_{TQ}=\frac1N\sum_i s_i\sum_{t,k}\mathcal P(T_{itk},C_{itk}^{TQ}),
\]
\[
L_A=\frac1N\sum_i\mathcal P(A_i,C_i^A),\qquad
L_{TA}=\frac1N\sum_{i,k}\mathcal P(T^A_{ik},C_{ik}^{TA}).
\]

\[
\boxed{
L_{\mathrm{actor}}=L_{\mathrm{GRPO,unchanged}}+\beta L_{\mathrm{KL}}-\eta_H H
+\lambda_s(L_Q+L_{TQ}+L_A+L_{TA}),
\qquad
\lambda_s=\lambda_{\max}\min(s/30,1).
}
\]

s 是 outer/global step；局部系数在 S30 达 full strength 后保持，不沿用 AGAM 退火到零。KL/entropy 若原函数已计入就不重复添加。

独立 clipped losses 相加与把 advantages 相加后只 clip 一次并不等价；本文件明确选择前者。不能为了方便把 local credits 合回原全 response token-mean advantage。

### 9.4 多轮 row、DP 和 accumulation 语义

N 始终是 outer batch 原始 trajectories，而不是实际 turn rows、active actions 或当前 microbatch 的 unique trajectory 数。

若 actor update 把每轮 response 展开成 M 个实际 sampling rows，且某个 PPO minibatch 含 m 个非 padding rows，令 n_r 表示 row r 的所有已定 credit 对应的 local numerator，则使用原始目标的 minibatch estimator：

\[
L_{\mathrm{local},b}=\frac{M}{Nm}\sum_{r\in b}n_r.
\]

M 是真正进入该 outer update sampler 的非 padding row 总数；被保留但 local=0 的 row 仍在 M 中。审计/adjustment copies 的 local numerator 必须为零，不可生成额外信用。上述 estimator 假定每个原始 sampling row 每个 PPO epoch 恰好出现一次；若现场有有放回重采样、重复均衡或不等抽样权重，须按真实 inclusion weights 还原同一原始 trajectory 目标，不能直接沿用 M/m。

对 frozen theta，必须满足：

\[
\sum_b\frac{m_b}{M}L_{\mathrm{local},b}=\frac1N\sum_r n_r.
\]

这说明 minibatch normalization 的语义，不声称多次 optimizer updates 与一次大 batch 更新完全相同。若现场已按 trajectory 为单位打包，直接使用相应 trajectory-mean estimator，无需生搬 M/m；必须与上面的目标在 frozen reference 上一致。

跨 DP ranks 的 m 采用全局计数。若框架平均 rank gradients，rank local numerator 应乘 world_size 抵消；microbatch accumulation 只能使用一次对应 optimizer-update denominator。禁止在已完成该归约后再次错误地除 accumulation steps。

不修改 global GRPO 的现有 sampler、归约或 padding 语义来迎合 local branch。

---

## 10. 与 veRL / IGPO 的实施边界

### 10.1 建议的数据流，不锁死函数路径

```text
原始 rollout / retriever results / terminal rewards
  → stable UID、actual visible prompt IDs、原始 action/chunk spans
  → CPU answer q 和同题同 reward residual
  → no-grad scorer request queue：real/null/control + 必要 responsibility masks
  → 冻结 U、rho、w、trajectory cap、local coefficients
  → 回填原始 training rows，随 shuffle/packing/DP 路由一起移动
  → 原 actor forward 得到 new log-prob
  → 原 GRPO/KL/entropy + 4 个 span-local PPO channels
  → 同一个 backward / optimizer
```

Scorer 可以分两批 RPC：先 utility，后只对非零 utility anchors 做 responsibility。这不会新增 rollout 或需要第二个 teacher。所有 scoring 必须发生在该 batch 第一次 actor update 之前。

本节的数据依赖、评分视角、mask 语义、归约目标和关闭项是方法契约；具体类名、函数路径、RPC 拆分、张量字段名和批处理组织不是方法的一部分。实施时同步修改真实代码，使其输出与这些契约一致，不为迁就旧 V1/V2/V3 接口保留另一套并行语义。

### 10.2 实施者拥有的工程选择

可自行选择 scorer microbatch/token budget、length bucketing、CPU 请求构造、合并 RPC、等价 mask 内核、日志字段和局部代码组织；根据现场接口决定复用当前 FSDP worker、增加窄 RPC，或重组 DataProto 字段。用户给的 compact review repo 和 IGPO 的 veRL 实现均只作参考，不要求照搬其文件布局。

不得为了复用 IGPO 而替换当前 trainer、reward、web-search service、group size、horizon、prompt 或整套 veRL/PyTorch/vLLM 版本。尤其不复制其 launcher 中关闭 Ray 内存监测等共享机器设置。

### 10.3 评分批量化的两层实现

**默认方案：batch-of-views。** 将独立的 `(prefix, gold target)` 或 `(masked prefix, sampled action target)` 请求合并到少量阶段化 RPC，内部按 token budget 分 microbatch，计算短目标的 teacher-forced log-prob。请求数可以很多，但不能由 controller 为每条样本或每个 view 单独发一次 GPU RPC/forward。此方案优先完成，不以高级 vectorization 未接通阻塞训练。

**可选等价加速：IGPO-style extended sequence。** IGPO 仓库中已有基于 veRL 的 teacher-forcing、target 对齐、position IDs、4D attention 与 sequential fallback 代码，可复用其计算组织。仅当现场 mask/backend 支持且小样本结果等价时使用；当前不同控制文档和 think 干预不是同一原始前缀的简单截断，不能把多个分支的 hidden states 无条件共享，也不能搬入 IGPO 自身 reward/advantage 配方。

若采用 GT copies 拼接：各副本只能读对应 prefix 和自己的前序 GT tokens，不能读其他副本、未来 observation 或 sampled answer；RoPE position IDs 要从各自 prefix 末尾续接；只评分 answer content，首目标 predictor 的边界必须单独检查。

IGPO 的公开脚本中 `use_vectorized_gt_logprob` 并非默认开启；不能把“代码里提供实现”写成“当前框架必然已用同一加速路径”。

### 10.4 额外成本的实际组成

Gold target 短，不代表长 prefix 前向免费。额外开销来自 utility views 和 masked responsibility forwards；local PPO 本身复用原有 new log-prob，不增加第二次带梯度 actor forward。

对 search anchor b，utility 的原始 teacher-forcing item 数为：

\[
N_U=\sum_b |\mathcal Y_{x_b}|(2+K),\qquad K=3,
\]

其中 2 对应 real/null。responsibility 只对 local utility 非零的 anchor 评分，masked item 数由 whole-mask 与实际 chunk 数决定。actual sampled action 的 full score 应优先从同一 \(\theta^-\) 下已有 `old_log_probs` 精确抽取；只有等价性检查不通过时才允许额外 full forward，并必须记录原因。

允许这种 teacher-forcing 定义本身使每 step 变慢，不设置脱离现场测量的固定速度门。禁止由代码组织制造额外下降：

- GPU forward 必须按 token budget/bucket 批量化；RPC 数随阶段与 microbatch 组数增长，而不是随单条 anchor/view 线性增长；
- controller 不逐样本等待 GPU，不在 Python 循环中发单条远程调用；CPU renderer、alias 展开和 control 选择先形成请求队列；
- real/null/control 与 masked views 不重复构造相同 tokenization；不重复计算可从 old-policy tensors 精确取得的 full-action log-prob；
- FSDP ranks 保持相同 collective 顺序，padding views 只用于形状/collective 对齐且永不产生信用；local PPO 继续复用原 new-log-prob forward。

所有 CPU/GPU 测试和长任务使用独立、短路径的任务专用 `TMPDIR`/`TMP`/`TEMP`，launcher 在正常退出与异常退出时统一清理。不得让 Transformers/checkpoint 原子写入产生的随机 `tmpXXXX` 长期遗留在代码根目录；清理前仍需确认目录属于本任务且没有活进程引用。

在纯 GRPO S150 诊断阶段先完成两级成本测量：小样本上比较逐条 reference 与批量实现的分数/梯度等价性；随后对一个完整 128×8 outer batch 记录实际 search/answer anchors、alias 数、utility/responsibility items、forward microbatches、RPC 数、prefix/target tokens、controller build/wait 时间、scorer wall time、rollout/update wall time、CPU/GPU peak memory 与 GPU 空转区间。若主要耗时来自逐样本调度、重复 tokenization/forward 或 rank 不均衡，必须先修实现；若结构已批量化而成本来自方法规定的长 prefix 和多视角，则如实接受并据此估算 5-step/300-step 时间。

只报告现场测得的开销，不能把 IG-Search 的约 6.4% 套到本方法，也不能为追求速度静默抽样正式训练 anchors、减少 K/alias 或改变 mask/utility 语义。

---

## 11. 必要的精确性检查

以下检查是为了防止实现错误，不是性能 gate。尽量复用原项目已有测试，不扩展成长篇审计工程。

| 类别 | 最小必测内容 |
|---|---|
| Utility | 两对照同正/同负/异号/边界；符号冲突不变成正奖；raw contrast 保留 |
| Repeat/budget | 当前可见完全重复禁止正奖；被截断旧证据不误判；不可消费 observation 为零 |
| Alias/probe | real/null/control targets 相同；目标长度和 causal shift 正确；无 gold 进入训练 tokens |
| Answer | 同题同 R 分组；全成功零；全失败可分有正负；singleton/同分零；跨 rank 一致 |
| Responsibility | no-mask identity；whole 与单 chunk 一致；零/负依赖无促进型路由；无 whole fallback |
| Mask | predecessor row、后续标签/动作 prefix、padding、跨样本隔离、future mutation |
| Routing | query 不扣除 rho；w 正 mass 时和为 1；无 mass 时和为 0；预算≤2.5 |
| Signed PPO | C 正/负、ratio 在 clip 上下界内外；empty spans 不产生 NaN |
| Local masks | query/answer/think content 外 local dL/dlogp=0；格式标签仅收 global |
| Reduction | CPU frozen global-trajectory reference；不等 microbatch、2-rank、不等 active counts、一 rank 全零 |
| Metadata boundary | 移除在线 metadata 前后 V4 credit/update 一致；scorer 和 actor batch 中无 supporting-fact/context 字段 |
| Cost structure | 同 comparison group 的批量评分稳定；不同 batch shape 漂移单列；无逐样本 RPC、无不必要的 full-action forward、controller 串行等待或 rank forward 失衡 |
| Regression | V4 off 时 global reward/advantage/loss 与 baseline 一致；旧分支没有残留 |
| Resume | 新配置、scales、RNG、UID、step 计数可恢复；不加载 smoke/旧方法 state |

CPU FP64 reference 的差异应在通常浮点精度范围内；GPU 使用相同精度/backend 的正常数值容差，不要求跨 backend bit-identical。只对 leak、错误符号、错位 mask、重复计数、错误归约设硬失败。

---

## 12. 最小校准数据：先利用 S150，不进行新验证项目

### 12.1 数据预算

首批从**训练集**随机选 64 HotpotQA + 64 NQ，共 128 个问题，每题 8 条 rollout，temperature=1，使用 `/data1/kongmu/model_checkpoints/base_grpo_3b_s150_hf` 和原检索环境。独立诊断 seed=20260908，保存 manifest；不按成功率、离线 metadata 命中或期望结论挑题。

若已有同 checkpoint、同协议、包含原始 tokens/observations 的随机 retained batch，可直接复用；缺少字段才重新采样。只含聚合 metrics 的日志不能替代原始样本。

最多再补一批同规模随机 128 题，仅用于技术覆盖不足、某数据源无足够合法 search 或身份/字段缺失；不为了追求显著性、更多正例或某个效果方向反复采样。总诊断上限为 256 个问题、2048 trajectories。

正式训练仍使用原合并数据 shuffle，不沿用诊断 1:1 配额。诊断来自训练集，不读取七集 held-out answers 来确定任何参数。

### 12.2 计算子集

CPU answer quality/residual 对整个 batch 计算。昂贵的 utility 初始至多评分 256 个 search anchors，尽量 128/source，并覆盖 first/later calls；按固定随机顺序选，不能先看 U 再选。

Responsibility 校准至多 64 个 query anchors + 64 个 answer anchors，优先包含局部信用非零的实际样本；不足可加入零 utility 的合法 anchors 校准 d，但这些 shadow anchors 不因被评分就产生信用。最多每条 think 6 chunks。

另取至多 32 条**完整 trajectory**用于 loss/gradient 归约检查，补齐这些 trajectory 的所有 search 和 terminal scorer，避免只评分一个 selected turn 却把它当成完整 trajectory cap。答案 stratum 仍从原来的完整 8-rollout group 计算。

不做新的 branch rollout、teacher rollout、任务 completion 重采样或大规模 Shapley/LOO 对照。

---

## 13. 三项最小数据判断与准入规则

这一步不选择新的方法族，仅做以下三个判断。质量指标不要求统计显著、不要求优于 held-out baseline，也不要求 source 与 meta-hit AUROC 超过某值。

### A. Utility scorer 是否按定义工作

检查两个数据源的合法 anchors 能否构造 real/null/control、有效分数比例、score 分布、sign conflict、dead zone、少量直观样本。scorer 对结构合格请求的正常成功率目标≥95%；剩余样本有明确原因，不静默丢失。

至少人工/代码辅助查看 12 个固定选例：强正、强负、冲突/死区各若干，分两个来源。只检查评分输入与分数是否因 gold 泄露、控制文档错位、模板错误而颠倒，不要求 Codex 给全部动作语义打分。

用至多 32 个原 anchor 换一组 control，报告非近零样本符号一致性。若超过一半的**大幅、远离死区**样本翻号且 n≥16，先排查 control/serialization；只允许一次定向工程修复。稀疏或近零样本的变号不构成失败，不启动 K/threshold 网格搜索。

数值结构正确、有非退化 direct utility 即可 `PASS` 或 `PASS_WITH_LIMITATION`。只有全体合法样本在补充后仍无可辨认信号、或存在系统性构造错误，才停止并报告 `UTILITY_NOT_USABLE`；不退回 fixed meta reward 冒充 V4。

### B. Responsibility 是否正确干预并能产生有限路由

在 Q/A 各至少 8–16 个可用 anchors 上测得 e_num,R；确认 mask 有实际作用、token IDs 与目标不变、rho/w 有界、无 mass 时不广播。相同请求、相同 padding/batch shape 和相同精度下的重复评分必须稳定；修改同批其他样本不得改变目标样本，修改被 barrier 屏蔽的历史 token 不得改变目标分数。

单条与批量结果仍需报告，但 BF16 下改变 batch shape 可能产生确定性数值漂移，因此它是量化诊断，不作为强制缩小 microbatch 的硬门。校准与正式训练必须固定相同 backend、精度、长度分桶、token budget 和每卡 microbatch；不能为了贴合逐样本结果而把正式 scorer 强制降为 microbatch=1。只有同 shape 重复不稳定、跨样本泄漏、mask 被忽略或目标错位时才判定精确性失败；若实际显存或速度测试要求改 batch，需记录新配置并重新完成上述检查。

不要求 rho 与最终 success 显著相关，不重跑旧的责任因果验证。Q/A 个别正依赖少时按第 8.4 节有限校准继续；若 mask 被忽略、跨样本泄露或路由始终错误，硬失败。

### C. Answer residual 的覆盖和算术正确性

核验 stratum、F0.5、zero-sum 原始残差；报告有非零残差的题组数、trajectory 比例、Q/A source 分层和正负范围。

不要求固定激活比例。单跳短答案、yes/no 或全相同失败答案自然可能没有 dense residual。若初批覆盖少，不专门追着 partial answers 重采样，也不把 `A_out` 或 readiness 奖励自动塞进来。

完全没有自然 answer residual 时标记 `ANSWER_COVERAGE_LIMITED`，保留公式和实际零覆盖证据；只要算术/合法 answer/mask 管线正确，可继续主训练，不能声称此时已经验证了 answer routing 的收益。若原因是 parser 丢掉合法 answer，则属于工程错误，必须修复。

### Gate 解释

`PASS_WITH_LIMITATION` 允许进入下一阶段，明确带上覆盖/近似限制；不得为了追求 `PASS` 反复采样、修改任务或抬高局部奖励。

精确性错误不是“门太严格”；泄露、符号错、错用 checkpoint 或 wrong denominator 必须修。统计不显著、短期 EM 不提升、某类 tied group 没信用，不是停止理由。

---

## 14. Full-strength 梯度尺度：只校准一个 lambda_max

不能再用 warm-up 的 1/30 强度通过校准，然后让正式 full strength 放大 30 倍。也不能把 scalar local loss/global loss 比当作参数梯度比。

在两份有非零 global PG 的 frozen 子批上，使用同一组原始轨迹、同一 actor snapshot、同一 new/old log-prob 和第 9 节实际归约，分别得到：

\[
g_G=\nabla_\theta L_{\mathrm{GRPO}},\qquad
 g_L=\nabla_\theta(L_Q+L_{TQ}+L_A+L_{TA}).
\]

只需总 local 与 global 的参数梯度范数/余弦；四角色可记录 dL/dlogp，不要求四次完整参数反传。若 FSDP 流程需要分两次 backward 并逐层累积 norm，可采用等价做法，不保留多份全梯度常驻显存。

令 h_b=||g_G,b||/max(||g_L,b||,epsilon)。排除因完全 tied 而 global gradient 为零的子批，不把零分母当成 local 过强。若所有子批都这样，使用原 batch 中实际 mixed groups 的一个子集补一次，不新增大采样。

\[
\lambda_0=\operatorname{clip}\left(0.1\operatorname{median}_b h_b,10^{-4},0.2\right),
\qquad
\boxed{\lambda_{\max}=\min(\lambda_0,\ 0.3\min_b h_b).}
\]

这以约 10% 参数梯度比作为工程起点，并限制校准批次的最强比例约≤30%。不是最优科学超参数，不做 0.01/0.02/0.05 的训练效果搜索。

若 local gradient 为零，先核验 source、mask 和归约；不通过无限增加 lambda 修复零信号。若结果异常小，报告原因，不设最低有效强度门强行放大。

在最终冻结前，用原始 Base 的一个 disposable shadow batch 再计算一次**full lambda、非 warm-up 缩小后的**比例，同时记录 S150 冻结的 utility dead-zone、饱和率和 Q/A responsibility 覆盖在 Base 上是否发生结构性塌缩。若梯度比例>0.3，允许一次仅降低 lambda 的安全修正；不因该 Base 批的比例偏小而上调已由 S150 确定的系数。shadow 不做 optimizer update，完成后重新从原始 Base 初始化正式 smoke。

只要求更新有限、真实存在且量级合理；不要求参数梯度与 GRPO 同向，也不要求一次更新立刻提高 EM。至少有一次实际非零 LR optimizer update；S1 LR=0 的反传不能冒充真实更新结果。

---

## 15. Freeze：明确选择结果，而不是留下待办参数

完成 S150 数值校准、完整 outer-batch 成本测量和 Base shadow 安全检查后，生成最终 `resolved_method.json`，至少包含：

```text
method_revision / code commit / working-tree diff hash
source dataset and retriever identity
original Base identity / diagnostic GRPO S150 identity
probe template version and hash / alias selection rule
utility: K=3, agreement, delta_U, s_U, negative_scale=0.1
metadata: online_pipeline=false, credit_enabled=false, diagnostics=offline_sidecar_only
responsibility: exact intervention name, Q/A delta_R, Q/A s_R
chunks: p=2, max_chunks=6, no-mass=zero
credit: alpha_Q=1, alpha_A=0.25, lambda_TQ=lambda_TA=1
budget: B_Q=2, original-trajectory denominator
local_loss: signed span PPO, lambda_max, warmup=30
scoring backend / dtype / microbatch or token budget
scoring cost report / structural efficiency checks
all relevant disable flags
calibration manifest / sample counts / PASS_WITH_LIMITATION notes
formal initialization and training protocol
```

数值全部写成实际值，不用 `auto`、`TODO` 或旧 V2 同名变量占位。保存 frozen reference 输出以便接入检查。文件冻结以后，不在正式训练中自动更新 delta、s_R 或 lambda。

仅评分后端的等价优化、batch size 等吞吐调整可以在不改变分数语义时进行，并记录新 hash；改变 utility、answer residual、mask 语义、local coefficient 或数据分布，属于新科学版本，不能无声拼接曲线。

---

## 16. 5-step disposable smoke 与真正启动

### 16.1 Smoke

从原始 Base、独立目录与独立 run ID 开始，使用正式 global batch/group、300-step scheduler 和经过 Base shadow 后最终冻结的方法运行 5 个 outer steps；仅 loop stop at 5，不能把 scheduler 的 total_steps 改成 5。

至少检查：

- 全局 reward/advantage 原样；所有 V1/V2/AGAM/V3 分支为零。
- 数据集 evidence metadata 不存在于在线 rollout/scorer/routing/actor-update batch；离线 sidecar 不回填训练。
- 两来源 query 均可进入评分；合法无 think 的 action 仍保留 action 信用。
- local 只影响定义的 content；所有 score/credit detached。
- 有至少一个 LR>0 的有限参数更新。
- 无 mask 被 backend 忽略、FSDP collective mismatch、batch reorder 错配或持续非有限值。
- 在 smoke S2 保存并恢复一次（或复用同等已验证 resume 路径），继续到 S5，检查 scales、RNG、step 和方法身份。

Smoke 不以 success 上升为目标，不因随机 5-step EM 波动反复调参。若仅显存不足，先降低 scorer microbatch、actor microbatch或使用已有 offload；保持 global batch 与 PPO 语义。

### 16.2 唯一正式 run

Smoke 结束后丢弃其训练状态，重新载入原始 Base，重置 optimizer/scheduler/RNG/dataloader，开始真正 300-step run。不得从 smoke S5、诊断 S150 或旧方法 checkpoint 继续并标成 Base→300。

条件满足、授权资源可用时，实际执行命令并核验进程。启动回报至少包含：

```text
run_id
code/config/model identities
实际启动命令（去掉凭据）
PID / tmux session / scheduler job ID
日志路径
checkpoint 输出目录
已完成 outer step / 至少一次非零 LR update 证据
```

仅提交 queue 时明确“已排队、尚未训练”；进程退出则明确退出及错误，不能把终端中曾经出现命令当成训练成功。

---

## 17. 训练期间的最少监测与安全动作

### 17.1 每步聚合指标

| 范围 | 最少指标 |
|---|---|
| 原任务 | 总体/Hotpot/NQ success、实际 executed calls、episode length、invalid、response lengths |
| Utility | eligible/score-success、g_inc/g_cf/g_dir 分位数、sign conflict、dead zone、U 正负/零 |
| 覆盖 | source、call bucket、全成功/全失败/mixed，仅分层日志，不作为 gate |
| Repeat/预算 | visible-exact-repeat、unconsumable O、source error；raw mass、cap active rate、effective mass |
| Answer | 合法终端比例、可分 stratum 数、U_A 正负/零、word overlap 与 EM 的分布 |
| Routing | rho Q/A、positive chunk mass、routed rate、no-mass=zero、think token fraction |
| Optimization | 四个 local scalar、local dL/dlogp、clip/KL、grad norm、lambda_s、nonfinite |
| Cost | rollout/scorer/update 时间、view 数、peak memory、score truncation/error counts |

S1–5、S30、S150、S300 保存完整或可追溯的 turn components；其他步固定 hash 抽样少量原始例子。始终包含被拒绝、零信用和 no-think 样本，不只保存 selected[:32]。

metadata 对比不属于每步在线必算指标。需要时只在上述 milestone 的已保存轨迹上离线生成 sidecar，并单独记录 join 覆盖率；训练日志不得依赖该 sidecar 才能解释 credit 或恢复训练。

### 17.2 少量健康检查

NaN/Inf、gold/future leak、预算/归约/mask 不变量被破坏、重复更新原始 row：立即暂停自己的 run，保存现场；不自动换 reward 继续训练。

合法请求的 scorer 成功率连续 3 步低于 90%，或已有合法 search/answer 却因未知原因全部被 auxiliary parser 拒绝：先暂停排查。这是覆盖故障，不是模型已经学会。

合法样本存在但本来都落在 dead zone、正确 answer stratum 全零、无 positive responsibility：记录原因即可，不能笼统按“active=0”自动判失败。连续异常变化要区分数据状态和工程故障。

calls 上升、训练 EM 暂时下降不单独触发自动方法修改。若伴随显著 invalid/长度膨胀、KL/clip 爆发、资源错误，则暂停诊断；不得边跑边改 lambda、negative_scale 或桥接开关来维持曲线。

至少在 S30 full-strength 时复核一批 local/global dL/dlogp；有明显风险才追加一次参数梯度检查，不每步重复昂贵多 backward。

---

## 18. Checkpoint、评估和最终交付

### 18.1 保存策略

沿用当前 25-step rolling full-resume 机制，至少保留最近一个完整可恢复 checkpoint，完成新 checkpoint 原子写入并核验后再清理旧的**本 run rolling**副本。S150、S300 保留不可被 rolling cleanup 删除的模型导出与身份文件；磁盘允许时保留 milestone full state。

空间盘点在启动前完成。不得删除别的实验、旧原始 rollout、用户模型或其他人的文件来腾空间。诊断数据和 scores 尽量保存标量/短 token IDs，不落盘全词表 logits。

### 18.2 七集评估

S150 和 S300 分别对 NQ、TriviaQA、PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle 进行当前标准 greedy 评估，每题一条轨迹。

沿用现有评估脚本和运行协议，不为 V4 修改 evaluator、batch aggregation 或直出 `success_rate` 语义。保留原始 trajectory 和最终答案，在最终报告中并列呈现三类结果：

1. 评估脚本直接输出的每集 `success_rate` 及七集等权均值，用于与 EviSD/SDAR、V1、V2 和纯 GRPO 对齐；
2. 从逐题原始输出计算的标准 EM：`correct_count / total_unique_questions`，七集 macro 为七项等权均值；
3. 从同一逐题输出计算的标准 word F1，并同时报告实际 calls、invalid 和样本数。

三者分别标明 provenance，不互相替代。不能把 shard/batch 均值广播到题后当成标准 EM，也不能把 Bamboogle 某个历史数值简单除以 2 当作修复；题级 EM/F1 的后处理不改变训练 reward 或现有评估脚本。

测试集不参与选 delta、lambda、alias、rescue 或 checkpoint；S150 和 S300 都报告，不因某个更好而改主预算。

### 18.3 目录建议（允许按现场习惯调整）

```text
reports/urcr_v4/
  identity.json
  calibration_report.md
  exactness_report.md
  implementation_and_launch_report.md
  final_training_report.md
artifacts/urcr_v4/
  diagnostic_manifest.jsonl
  probe_template.json
  frozen_scores.parquet
  resolved_method.json
runtime/urcr_v4/<unique_run_id>/
  resolved_config.yaml
  train.log
  step_metrics/
  turn_components/
  checkpoints/
  evaluations/S150/
  evaluations/S300/
```

不是要求建立几十份重复报告。可以合并小报告，但 identity、frozen decisions、原始诊断样本、正式日志和 eval 必须可追溯。

### 18.4 最终执行报告的核心内容

给出实际方法与本文件差异（若无则明确无）、冻结参数、诊断覆盖限制、数值/归约结果、实际训练起点、完成的 steps、S150/S300 七集 `success_rate`、标准 EM、标准 F1、运行成本和 artifacts 路径。

报告必须区分：方法规格已完成、代码已实现、检查已通过、训练已启动、300 steps 已完成、评估已完成。不能用前一项代替后一项。

---

## 19. 最终配置摘要（语义示例，不要求这些就是现有 Hydra 键）

```yaml
method:
  name: urcr_v4_unified_typed
  revision: urcr_v4_unified_typed_r1_final
  global_grpo_unchanged: true
  formal_init: original_qwen2_5_3b_instruct
  total_outer_steps: 300
  train_questions: 128
  rollouts_per_question: 8

probe:
  actor_snapshot: pre_first_update_of_outer_batch
  grad: false
  temperature: 1.0
  targets: fixed_alias_subset_mean
  max_aliases: 3
  score_tokens: answer_content_only
  template: native_neutral_answer_probe
  max_total_tokens: 8192  # 不超过模型实际上限；不改变 rollout 上限
  default_backend: independent_views_batched_rpc

search_utility:
  controls: 3
  same_question_control: false
  select_controls_by_gold_alias_absence: false
  combine: same_sign_min_magnitude
  delta: resolved_numeric_from_calibration
  scale: resolved_numeric_from_calibration
  squash: continuous_deadzone_tanh
  negative_scale: 0.1
  visible_exact_repeat_positive: false
  unconsumable_observation_credit: 0.0
  group_whitening: false

metadata:
  online_pipeline: false
  credit_enabled: false
  diagnostics: offline_sidecar_only

answer_utility:
  quality: evaluator_consistent_word_fbeta
  beta: 0.5
  residual_group: rollout_group_id_and_binary_em
  baseline: within_stratum_mean_including_self
  std_normalization: false
  singleton_credit: 0.0
  all_equal_credit: 0.0
  dependence_on_a_out: false

responsibility:
  targets: actual_sampled_action_content
  intervention: position_preserving_outgoing_information_barrier
  mapping: positive_exponential
  query_delta: resolved_numeric_from_calibration
  answer_delta: resolved_numeric_from_calibration
  query_scale: resolved_numeric_from_calibration
  answer_scale: resolved_numeric_from_calibration
  chunk_power: 2.0
  max_chunks: 6
  no_mass: zero
  whole_think_fallback: false

credit:
  alpha_query: 1.0
  alpha_answer: 0.25
  lambda_think_query: 1.0
  lambda_think_answer: 1.0
  query_think_split: false
  search_trajectory_absolute_cap: 2.0
  terminal_broadcast_to_old_thinks: false

local_objective:
  span_reduction: mean_once
  surrogate: signed_tokenwise_ppo
  population_denominator: original_trajectories
  minibatch_estimator: preserve_outer_trajectory_mean
  lambda_max: resolved_numeric_from_full_strength_gradient_calibration
  warmup_outer_steps: 30
  final_decay: false
  same_actor_forward_for_training: true

legacy:
  fixed_support_reward: false
  legacy_v1_residual: false
  agam: false
  visible_focus_kl: false
  evisd_teacher: false
  evisd_search_pi: false
  evisd_answer_pi: false
  old_s_local: null
  n1_reference: false
```

上面的 `resolved_numeric_...` 在本计划中表示待一次性测得的值；在正式 `resolved_method.json` 和启动配置里必须替换为实际数字，不能以占位符启动。

---

## 20. 方法参考、阅读结果与未完成的外部核验

### 20.1 直接设计来源

- 合作者 C 上传稿：《URCR-V4：从“固定证据奖励”重构为“动作效用—思考责任—残差信用路由”》，本轮提供的 1308 行版本。源文件 SHA256：`88433ae5cc851444c02efa856d5834b2b7b5d141842d9e92f8193e401fab83d3`。
- 当前会话的 B 版 V4 架构讨论：提供动作与 think baseline 区分、双参照、候选动作和时序变体；本次只保留与 C 的统一残差主线一致的部分。
- 已保存的 URCR V2 实现/尺度审计：用于复用 no-grad responsibility、span masks、独立 local loss 和 DP 归约经验；旧数值不直接继承为 V4 参数。

### 20.2 公开代码与论文参考

- URCR compact review repository：<https://github.com/CrilwaKl/urcr-core-review>
- IGPO repository：<https://github.com/GuoqingWang1/IGPO>
- IGPO vectorized scorer：<https://github.com/GuoqingWang1/IGPO/blob/main/scrl/llm_agent/vectorized_gt_logprob.py>
- IGPO actor log-prob / update：<https://github.com/GuoqingWang1/IGPO/blob/main/verl/workers/actor/dp_actor.py>
- IGPO launcher：<https://github.com/GuoqingWang1/IGPO/blob/main/train.sh>
- IG-Search：<https://arxiv.org/abs/2604.15148>
- veRL extension guide：<https://verl.readthedocs.io/en/latest/advance/dpo_extension.html>

本轮已成功读取 IGPO 的 vectorized scorer、actor、trainer 和 launcher 相关公开内容：有批量 RPC 与 extended-sequence 两类实现思路，涉及 causal mask、position IDs、answer target 范围及后端 fallback。只借鉴计算组织，不将其 reward/advantage 配方一并迁移。

本轮对 URCR compact repo 的网页、raw 与历史 commit 读取没有成功返回正文。因此本文不声称已逐行审查当前仓库或当前现场 commit；对现有 URCR 的工程描述基于用户历史审计。Codex 必须以现场实际源码为准，记录本地 commit/diff，必要时自行读取该参考仓库。不能因网页本轮未取到就阻塞拥有本地源码的实施。

已访问的 IGPO `main` 页面不是永久版本固定；执行时记录真正借鉴的 commit/file hash，保留相关 license headers。若当前版本变化，以实际读取实现为准，不假定某个函数签名永远不变。

---

## 21. 一句话执行原则

**冻结统一动作效用—促进型思考责任—邻接残差路由的主线；只做让评分、尺度与数据流可用的最小检查；不回退固定 meta 奖励、不复制 AGAM、不扩大架构搜索。检查通过后，从原始 Base 实际运行唯一一条 300-step URCR-V4 正式训练。**
