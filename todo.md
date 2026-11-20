# llmsim 项目大纲

> 复选框由用户控制：勾选 = 已验收并冻结要求与行为；未勾选 = 仍有改进空间。
> AI 使用“实现状态”记录进度，不自行替用户勾选。

## 项目目标

解析式（纯数值、不搬运真实数据）的推理性能模拟器，用来快速验证想法。当前只关注 MoE 层：
统计 EP all-to-all 在不同硬件、不同 EP 配置下的传输时间，硬件以按 payload 的访存带宽为参数。
Attention、其他 Layer、Engram 等由用户后续逐步要求添加（2026-11-05 用户要求删除这些代码，只保留 MoE）。

## 共用要求

- 保留已完成功能；修改共用代码时验证受影响场景。
- 硬件、模型、部署、负载分别配置；新增硬件只需一个 TOML，新增模型只需一个 HF 风格 config.json。
- 只用 Python 标准库，Python ≥ 3.11。
- 硬件参数中的带宽一律为每卡单向带宽。
- 代码只包含当前范围需要的部分，不留冗余代码；注意码风（用户会审阅代码）。
- 输出 / 可视化代码与仿真主逻辑分开封装，审阅主逻辑时不需要看图表代码。

## 始终禁止

- 不得由 AI 自行勾选用户验收。
- 不得默默丢弃已经完成的功能或隐瞒验证缺口。
- 如用户指定不可修改文件，按“禁止修改文件：相对路径”逐项记录；不要凭空添加文件限制。

## 场景 S-001：配置基础

- [ ] **[F-001] 硬件实例参数化**
  - 实现状态：待验证
  - 要求：硬件分三层：GPU（显存容量、本卡读写带宽）、NIC（跨服务器读写带宽）、服务器（顶层：GPU + NIC + n×n 卡间拓扑与各链路类型的读写带宽）。带宽可为实测曲线或固定带宽 + 延迟。算力相关字段已随 2026-11-05 精简移除。2026-11-10 用户要求：服务器按 8 卡一体机，网卡暂不使用（NIC 定义保留，不参与仿真，EP 不超过单机卡数）。
  - 验收行为：`--set server=s3x8|h20x8|h800x8|b300x8` 均能运行；GPU / NIC 规格数值经用户按 datasheet 核对。
  - 实现位置：llmsim/hardware.py（GPU、NIC、Server、Access）；configs/gpu/、configs/nic/、configs/server/
  - 验证：tests/test_hardware.py 通过（S3 本卡曲线、S3x8 所有卡对的对端曲线、S3x8 无 NIC、H800x8 NVLink）；EP 超过 8 时场景构建报错；GPU / NIC 规格未与 datasheet 核对。
- [ ] **[F-002] 模型 MoE 结构从 HF config.json 解析**
  - 实现状态：已完成
  - 要求：只解析 MoE 相关字段（hidden、层数、路由专家数、topk、专家中间维度、shared expert、前 k 层 dense、MoE 层间隔、专家量化精度）；支持多模态模型的 text_config。Attention 等结构解析已随 2026-11-05 精简移除。
  - 验收行为：deepseek_v4_1、deepseek_v3、qwen3_30b_a3b 都能解析。
  - 实现位置：llmsim/model.py；configs/models/
  - 验证：三个模型均已解析（V4.1：5120 / 384 专家 / top6 / 2304 / 40 层 / fp4；V3：58 个 MoE 层；Qwen3：48 层）；test_moe.ModelTest 通过。
- [ ] **[F-003] 场景配置与命令行覆盖 / 扫描**
  - 实现状态：已完成
  - 要求：场景 TOML 组合硬件、模型、EP、精度、传输方式、阶段 token 数；`--set` 用点号路径覆盖，`--sweep` 可重复做组合扫描，`--json` 输出结构化结果。
  - 验收行为：`python -m llmsim configs/scenarios/deepseek_v41_s3.toml --sweep parallel.ep=2,4,8 --sweep a2a.concurrency=1,7` 输出对比表。
  - 实现位置：llmsim/scenario.py、llmsim/cli.py、llmsim/report.py
  - 验证：已运行单次、--sweep concurrency、--sweep ep、--json、--show-curves。

## 场景 S-002：已移除的完整 decode 仿真

- [ ] **[F-004] 算子 roofline 代价模型**
  - 实现状态：已移除（2026-11-05 用户要求只保留 MoE，删除 Layer / Attention 相关代码）
  - 要求：算子耗时 = max(FLOPs/有效算力, 字节/有效带宽) + kernel 开销。
  - 验证：原实现曾通过单元测试；代码已删除，需要时按用户要求重新添加。
- [ ] **[F-005] EP all-to-all 与负载不均衡**
  - 实现状态：已合并到 F-010 / F-011（路由抽样、EPLB 放置与冗余专家、负载不均衡保留在 llmsim/routing.py）
  - 要求：逐 token 抽样路由（Zipf 热度）；static / EPLB 放置与冗余专家；EP 阶段取最慢卡并同时报告平均卡。
  - 验收行为：EPLB+冗余专家降低不均衡；按 rank 去重减少传输份数。
  - 实现位置：llmsim/routing.py
  - 验证：tests/test_routing.py 通过（EPLB 降低不均衡、按 rank 去重减少份数、份数守恒）。
- [ ] **[F-006] 计算/通信重叠调度（none / async / tbo）**
  - 实现状态：已移除（2026-11-05 用户要求只保留 MoE）
  - 要求：两条资源上的列表调度；TBO 拆两个 micro-batch。
  - 验证：原实现曾通过单元测试；代码已删除。
- [ ] **[F-007] 结果校准**
  - 实现状态：待开发
  - 要求：用公开数据或实测数据校准传输时间，把误差控制在约 ±20%。
  - 验收行为：给出校准场景、参考数值与仿真数值对照。
  - 验证：未执行。
- [ ] **[F-008] DeepSeek V4.1 结构**
  - 实现状态：已完成
  - 要求：接入用户提供的 V4.1 config.json；当前只用到 MoE 字段，其余新结构（Engram、MTP、KV 压缩等）待用户要求时再接入。
  - 验收行为：V4.1 场景可运行。
  - 实现位置：configs/models/deepseek_v4_1.json（用户提供）、llmsim/model.py
  - 验证：默认场景已运行，V4.1 MoE 字段解析正确。

## 场景 S-003：MoE 层通信仿真（不依赖算力）

- [ ] **[F-009] S3 硬件与按 payload 的访存带宽曲线**
  - 实现状态：已完成
  - 要求：新增硬件 S3（每芯片 96GB，8 卡机）；按用户给的实测表提供不同 payload 下的读写带宽（GB/s），拟合为分段函数，每段对应各自的带宽。2026-11-09 用户要求：本卡 load / store 留在 GPU（GPU 特性），对端 load / store 放到顶层服务器，用 8×8 拓扑表达，每张卡对的读写带宽仍用用户给的数据（当前机器是对称拓扑）。
  - 验收行为：曲线经过 1KiB–4MiB 的全部 28 个测量点；S3x8 的 8×8 拓扑中每个卡对都走 peer 链路；`--show-curves` 打印 GPU 本卡、各链路与 NIC 的带宽。
  - 实现位置：llmsim/hardware.py（PayloadCurve、Access、GPU、Server）；configs/gpu/s3.toml、configs/server/s3x8.toml；llmsim/report.py（format_curves）
  - 验证：test_hardware 通过（本卡 14 点、所有 56 个卡对的对端 14 点全部吻合，32KiB 为几何平均，外推规则）；--show-curves 已运行。拟合方式（log-log 分段线性）、GB/s = 1e9 字节/秒、KiB 按 1024 进制为 AI 的假设，待用户确认。
- [ ] **[F-010] EP all-to-all 传输时间：prefill / decode × 不同 EP 配置**
  - 实现状态：已完成
  - 要求：只统计 MoE 层 token all-to-all（dispatch + combine）的传输时间，不算 kernel；每层拆成卡到卡传输，payload 查带宽曲线；支持 push / pull、每卡并发传输数、总带宽上限、本卡拷贝、FP8 scale 开销；报告每卡 MoE 权重。
  - 验收行为：`python -m llmsim configs/scenarios/deepseek_v41_s3.toml` 输出 prefill / decode 每层 dispatch、combine 与每步时间。
  - 实现位置：llmsim/moe.py、llmsim/report.py、configs/scenarios/deepseek_v41_s3.toml
  - 验证：test_moe 通过；默认场景与 ep=2,4,8、concurrency=1,2,4,7 扫描已运行。每卡并发传输数（默认 1）待用户确认；本卡拷贝占用一条并发通道。2026-11-06 起改为逐层仿真（见 F-014），每步时间 = 各层最慢卡之和。
- [ ] **[F-011] 大规模路由的聚合近似抽样**
  - 实现状态：已完成
  - 要求：每层 token 数 × topk 超过 5 万时改用按概率抽计数的近似；结果与逐 token 抽样一致到约 5%。
  - 验收行为：默认场景 prefill 用 aggregate、decode 用 token；两种方式的总传输份数差异 < 5%。
  - 实现位置：llmsim/routing.py（sample_aggregate、resolve_method）
  - 验证：选中概率改为 π = 1 - (1 - p)^t 后，与逐 token 抽样的份数比为 0.981（rank 去重）/ 1.001（不去重），不均衡 1.138 / 1.174 对 1.118；默认场景 prefill 用 aggregate、decode 用 token。阈值随逐层仿真从 20 万降到 5 万（每层），对比测试在 4.9 万规模上通过。
- [ ] **[F-012] 部署并行参数与每卡推理权重**
  - 实现状态：部分移除（2026-11-05 用户要求只保留 MoE）：DP / TP / SP、attention 与 Engram 权重已删除，保留每卡 MoE 权重（路由专家 + shared expert）
  - 要求：每卡 MoE 权重 = (专家数 + 冗余) / ep 个路由专家 + shared expert，所有 MoE 层合计；DP / TP / SP 待用户要求加入 attention 时再加。
  - 验收行为：报告每卡 MoE 权重与 HBM 剩余。
  - 实现位置：llmsim/moe.py（expert_memory）
  - 验证：test_moe.test_expert_memory 通过；EP8 每卡 37.5 GB（48 个专家 36.1 GB + shared 1.4 GB），EP2 每卡 145.8 GB 超过 96 GB。FP4 专家每 32 元素 1 字节 scale 为 AI 的假设，待确认。

- [ ] **[F-015] dispatch / combine 的顺序变换（device-major ↔ expert-major）**
  - 实现状态：已完成
  - 要求：dispatch 收到的 token 按来源卡排列，需要变换成按本地专家排列后才能做专家计算；combine 发送前做反向变换。2026-11-10 用户要求补上顺序变换算法本身的延迟：算顺序（读 topk 专家 id、写每专家计数 / 偏移与置换索引）和搬数据两步都计入，路由元数据（topk 专家 id 与权重）随 token 一起 dispatch。
  - 验收行为：报告与页面分别给出每层 permute / unpermute 的耗时及其中算顺序的耗时；`a2a.permute = segment | buffer | none` 切换搬数据的计法（none 仍计算顺序）。
  - 实现位置：llmsim/layers/dispatcher.py（TokenDispatcher.permute / unpermute、TOPK_META_BYTES）；llmsim/scenario.py（Transfer.permute）
  - 验证：tests/test_dispatcher.py 通过（算顺序与搬数据的访存公式、buffer / none 模式、路由元数据字节）；默认场景 decode 每层算顺序 permute 9.8 µs / unpermute 3.2 µs，每步 46.67 → 47.23 ms，prefill 2636.6 → 2640.8 ms。算顺序的访存步骤和按段 / 整块粒度是 AI 的建模，待用户确认。
- [ ] **[F-016] 按推理框架结构组织代码**
  - 实现状态：已完成
  - 要求：与推理框架一致：Transformer 中堆叠多个层（目前只有 MoE），forward 过程中计算开销；MoE 层内部为 router → dispatch → permute → experts → unpermute → combine。
  - 验收行为：`Transformer.forward(batch)` 逐层调用 `MoELayer.forward`，返回每层每卡的开销；数值与重构前一致（除本轮新增的算顺序与路由元数据）。
  - 实现位置：llmsim/config.py（ModelConfig）、llmsim/model.py（Transformer）、llmsim/layers/moe.py（MoELayer、Router）、llmsim/layers/dispatcher.py（TokenDispatcher）、llmsim/batch.py、llmsim/runner.py
  - 验证：run_tests.sh 38 项通过；permute=none 时 decode 15.41 ms，与旧版不计重排的 14.85 ms 相差为新增的算顺序与路由元数据。

## 场景 S-004：模拟器任务

- [ ] **[F-013] 任务 1：标准 batch 下 8 卡机 MoE 层的 prefill / decode 传输时间**
  - 实现状态：已完成
  - 要求：按 SGLang 或 vLLM 的一个标准 batch 配置，给出 S3 8 卡机上 DeepSeek V4.1 每个 MoE 层 prefill 和 decode 的传输时间。
  - 验收行为：给出 EP8、prefill 每卡 8192 token、decode 每卡 128 token 下每层 dispatch、combine 与每步合计时间，并说明对并发数的敏感性。
  - 实现位置：configs/scenarios/deepseek_v41_s3.toml
  - 验证：push、concurrency=1、skew=0.6：prefill 每层 dispatch 7.99 ms + combine 15.75 ms = 23.7 ms（40 层 949 ms）；decode 每层 132 µs + 239 µs = 371 µs（40 层 14.8 ms）。concurrency=4 时 prefill 242 ms / decode 3.84 ms。标准 batch 取值来自对 SGLang / vLLM / DeepEP 默认值的记忆，待核对。逐层仿真后：prefill 950.6 ms、decode 14.85 ms（差异来自逐层抽样）。

## 场景 S-005：可视化

- [ ] **[F-014] 逐层仿真与 HTML 图表**
  - 实现状态：已完成
  - 要求：PD 两阶段下展示每张卡经过的 MoE 层数，以及每层每张卡迁移的数据量；先只做 token 分发，为后续 expert weight 迁移算法留出直观的对比方式。每个 MoE 层独立的专家热度与放置（prefill / decode 共用），逐层记录每卡发出 / 收到 / 本卡字节与耗时、卡到卡 token 份数。
  - 验收行为：`python -m llmsim configs/scenarios/deepseek_v41_s3.toml --html out/moe.html` 生成自包含页面，可切换 prefill / decode、dispatch / combine、发出 / 收到；包含每层传输时间柱图、卡 × 层数据量热力图、选中层卡到卡矩阵、选中层每卡耗时，以及表格视图。
  - 实现位置：llmsim/moe.py（LayerResult、逐层 simulate）；可视化封装在 llmsim/viz/（data.py 取页面数据，page.html / page.css / page.js 为页面），主逻辑只有 llmsim/cli.py（--html）引用它
  - 验证：run_tests.sh 27 项通过（test_moe 校验逐层字节与路由一致；test_viz 校验页面拼装与 2×40 层、每层 8 卡数据）；headless Edge 截图检查默认与 skew=1.2 static 两个页面，脚本点击切换阶段 / 阶段类型 / 方向、选中层、悬停提示、展开表格均正常且无 JS 报错；2026-11-06 封装为 viz 子包后重新截图确认。浅色已截图检查，深色模式未截图检查。色阶按每个视图的最小值到最大值映射（卡间差异只有几个百分点），图例标注最小 / 最大值。--json 只保留每层的 dispatch / combine 耗时与不均衡，逐卡数据只在页面里。

## 触发人工检查

当前未指定报错规则。用户可用“报错包含：具体关键词”或“报错匹配：正则表达式”添加；命中后停止自动操作，交由人工检查。

## 下一步待做

1. F-015：确认顺序变换的建模（算顺序的访存步骤、搬数据按段 / 整块、是否使用表中的本卡读写带宽）；当前顺序变换约占每卡耗时的 2/3。
2. F-016：用户 review 新的代码结构（Transformer / MoELayer / TokenDispatcher）。
3. F-010：确认 S3 每卡可并发的对端传输数、dispatch 用 push 还是 pull。
4. F-014：用户查看 out/moe.html，确认图表形式是否符合预期。
5. F-013：核对 SGLang / vLLM 标准 batch 的取值。
6. F-009：确认带宽单位（GB/s）与分段拟合方式是否符合预期。
7. 后续（等用户要求）：专家计算开销及计算与通信争用 SM 的建模；expert weight 迁移算法；跨服务器（NIC）与超节点；attention 与 DP / TP / SP、Engram、MTP 层。


<!-- selfmanager:history:start -->
## 更新记录

### 2026-11-05T10:46:53+08:00 · 新建 llmsim 解析式模拟器骨架：硬件/模型/场景配置、roofline 算子、EP 路由与 all-to-all、负载不均衡、EPLB、none/async/tbo 调度、显存估算、CLI 扫描
- 本轮已完成：F-002、F-003、F-004、F-005、F-006
- 本轮待处理：F-001、F-007、F-008
- 保留的已验收任务：暂无用户验收
- 验证：unittest 21 项全部通过；手动运行 DeepSeek H800 EP32/EP144、硬件/skew/冗余/EP/kv_len 扫描及 --json；硬件规格未与 datasheet 核对，结果未校准

### 2026-11-05T17:20:43+08:00 · 删除旧的 Layer/Attention/decode 代码与配置，只保留 MoE 通信仿真；修正聚合抽样的专家选中概率（π = 1-(1-p)^t）；新增 run_tests.sh
- 本轮已完成：F-002、F-003、F-008、F-009、F-010、F-011、F-013
- 本轮待处理：F-001、F-007
- 保留的已验收任务：暂无用户验收
- 验证：bash run_tests.sh：23 项全部通过；默认场景、ep / concurrency 扫描、--json、--show-curves 已运行；三个模型配置均可解析。快照在本轮开始时建立，此前一轮的精简改动发生在快照之前

### 2026-11-06T10:49:30+08:00 · MoE 改为逐层仿真（每层独立专家热度与放置，逐层记录每卡收发字节与耗时）；新增 --html 交互页面（每层时间、卡×层热力图、卡到卡矩阵、每卡耗时、表格视图）；聚合抽样阈值降为每层 5 万
- 本轮已完成：F-014、F-010、F-011、F-013
- 本轮待处理：F-001、F-004、F-005、F-006、F-007、F-012
- 保留的已验收任务：暂无用户验收
- 验证：run_tests.sh 26 项通过；headless Edge 截图检查默认与 skew=1.2 static 页面，脚本化点击/悬停/表格正常无 JS 报错；深色模式未截图检查

### 2026-11-06T11:02:23+08:00 · 可视化封装为 llmsim/viz 子包（data.py + page.html/css/js），主逻辑只有 cli.py 引用；report --json 只保留每层摘要；删掉仅为页面加的 RoutingStats.load；HTML 测试移到 tests/test_viz.py
- 本轮已完成：F-014
- 本轮待处理：F-001、F-004、F-005、F-006、F-007、F-012
- 保留的已验收任务：暂无用户验收
- 验证：run_tests.sh 27 项通过；重新生成两个页面并用 headless Edge 脚本点击截图，渲染正常无 JS 报错

### 2026-11-09T18:22:11+08:00 · 硬件拆为 GPU（本卡读写）/ NIC / Server（顶层，8x8 拓扑与链路读写带宽）；新增 dispatch / combine 的 device-major ↔ expert-major 布局重排延迟；configs 改为 gpu/nic/server 三个目录；报告、页面、测试相应更新
- 本轮已完成：F-009、F-015
- 本轮待处理：F-001
- 保留的已验收任务：暂无用户验收
- 验证：run_tests.sh 33 项通过；默认场景与 reorder=none/buffer/segment 扫描已运行；页面重新生成并 headless 截图检查堆叠条与提示，无 JS 报错

### 2026-11-10T10:02:08+08:00 · 按推理框架结构重组代码（ModelConfig / Transformer / MoELayer / Router / TokenDispatcher / runner）；顺序变换补上算顺序的访存与路由元数据传输；服务器限定为单台 8 卡机，NIC 保留定义不参与仿真
- 本轮已完成：F-015、F-016
- 本轮待处理：F-001
- 保留的已验收任务：暂无用户验收
- 验证：run_tests.sh 38 项通过；默认场景与 permute 扫描已运行；页面重新生成并 headless 截图检查 combine 阶段的执行顺序与提示

<!-- selfmanager:history:end -->
