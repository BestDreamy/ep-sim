# llmsim

MoE 层 EP all-to-all 传输时间仿真。只做数值计算，不搬运真实数据，不计算任何 kernel。
纯 Python 标准库，需要 Python ≥ 3.11。

## 用法

```bash
python -m llmsim configs/scenarios/deepseek_v41_s3.toml --show-curves
```

`--set` 覆盖字段，`--sweep` 扫描字段（可重复，取所有组合），`--json` 输出结构化结果：

```bash
python -m llmsim configs/scenarios/deepseek_v41_s3.toml --sweep parallel.ep=2,4,8 --sweep a2a.concurrency=1,7
```

`--html` 另外生成一个自包含的交互页面（数据内嵌，浏览器直接打开）：

```bash
python -m llmsim configs/scenarios/deepseek_v41_s3.toml --html out/moe.html
```

页面可切换 prefill / decode、dispatch / combine、发出 / 收到，包含：每层传输时间（最慢卡）、
卡 × 层的数据量热力图、选中层的卡到卡数据量矩阵、选中层每张卡的耗时（看谁在等谁），以及表格视图。

```bash
bash run_tests.sh
```

## 结构

代码按推理框架的方式组织：`Transformer` 里堆叠各层（目前只有 MoE 层），`forward(batch)` 逐层累计开销。

| 文件 | 作用 |
|---|---|
| `configs/gpu/*.toml` | GPU：显存容量、本卡读写带宽（`[local]`） |
| `configs/nic/*.toml` | NIC：跨服务器读写带宽（`[access]`）；目前只仿真单台服务器，NIC 不参与计算 |
| `configs/server/*.toml` | 服务器（顶层）：GPU、n×n 卡间拓扑 `topology` 与各链路类型的读写带宽 `[links.*]` |
| `configs/models/*.json` | HuggingFace 风格 `config.json`，只读取 MoE 相关字段 |
| `configs/scenarios/*.toml` | 场景：服务器 + 模型 + EP + 精度 + 传输方式 + 每阶段 token 数 |
| `llmsim/hardware.py` | `PayloadCurve` / `AlphaBeta` 带宽、`Access`（load / store）、`GPU`、`NIC`、`Server` |
| `llmsim/config.py` | `ModelConfig`：解析 config.json（含多模态模型的 `text_config`） |
| `llmsim/scenario.py` | 场景配置：部署、精度、传输方式、负载 |
| `llmsim/model.py` | `Transformer`：层列表、`forward(batch) -> ForwardCost`、每卡权重 |
| `llmsim/layers/moe.py` | `MoELayer.forward`：router → dispatch → permute → experts（暂不计）→ unpermute → combine |
| `llmsim/layers/dispatcher.py` | `TokenDispatcher`：all-to-all 与顺序变换的开销 |
| `llmsim/routing.py` | 专家热度、EPLB 放置、路由抽样（router 用它代替 gate 计算） |
| `llmsim/batch.py`、`llmsim/runner.py` | forward 的输入（每卡 token 数）；`simulate`：建模型、每个阶段 forward 一次 |
| `llmsim/report.py`、`llmsim/cli.py` | 文本 / JSON 报告与命令行 |
| `llmsim/viz/` | HTML 可视化，独立于仿真主逻辑：`data.py` 从 `SimResult` 取页面数据，`page.html` / `page.css` / `page.js` 是页面本身 |

## 怎么算

- **硬件层次**：本卡读写带宽属于 GPU；卡间读写带宽属于服务器，`topology[a][b]` 是卡 a 访问卡 b 显存所走的链路类型（S3x8 是对称的，所有卡对都走 `peer`），可以按卡对写不同链路来表达非对称拓扑。每种访存路径的带宽可以是实测曲线（`payload` / `load` / `store`）或固定带宽（`gbps` + `latency_us`）。EP 不能超过一台服务器的卡数。
- **带宽曲线**：相邻测量点之间按 `bw(s) = bw_i·(s/s_i)^k_i` 拟合（log-log 分段线性，经过每个测量点）；小于最小点时耗时不变，大于最大点时带宽保持最后一个值。
- **逐层**：每个 MoE 层有自己的专家热度（Zipf，`expert_skew`，各层热点专家不同）和放置，prefill 与 decode 共用同一套放置。
- **路由**：token 按专家热度无放回选 topk 个专家；`dedup = "rank"` 时发往同一张卡的多个专家只传一次。每层 token 数 × topk 超过 5 万时（典型是 prefill）改用按概率抽计数的聚合近似。
- **传输**：每层 dispatch 拆成卡到卡的传输，payload = 发给该对端的 token 份数 × 每 token 字节数。dispatch 每 token 包括 hidden（FP8 另计每 128 个元素一个 fp32 scale）和路由元数据（topk 个 int32 专家 id + fp32 权重）。push 是发送卡 store 对端显存，pull 是接收卡 load 对端显存，发往本卡专家的部分走本卡读写。`concurrency` 条通道并发，每卡取发送侧与接收侧的较大者。combine 是 dispatch 的反向。
- **顺序变换**：dispatch 收到的 token 按来源卡排列（device-major），专家计算要按本地专家排列（expert-major）。permute 分两步，都用 GPU 本卡读写带宽：
  - 算顺序：读收到 token 的 topk 专家 id，写每个本地专家的 token 数（前缀和得到偏移）和每一行的置换索引；
  - 搬数据：按索引把 token 从 device-major 搬到 expert-major（`dedup = "rank"` 时一个 token 可能写给本卡多个专家，写的份数多于读）。`permute = "segment"` 按段计（每个来源卡一段读、每个本地专家一段写），`buffer` 按整块读写，`none` 不计搬数据（算顺序仍计）。
  - combine 发送前做 unpermute：读一次置换索引，再把专家输出搬回按目标卡分段。
- **每层时间**：dispatch 阶段（传输 + permute）与 combine 阶段（unpermute + 传输）各取最慢的卡再相加；每步时间是各 MoE 层之和。
- **MoE 权重**：每卡 `(专家数 + 冗余) / ep` 个路由专家加 shared expert，所有 MoE 层合计。

## 已知简化

- 只仿真单台 8 卡服务器，不走 NIC；超节点尚未建模；每卡可并发的传输数待确认，默认 1。
- 顺序变换的访存粒度（按段 / 整块）、算顺序的访存步骤，以及是否用与表中相同的本卡带宽，待确认。
- 专家计算、通信占用 SM 的影响都还没计；attention、MTP 层、Engram 等后续再加。
