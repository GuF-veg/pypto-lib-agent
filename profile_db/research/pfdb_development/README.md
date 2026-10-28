# pfdb 开发实验：复查与重放

这些脚本是数据库之外的研究执行器，不属于安装后的 pfdb query/API。它们负责实验隔离、现有工件导出、模型调用，以及明确启用后的设备验证；pfdb 本身仍只分析已有工件。

研究目标与验收见[开发计划](../../../docs/debug-and-tune/pfdb-development-plan.md)，结论与偏离协议见[实施记录](../../../docs/debug-and-tune/pfdb-development-results.md)。[接口审查](interface_decisions_review.md)是独立输入，不替代最终决策台账。

[接口决定](interface_decisions.csv)记录 41 个 MCP 工具的取舍；[能力清单](capability_inventory.csv)展开到 view、selector、通道和有效导出组合，共 95 项（0.6.0 增补两项候选模式，0.7.0 增补诊断 view），不是新增工具。`inventory_capabilities.py --output <new.csv>` 可从当前 schema 重新生成清单，拒绝覆盖已有文件；完整性测试检查引用、MCP 覆盖与实际支持的 13 个 render 组合。它不是全部参数笛卡尔积，也不是每项优化价值已经得到实测的声明。

## 环境与隔离

所有命令先进入 `conda activate pypto`；安装与 pin 链沿用[项目环境说明](../../../docs/get-started/installation.md)。设备命令统一经 `npu-run`，使用逻辑设备 0。原始数据库、capture、PyPTO/runtime 都保持不变；输出写入忽略的 `build_output/pfdb_development/`。

OpenCode 使用已验证的 `opencode-go/deepseek-v4.1-flash`。提供方 alias 没有暴露不可变模型 revision；replicate 是重复编号，不是服务端可复现随机种子。每个静态题使用新会话；优化反馈只继续同一优化会话。模型隐藏推理不参与评分或报告。

静态/优化模型在独立工作目录中使用名为 `pfdb-eval` 的 deny-all profile。已验证的 V2 配置形式为：

```json
{
  "snapshots": false,
  "permissions": [{"action": "*", "resource": "*", "effect": "deny"}],
  "agents": {
    "pfdb-eval": {
      "mode": "primary",
      "steps": 2,
      "system": "Use only supplied evidence. No tools. Return the requested JSON.",
      "permissions": [{"action": "*", "resource": "*", "effect": "deny"}]
    }
  }
}
```

执行器同步子进程 `cwd/PWD`，并核验实际加载的 agent 与权限。客户端有时首次返回空目录，因此仅对只读预检限定最多三次，并保存目录结果；这不是模型失败重试。实际调用的 schema、输出、错误与协议均保留。不要把只初始化 Git 或旧配置字段的隐式转换当作隔离证明。

## 现有证据与静态实验

先核对源工件是否还存在。`prepare_calibration.py` 是最初两题校准；`prepare_decision_tasks.py` 导出十题的 D0/D1。二者不会采集数据。示例中的输入必须是已经验证的研究 capture：

```bash
python profile_db/research/pfdb_development/prepare_decision_tasks.py \
  --swa-database build_output/pfdb_development/swa_current.duckdb \
  --compressor-database .pfdb/compressor_ratio4_gap.duckdb \
  --swa-capture build_output/pfdb_development/swa_agent_capture \
  --output build_output/pfdb_development/decision-corpus-new
```

生成目录里的 `cases.json` 是评审答案索引；模型只收到选定问题和单个 packet。各 packet 保留稳定 fact ID、query/selector、source hash 和适用范围。D1 声明新增哪些信息；不能将新增定义、身份或映射算作纯格式优化。

```bash
python profile_db/research/pfdb_development/run_offline.py \
  --corpus build_output/pfdb_development/decision-corpus-new/d0/cases.json \
  --output build_output/pfdb_development/static-new \
  --workspace <isolated-opencode-workspace> --replicates 3 --parallel 2
```

默认 facts/columns 是可逆表示对照。仅运行一个信息条件时使用 `--conditions facts`，并用 `--intervention` 写清改动。每次输出目录必须全新；完整协议复制 corpus、packet 和 runner，记录 SHA。错误和超时保留，不自动补跑直到成功。

`prepare_rare_tasks.py` 复用已有 synthetic fixtures，另可导出两个固定真实 holdout；`collect_holdouts.py` 是外部一次采集入口。`prepare_memory_tasks.py` 生成明确标注的合成历史和 prune 前后副本。它们不把 authored fixture 数值冒充真实优化结果。

```bash
python profile_db/research/pfdb_development/prepare_rare_tasks.py \
  --output build_output/pfdb_development/e3-new \
  --holdouts-root build_output/pfdb_development/holdouts
python profile_db/research/pfdb_development/prepare_memory_tasks.py \
  --output build_output/pfdb_development/memory-new
```

`run_interactive.py` 是原 compressor run6/task 的单题 MCP 导航原型，不是通用十题执行器。默认固定 Code Mode；`mcp_experiment.py` 在自身进程内改变一项描述，并记录实际后端调用。区分 `execute`、pfdb queries 和工具目录获取，不能用后端工具总数推算模型 prompt。

## 外部优化与独立确认

`collect_swa.py` 和既有 compressor 研究 harness 使用固定 reference/comparator，支持 frozen golden；新 fixture 先由原始实现生成并验证。源代码、golden 与 specs 绑定在 manifest 中。补采和性能运行均由外部 harness 执行。

`prepare_optimization.py` 从已经通过的四个起点、搜索 benchmark、capture 和 heldout 构造输入。`run_optimization.py` 默认只准备 24 个会话；`--execute` 才启动模型及设备。每会话最多两个候选，失败计预算并停止该会话。AST guard 限定修改范围；数值与同步正确性仍由验证及审查负责。

```bash
python profile_db/research/pfdb_development/run_optimization.py \
  --config <registered-config.json> \
  --output build_output/pfdb_development/pilot-new
```

搜索使用无插桩 median、20 rounds / 5 warmup、原始 samples。其最小值和 provisional incumbent 都不是确认收益。独立确认只接受完整的 24 会话结果：

```bash
python profile_db/research/pfdb_development/confirm_optimization.py \
  --pilot build_output/pfdb_development/pilot-new \
  --output build_output/pfdb_development/confirmation-new
```

默认也只准备；明确 `--execute` 后，先验证保留的不同合法 fixture，再采三个独立进程 strata，每个 100/5。相同实现与 workload 的确认会复用并明确标记，不能当作多个独立性能样本。采用外部 median bootstrap；现有 pfdb mean bootstrap 不参与 median 接受。缺失、回退、确认失败与基础设施限制都进入结果。

已匹配的编译目录先完整复制到本次确认的私有目录，再作为 replay 目标；原 capture/build 不直接 replay。运行器可能更新缓存或重建程序，因此不能将“复用编译结果”理解为对原目录绝对只读。来源目录、metadata 与编译文件 hash 保留，后续 strata 继续检查程序身份。

## 输入契约、候选身份与关联的后续实验

[后续登记](followup_registration.json)另行冻结六题、三条件、三个重复。
`prepare_followup.py` 只从 21 个严格 hash 匹配的既有文件导出，不采集；
每次使用新目录。`cases.json` 是 scorer 输入，`reviewer.json` 保存处理映射，
二者不整体发给模型。模型仅见问题与单个 packet。

```bash
python profile_db/research/pfdb_development/prepare_followup.py \
  --output build_output/pfdb_development/followup-new/corpus
```

三个条件编码成 18 个匿名 case；使用既有执行器的 `--conditions facts`
与 `--replicates 3 --parallel 2` 才是登记的 54 次尝试。D0 保留正确 unknown；
D1 加候选原始材料；R 完整保留 D1 再加派生关系，不是纯格式变化。
契约 D0/D1 相同是负对照，问题含明确字段提示，因此有校准与到顶限制。
不得将字段分数解释为真实优化能力提升。

`run_offline.py` 的只读预检现在通过 `agent.list` 显式指定
`location[directory]`，核对返回目录，再核对 deny-all 权限。
`debug agents` 的默认上下文可能与模型 workspace 不同；仅设置 cwd/PWD
不能证明它返回了正确目录。本轮首次预检在模型调用前停止，保留原日志；
修正通过六项 regression 后，在新目录继续，prompt/packet/scorer 未变。

`validate_contract.py prepare <new-directory>` 在外部准备两份单 request
fixture、canonical expected 和四个私有 build；不触碰设备。
随后每次 `npu-run python .../validate_contract.py run <directory> <job>`
只验证一个 job：`parent-affine`、`parent-gap`、`scan-affine`、`scan-gap`。
benchmark 关闭、拒绝重试/覆盖，reference/comparator 不改。
有序缺口是 oracle-defined diagnostic，业务输入域 authority 仍未知。

结果、实际成本与限制见[后续研究报告](../../../docs/debug-and-tune/pfdb-followup-results.md)。
完整日志及原始响应留在忽略目录；独立审查只读匿名化公开响应与各自证据，
不读取模型隐藏推理。旧实验快照不因当前预检修正而改写。

## 成本与验收

`collect_usage.py` 仅读取 session 级 cost/token metadata，不导出会话或隐藏推理；同一多轮会话只计一次，避免再加 message totals。`summarize_study.py` 输出带来源 hash 的小摘要，将字段正确、引用 ID/覆盖、基础设施缺失分开；自由文本的证据支持与合法性仍需评审。

本轮静态结果见 [study_summary.json](study_summary.json)，24 会话及独立确认的缩减公开账本见 [optimization_summary.json](optimization_summary.json)。后者保留逐会话/共享 source 身份、gate、失败、usage 与未观察到收益的停止终点，不导出 transcript；完整 raw 证据仍在其引用 SHA 对应的忽略目录中。它只确认已声明 fixture，scan 缩减的连续性前提另行标记，不能作为通用合法优化记录。

普通 CI 只执行确定性 fixture/契约测试；付费模型与设备实验显式触发。一次实验通过不证明一般优化能力提升，未调用能力也不自动冗余。以后更换 runtime、模型、客户端通道或算子结构时，按[唯一 runtime 台账](../../../docs/debug-and-tune/pfdb-runtime-integration.md)和保留的决策理由复审。


## 0.7 交接和原生接口验证

[实施与结果](../../../docs/debug-and-tune/pfdb-native-interface-results.md)将 setup、严格评分、事实审计和未验证范围分别记账。`run_new_operators.acquire` 现在在进程结束后调用 `export_new_candidate.py`，写入 `candidate_manifest.json`；原有 capture 不被回写。没有完整构建/工具链来源时保持 build unavailable。manifest 由外部调用者使用现有 trial/attach/bind 接口导入，数据库不启动采集。

`probe_native_interface.py` 使用已存在的 replay 数据库，只准备协议时不调用模型；明确 `--execute` 才进行三题各两次原生查询，不生成算子、不采集设备数据：

```bash
python profile_db/research/pfdb_development/probe_native_interface.py \
  --database build_output/pfdb_development/new-operators-v1/v06-replay/evidence.duckdb \
  --output build_output/pfdb_development/native-query-new \
  --workspace /tmp/pfdb-native-query-new
```

执行模式需使用另一全新 output/workspace，并追加 `--execute`。为保留相对工件定位，查询副本建立在输入 DB 同级目录；已有副本拒绝覆盖。使用独立 workspace 配置、只读 MCP、`codemode=false`；先等待实际 agent 和 MCP ready，再在同一 background location 发请求。未就绪会在模型调用前停止。Code Mode 的 `execute` 在这个实验中不开放。模型错误不自动补跑，隐藏推理正文不参与分析。


## 0.8：目录与语义的独立对照

[后续结果第 9–11 节](../../../docs/debug-and-tune/pfdb-native-interface-results.md)记录 24 次目录对照、12 次问题定义/scope 的 2×2 对照，以及 0.8 的实现依据。完整目录与三工具目录的判断不同于权限筛选：生产只读 MCP 现暴露 33 项，可写模式仍为全部 41 项；三工具方案仅在研究 server 中存在。

实际源码提案的下一阶段见[原生闭环可行性研究](../../../docs/debug-and-tune/pfdb-native-operator-study.md)。`run_native_operators.py` 将现有采集、manifest、trial、只读 MCP 和源码 guard 接在一起；原生文件组与 pfdb 诊断组读取逐字节一致的注册文本。编译、NPU、profile 仍由数据库以外的执行器负责。权限校准失败和客户端 step 上限造成的总结输出单列，不混入有效性评分。v3 已结束但服务错误占 46/64，不能解释为算子成功率研究；两个 HC 源码另有独立确认。用户恢复认证后，v4 从原 bootstrap 开始，另行冻结协议。`summarize_native_operators.py <experiment> --output <new.json>` 检查冻结脚本、候选身份和汇总一致性，区分服务失败、诊断读取、提案、数值验证与尚未确认的搜索计时。

`probe_catalog.py` 默认冻结四题、两个目录、三次重复的协议；追加 `--execute` 才发起 24 个 OpenCode 会话。所需事实在两组均可达，但三工具组不能访问其他分析，不能视作保留全部能力。输出与 workspace 必须全新：

```bash
python profile_db/research/pfdb_development/probe_catalog.py \
  --database build_output/pfdb_development/new-operators-v1/v06-replay/evidence.duckdb \
  --output build_output/pfdb_development/catalog-new \
  --workspace /tmp/pfdb-catalog-new
```

当前版本的完整只读目录为 33 项；原试验在 0.7 下的完整目录为 41 项，历史预注册仍记录 41，不能混作同一条件。`probe_resource_scope.py` 是一次针对原来未加 scope 的 0.7 的 12 会话试验；它会拒绝把已经带 scope 的 0.8 当作旧基线。复查原试验需在临时、匹配的源码布局中使用保存的旧包/执行器和现存工件，不回退工作树或修改 runtime。

使用 `collect_usage.py` 获取 session aggregate metadata；它现支持原生会话的 `stdout.log`，相同 SID 与 JSONL 重复记录会去重。`summarize_catalog.py --experiment <directory> --output <new.json>` 导出目录结果；语义试验追加 `--study resource-scope`。分析器拒绝不完整/重排结果、变更的冻结脚本/包源码和不匹配目录，且不导出本机路径、诊断正文或隐藏推理。

两项历史试验的执行器在调用前复制、包源码 hash 在调用前冻结；完整包源码后来新增到 protocol/package，其中三个已经升级的源文件仅在归档中逆转小改动并验证所有字节的 SHA 与预注册值一致，过程标记为事后重建。后续准备默认直接复制包源码及 hash。模型服务未提供不可变 revision，这些材料用于定位和复核，不能保证模型逐字重现。

0.8.1 的闭环终局：v4 因 ENOSPC 停在58个会话、54份结果、12条完成轨迹，不能估计入口效果；8份源码另有独立确认，见 [native_operator_v4_summary.json](native_operator_v4_summary.json)。已有入口增加时间范围/关键路径定义/空counter状态，未声明验证默认unknown。外部producer/3继承错误报告交接，历史原始清单不回写。已有系统盘归档保留完整内容；冻结tensor的链接目标必须保留。
