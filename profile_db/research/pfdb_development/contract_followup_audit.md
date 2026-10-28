# Compressor ratio4 输入契约与扫描缩减的后续审计

日期：2026-10-22。只读代码、文档和既有候选；仅新增本报告。未执行模型、golden 数值计算、编译或 NPU 测试，未修改参考数学、runtime、原算子、既有输入或工件。下面的集合与数值推导是静态分析，设备验证由外部实验执行器另行完成。

**结论：当前主源码声明了 shape/dtype、独立 request 和有序物理 token tile；默认 fixture 生成连续位置，但没有找到明确授权“任意非连续位置”或明确禁止它的算子输入域条款。** 因此不能将一个有序缺口输入称为已经证明业务合法的反例。可以构造满足已知 shape、范围、边界和一致 paging 关系的诊断输入，证明 scan 候选的候选集合会漏项；其结果应标为“单 request、指定稀疏输入上的 canonical-golden 诊断”，不能直接升级为通用正确性或采用判决。

还有两个直接影响实验设计的事实：

- **mapping 的物理地址不连续，不等于 position 不连续。** 在位置为 affine 序列时，任意一致的 paging 重排或 `-1` 掩码都不会单独破坏这次扫描缩减，因为候选保留了原来的 mapping/位置判断。
- **本轮必须限制单 request。** 当前 standalone canonical golden 固定读取 state block table 的第 0 行、按全局物理行分 512-row tile，未使用 `query_start_loc`。多 request/填充不能用这个 golden 直接裁定；request-aware oracle 不在本轮范围内。

## 1. 哪些是声明，哪些只是生成模式

| 主证据 | 明确拥有的约束/行为 | 不能从这里推出什么 |
| --- | --- | --- |
| [canonical 算子](../../../models/deepseek_v4_flash_dspark/prefill_compressor_ratio4.py)，第 498–529 行 | `x` 为 `[T,D]` BF16；position 为 `[T]` INT32；两个 slot mapping 为 `[T]` INT64；state/cache 为 InOut；动态轴绑定；table 有 request 轴 | 未声明 position 的值必须为 `p0+j`；dtype 和 shape 不是位置连续性证明 |
| 同文件第 519、557–584 行 | packed requests 独立处理；每 request 按物理行切最多 512 行；tile 注释第 95 行称 ordered physical token tile | “有序”没有写成“无缺口连续”；也不能把“未写禁止”视为业务授权 |
| 同文件第 724–729 行 | fixture builder 要求 `1≤token_count≤MAX_SEQ_LEN`、`start_pos≥0`、`start_pos+token_count≤MAX_SEQ_LEN` | 这是 builder 参数检查，不是对运行时任意 `position_ids` 的完整域验证 |
| 同文件第 774–811、815、840–842 行 | position 初始化为 `arange(start,start+T)`；query starts 为 `[0,T]`；mapping 与这些位置共同生成 | 连续性是实际生成的 fixture 模式，不能自动扩大成所有调用方的契约 |
| [TensorSpec](../../../golden/spec.py)，第 33–52、153–172 行 | shape/dtype/初始化策略；`init_value` 是初始 host 内容 | callable 生成了某种数据，不代表所有其他同 shape/dtype 数据不合法，也不代表它们都合法 |
| [模型说明](../../../docs/models/deepseek_v4_flash_dspark/index.md)，第 24 行 | prefill 为 packed stream，长 prompt 分块对 resident prefix 执行 | 未给出位置缺口、重排、重复的处理契约 |
| [已有 ragged fixture](../../../models/deepseek_v4_flash_dspark/prefill_csa.py)，第 1746–1754、1777–1788 行 | 两 request 的位置分别为 `[126,127,128]`、`[30,31,32,33]`，再拼接 padding；mapping 按 request table 分别生成 | 全局拼接位置不连续，不代表单 request 内允许缺口；也不是 standalone golden 对多 request 正确的证据 |

本轮 agent 问题包可以如实说：“已验证范围是给定 fixture；一般输入域中的位置连续性尚未被权威来源明确裁定。”不能伪造一个 owner policy，要求 agent 将连续性当成真实通用契约。

## 2. Golden 实际定义的候选集合

canonical golden 的第 602、624–625 行固定使用 table `[0]` 和全局物理 tile；第 659–679 行对 tile 内所有物理 token 做 overlay。设当前 tile 的物理起点为 `b`、长度为 `n`，`p_j=position_ids[b+j]`，`m_j=state_slot_mapping[b+j]`，当前压缩输出位置为 `w`。ratio=4 时：

\[
\text{cur}=w-3,\quad\text{prev}=w-7,\qquad
S(w)=\{j\mid0\le j<n,\ m_j\ge0,\ w-7\le p_j\le w\}.
\]

`p_j<cur` 的 token 使用投影前半 HEAD_DIM，其余使用后半。历史 state 先加载，再由当前 tile 的匹配 token 覆盖；所有压缩输出处理结束之后才发布 tail state（第 641–679、709–716 行）。因此诊断不能只修改 positions，却沿用与之不一致的 mapping、RoPE 或历史状态。

既有 [候选 `6fded292…`](../../../build_output/pfdb_development/optimization-pilot-v1/c4_manual-1-baseline/attempt-1/candidate.py) 第 276–284 行把扫描范围改为：

\[
\ell=\max(w-7-p_0,0),\quad h=\min(n,w+1-p_0),\qquad
S'(w)=\{j\in S(w)\mid\ell\le j<h\}.
\]

若 tile 内 **`p_j=p_0+j`**，则 `S'=S`，而且原 mapping 和位置判断仍保留。这个前提足以证明候选集合等价；不同 state/cache 物理布局不会改变证明。只有“单调递增”并不足够，缺口会破坏 `position−p0=physical offset` 的反解。

原 agent 的 [v1](../../../benchmarks/implementation/deepseek-v4.1-flash/deepseek_v4_flash_dspark/prefill_compressor_ratio4_v1.py) 第 236–256 行也混合使用物理邻近窗口 `tid-15` 与 position mask；它不是已经证明支持任意稀疏 position 的独立 oracle。此次应以 unchanged canonical golden 比较原 manual 与 scan 候选，不能以“agent 也失败/通过”替代契约判定。

## 3. 一致的 metadata 构造公式

使用单 request：`query_start_loc=[0,n]`，无 padding。所有位置非负、递增且 `<MAX_SEQ_LEN`。本轮下面的具体输入只用 0–12。

沿用 canonical fixture 的 state table（第 731–743 行），state page 大小为 2、每 request 有 8 个 state blocks（[config](../../../models/deepseek_v4_flash_dspark/config.py)，第 255、268–273 行）：

\[
B_s[q]=(17q+3)\bmod8,\qquad
M_s(p)=2B_s[\lfloor p/2\rfloor]+p\bmod2.
\]

表按 canonical shape `[1,CSA_STATE_MAX_BLOCKS]` 填充；state tensor 沿用 canonical 容量，不缩小。`state_slot_mapping[j]=M_s(p_j)`。所有这些访问均在 physical state 范围内。

沿用 standalone compressed-cache 的 identity page 布局（第 796–804 行）：

\[
M_c(p)=\begin{cases}
\lfloor p/4\rfloor,&(p+1)\bmod4=0,\\
-1,&\text{otherwise}.
\end{cases}
\]

这是 canonical initializer 的 `(p+1)//4-1` 在合法边界上的等价式。一般 paging 公式见 [utils](../../../models/deepseek_v4_flash_dspark/utils.py) 第 325–342、369–383 行：state/cache 的 logical page 分别查自己的 table，再乘对应 page 大小加 intra-page offset；table 的 `-1` 产生未映射位置。不要将非边界 token 随意标成 compressed write，也不要生成超过 `MAX_CMP_WRITES=128` 个输出、越界地址或有冲突的并行写作为反例。

RoPE 应由同一 `p_j` 派生：boundary 时 compressed RoPE position 为 `p_j-3`，其余为 0，继续使用 canonical `token_local_rope` 与 dtype（第 777–794 行）。历史 state 从相同 start/paging 生成；本诊断 start=0，没有历史前缀，canonical 零初始化即可。新测试输入和 golden 必须写入新的外部实验目录，不能改写已冻结数据。

## 4. 有序缺口诊断：可精确定位一条漏项

建议保持与小控制组相同的 `n=12`，只改变第一个 request 内的位置序列，并联动重建 metadata：

```text
position_ids       = [0,1,2,4,5,6,7,8,9,10,11,12]
state_slot_mapping = [6,7,8,10,11,12,13,14,15,0,1,2]
cmp_slot_mapping   = [-1,-1,-1,-1,-1,-1,1,-1,-1,-1,2,-1]
query_start_loc    = [0,12]
```

这是一个严格递增、只缺 position 3 的小输入。两个 compressed writes 均在真实 ratio-4 边界，目标 cache row 分别为 1、2；12 个 state rows 无重复，相关历史/当前窗口所需 state page 均映射且在范围内，write 数量为 2≤128。

对 position **11** 的输出，合法位置窗口是 `[4,11]`。完整扫描得到物理下标 `[3,4,5,6,7,8,9,10]`；候选算出 `lo=4, hi=12`，加原判断后得到 `[4,5,6,7,8,9,10]`，**漏掉物理下标 3 的 position 4**。这一集合差异由源码静态证明；尚没有本报告新采集的设备失败。

为避免随机投影偶然隐藏差异，可另生成一份有限、可精确表示的 BF16 哨兵输入，保持原参考数学与比较器不动：

- `x[:,0]=1`；仅物理下标 3 令 `x[3,1]=1`，其余输入特征为 0。
- `wkv[:,0]=1`，另令 `wkv[0,1]=8`；其余权重为 0。这样普通 token 的 1024 个投影分量均为 1，被漏 token 的前半第 0 分量为 9。
- `wgate=0`、`ape=0`、`norm_w=1`、初始 state/cache 为 0；RoPE 继续由真实 positions 生成。

对 `w=11`，原扫描的 8 个窗口位置都由当前 tile 覆盖，score 均为 0，均匀池化得到 `[2,1,…,1]`。候选遗漏 position 4，该 slot 留下零初始化 state，score 仍为 0，得到 `[7/8,…,7/8]`。在 NOPE channel 0，RMSNorm 后、BF16 cast 前分别约 **1.9941653 与 0.99999935**；RoPE 不触及这个 channel。canonical `cmp_kv` 比较器是 `atol=1e-4, rtol=1/128, max_error_ratio=0`（原文件第 896–899 行；[比较器实现](../../../golden/validation.py) 第 424–437 行），这个数学差异远大于容差，且不会被巨大 cache tensor 的稀释比例放过。

这些权重值和位置值没有违反已找到的明确数值约束，但**这仍不能补上业务域的缺口**。若 owner 将一般 operator 限定为 request 内无缺口的连续 prefill，那么该实验只说明候选需要这个前提，不能记为一般合法输入上的失败。若 owner 明确允许此类稀疏位置，则它才成为通用契约的有效反例。此次不创设任何 owner policy。

## 5. 最小验证矩阵与判决门槛

建议先用相同 K256 的 source pair 隔离扫描因素：未改扫描的 SHA **`c7ea6e9240a405e8…`**（[既有候选](../../../build_output/pfdb_development/optimization-pilot-v1/c4_manual-0-baseline/attempt-0/candidate.py)）与 scan+K256 的 **`6fded29273bcc107…`**。原 manual 可作为额外基线。所有版本调用相同 unchanged canonical golden 和原比较器；不复用旧位置对应的 expected 输出。

| 用例 | 新输入构造 | 能回答什么 | 采用口径 |
| --- | --- | --- | --- |
| C0：小连续控制 | `n=12, p=arange(0,12)`；canonical table、mapping、RoPE；哨兵或冻结随机数 | 短尾 shape、harness、oracle、基础数学是否可用 | 两版本均应先通过；不通过则停止解释 gap 诊断 |
| C1：连续 + paging 重排 | 同 C0；对 8 个 state physical block 应用双射 `π(b)=(b+1)%8`，table、mapping、初始 state 同步重排；cache 目标也可按独立双射重排 | 排除“物理 mapping 不连续就是不等价”的误判 | 使用 existing paging 关系；不改变 position 域；两版本均应通过 |
| C2：非零 start + 历史控制 | `n=12, p=arange(4,16)`；canonical builder 的 start=4 历史、paging 与 RoPE | 检查 `p0≠0` 和历史 state，不混入缺口 | 原 initializer 明确支持的配置；两版本均应通过 |
| D0：有序缺口诊断 | 本文 `p=[0,1,2,4,…,12]`，一致 metadata，哨兵数据 | 直接检验一次 position→physical-offset 漏项 | 单独标记 oracle-defined diagnostic；业务合法域未定，不纳入采用或通用正确率 |

用 C1 时 cache 重排不是必需，避免一次引入过多变量即可先只重排 state；非负 mapping 必须与自己的 table 和容量一致。mapping 为 `-1` 的缺页可另做覆盖，但不是此次扫描推导的独立反例。

判决依次为：**输入关系与访问范围检查 → unchanged golden 生成且有限 → 未改扫描版本对 golden 通过 → 才比较 scan 候选**。保留 source/reference/input SHA、完整 mapping 与 positions、输出维度、比较器、具体失败坐标；区分编译、运行、数值、oracle 范围和业务域未定。若 D0 的未改扫描版本不通过，不能把后续差异直接归因于 scan 变换。多 request、query padding、乱序/重复位置、输出槽冲突都留在本轮之外。

本轮控制问题应区分：“在当前指定 fixture 上优化”与“保持一般 operator 输入域正确”。前者可以解释为什么连续 fixture 上的特化有性能价值；后者还缺权威输入域条款。**profiling 数据能指出昂贵扫描，不能替代这一语义裁决。**

## 6. 审计锚点

本次读取时的 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| canonical manual `models/deepseek_v4_flash_dspark/prefill_compressor_ratio4.py` | `d2cca4d12fac58b50a61262cd8104f04435eb6cdcaf56ea63e101174afafafd2` |
| original agent `benchmarks/implementation/deepseek-v4.1-flash/deepseek_v4_flash_dspark/prefill_compressor_ratio4_v1.py` | `4c7151590d7406116601949bc0604aaac4ee519c078a5c905b89f709a08196ba` |
| scan+K256 candidate | `6fded29273bcc107963a853cbcdefacb62d9e299866928388c55df891921db2e` |

历史性能结论与扫描前提另见 [compressor 案例报告](../../../benchmarks/implementation/deepseek-v4.1-flash/compressor-ratio4-performance-gap-study.md) §5.1、[公开决策复核](online_decision_review.md) §4。历史测量不构成本报告新输入上的正确性或性能验证。
