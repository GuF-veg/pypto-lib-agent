# 扫描范围特化的四次设备诊断复核

日期：2026-10-22。四个登记验证已完成：**K256 parent 在 affine 和 gap 两个输入上均通过；scanK256 在 affine 上通过，在 gap 上仅 `cmp_kv` 验证失败。** 这是同 K 配置、同参考数学、同比较器的扫描范围对照。结果支持“该扫描特化依赖位置与物理下标的连续对应关系”，但**跳过位置是否属于一般业务输入域仍未得到权威裁定**；不据此给出通用契约、采用或编译器缺陷结论。

本报告只读取四次执行已有工件；未运行模型、编译器或 NPU，未重试、提取设备输出或修改冻结 helper。四次设备预算已用完。没有 benchmark、level-4 或 PMU 采集，执行进程耗时不作算子性能比较。

## 1. 对照和身份

证据根目录为 [contract](../../../build_output/pfdb_development/followup-v1/contract/)，[protocol.json](../../../build_output/pfdb_development/followup-v1/contract/protocol.json) 的 SHA-256 为 `170509ed7f8ea760d32a689f525f0d5ec99864a724bba57c2c0dc4079c50af1c`。协议固定 a2a3、逻辑 device 0、swimlane/PMU=0、benchmark=false、四次 one-shot、无重试。

| 身份 | SHA-256 |
| --- | --- |
| parent：K256、原扫描 | `c7ea6e9240a405e8d3ae1f011c2767d9bbc3515124c1da16ffd268619c01d20a` |
| scan：K256、缩窄扫描 | `6fded29273bcc107963a853cbcdefacb62d9e299866928388c55df891921db2e` |
| unchanged canonical reference | `d2cca4d12fac58b50a61262cd8104f04435eb6cdcaf56ea63e101174afafafd2` |
| affine 输入和 expected 输出 tree | `a99c79e7d02e1742fe70f658edaf56463c64375064080a942e17e4b20ec94314` |
| gap 输入和新 expected 输出 tree | `cfb777b87304781b9e91c244cc17f76ef86c3566c421f6fdb41bb90e41a46e69` |

直接比较四个私有 `source.py`，各自 SHA 与登记值一致；比较 [parent 源码](../../../build_output/pfdb_development/followup-v1/contract/jobs/parent-affine/source.py) 与 [scan 源码](../../../build_output/pfdb_development/followup-v1/contract/jobs/scan-affine/source.py)，**唯一 diff** 是把 `for pool_local in pl.range(tile_rows)` 改为读取 `tile_pos0`、计算 `pool_lo/pool_hi`、扫描 `[pool_lo,pool_hi)`。两者均为 K256，原 overlay body、mapping 判断、投影、state publish、RMS/RoPE 没有差异。

两种 fixture 都是单 request、512 个物理 token、`query_start_loc=[0,512]`。affine 为旧冻结数据的完整复制；gap 的 `p=arange(512)`、`p[3:]+=1`，即缺 position 3，最大 position 为 512。两份输入按字节比较，只有 **position、state slot、compressed slot、cos、sin 五个输入文件改变**；x、权重、APE、norm、初始 state/cache、block table 和 query starts 均保留。gap 的 mapping/RoPE 按新 positions 一致重建，两个 expected 输出由 unchanged canonical golden 重新生成，保存于新目录，未沿用旧 positions 的 expected 输出。

四次日志均显示 `runtime_only`、`.pto` 没有变化、复用缓存二进制，以及输入/expected 输出 cache hit。四份 `result.json` 的 `build_frozen` 与 `build_after` 对比，每次登记的 **53 个编译元数据/生成物文件均无内容变化**；每次执行都有自己的私有 replay build。运行驱动在执行前后验证共同 immutable 文件；结果文件已保存原始输入、两个新 fixture、reference、比较器实现和 helper/source 的 SHA 锚点。此次诊断未覆盖或改写原捕获目录。

## 2. 四个完整结果

比较器保持原设置：`compress_state` 为 `atol=1e-3, rtol=1e-3, max_error_ratio=0`；`cmp_kv` 为 `atol=1e-4, rtol=1/128, max_error_ratio=0`。两者都不忽略 NaN，不裁剪有效行。

| 执行结果 | compress_state | cmp_kv | 进程退出码 |
| --- | --- | --- | --- |
| [parent-affine](../../../build_output/pfdb_development/followup-v1/contract/jobs/parent-affine/validation/result.json) | PASS | PASS | 0 |
| [parent-gap](../../../build_output/pfdb_development/followup-v1/contract/jobs/parent-gap/validation/result.json) | PASS | PASS | 0 |
| [scan-affine](../../../build_output/pfdb_development/followup-v1/contract/jobs/scan-affine/validation/result.json) | PASS | PASS | 0 |
| [scan-gap](../../../build_output/pfdb_development/followup-v1/contract/jobs/scan-gap/validation/result.json) | PASS | FAIL | 1 |

scan-gap 的退出码 1 对应已经完成 runtime 后的 `cmp_kv` 数值验证失败，完整错误保存在结果 JSON；不是一次未获得结果的实验。其 [stdout](../../../build_output/pfdb_development/followup-v1/contract/jobs/scan-gap/validation/stdout.log) 第 13–16 行同时保留 state PASS、cache FAIL 和 `RESEARCH_RESULT`。

## 3. 已保留的数值差异和分布限制

scan-gap 的 `cmp_kv` shape 为 `(8192,32,1,512)`，共 **134,217,728** 个元素。现有错误记录：

- 超容差元素 **63,501 / 134,217,728**，全 tensor 比例 **0.0473%**。
- 允许比例为 **0**，允许超容差元素数为 **0**；因此大 cache 中的较小全局比例仍明确判失败。
- 最大绝对误差 **2.07422**，位置 **`(1,14,0,218)`**，该点容差 **0.0104149**。
- 第一个超容差 flat index 为 **1024**，对应 **`(0,2,0,0)`**，即 flattened compressed row 2、channel 0。实际值 **−0.984375**，expected **−0.9296875**，绝对差 **0.0546875**，容差约 **0.007363**。
- 前十个记录为 flat index **1024–1033**，即同一 compressed row 的前十个 channel；不是只在 RoPE 尾部出现的误差。

gap fixture 的 compressed writes 只有 127 行，对应 65,024 个输出元素，远小于完整 cache；全 tensor 的比例受到大量未写 cache 行影响。**现有日志只提供总计、最大误差和前十个超差点，没有完整设备输出，不能恢复每一行/每个 channel 的误差直方图，也不能从前十个点推断全部错误分布。** 本报告没有为此重跑。

## 4. 机制证据与适用边界

对 position 11 的输出，原逻辑应纳入位置 `[4,11]`。在本次 gap 输入中，它们的物理下标是 `[3,4,5,6,7,8,9,10]`；scan 使用 `position−tile_pos0` 作为物理下标反解，算出 `pool_lo=4`、`pool_hi=12`，保留原 position 判断后只扫描到 `[4,5,6,7,8,9,10]`，漏掉物理下标 3 的 position 4。对应的 compressed row 为 `11//4=2`，与**首个保留错误位置**相符。

此处有四层相互支持的证据：同 K256 的唯一源码 diff、只改变一致 metadata 的输入、parent 对新 golden 的 PASS、scan 对同一新 golden 的数值 FAIL。由此可以把**本 fixture 上的数值差异归到扫描范围特化**，并把连续 position→physical-offset 前提加入候选说明。没有中间 pool tensor 或完整设备输出，不能进一步报告每个误差点的传播细节。

一般域判断仍受 [契约审计](contract_followup_audit.md) 的限制：default initializer 生成 affine 位置；公开源码和文档没有明确裁定 skipped-position 的一般业务合法性；standalone canonical golden 也没有实现 request-aware 的完整 oracle。这里已限定一个 request，避免把该 oracle 的 request 边界缺口混进扫描对照。**parent 在这个 gap 例子通过，并不会自动赋予所有稀疏 position 业务合法性；scan 在这里失败，也不自动构成通用契约或编译器 bug。**

供 pfdb/agent 反馈的具体信息应是：候选在 affine fixture 已验证；同 K256 的新单 request 缺口诊断失败于 `cmp_kv`；扫描优化需要位置连续对应关系；一般输入域对这个前提的授权未知。原始 profiling 能指出昂贵扫描，这次执行补上的是**特化前提和对应验证覆盖**，并未证明增加某类信息后 agent 的搜索能力改善。
