# 模型算子调用依赖

本文记录 `models/` 下各模型 PyPTO 算子之间的调用依赖关系，并按模型列出**基本算子**与**非基本算子**，同时标注每个算子的 **golden 校验状态**（是否被 torch 参考实现校验、由哪个 golden 函数校验）。

**算子判定**：凡被 `@pl.jit` 系列装饰器（`pl.jit`、`pl.jit.inline`、`pl.jit.host`、`pl.jit.incore`、`pl.jit.extern`）标记的函数，以及由「工厂函数」创建的 jit 闭包实例（见下方 `*` 标记），均记为一个 PyPTO 算子。

**基本算子**：函数体内不调用任何其它 PyPTO 算子的算子（叶子算子，只调用 `pl.*` 语言原语）。

**非基本算子**：函数体内调用了至少一个其它 PyPTO 算子的算子。

**标记说明**（算子名后）：

- 无标记 — `@pl.jit.inline` 内联算子；`[jit]` — `@pl.jit` 设备侧入口；`[host]` — `@pl.jit.host` host 侧入口；`[incore]` — `@pl.jit.incore`；`[extern]` — `@pl.jit.extern` 外部（CANN）内核
- `name`*（make_xxx）— 由工厂函数 `make_xxx` 返回的 jit 闭包实例；被其它算子调用但未在模块级命名的匿名实例，以 `工厂(区分参数)→闭包` 形式出现
- 文件名后的 ⚠ 表示 `_draft.py` 草稿文件（未纳入可运行集）

**golden 状态标记**（紧随类型标记之后，每个算子都标注）：

| 标记 | 状态 | 含义 |
| --- | --- | --- |
| `⊕` | 独立 golden 函数 | 该算子是某个 `run(fn=…, golden_fn=<torch 参考>)` 的被测入口，拥有专属 torch 参考直接对比 |
| `◉` | 专属入口直连 golden | 基本算子是某个 `⊕` 入口**唯一**调用的 PyPTO 算子：该入口的 golden 偏差可直接归因到它（除 `pl.*` 原语外无其它算子混入） |
| `⊖` | 数据回放 | 直接校验但接线为 `golden_data=<目录>`，库内没有现成 golden 函数 |
| `○` | 传递覆盖 | 被某个 `⊕` 算子直接或间接调用，输出汇入其 golden 对比 |
| `T` | 同体孪生已校验 | 与某已校验算子包装同一函数体（`X` ↔ `X_test`），代码等同 |
| `H` | host 入口 | `@pl.jit.host` serving 参数适配封装，设计上不单独校验 |
| `P` | perf-only | 被 `run` 运行但 `golden_fn=None`，不做数值校验 |
| `✗` | 未覆盖 | 以上皆非 |

判定优先级：`⊕`/`⊖`（直接接线）> `P` > `H` > `T` > `◉`/`○` > `✗`；`◉` 是 `○` 的细分，不改变覆盖统计。golden 接线由静态解析各文件 `__main__` / `main()` / `validate()` 区域内的 `run(fn=…, golden_fn=…)` 得到，跨文件辅助封装（如 `run_attention(args, program, specs, golden_fn)`）的形参按调用点实际传参展开。

> 本文档由静态 AST 分析生成（分析 `models/` 全部 Python 源码中的算子定义与调用点，宿主侧 golden 测试驱动代码如 `run(fn=...)` 不计为算子间调用）。源码变动后需重新生成。

## 总览

全仓库 824 个算子中，690 个进入 golden 校验链路（`⊕`/`⊖`/`◉`/`○`/`T`）；其中**拥有独立 golden 函数的基本算子 64 个**，按模型汇总在下方各模型的「有独立 golden 函数的基本算子」小节。

| 模型 | 算子总数 | 基本算子 | 非基本算子 | golden 覆盖 | 有独立 golden 函数的基本算子 |
| --- | --- | --- | --- | --- | --- |
| [qwen3_14b](qwen3_14b/index.md) | 47 | 21 | 26 | 28/47 | 3 |
| [deepseek_v4_flash_mtp](deepseek_v4_flash_mtp/index.md) | 160 | 74 | 86 | 141/160 | 15 |
| [deepseek_v4_flash_dspark](deepseek_v4_flash_dspark/index.md) | 228 | 80 | 148 | 204/228 | 13 |
| [deepseek_v4_pro](deepseek_v4_pro/index.md) | 101 | 40 | 61 | 90/101 | 15 |
| [deepseek_v4_1_flash](deepseek_v4_1_flash/index.md) | 230 | 75 | 155 | 201/230 | 6 |
| [glm5_3_flash](glm5_3_flash/index.md) | 58 | 44 | 14 | 26/58 | 12 |
| 合计 | 824 | 334 | 490 | 690/824 | 64 |

## qwen3_14b

Qwen3-14B BF16 prefill 与 decode，附带 A8W8 / TurboQuant 量化变体与采样组件（serving 支持）。

共 47 个算子：基本算子 21 个，非基本算子 26 个。golden 覆盖 28/47（`⊕` 9、`◉` 2、`○` 17）；host 6、未覆盖 13。其中**有独立 golden 函数的基本算子 3 个**（见下方同名小节）。

顶层入口（未被其它算子调用，共 15 个）：

- `contract.py`：`qwen3_prefill_host`, `qwen3_decode_host`, `qwen3_greedy_sample_host`
- `decode_fwd.py`：`decode_fwd_layers`, `qwen3_decode_host`
- `decode_layer_a8w8.py`：`decode_fwd`
- `decode_tq_draft.py`：`decode_fwd_tq`
- `paged_attention_cce.py`：`qwen_decode_attention_cce`, `qwen_decode_attention_cache_offset_test`
- `prefill_fwd.py`：`qwen3_prefill_host`
- `prefill_fwd_a8w8.py`：`prefill_hidden_a8w8`
- `prefill_tq_draft.py`：`prefill_fwd_tq`
- `rope_qkv_regen.py`：`rope_qkv_regen`
- `test_paged_attention_pypto.py`：`paged_attention_pypto_dynamic`
- `topk_select.py`：`qwen3_topk_select_host`

### 基本算子（21 个）

算子名后第二组标记为 golden 状态（图例见文首）。

| 文件 | 基本算子 |
| --- | --- |
| `decode_fwd.py` | `_token_embed_inline`✗, `_greedy_sample_inline`✗ |
| `decode_layer_a8w8.py` | `_decode_layer`✗ |
| `greedy_sample.py` | `greedy_sample_fwd`[jit]⊕ |
| `paged_attention_cce.py` | `paged_attention_cce`[extern]○, `paged_attention_rope_cce`[extern]✗, `paged_attention_tiling_cce`[extern]○ |
| `paged_attention_pypto.py` | `paged_attention_pypto_swpipe`◉ |
| `prefill_fwd.py` | `_attention_phase_window`○, `_attention_phase_window_full_single_block`○, `_rope_kv_cache_phase`○, `_out_proj_aic_phase`○ |
| `prefill_fwd_a8w8.py` | `prefill_layer`✗ |
| `rms_lm_head.py` | `rms_lm_head`○, `rms_lm_head_fp32`✗, `rms_only`✗, `rms_lm_head_single_chunk`✗ |
| `rope_qkv_regen.py` | `rope_qkv_regen`[jit]✗ |
| `topk_select.py` | `_topk_group_pairs`◉ |
| `turboquant_kv.py` | `turboquant_kv_dequant_chunk`○, `turboquant_kv_quantize`○ |

### 有独立 golden 函数的基本算子（3 个）

这些叶子算子由专属 torch 参考对比：`⊕` 表示该算子本身就是被测入口，`◉` 表示它经由唯一调用它的测试入口对比。

| 文件 | 基本算子 | golden 函数 | 接线 |
| --- | --- | --- | --- |
| `greedy_sample.py` | `greedy_sample_fwd` | `golden_greedy_sample` | 自身接线 |
| `paged_attention_pypto.py` | `paged_attention_pypto_swpipe` | `golden_attention` | 经 `paged_attention_pypto_dynamic` |
| `topk_select.py` | `_topk_group_pairs` | `golden_topk_select` | 经 `topk_select_fwd` |

### 非基本算子（26 个）

| 文件 | 算子 | 调用的算子 | golden |
| --- | --- | --- | --- |
| `contract.py` | `qwen3_prefill_host`[host]H | `prefill_fwd` | H host 入口不单独校验 |
| `contract.py` | `qwen3_decode_host`[host]H | `decode_fwd（decode_fwd.py）` | H host 入口不单独校验 |
| `contract.py` | `qwen3_greedy_sample_host`[host]H | `greedy_sample_fwd` | H host 入口不单独校验 |
| `decode_fwd.py` | `_run_paged_attention`○ | `paged_attention_pypto_swpipe` | ○ 传递覆盖 |
| `decode_fwd.py` | `_decode_layer`○ | `_run_paged_attention` | ○ 传递覆盖 |
| `decode_fwd.py` | `_decode_fwd_body`✗ | `_decode_layer（decode_fwd.py）`, `_greedy_sample_inline`, `_token_embed_inline`, `rms_lm_head_fp32` | ✗ 未覆盖 |
| `decode_fwd.py` | `decode_fwd`[jit]✗ | `_decode_fwd_body` | ✗ 未覆盖 |
| `decode_fwd.py` | `_decode_fwd_layers_body`○ | `_decode_layer（decode_fwd.py）` | ○ 传递覆盖 |
| `decode_fwd.py` | `decode_fwd_layers`[jit]⊕ | `_decode_fwd_layers_body` | ⊕ 独立 golden `golden_decode_layer` |
| `decode_fwd.py` | `qwen3_decode_host`[host]H | `decode_fwd（decode_fwd.py）` | H host 入口不单独校验 |
| `decode_layer_a8w8.py` | `decode_fwd`[jit]✗ | `_decode_layer（decode_layer_a8w8.py）`, `rms_lm_head` | ✗ 未覆盖 |
| `decode_tq_draft.py` ⚠ | `decode_layer_tq`○ | `turboquant_kv_dequant_chunk`, `turboquant_kv_quantize` | ○ 传递覆盖 |
| `decode_tq_draft.py` ⚠ | `decode_fwd_tq`[jit]⊕ | `decode_layer_tq`, `rms_lm_head` | ⊕ 独立 golden `golden_decode_fwd_tq` |
| `paged_attention_cce.py` | `build_paged_attention_metadata`○ | `paged_attention_tiling_cce` | ○ 传递覆盖 |
| `paged_attention_cce.py` | `qwen_decode_attention_cce`[jit]⊕ | `build_paged_attention_metadata`, `paged_attention_cce` | ⊕ 独立 golden `golden_attention` |
| `paged_attention_cce.py` | `qwen_decode_attention_cache_offset_test`[jit]⊕ | `build_paged_attention_metadata`, `paged_attention_cce` | ⊕ 独立 golden `golden_attention` |
| `prefill_fwd.py` | `_manual_rope_kv_cache_phase`○ | `_rope_kv_cache_phase` | ○ 传递覆盖 |
| `prefill_fwd.py` | `prefill_layer`○ | `_attention_phase_window`, `_attention_phase_window_full_single_block`, `_manual_rope_kv_cache_phase`, `_out_proj_aic_phase` | ○ 传递覆盖 |
| `prefill_fwd.py` | `prefill_fwd`[jit]⊕ | `prefill_layer（prefill_fwd.py）`, `rms_lm_head` | ⊕ 独立 golden `golden_qwen3_14b_prefill` |
| `prefill_fwd.py` | `qwen3_prefill_host`[host]H | `prefill_fwd` | H host 入口不单独校验 |
| `prefill_fwd_a8w8.py` | `prefill_hidden_a8w8`[jit]✗ | `prefill_layer（prefill_fwd_a8w8.py）` | ✗ 未覆盖 |
| `prefill_tq_draft.py` ⚠ | `prefill_layer_tq`○ | `turboquant_kv_dequant_chunk`, `turboquant_kv_quantize` | ○ 传递覆盖 |
| `prefill_tq_draft.py` ⚠ | `prefill_fwd_tq`[jit]⊕ | `prefill_layer_tq`, `rms_lm_head` | ⊕ 独立 golden `golden_qwen3_14b_prefill_tq` |
| `test_paged_attention_pypto.py` | `paged_attention_pypto_dynamic`[jit]⊕ | `paged_attention_pypto_swpipe` | ⊕ 独立 golden `golden_attention` |
| `topk_select.py` | `topk_select_fwd`[jit]⊕ | `_topk_group_pairs` | ⊕ 独立 golden `golden_topk_select` |
| `topk_select.py` | `qwen3_topk_select_host`[host]H | `topk_select_fwd` | H host 入口不单独校验 |

## deepseek_v4_flash_mtp

DeepSeek V4-Flash（MTP=1、每卡 batch 4）：单层算子、layer / MTP 组合以及 prefill / decode 全前向。

共 160 个算子：基本算子 74 个，非基本算子 86 个。golden 覆盖 141/160（`⊕` 38、`◉` 7、`○` 80、`T` 15）；数据回放 1、perf-only 2、未覆盖 17。其中**有独立 golden 函数的基本算子 15 个**（见下方同名小节）。

顶层入口（未被其它算子调用，共 41 个）：

- `decode_compressor_ratio128.py`：`compressor_test`
- `decode_compressor_ratio4.py`：`compressor_test`
- `decode_csa.py`：`attention_csa_test`
- `decode_fwd.py`：`l3_decode_fwd`
- `decode_fwd_mtp.py`：`l3_decode_fwd_mtp`
- `decode_hca.py`：`attention_hca_test`
- `decode_indexer.py`：`indexer_test`
- `decode_indexer_compressor.py`：`compressor_test`
- `decode_layer.py`：`l3_decode_layer`
- `decode_moe.py`：`l3_moe`
- `decode_mtp.py`：`l3_decode_mtp`
- `decode_sparse_attn_csa.py`：`sparse_attn_test`
- `decode_sparse_attn_hca.py`：`sparse_attn_test`
- `decode_sparse_attn_swa.py`：`sparse_attn_test`
- `decode_swa.py`：`attention_swa_test`
- `expert_routed.py`：`expert_routed_test`
- `expert_shared.py`：`expert_shared_test`
- `hc_head.py`：`hc_head_test`
- `hc_post.py`：`hc_post_test`
- `hc_pre.py`：`hc_pre_test`
- `lm_head.py`：`l3_lm_head`, `l3_lm_head_projection`
- `lookup_embedding.py`：`lookup_embedding_test`
- `mtp_projection.py`：`mtp_projection_test`
- `prefill_compressor_ratio128.py`：`prefill_compressor_ratio128_test`
- `prefill_compressor_ratio4.py`：`prefill_compressor_ratio4_test`
- `prefill_cp_exchange.py`：`prefill_cp_exchange_test`
- `prefill_cp_zigzag.py`：`prefill_cp_zigzag_kv_tail_exchange_test`
- `prefill_csa.py`：`prefill_cp_csa_test`
- `prefill_fwd.py`：`l3_prefill_fwd`
- `prefill_hca.py`：`prefill_cp_hca_test`
- `prefill_indexer.py`：`prefill_indexer_test`
- `prefill_indexer_compressor.py`：`prefill_indexer_compressor_test`
- `prefill_moe.py`：`l3_prefill_moe`
- `prefill_mtp.py`：`l3_mtp_prefill_fwd`
- `prefill_sparse_attn.py`：`prefill_sparse_attn_test`
- `prefill_swa.py`：`prefill_mtp_attention_swa_test`, `prefill_cp_swa_test`
- `qkv_proj_rope.py`：`qkv_proj_rope_test`
- `rmsnorm.py`：`rms_norm_test`
- `sample.py`：`sample_test`

### 基本算子（74 个）

算子名后第二组标记为 golden 状态（图例见文首）。

| 文件 | 基本算子 |
| --- | --- |
| `decode_compressor_ratio128.py` | `compressor_ratio128`○ |
| `decode_compressor_ratio4.py` | `compressor_ratio4`○ |
| `decode_fwd_mtp.py` | `prepare_decode_from_device_state`✗, `prepare_mtp_sampling_from_device_state`✗, `advance_decode_device_state`✗, `verify_and_pack_mtp_tokens`✗ |
| `decode_indexer_compressor.py` | `indexer_compressor`○ |
| `decode_moe.py` | `dispatch`○, `combine`○, `clear_moe_signals`○ |
| `decode_prepare.py` | `gather_swa_rope_rows`○, `gather_decode_rope_rows`○, `build_swa_metadata`○, `pack_x_hc`○, `pack_mtp_hidden`✗ |
| `decode_sparse_attn_csa.py` | `sparse_attn_csa`*（`包装 _sparse_attn_csa`）⊕, `sparse_attn_test`*[jit]（`包装 _sparse_attn_csa`）T |
| `decode_sparse_attn_hca.py` | `sparse_attn_hca`*（`包装 _sparse_attn_hca`）⊕, `sparse_attn_test`*[jit]（`包装 _sparse_attn_hca`）T |
| `decode_sparse_attn_swa.py` | `sparse_attn_swa`◉ |
| `expert_routed.py` | `expert_routed_tile`◉ |
| `expert_shared.py` | `expert_shared`*（`包装 _expert_shared`）⊕, `expert_shared_test`*[jit]（`包装 _expert_shared`）T |
| `hc_head.py` | `hc_head`*（`包装 _hc_head`）⊕, `hc_head_test`*[jit]（`包装 _hc_head`）T |
| `hc_post.py` | `hc_post`*（`包装 _hc_post`）⊕, `hc_post_test`*[jit]（`包装 _hc_post`）T, `hc_post_prefill`○ |
| `hc_pre.py` | `hc_pre`*（`包装 _hc_pre`）⊕, `hc_pre_test`*[jit]（`包装 _hc_pre`）T |
| `lm_head.py` | `lm_head`*（`包装 _lm_head`）◉, `lm_head_test`*[jit]（`包装 _lm_head`）○ |
| `lookup_embedding.py` | `lookup_embedding`*（`包装 _lookup_embedding`）⊕, `lookup_embedding_test`*[jit]（`包装 _lookup_embedding`）T |
| `mtp_projection.py` | `mtp_projection`*（`包装 _mtp_projection`）⊕, `mtp_projection_test`*[jit]（`包装 _mtp_projection`）T |
| `prefill_compressor_ratio128.py` | `prefill_compressor_ratio128`◉ |
| `prefill_compressor_ratio4.py` | `compressor_ratio4`◉ |
| `prefill_cp_exchange.py` | `_prefill_cp_request_header`○, `_prefill_cp_scatter_request`[incore]○, `_clear_prefill_cp_exchange_signals`○, `_prefill_cp_hidden_tail_exchange_wave`○, `_prefill_cp_hca_history_exchange`○, `_prefill_cp_hca_compact_exchange_commit_wave`○, `_prefill_cp_csa_compact_transport_wave`○, `_prefill_cp_csa_compact_finish_wave`○, `_prefill_cp_gather_hidden`○, `_prefill_cp_request_barrier`○ |
| `prefill_cp_zigzag.py` | `prefill_cp_zigzag_kv_tail_exchange_core`[incore]○ |
| `prefill_fwd.py` | `_fwd_attention_stage_barrier_from_completion`✗, `_fwd_attention_stage_barrier_from_x_attn`✗, `_fwd_prepare_moe_inputs`✗, `_fwd_restore_hidden_layout`✗, `_fwd_wait_previous_moe`✗, `_prefill_cp_metadata`✗ |
| `prefill_indexer.py` | `_cp_topk512_query`[incore]○ |
| `prefill_indexer_compressor.py` | `_prefill_indexer_compressor_with_completion`◉ |
| `prefill_moe.py` | `clear_prefill_moe_signals`○ |
| `prefill_sparse_attn.py` | `_staged_attn_prepare_rope`○, `_staged_sparse_wave`○, `_staged_attn_o_proj`○, `_hca_segment_heads`○ |
| `prefill_swa.py` | `_cp_swa_history_exchange`○, `_cp_swa_stage_sources`○ |
| `qkv_proj_rope.py` | `materialize_rope_rows`○, `rope_prepare`○, `q_proj_rope`○, `kv_proj_rope`○ |
| `rmsnorm.py` | `rms_norm`◉ |
| `rope_interleave.py` | `rope_interleave`○ |
| `sample.py` | `_counter_gumbel`○, `_greedy_sample_logits`○, `apply_temperature`○, `apply_top_k`○ |

### 有独立 golden 函数的基本算子（15 个）

这些叶子算子由专属 torch 参考对比：`⊕` 表示该算子本身就是被测入口，`◉` 表示它经由唯一调用它的测试入口对比。

| 文件 | 基本算子 | golden 函数 | 接线 |
| --- | --- | --- | --- |
| `decode_sparse_attn_csa.py` | `sparse_attn_csa` | `golden_sparse_attn` | 经 `sparse_attn_test` |
| `decode_sparse_attn_hca.py` | `sparse_attn_hca` | `golden_sparse_attn` | 经 `sparse_attn_test` |
| `decode_sparse_attn_swa.py` | `sparse_attn_swa` | `golden_sparse_attn` | 经 `sparse_attn_test` |
| `expert_routed.py` | `expert_routed_tile` | `golden_expert_routed` | 经 `expert_routed_test` |
| `expert_shared.py` | `expert_shared` | `golden_expert_shared` | 经 `expert_shared_test` |
| `hc_head.py` | `hc_head` | `golden_hc_head` | 经 `hc_head_test` |
| `hc_post.py` | `hc_post` | `golden_hc_post` | 经 `hc_post_test` |
| `hc_pre.py` | `hc_pre` | `golden_hc_pre` | 经 `hc_pre_test` |
| `lm_head.py` | `lm_head` | `golden_lm_head` | 经 `l3_lm_head_projection` |
| `lookup_embedding.py` | `lookup_embedding` | `golden_lookup_embedding_test` | 经 `lookup_embedding_test` |
| `mtp_projection.py` | `mtp_projection` | `golden_mtp_projection` | 经 `mtp_projection_test` |
| `prefill_compressor_ratio128.py` | `prefill_compressor_ratio128` | `golden_prefill_compressor_ratio128` | 经 `prefill_compressor_ratio128_test` |
| `prefill_compressor_ratio4.py` | `compressor_ratio4` | `golden_prefill_compressor_ratio4` | 经 `prefill_compressor_ratio4_test` |
| `prefill_indexer_compressor.py` | `_prefill_indexer_compressor_with_completion` | `golden_prefill_indexer_compressor` | 经 `prefill_indexer_compressor_test` |
| `rmsnorm.py` | `rms_norm` | `golden_rms_norm_test` | 经 `rms_norm_test` |

### 非基本算子（86 个）

| 文件 | 算子 | 调用的算子 | golden |
| --- | --- | --- | --- |
| `decode_compressor_ratio128.py` | `compressor_test`[jit]⊕ | `compressor_ratio128`, `rope_interleave` | ⊕ 独立 golden `golden_compressor` |
| `decode_compressor_ratio4.py` | `compressor_test`[jit]⊕ | `compressor_ratio4（decode_compressor_ratio4.py）`, `rope_interleave` | ⊕ 独立 golden `golden_compressor` |
| `decode_csa.py` | `attention_csa`*（`包装 _attention_csa`）⊕ | `compressor_ratio4（decode_compressor_ratio4.py）`, `indexer`, `qkv_proj_rope`, `rms_norm`, `rope_interleave`, `sparse_attn_csa`, `hc_post`, `hc_pre` | ⊕ 独立 golden `golden_attention_csa`（入口 `attention_csa_test`） |
| `decode_csa.py` | `attention_csa_test`*[jit]（`包装 _attention_csa`）T | `compressor_ratio4（decode_compressor_ratio4.py）`, `indexer`, `qkv_proj_rope`, `rms_norm`, `rope_interleave`, `sparse_attn_csa`, `hc_post`, `hc_pre` | T 同体孪生已校验（`golden_attention_csa`） |
| `decode_fwd.py` | `decode_fwd`*（`包装 _decode_fwd`）T | `clear_moe_signals`, `moe`, `build_decode_metadata`, `gather_decode_rope_rows`, `pack_x_hc`, `rms_norm`, `attention_csa`, `attention_hca`, `attention_swa`, `hc_head`, `lm_head_with_sampling` | T 同体孪生已校验 |
| `decode_fwd.py` | `l2_decode_fwd`*[jit]（`包装 _decode_fwd`）○ | `clear_moe_signals`, `moe`, `build_decode_metadata`, `gather_decode_rope_rows`, `pack_x_hc`, `rms_norm`, `attention_csa`, `attention_hca`, `attention_swa`, `hc_head`, `lm_head_with_sampling` | ○ 传递覆盖 |
| `decode_fwd.py` | `l3_decode_fwd`[host]⊖ | `l2_decode_fwd` | ⊖ 数据回放（golden_data，无在库函数） |
| `decode_fwd_mtp.py` | `l2_decode_fwd_mtp`[jit]✗ | `advance_decode_device_state`, `prepare_decode_from_device_state`, `prepare_mtp_sampling_from_device_state`, `verify_and_pack_mtp_tokens`, `build_swa_metadata`, `pack_mtp_hidden`, `decode_fwd`, `decode_mtp`, `lookup_embedding` | ✗ 未覆盖 |
| `decode_fwd_mtp.py` | `l3_decode_fwd_mtp`[host]P | `l2_decode_fwd_mtp` | P 运行但 golden_fn=None |
| `decode_hca.py` | `attention_hca`*（`包装 _attention_hca`）⊕ | `compressor_ratio128`, `qkv_proj_rope`, `rms_norm`, `rope_interleave`, `sparse_attn_hca`, `hc_post`, `hc_pre` | ⊕ 独立 golden `golden_attention_hca`（入口 `attention_hca_test`） |
| `decode_hca.py` | `attention_hca_test`*[jit]（`包装 _attention_hca`）T | `compressor_ratio128`, `qkv_proj_rope`, `rms_norm`, `rope_interleave`, `sparse_attn_hca`, `hc_post`, `hc_pre` | T 同体孪生已校验（`golden_attention_hca`） |
| `decode_indexer.py` | `indexer`○ | `indexer_compressor` | ○ 传递覆盖 |
| `decode_indexer.py` | `indexer_test`[jit]⊕ | `indexer`, `rope_interleave` | ⊕ 独立 golden `golden_indexer` |
| `decode_indexer_compressor.py` | `compressor_test`[jit]⊕ | `indexer_compressor`, `rope_interleave` | ⊕ 独立 golden `golden_compressor` |
| `decode_layer.py` | `decode_layer`[jit]○ | `clear_moe_signals`, `moe`, `attention_csa`, `attention_hca`, `attention_swa` | ○ 传递覆盖 |
| `decode_layer.py` | `l3_decode_layer`[host]⊕ | `decode_layer` | ⊕ 独立 golden `golden_decode_layer_auto` |
| `decode_moe.py` | `expert_routed_scatter`○ | `expert_routed_tile` | ○ 传递覆盖 |
| `decode_moe.py` | `moe`○ | `combine`, `dispatch`, `expert_routed_scatter`, `expert_shared`, `hc_post`, `hc_pre` | ○ 传递覆盖 |
| `decode_moe.py` | `l2_moe`[jit]○ | `clear_moe_signals`, `moe` | ○ 传递覆盖 |
| `decode_moe.py` | `l3_moe`[host]⊕ | `l2_moe` | ⊕ 独立 golden `golden_moe` |
| `decode_mtp.py` | `decode_mtp`*（`包装 _decode_mtp`）T | `clear_moe_signals`, `moe`, `gather_swa_rope_rows`, `rms_norm`, `attention_swa`, `hc_head`, `lm_head_with_sampling`, `mtp_projection` | T 同体孪生已校验 |
| `decode_mtp.py` | `l2_decode_mtp`*[jit]（`包装 _decode_mtp`）○ | `clear_moe_signals`, `moe`, `gather_swa_rope_rows`, `rms_norm`, `attention_swa`, `hc_head`, `lm_head_with_sampling`, `mtp_projection` | ○ 传递覆盖 |
| `decode_mtp.py` | `l3_decode_mtp`[host]⊕ | `l2_decode_mtp` | ⊕ 独立 golden `golden_decode_mtp` |
| `decode_prepare.py` | `build_decode_metadata`○ | `build_swa_metadata` | ○ 传递覆盖 |
| `decode_sparse_attn_swa.py` | `sparse_attn_test`[jit]⊕ | `sparse_attn_swa` | ⊕ 独立 golden `golden_sparse_attn` |
| `decode_swa.py` | `attention_swa`*（`包装 _attention_swa`）⊕ | `sparse_attn_swa`, `qkv_proj_rope`, `rms_norm`, `hc_post`, `hc_pre` | ⊕ 独立 golden `golden_attention_swa`（入口 `attention_swa_test`） |
| `decode_swa.py` | `attention_swa_test`*[jit]（`包装 _attention_swa`）T | `sparse_attn_swa`, `qkv_proj_rope`, `rms_norm`, `hc_post`, `hc_pre` | T 同体孪生已校验（`golden_attention_swa`） |
| `expert_routed.py` | `expert_routed_test`[jit]⊕ | `expert_routed_tile` | ⊕ 独立 golden `golden_expert_routed` |
| `lm_head.py` | `lm_head_with_sampling`*（`包装 _lm_head_with_sampling`）○ | `sample`, `lm_head` | ○ 传递覆盖 |
| `lm_head.py` | `lm_head_with_sampling_test`*[jit]（`包装 _lm_head_with_sampling`）○ | `sample`, `lm_head` | ○ 传递覆盖 |
| `lm_head.py` | `l3_lm_head`[host]⊕ | `lm_head_with_sampling_test` | ⊕ 独立 golden `golden_lm_head` |
| `lm_head.py` | `l3_lm_head_projection`[host]⊕ | `lm_head_test` | ⊕ 独立 golden `golden_lm_head` |
| `prefill_compressor_ratio128.py` | `prefill_compressor_ratio128_test`[jit]⊕ | `prefill_compressor_ratio128` | ⊕ 独立 golden `golden_prefill_compressor_ratio128` |
| `prefill_compressor_ratio4.py` | `prefill_compressor_ratio4_test`[jit]⊕ | `compressor_ratio4（prefill_compressor_ratio4.py）` | ⊕ 独立 golden `golden_prefill_compressor_ratio4` |
| `prefill_cp_exchange.py` | `_prefill_cp_request_test`[jit]○ | `_prefill_cp_gather_hidden`, `_prefill_cp_request_barrier`, `_prefill_cp_request_header`, `_prefill_cp_scatter_request` | ○ 传递覆盖 |
| `prefill_cp_exchange.py` | `prefill_cp_exchange_test`[host]⊕ | `_prefill_cp_request_test` | ⊕ 独立 golden `golden_prefill_cp_exchange` |
| `prefill_cp_zigzag.py` | `prefill_cp_zigzag_kv_tail_exchange`[jit]○ | `prefill_cp_zigzag_kv_tail_exchange_core` | ○ 传递覆盖 |
| `prefill_cp_zigzag.py` | `prefill_cp_zigzag_kv_tail_exchange_test`[host]⊕ | `prefill_cp_zigzag_kv_tail_exchange` | ⊕ 独立 golden `golden_prefill_cp_zigzag_kv_tail_exchange` |
| `prefill_csa.py` | `_cp_csa_compress_pack_part`○ | `compressor_ratio4（prefill_compressor_ratio4.py）`, `_prefill_indexer_compressor_with_completion` | ○ 传递覆盖 |
| `prefill_csa.py` | `_prefill_cp_csa_history_exchange`○ | `_prefill_cp_csa_compact_finish_wave` | ○ 传递覆盖 |
| `prefill_csa.py` | `prefill_attention_csa`○ | `_prefill_cp_csa_compact_finish_wave`, `_prefill_cp_csa_compact_transport_wave`, `_prefill_cp_hidden_tail_exchange_wave`, `_cp_csa_compress_pack_part`, `_prefill_cp_csa_history_exchange`, `_prefill_indexer_cp_score_topk`, `prefill_physical_attention`, `kv_proj_rope`, `materialize_rope_rows`, `prefill_attention_prolog`, `rope_prepare` | ○ 传递覆盖 |
| `prefill_csa.py` | `prefill_cp_csa_rank`[jit]○ | `prefill_attention_csa` | ○ 传递覆盖 |
| `prefill_csa.py` | `prefill_cp_csa_test`[host]⊕ | `prefill_cp_csa_rank` | ⊕ 独立 golden `golden_prefill_cp_csa` |
| `prefill_fwd.py` | `cp_prefill_moe`*（`make_prefill_moe(layout=CP_MOE_LAYOUT)→prefill_moe`）✗ | `make_prefill_expert_grouped(grouped_capacity=layout.grouped_capacity)→prefill_expert_grouped`, `expert_shared`, `hc_post`, `hc_pre` | ✗ 未覆盖 |
| `prefill_fwd.py` | `_fwd_moe_tail`✗ | `cp_prefill_moe`, `_fwd_prepare_moe_inputs`, `_fwd_restore_hidden_layout` | ✗ 未覆盖 |
| `prefill_fwd.py` | `prefill_fwd`✗ | `_clear_prefill_cp_exchange_signals`, `prefill_attention_csa`, `_fwd_attention_stage_barrier_from_completion`, `_fwd_attention_stage_barrier_from_x_attn`, `_fwd_moe_tail`, `_fwd_wait_previous_moe`, `prefill_attention_hca`, `clear_prefill_moe_signals`, `prefill_attention_swa`, `rms_norm`, `hc_head` | ✗ 未覆盖 |
| `prefill_fwd.py` | `_prefill_request`[jit]✗ | `_prefill_cp_gather_hidden`, `_prefill_cp_request_barrier`, `_prefill_cp_request_header`, `_prefill_cp_scatter_request`, `_prefill_cp_metadata`, `prefill_fwd` | ✗ 未覆盖 |
| `prefill_fwd.py` | `l3_prefill_fwd`[host]P | `_prefill_request`, `lm_head_test` | P 运行但 golden_fn=None |
| `prefill_hca.py` | `prefill_attention_hca`○ | `hc_post_prefill`, `prefill_compressor_ratio128`, `_prefill_cp_hca_compact_exchange_commit_wave`, `_prefill_cp_hca_history_exchange`, `_prefill_cp_hidden_tail_exchange_wave`, `hca_attn`, `kv_proj_rope`, `materialize_rope_rows`, `prefill_attention_prolog`, `rope_prepare` | ○ 传递覆盖 |
| `prefill_hca.py` | `prefill_cp_hca_rank`[jit]○ | `prefill_attention_hca` | ○ 传递覆盖 |
| `prefill_hca.py` | `prefill_cp_hca_test`[host]⊕ | `prefill_cp_hca_rank` | ⊕ 独立 golden `golden_prefill_cp_hca` |
| `prefill_indexer.py` | `prefill_indexer`○ | `prefill_indexer_compressor` | ○ 传递覆盖 |
| `prefill_indexer.py` | `_prefill_indexer_cp_score_topk`○ | `_cp_topk512_query` | ○ 传递覆盖 |
| `prefill_indexer.py` | `prefill_indexer_test`[jit]⊕ | `prefill_indexer` | ⊕ 独立 golden `golden_prefill_indexer` |
| `prefill_indexer_compressor.py` | `prefill_indexer_compressor`○ | `_prefill_indexer_compressor_with_completion` | ○ 传递覆盖 |
| `prefill_indexer_compressor.py` | `prefill_indexer_compressor_test`[jit]⊕ | `_prefill_indexer_compressor_with_completion` | ⊕ 独立 golden `golden_prefill_indexer_compressor` |
| `prefill_moe.py` | `prefill_moe`*（`make_prefill_moe(layout=PREFILL_MOE_LAYOUT)→prefill_moe`）○ | `make_prefill_expert_grouped(grouped_capacity=layout.grouped_capacity)→prefill_expert_grouped`, `expert_shared`, `hc_post`, `hc_pre` | ○ 传递覆盖 |
| `prefill_moe.py` | `prefill_moe_test`[jit]○ | `prefill_moe（prefill_moe.py）`, `clear_prefill_moe_signals` | ○ 传递覆盖 |
| `prefill_moe.py` | `l3_prefill_moe`[host]⊕ | `prefill_moe_test` | ⊕ 独立 golden `golden_prefill_moe` |
| `prefill_mtp.py` | `prefill_moe`*（`make_prefill_moe(layout=PREFILL_MOE_LAYOUT)→prefill_moe`）✗ | `make_prefill_expert_grouped(grouped_capacity=layout.grouped_capacity)→prefill_expert_grouped`, `expert_shared`, `hc_post`, `hc_pre` | ✗ 未覆盖 |
| `prefill_mtp.py` | `mtp_prefill_fwd`[jit]○ | `prefill_moe（prefill_moe.py）`, `clear_prefill_moe_signals`, `rms_norm`, `hc_head`, `mtp_projection`, `prefill_mtp_attention_swa` | ○ 传递覆盖 |
| `prefill_mtp.py` | `l3_mtp_prefill_fwd`[host]⊕ | `mtp_prefill_fwd`, `lm_head_test` | ⊕ 独立 golden `golden_mtp_prefill_fwd` |
| `prefill_sparse_attn.py` | `prefill_sparse_attn_test`*[jit]（`包装 _sparse_attn`）T | `physical_sparse_attn` | T 同体孪生已校验（`golden_prefill_sparse_attn`） |
| `prefill_sparse_attn.py` | `sparse_attn`*（`包装 _sparse_attn`）⊕ | `physical_sparse_attn` | ⊕ 独立 golden `golden_prefill_sparse_attn`（入口 `prefill_sparse_attn_test`） |
| `prefill_sparse_attn.py` | `_staged_swa_heads`○ | `_staged_attn_prepare_rope`, `_staged_sparse_wave` | ○ 传递覆盖 |
| `prefill_sparse_attn.py` | `_physical_sparse_wave`○ | `_staged_sparse_wave` | ○ 传递覆盖 |
| `prefill_sparse_attn.py` | `_physical_sparse_heads`○ | `_physical_sparse_wave`, `_staged_attn_prepare_rope` | ○ 传递覆盖 |
| `prefill_sparse_attn.py` | `_hca_heads`○ | `_hca_segment_heads`, `_staged_attn_prepare_rope` | ○ 传递覆盖 |
| `prefill_sparse_attn.py` | `hca_attn`○ | `_hca_heads`, `_staged_attn_o_proj` | ○ 传递覆盖 |
| `prefill_sparse_attn.py` | `physical_sparse_attn`○ | `_physical_sparse_heads`, `_staged_attn_o_proj` | ○ 传递覆盖 |
| `prefill_sparse_attn.py` | `staged_sparse_attn`○ | `_staged_attn_o_proj`, `_staged_swa_heads` | ○ 传递覆盖 |
| `prefill_sparse_attn.py` | `prefill_staged_attention`○ | `hc_post_prefill`, `staged_sparse_attn` | ○ 传递覆盖 |
| `prefill_sparse_attn.py` | `prefill_physical_attention`○ | `hc_post_prefill`, `physical_sparse_attn` | ○ 传递覆盖 |
| `prefill_swa.py` | `prefill_mtp_attention_swa`*（`包装 _prefill_mtp_attention_swa`）⊕ | `prefill_staged_attention`, `kv_proj_rope`, `materialize_rope_rows`, `prefill_attention_prolog` | ⊕ 独立 golden `golden_prefill_attention_swa`（入口 `prefill_mtp_attention_swa_test`） |
| `prefill_swa.py` | `prefill_mtp_attention_swa_test`*[jit]（`包装 _prefill_mtp_attention_swa`）T | `prefill_staged_attention`, `kv_proj_rope`, `materialize_rope_rows`, `prefill_attention_prolog` | T 同体孪生已校验（`golden_prefill_attention_swa`） |
| `prefill_swa.py` | `prefill_attention_swa`○ | `_prefill_cp_hidden_tail_exchange_wave`, `prefill_staged_attention`, `_cp_swa_history_exchange`, `_cp_swa_stage_sources`, `kv_proj_rope`, `materialize_rope_rows`, `q_proj_rope`, `rope_prepare`, `rms_norm`, `hc_pre` | ○ 传递覆盖 |
| `prefill_swa.py` | `prefill_cp_swa_rank`[jit]○ | `_clear_prefill_cp_exchange_signals`, `prefill_attention_swa` | ○ 传递覆盖 |
| `prefill_swa.py` | `prefill_cp_swa_test`[host]⊕ | `prefill_cp_swa_rank` | ⊕ 独立 golden `golden_prefill_cp_swa` |
| `qkv_proj_rope.py` | `qkv_proj_rope`○ | `rope_prepare` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `prefill_attention_prolog`○ | `q_proj_rope`, `rope_prepare`, `rms_norm`, `hc_pre` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `qkv_proj_rope_test`[jit]⊕ | `qkv_proj_rope` | ⊕ 独立 golden `golden_qkv_proj_rope` |
| `rmsnorm.py` | `rms_norm_test`[jit]⊕ | `rms_norm` | ⊕ 独立 golden `golden_rms_norm_test` |
| `sample.py` | `_sample_filtered_logits`○ | `_counter_gumbel` | ○ 传递覆盖 |
| `sample.py` | `gumbel_sample`○ | `_greedy_sample_logits`, `_sample_filtered_logits` | ○ 传递覆盖 |
| `sample.py` | `sample`○ | `apply_temperature`, `apply_top_k`, `gumbel_sample` | ○ 传递覆盖 |
| `sample.py` | `sample_test`[jit]⊕ | `sample` | ⊕ 独立 golden `golden_sample` |

## deepseek_v4_flash_dspark

同一 V4-Flash 权重在每卡 batch 64、S=8 DSpark 投机解码下的 TP 分片 DSA-CP 注意力（开发中）。

共 228 个算子：基本算子 80 个，非基本算子 148 个。golden 覆盖 204/228（`⊕` 50、`◉` 8、`○` 130、`T` 16）；perf-only 2、未覆盖 22。其中**有独立 golden 函数的基本算子 13 个**（见下方同名小节）。

顶层入口（未被其它算子调用，共 54 个）：

- `decode_compressor_ratio128.py`：`compressor_test`
- `decode_compressor_ratio4.py`：`compressor_test`
- `decode_cp_allgather.py`：`l3_decode_cp_allgather_fixture`
- `decode_csa.py`：`decode_csa_tp1_test`, `l3_decode_csa`
- `decode_fwd.py`：`l3_decode_fwd`
- `decode_fwd_dspark.py`：`l3_decode_fwd_dspark`
- `decode_hca.py`：`decode_hca_tp1_test`, `l3_decode_hca`
- `decode_indexer.py`：`indexer_test`
- `decode_indexer_compressor.py`：`compressor_test`
- `decode_layer.py`：`l3_decode_layer_swa`, `l3_decode_layer_hca`, `l3_decode_layer_csa`
- `decode_o_proj.py`：`l3_o_group_a2a`
- `decode_sparse_attn_csa.py`：`sparse_attn_csa_test`
- `decode_sparse_attn_hca.py`：`sparse_attn_hca_test`
- `decode_sparse_attn_swa.py`：`sparse_attn_swa_test`
- `decode_swa.py`：`decode_swa_tp1_test`, `l3_decode_swa`
- `dspark_attention.py`：`dspark_attention_test`
- `dspark_context_kv.py`：`dspark_context_kv_test`
- `dspark_drafter.py`：`l3_dspark_drafter`
- `dspark_markov.py`：`markov_sample`, `l3_distributed_markov_sample`
- `expert_routed.py`：`expert_routed_test`
- `expert_shared.py`：`expert_shared_test`
- `gate.py`：`gate_test`
- `hc_head.py`：`hc_head_test`
- `hc_post.py`：`hc_post_test`
- `hc_pre.py`：`hc_pre_test`
- `lm_head.py`：`l3_lm_head_sample`, `l3_lm_head`
- `lookup_embedding.py`：`lookup_embedding_test`
- `markov_head.py`：`markov_head_test`
- `moe.py`：`l3_moe`
- `prefill_compressor_ratio128.py`：`prefill_compressor_ratio128_test`
- `prefill_compressor_ratio4.py`：`prefill_compressor_ratio4_test`
- `prefill_cp_token_allgather.py`：`l3_prefill_cp_token_allgather_fixture`
- `prefill_csa.py`：`prefill_attention_csa_test`, `l3_prefill_attention_csa_cp`
- `prefill_fwd.py`：`l3_prefill_fwd`
- `prefill_hca.py`：`prefill_attention_hca_test`, `l3_prefill_attention_hca_cp`
- `prefill_indexer.py`：`prefill_indexer_test`
- `prefill_indexer_compressor.py`：`prefill_indexer_compressor_test`
- `prefill_layer.py`：`l3_prefill_layer`
- `prefill_metadata.py`：`prefill_metadata_test`
- `prefill_sparse_attn.py`：`prefill_sparse_attn_test`
- `prefill_swa.py`：`prefill_attention_swa_test`, `l3_prefill_attention_swa_cp`
- `qkv_proj_rope.py`：`qkv_proj_rope_test`, `q_kv_split_test`
- `rmsnorm.py`：`rms_norm_test`

### 基本算子（80 个）

算子名后第二组标记为 golden 状态（图例见文首）。

| 文件 | 基本算子 |
| --- | --- |
| `decode_compressor_ratio128.py` | `compressor_ratio128_project`○, `compressor_ratio128_projected`○ |
| `decode_compressor_ratio4.py` | `compressor_ratio4_project`○, `compressor_ratio4_pool_projected`○, `compressor_ratio4_cache_write`○ |
| `decode_cp_allgather.py` | `cp_hca_projection_allgather_readback`[incore]○, `decode_cp_csa_main_typed_allgather_step`○, `decode_cp_csa_aux_typed_allgather_step`○, `decode_cp_kv_allgather_step`○, `decode_cp_allgather_fixture`[jit]◉ |
| `decode_indexer.py` | `merge2_top512_pairs`○, `indexer_topk_half_leaf`○, `indexer_topk_leaf_publish`○, `indexer_qr_rope`○, `indexer_qr_hadamard_mm`○ |
| `decode_indexer_compressor.py` | `indexer_compressor_project`○, `indexer_compressor_pool_projected`○, `indexer_compressor_write`○ |
| `decode_o_proj.py` | `_decode_o_proj_tp1_tiled`○, `o_group_a2a_gather`[incore]○, `tp_o_rs_reduce`[incore]○ |
| `decode_prepare.py` | `gather_group_decode_rope_rows`✗, `build_group_decode_metadata`✗, `prepare_target_group_from_device_state`✗, `accept_target_into_device_state`✗, `commit_drafts_to_device_state`✗, `fence_drafter_head_hidden`✗, `_allgather_metadata`✗, `_allgather_rope`✗ |
| `decode_sparse_attn_csa.py` | `sparse_attn_csa`○ |
| `decode_sparse_attn_hca.py` | `_cmp_query_kv`○ |
| `decode_sparse_attn_swa.py` | `sparse_attn_swa`○ |
| `dspark_drafter.py` | `build_dspark_metadata`○ |
| `dspark_markov.py` | `normalize_head_hidden`○ |
| `expert_routed.py` | `expert_routed_tile`◉ |
| `expert_shared.py` | `expert_shared`*（`包装 _expert_shared`）⊕, `expert_shared_test`*[jit]（`包装 _expert_shared`）T |
| `gate.py` | `gate`*（`包装 _gate`）⊕, `gate_test`*[jit]（`包装 _gate`）T |
| `hc_head.py` | `hc_head`*（`包装 _hc_head`）⊕, `hc_head_test`*[jit]（`包装 _hc_head`）T |
| `hc_post.py` | `hc_post`*（`包装 _hc_post`）⊕, `hc_post_test`*[jit]（`包装 _hc_post`）T, `hc_post_prefill`○ |
| `hc_pre.py` | `hc_pre_gates`◉ |
| `lm_head.py` | `lm_head`○, `greedy_sample`○ |
| `lookup_embedding.py` | `lookup_embedding`*（`包装 _lookup_embedding`）⊕, `lookup_embedding_test`*[jit]（`包装 _lookup_embedding`）T |
| `markov_head.py` | `markov_head`◉ |
| `moe.py` | `clear_moe_signals`○, `clear_prefill_moe_signals`○, `dispatch`○, `combine`○ |
| `prefill_compressor_ratio128.py` | `_prefill_compressor_ratio128_tile`◉ |
| `prefill_compressor_ratio4.py` | `_prefill_compressor_ratio4_tile`◉ |
| `prefill_cp_token_allgather.py` | `prefill_cp_token_allgather_fixture`*[jit]（`包装 _prefill_cp_token_allgather_step`）◉, `prefill_cp_token_allgather_step`*（`包装 _prefill_cp_token_allgather_step`）○ |
| `prefill_fwd.py` | `_copy_target_hc_row`[incore]○ |
| `prefill_indexer.py` | `_merge2_top512_pairs`○, `_topk_leaf`○ |
| `prefill_indexer_compressor.py` | `_prefill_indexer_compressor_tile`○ |
| `prefill_metadata.py` | `lower_local_request_ids`◉ |
| `prefill_o_proj.py` | `gather_o_proj_full_weights`○, `retire_o_proj_weight_signals`○ |
| `prefill_sparse_attn.py` | `_prepare_sparse_attn_rope`○, `_hca_streaming_wave`○, `_sparse_attn_wave`○, `_sparse_attn_o_proj`○ |
| `qkv_proj_rope.py` | `materialize_rope_rows_dynamic`✗, `materialize_rope_rows`✗, `rope_prepare`○, `q_proj_qr`○, `q_proj_q_matmul`○, `q_proj_q_dequant`○, `kv_proj_rope`○ |
| `rmsnorm.py` | `rms_norm_inverse`○, `rms_norm_apply`○, `_rms_norm_tail_tile`○ |
| `rope_interleave.py` | `rope_interleave`○ |

### 有独立 golden 函数的基本算子（13 个）

这些叶子算子由专属 torch 参考对比：`⊕` 表示该算子本身就是被测入口，`◉` 表示它经由唯一调用它的测试入口对比。

| 文件 | 基本算子 | golden 函数 | 接线 |
| --- | --- | --- | --- |
| `decode_cp_allgather.py` | `decode_cp_allgather_fixture` | `golden_decode_cp_allgather` | 经 `l3_decode_cp_allgather_fixture` |
| `expert_routed.py` | `expert_routed_tile` | `golden_expert_routed` | 经 `expert_routed_test` |
| `expert_shared.py` | `expert_shared` | `golden_expert_shared` | 经 `expert_shared_test` |
| `gate.py` | `gate` | `golden_gate_core` | 经 `gate_test` |
| `hc_head.py` | `hc_head` | `golden_hc_head` | 经 `hc_head_test` |
| `hc_post.py` | `hc_post` | `golden_hc_post` | 经 `hc_post_test` |
| `hc_pre.py` | `hc_pre_gates` | `golden_hc_pre` | 经 `hc_pre` |
| `lookup_embedding.py` | `lookup_embedding` | `golden_lookup_embedding_test` | 经 `lookup_embedding_test` |
| `markov_head.py` | `markov_head` | `golden_markov_head` | 经 `markov_head_test` |
| `prefill_compressor_ratio128.py` | `_prefill_compressor_ratio128_tile` | `golden_prefill_compressor_ratio128` | 经 `prefill_compressor_ratio128` |
| `prefill_compressor_ratio4.py` | `_prefill_compressor_ratio4_tile` | `golden_prefill_compressor_ratio4` | 经 `compressor_ratio4` |
| `prefill_cp_token_allgather.py` | `prefill_cp_token_allgather_fixture` | `golden_prefill_cp_token_allgather` | 经 `l3_prefill_cp_token_allgather_fixture` |
| `prefill_metadata.py` | `lower_local_request_ids` | `golden_prefill_metadata` | 经 `prefill_metadata_test` |

### 非基本算子（148 个）

| 文件 | 算子 | 调用的算子 | golden |
| --- | --- | --- | --- |
| `decode_compressor_ratio128.py` | `compressor_ratio128`○ | `compressor_ratio128_project`, `compressor_ratio128_projected` | ○ 传递覆盖 |
| `decode_compressor_ratio128.py` | `compressor_test`[jit]⊕ | `compressor_ratio128`, `rope_interleave` | ⊕ 独立 golden `golden_compressor` |
| `decode_compressor_ratio4.py` | `compressor_ratio4_pool`○ | `compressor_ratio4_pool_projected`, `compressor_ratio4_project` | ○ 传递覆盖 |
| `decode_compressor_ratio4.py` | `compressor_ratio4`○ | `compressor_ratio4_cache_write`, `compressor_ratio4_pool` | ○ 传递覆盖 |
| `decode_compressor_ratio4.py` | `compressor_test`[jit]⊕ | `compressor_ratio4（decode_compressor_ratio4.py）` | ⊕ 独立 golden `golden_compressor` |
| `decode_cp_allgather.py` | `decode_cp_hca_projection_allgather_step`○ | `cp_hca_projection_allgather_readback` | ○ 传递覆盖 |
| `decode_cp_allgather.py` | `csa_main_allgather_fixture_step`✗ | `decode_cp_csa_main_typed_allgather_step` | ✗ 未覆盖 |
| `decode_cp_allgather.py` | `csa_aux_allgather_fixture_step`✗ | `decode_cp_csa_aux_typed_allgather_step` | ✗ 未覆盖 |
| `decode_cp_allgather.py` | `l3_decode_cp_allgather_fixture`[host]⊕ | `decode_cp_allgather_fixture` | ⊕ 独立 golden `golden_decode_cp_allgather` |
| `decode_csa.py` | `decode_csa`*（`包装 _decode_csa`）○ | `compressor_ratio4_cache_write`, `compressor_ratio4_pool_projected`, `compressor_ratio4_project`, `decode_cp_csa_aux_typed_allgather_step`, `decode_cp_csa_main_typed_allgather_step`, `indexer_qr_hadamard_mm`, `indexer_qr_rope`, `indexer_weights_score`, `indexer_compressor_pool_projected`, `indexer_compressor_project`, `indexer_compressor_write`, `o_group_a2a`, `o_proj_reduce_scatter`, `sparse_attn_csa`, `hc_pre_norm`, `kv_proj_rope`, `q_proj_q_dequant`, `q_proj_q_matmul`, `q_proj_qr`, `rope_prepare`, `hc_post` | ○ 传递覆盖 |
| `decode_csa.py` | `decode_csa_test`*[jit]（`包装 _decode_csa`）○ | `compressor_ratio4_cache_write`, `compressor_ratio4_pool_projected`, `compressor_ratio4_project`, `decode_cp_csa_aux_typed_allgather_step`, `decode_cp_csa_main_typed_allgather_step`, `indexer_qr_hadamard_mm`, `indexer_qr_rope`, `indexer_weights_score`, `indexer_compressor_pool_projected`, `indexer_compressor_project`, `indexer_compressor_write`, `o_group_a2a`, `o_proj_reduce_scatter`, `sparse_attn_csa`, `hc_pre_norm`, `kv_proj_rope`, `q_proj_q_dequant`, `q_proj_q_matmul`, `q_proj_qr`, `rope_prepare`, `hc_post` | ○ 传递覆盖 |
| `decode_csa.py` | `decode_csa_tp1`*（`包装 _decode_csa_tp1`）⊕ | `compressor_ratio4（decode_compressor_ratio4.py）`, `indexer`, `indexer_compressor`, `decode_o_proj_tp1`, `sparse_attn_csa_tp1`, `hc_pre_norm`, `qkv_proj_rope`, `hc_post` | ⊕ 独立 golden `golden_decode_csa_tp1`（入口 `decode_csa_tp1_test`） |
| `decode_csa.py` | `decode_csa_tp1_test`*[jit]（`包装 _decode_csa_tp1`）T | `compressor_ratio4（decode_compressor_ratio4.py）`, `indexer`, `indexer_compressor`, `decode_o_proj_tp1`, `sparse_attn_csa_tp1`, `hc_pre_norm`, `qkv_proj_rope`, `hc_post` | T 同体孪生已校验（`golden_decode_csa_tp1`） |
| `decode_csa.py` | `l3_decode_csa`[host]⊕ | `decode_csa_test` | ⊕ 独立 golden `golden_decode_csa` |
| `decode_fwd.py` | `decode_fwd`*（`包装 _decode_fwd`）✗ | `decode_embedding_preamble`, `greedy_sample`, `lm_head`, `clear_moe_signals`, `moe`, `rms_norm`, `decode_csa`, `decode_csa_tp1`, `decode_hca`, `decode_hca_tp1`, `decode_swa`, `decode_swa_tp1`, `hc_head` | ✗ 未覆盖 |
| `decode_fwd.py` | `l2_decode_fwd`*[jit]（`包装 _decode_fwd`）✗ | `decode_embedding_preamble`, `greedy_sample`, `lm_head`, `clear_moe_signals`, `moe`, `rms_norm`, `decode_csa`, `decode_csa_tp1`, `decode_hca`, `decode_hca_tp1`, `decode_swa`, `decode_swa_tp1`, `hc_head` | ✗ 未覆盖 |
| `decode_fwd.py` | `decode_embedding_preamble`✗ | `lookup_embedding` | ✗ 未覆盖 |
| `decode_fwd.py` | `l3_decode_fwd`[host]P | `l2_decode_fwd` | P 运行但 golden_fn=None |
| `decode_fwd_dspark.py` | `l2_decode_fwd_dspark`[jit]✗ | `accept_target_into_device_state`, `build_group_decode_metadata`, `commit_drafts_to_device_state`, `fence_drafter_head_hidden`, `gather_group_decode_rope_rows`, `prepare_drafter_after_target`, `prepare_target_group_from_device_state`, `decode_fwd`, `dspark_drafter`, `distributed_markov_sample` | ✗ 未覆盖 |
| `decode_fwd_dspark.py` | `l3_decode_fwd_dspark`[host]P | `l2_decode_fwd_dspark` | P 运行但 golden_fn=None |
| `decode_hca.py` | `decode_hca`*（`包装 _decode_hca`）○ | `compressor_ratio128_project`, `compressor_ratio128_projected`, `decode_cp_hca_projection_allgather_step`, `o_group_a2a`, `o_proj_reduce_scatter`, `sparse_attn_hca`, `hc_pre_norm`, `kv_proj_rope`, `q_proj_rope`, `rope_prepare`, `rope_interleave`, `hc_post` | ○ 传递覆盖 |
| `decode_hca.py` | `decode_hca_test`*[jit]（`包装 _decode_hca`）○ | `compressor_ratio128_project`, `compressor_ratio128_projected`, `decode_cp_hca_projection_allgather_step`, `o_group_a2a`, `o_proj_reduce_scatter`, `sparse_attn_hca`, `hc_pre_norm`, `kv_proj_rope`, `q_proj_rope`, `rope_prepare`, `rope_interleave`, `hc_post` | ○ 传递覆盖 |
| `decode_hca.py` | `decode_hca_tp1`*（`包装 _decode_hca_tp1`）⊕ | `compressor_ratio128`, `decode_o_proj_tp1`, `sparse_attn_hca_tp1`, `hc_pre_norm`, `qkv_proj_rope`, `rope_interleave`, `hc_post` | ⊕ 独立 golden `golden_decode_hca_tp1`（入口 `decode_hca_tp1_test`） |
| `decode_hca.py` | `decode_hca_tp1_test`*[jit]（`包装 _decode_hca_tp1`）T | `compressor_ratio128`, `decode_o_proj_tp1`, `sparse_attn_hca_tp1`, `hc_pre_norm`, `qkv_proj_rope`, `rope_interleave`, `hc_post` | T 同体孪生已校验（`golden_decode_hca_tp1`） |
| `decode_hca.py` | `l3_decode_hca`[host]⊕ | `decode_hca_test` | ⊕ 独立 golden `golden_decode_hca` |
| `decode_indexer.py` | `indexer_topk_query_merge_one`○ | `merge2_top512_pairs` | ○ 传递覆盖 |
| `decode_indexer.py` | `indexer_topk_query_merge`[incore]○ | `indexer_topk_query_merge_one` | ○ 传递覆盖 |
| `decode_indexer.py` | `indexer_topk_single_leaf_publish`[incore]○ | `indexer_topk_leaf_publish` | ○ 传递覆盖 |
| `decode_indexer.py` | `indexer_score_topk_forest`○ | `indexer_topk_half_leaf`, `indexer_topk_query_merge`, `indexer_topk_single_leaf_publish` | ○ 传递覆盖 |
| `decode_indexer.py` | `indexer_qr_hadamard`○ | `indexer_qr_hadamard_mm`, `indexer_qr_rope` | ○ 传递覆盖 |
| `decode_indexer.py` | `indexer_weights_score`○ | `indexer_score_topk_forest` | ○ 传递覆盖 |
| `decode_indexer.py` | `indexer`○ | `indexer_qr_hadamard`, `indexer_weights_score` | ○ 传递覆盖 |
| `decode_indexer.py` | `indexer_test`[jit]⊕ | `indexer`, `indexer_compressor` | ⊕ 独立 golden `golden_indexer` |
| `decode_indexer_compressor.py` | `indexer_compressor_pool`○ | `indexer_compressor_pool_projected`, `indexer_compressor_project` | ○ 传递覆盖 |
| `decode_indexer_compressor.py` | `indexer_compressor`○ | `indexer_compressor_pool`, `indexer_compressor_write` | ○ 传递覆盖 |
| `decode_indexer_compressor.py` | `compressor_test`[jit]⊕ | `indexer_compressor` | ⊕ 独立 golden `golden_compressor` |
| `decode_layer.py` | `decode_layer_swa`○ | `moe`, `decode_swa`, `decode_swa_tp1` | ○ 传递覆盖 |
| `decode_layer.py` | `decode_layer_swa_test`[jit]○ | `decode_layer_swa`, `clear_moe_signals` | ○ 传递覆盖 |
| `decode_layer.py` | `l3_decode_layer_swa`[host]⊕ | `decode_layer_swa_test` | ⊕ 独立 golden `golden_decode_layer_swa` |
| `decode_layer.py` | `decode_layer_hca`○ | `moe`, `decode_hca`, `decode_hca_tp1` | ○ 传递覆盖 |
| `decode_layer.py` | `decode_layer_hca_test`[jit]○ | `decode_layer_hca`, `clear_moe_signals` | ○ 传递覆盖 |
| `decode_layer.py` | `l3_decode_layer_hca`[host]⊕ | `decode_layer_hca_test` | ⊕ 独立 golden `golden_decode_layer_hca` |
| `decode_layer.py` | `decode_layer_csa`○ | `moe`, `decode_csa`, `decode_csa_tp1` | ○ 传递覆盖 |
| `decode_layer.py` | `decode_layer_csa_test`[jit]○ | `decode_layer_csa`, `clear_moe_signals` | ○ 传递覆盖 |
| `decode_layer.py` | `l3_decode_layer_csa`[host]⊕ | `decode_layer_csa_test` | ⊕ 独立 golden `golden_decode_layer_csa` |
| `decode_o_proj.py` | `decode_o_proj_tp1`○ | `_decode_o_proj_tp1_tiled` | ○ 传递覆盖 |
| `decode_o_proj.py` | `o_group_a2a`○ | `o_group_a2a_gather` | ○ 传递覆盖 |
| `decode_o_proj.py` | `l2_o_group_a2a`[jit]○ | `o_group_a2a` | ○ 传递覆盖 |
| `decode_o_proj.py` | `l3_o_group_a2a`[host]⊕ | `l2_o_group_a2a` | ⊕ 独立 golden `golden_o_group_a2a` |
| `decode_o_proj.py` | `o_proj_reduce_scatter`○ | `tp_o_rs_reduce` | ○ 传递覆盖 |
| `decode_prepare.py` | `prepare_drafter_after_target`✗ | `_allgather_metadata`, `_allgather_rope` | ✗ 未覆盖 |
| `decode_sparse_attn_csa.py` | `sparse_attn_csa_tp1`○ | `sparse_attn_csa` | ○ 传递覆盖 |
| `decode_sparse_attn_csa.py` | `sparse_attn_csa_test`[jit]⊕ | `sparse_attn_csa_tp1` | ⊕ 独立 golden `golden_sparse_attn` |
| `decode_sparse_attn_hca.py` | `sparse_attn_hca`○ | `_cmp_query_kv` | ○ 传递覆盖 |
| `decode_sparse_attn_hca.py` | `sparse_attn_hca_tp1`○ | `sparse_attn_hca` | ○ 传递覆盖 |
| `decode_sparse_attn_hca.py` | `sparse_attn_hca_test`[jit]⊕ | `sparse_attn_hca_tp1` | ⊕ 独立 golden `golden_sparse_attn` |
| `decode_sparse_attn_swa.py` | `sparse_attn_swa_tp1`○ | `sparse_attn_swa` | ○ 传递覆盖 |
| `decode_sparse_attn_swa.py` | `sparse_attn_swa_test`[jit]⊕ | `sparse_attn_swa_tp1` | ⊕ 独立 golden `golden_sparse_attn` |
| `decode_swa.py` | `decode_swa`*（`包装 _decode_swa`）○ | `decode_cp_kv_allgather_step`, `o_group_a2a`, `o_proj_reduce_scatter`, `sparse_attn_swa`, `hc_pre_norm`, `kv_proj_rope`, `q_proj_rope`, `rope_prepare`, `hc_post` | ○ 传递覆盖 |
| `decode_swa.py` | `decode_swa_test`*[jit]（`包装 _decode_swa`）○ | `decode_cp_kv_allgather_step`, `o_group_a2a`, `o_proj_reduce_scatter`, `sparse_attn_swa`, `hc_pre_norm`, `kv_proj_rope`, `q_proj_rope`, `rope_prepare`, `hc_post` | ○ 传递覆盖 |
| `decode_swa.py` | `decode_swa_tp1`*（`包装 _decode_swa_tp1`）⊕ | `decode_o_proj_tp1`, `sparse_attn_swa_tp1`, `hc_pre_norm`, `qkv_proj_rope`, `hc_post` | ⊕ 独立 golden `golden_decode_swa_tp1`（入口 `decode_swa_tp1_test`） |
| `decode_swa.py` | `decode_swa_tp1_test`*[jit]（`包装 _decode_swa_tp1`）T | `decode_o_proj_tp1`, `sparse_attn_swa_tp1`, `hc_pre_norm`, `qkv_proj_rope`, `hc_post` | T 同体孪生已校验（`golden_decode_swa_tp1`） |
| `decode_swa.py` | `l3_decode_swa`[host]⊕ | `decode_swa_test` | ⊕ 独立 golden `golden_decode_swa` |
| `dspark_attention.py` | `dspark_attention`○ | `kv_proj_rope`, `q_proj_rope`, `rope_prepare` | ○ 传递覆盖 |
| `dspark_attention.py` | `dspark_attention_test`[jit]⊕ | `dspark_attention` | ⊕ 独立 golden `golden_dspark_attention` |
| `dspark_context_kv.py` | `dspark_context_kv`○ | `kv_proj_rope`, `rope_prepare` | ○ 传递覆盖 |
| `dspark_context_kv.py` | `dspark_context_kv_query`✗ | `kv_proj_rope`, `rope_prepare` | ✗ 未覆盖 |
| `dspark_context_kv.py` | `dspark_context_kv_test`[jit]⊕ | `dspark_context_kv` | ⊕ 独立 golden `golden_dspark_context_kv` |
| `dspark_drafter.py` | `dspark_drafter`*（`包装 _dspark_drafter`）T | `dspark_context_kv`, `build_dspark_metadata`, `draft_layer`, `clear_moe_signals`, `rms_norm`, `hc_head`, `lookup_embedding`, `prefill_cp_token_allgather_step` | T 同体孪生已校验 |
| `dspark_drafter.py` | `l2_dspark_drafter`*[jit]（`包装 _dspark_drafter`）○ | `dspark_context_kv`, `build_dspark_metadata`, `draft_layer`, `clear_moe_signals`, `rms_norm`, `hc_head`, `lookup_embedding`, `prefill_cp_token_allgather_step` | ○ 传递覆盖 |
| `dspark_drafter.py` | `dspark_proj`✗ | `rms_norm` | ✗ 未覆盖 |
| `dspark_drafter.py` | `prepare_dspark_inputs`✗ | `dspark_proj`, `lookup_embedding` | ✗ 未覆盖 |
| `dspark_drafter.py` | `draft_layer`○ | `o_group_a2a`, `o_proj_reduce_scatter`, `dspark_attention`, `hc_post_prefill`, `moe`, `rms_norm`, `hc_pre`, `prefill_cp_token_allgather_step` | ○ 传递覆盖 |
| `dspark_drafter.py` | `l3_dspark_drafter`[host]⊕ | `l2_dspark_drafter` | ⊕ 独立 golden `golden_dspark_drafter` |
| `dspark_markov.py` | `distributed_markov_sample`*（`包装 _distributed_markov_sample`）T | `normalize_head_hidden`, `sample_from_base_logits`, `lm_head` | T 同体孪生已校验 |
| `dspark_markov.py` | `l2_distributed_markov_sample`*[jit]（`包装 _distributed_markov_sample`）○ | `normalize_head_hidden`, `sample_from_base_logits`, `lm_head` | ○ 传递覆盖 |
| `dspark_markov.py` | `compute_base_logits`○ | `normalize_head_hidden` | ○ 传递覆盖 |
| `dspark_markov.py` | `greedy_markov_step`○ | `markov_head` | ○ 传递覆盖 |
| `dspark_markov.py` | `sample_from_base_logits`○ | `greedy_markov_step` | ○ 传递覆盖 |
| `dspark_markov.py` | `markov_sample`[jit]⊕ | `compute_base_logits`, `sample_from_base_logits` | ⊕ 独立 golden `golden_nonzero_markov` |
| `dspark_markov.py` | `l3_distributed_markov_sample`[host]⊕ | `l2_distributed_markov_sample` | ⊕ 独立 golden `golden_distributed_markov` |
| `expert_routed.py` | `expert_routed_test`[jit]⊕ | `expert_routed_tile` | ⊕ 独立 golden `golden_expert_routed` |
| `hc_pre.py` | `hc_pre`*（`包装 _hc_pre`）⊕ | `hc_pre_gates` | ⊕ 独立 golden `golden_hc_pre`（入口 `hc_pre_test`） |
| `hc_pre.py` | `hc_pre_test`*[jit]（`包装 _hc_pre`）T | `hc_pre_gates` | T 同体孪生已校验（`golden_hc_pre`） |
| `hc_pre.py` | `hc_pre_norm`○ | `hc_pre_gates`, `rms_norm_apply`, `rms_norm_inverse` | ○ 传递覆盖 |
| `lm_head.py` | `l2_lm_head`[jit]○ | `lm_head` | ○ 传递覆盖 |
| `lm_head.py` | `l2_lm_head_sample`[jit]○ | `greedy_sample`, `lm_head` | ○ 传递覆盖 |
| `lm_head.py` | `l3_lm_head_sample`[host]⊕ | `l2_lm_head_sample` | ⊕ 独立 golden `golden_lm_head_sample` |
| `lm_head.py` | `l3_lm_head`[host]⊕ | `l2_lm_head` | ⊕ 独立 golden `golden_lm_head` |
| `markov_head.py` | `markov_head_test`[jit]⊕ | `markov_head` | ⊕ 独立 golden `golden_markov_head` |
| `moe.py` | `_moe_tile`○ | `expert_routed_tile`, `combine`, `dispatch`, `expert_shared`, `gate` | ○ 传递覆盖 |
| `moe.py` | `moe`○ | `_moe_tile`, `hc_post`, `hc_pre` | ○ 传递覆盖 |
| `moe.py` | `prefill_moe`○ | `_moe_tile`, `hc_post`, `hc_pre`, `prefill_cp_token_allgather_step` | ○ 传递覆盖 |
| `moe.py` | `moe_test`[jit]○ | `clear_moe_signals`, `moe` | ○ 传递覆盖 |
| `moe.py` | `l3_moe`[host]⊕ | `moe_test` | ⊕ 独立 golden `golden_moe_rounds` |
| `prefill_compressor_ratio128.py` | `prefill_compressor_ratio128`*（`包装 _prefill_compressor_ratio128`）⊕ | `_prefill_compressor_ratio128_tile` | ⊕ 独立 golden `golden_prefill_compressor_ratio128`（入口 `prefill_compressor_ratio128_test`） |
| `prefill_compressor_ratio128.py` | `prefill_compressor_ratio128_test`*[jit]（`包装 _prefill_compressor_ratio128`）T | `_prefill_compressor_ratio128_tile` | T 同体孪生已校验（`golden_prefill_compressor_ratio128`） |
| `prefill_compressor_ratio4.py` | `compressor_ratio4`*（`包装 _compressor_ratio4`）⊕ | `_prefill_compressor_ratio4_tile` | ⊕ 独立 golden `golden_prefill_compressor_ratio4`（入口 `prefill_compressor_ratio4_test`） |
| `prefill_compressor_ratio4.py` | `prefill_compressor_ratio4_test`*[jit]（`包装 _compressor_ratio4`）T | `_prefill_compressor_ratio4_tile` | T 同体孪生已校验（`golden_prefill_compressor_ratio4`） |
| `prefill_cp_token_allgather.py` | `l3_prefill_cp_token_allgather_fixture`[host]⊕ | `prefill_cp_token_allgather_fixture` | ⊕ 独立 golden `golden_prefill_cp_token_allgather` |
| `prefill_csa.py` | `prefill_attention_csa`*（`包装 _prefill_attention_csa`）⊕ | `prefill_indexer`, `sparse_attn_physical`, `qkv_proj_rope`, `rms_norm`, `hc_post`, `hc_pre`, `compressor_ratio4（prefill_compressor_ratio4.py）` | ⊕ 独立 golden `golden_prefill_attention_csa`（入口 `prefill_attention_csa_test`） |
| `prefill_csa.py` | `prefill_attention_csa_test`*[jit]（`包装 _prefill_attention_csa`）T | `prefill_indexer`, `sparse_attn_physical`, `qkv_proj_rope`, `rms_norm`, `hc_post`, `hc_pre`, `compressor_ratio4（prefill_compressor_ratio4.py）` | T 同体孪生已校验（`golden_prefill_attention_csa`） |
| `prefill_csa.py` | `prefill_attention_csa_cp_core`○ | `prefill_indexer_query`, `prefill_indexer_compressor`, `sparse_attn_physical`, `kv_proj_rope`, `q_proj_rope`, `rope_prepare`, `compressor_ratio4（prefill_compressor_ratio4.py）` | ○ 传递覆盖 |
| `prefill_csa.py` | `prefill_attention_csa_cp`○ | `prefill_attention_csa_cp_core`, `gather_o_proj_full_weights`, `rms_norm`, `hc_post`, `hc_pre`, `prefill_cp_token_allgather_step` | ○ 传递覆盖 |
| `prefill_csa.py` | `prefill_attention_csa_cp_test`[jit]○ | `prefill_attention_csa_cp` | ○ 传递覆盖 |
| `prefill_csa.py` | `l3_prefill_attention_csa_cp`[host]⊕ | `prefill_attention_csa_cp_test` | ⊕ 独立 golden `golden_prefill_attention_csa_cp` |
| `prefill_fwd.py` | `prefill_fwd`[jit]○ | `greedy_sample`, `lm_head`, `clear_prefill_moe_signals`, `prefill_moe`, `prefill_attention_csa_cp`, `_copy_target_hc_row`, `prefill_attention_hca_cp`, `lower_local_request_ids`, `retire_o_proj_weight_signals`, `prefill_attention_swa_cp`, `rms_norm`, `hc_head` | ○ 传递覆盖 |
| `prefill_fwd.py` | `l3_prefill_fwd`[host]⊕ | `prefill_fwd` | ⊕ 独立 golden `golden_prefill_fwd` |
| `prefill_hca.py` | `prefill_attention_hca`*（`包装 _prefill_attention_hca`）⊕ | `hca_streaming_attn_physical`, `qkv_proj_rope`, `rms_norm`, `hc_post`, `hc_pre`, `prefill_compressor_ratio128` | ⊕ 独立 golden `golden_prefill_attention_hca`（入口 `prefill_attention_hca_test`） |
| `prefill_hca.py` | `prefill_attention_hca_test`*[jit]（`包装 _prefill_attention_hca`）T | `hca_streaming_attn_physical`, `qkv_proj_rope`, `rms_norm`, `hc_post`, `hc_pre`, `prefill_compressor_ratio128` | T 同体孪生已校验（`golden_prefill_attention_hca`） |
| `prefill_hca.py` | `prefill_attention_hca_cp_core`○ | `hca_streaming_attn_physical`, `kv_proj_rope`, `q_proj_rope`, `rope_prepare`, `prefill_compressor_ratio128` | ○ 传递覆盖 |
| `prefill_hca.py` | `prefill_attention_hca_cp`○ | `prefill_attention_hca_cp_core`, `gather_o_proj_full_weights`, `rms_norm`, `hc_post`, `hc_pre`, `prefill_cp_token_allgather_step` | ○ 传递覆盖 |
| `prefill_hca.py` | `prefill_attention_hca_cp_test`[jit]○ | `prefill_attention_hca_cp` | ○ 传递覆盖 |
| `prefill_hca.py` | `l3_prefill_attention_hca_cp`[host]⊕ | `prefill_attention_hca_cp_test` | ⊕ 独立 golden `golden_prefill_attention_hca_cp` |
| `prefill_indexer.py` | `_merge_topk_level_pairs`○ | `_merge2_top512_pairs` | ○ 传递覆盖 |
| `prefill_indexer.py` | `_topk_group_wave`[incore]○ | `_merge2_top512_pairs`, `_topk_leaf` | ○ 传递覆盖 |
| `prefill_indexer.py` | `_topk_query_merge`[incore]○ | `_merge_topk_level_pairs` | ○ 传递覆盖 |
| `prefill_indexer.py` | `_prefill_indexer_score_topk`○ | `_topk_group_wave`, `_topk_query_merge` | ○ 传递覆盖 |
| `prefill_indexer.py` | `_prefill_indexer_dense_tile`○ | `_prefill_indexer_score_topk` | ○ 传递覆盖 |
| `prefill_indexer.py` | `prefill_indexer`○ | `prefill_indexer_query`, `prefill_indexer_compressor` | ○ 传递覆盖 |
| `prefill_indexer.py` | `prefill_indexer_query`○ | `_prefill_indexer_dense_tile` | ○ 传递覆盖 |
| `prefill_indexer.py` | `prefill_indexer_test`[jit]⊕ | `prefill_indexer` | ⊕ 独立 golden `golden_prefill_indexer` |
| `prefill_indexer_compressor.py` | `prefill_indexer_compressor`○ | `_prefill_indexer_compressor_tile` | ○ 传递覆盖 |
| `prefill_indexer_compressor.py` | `prefill_indexer_compressor_test`[jit]⊕ | `prefill_indexer_compressor` | ⊕ 独立 golden `golden_prefill_indexer_compressor` |
| `prefill_layer.py` | `prefill_layer_attention`[jit]○ | `prefill_attention_csa_cp`, `prefill_attention_hca_cp`, `lower_local_request_ids`, `prefill_attention_swa_cp` | ○ 传递覆盖 |
| `prefill_layer.py` | `prefill_layer_moe`[jit]○ | `clear_prefill_moe_signals`, `prefill_moe` | ○ 传递覆盖 |
| `prefill_layer.py` | `l3_prefill_layer`[host]⊕ | `prefill_layer_attention`, `prefill_layer_moe` | ⊕ 独立 golden `golden_prefill_layer` |
| `prefill_metadata.py` | `prefill_metadata_test`[jit]⊕ | `lower_local_request_ids` | ⊕ 独立 golden `golden_prefill_metadata` |
| `prefill_sparse_attn.py` | `_hca_streaming_attn_tile`○ | `_hca_streaming_wave`, `_prepare_sparse_attn_rope`, `_sparse_attn_o_proj` | ○ 传递覆盖 |
| `prefill_sparse_attn.py` | `_sparse_attn_heads`○ | `_prepare_sparse_attn_rope`, `_sparse_attn_wave` | ○ 传递覆盖 |
| `prefill_sparse_attn.py` | `hca_streaming_attn_physical`○ | `_hca_streaming_attn_tile` | ○ 传递覆盖 |
| `prefill_sparse_attn.py` | `sparse_attn_compute`○ | `_sparse_attn_heads`, `_sparse_attn_o_proj` | ○ 传递覆盖 |
| `prefill_sparse_attn.py` | `sparse_attn_physical`○ | `sparse_attn_compute` | ○ 传递覆盖 |
| `prefill_sparse_attn.py` | `prefill_sparse_attn_test`[jit]⊕ | `sparse_attn_physical` | ⊕ 独立 golden `golden_prefill_sparse_attn` |
| `prefill_swa.py` | `prefill_attention_swa`*（`包装 _prefill_attention_swa`）⊕ | `sparse_attn_physical`, `qkv_proj_rope`, `rms_norm`, `hc_post`, `hc_pre` | ⊕ 独立 golden `golden_prefill_attention_swa`（入口 `prefill_attention_swa_test`） |
| `prefill_swa.py` | `prefill_attention_swa_test`*[jit]（`包装 _prefill_attention_swa`）T | `sparse_attn_physical`, `qkv_proj_rope`, `rms_norm`, `hc_post`, `hc_pre` | T 同体孪生已校验（`golden_prefill_attention_swa`） |
| `prefill_swa.py` | `prefill_attention_swa_cp_core`○ | `sparse_attn_physical`, `kv_proj_rope`, `q_proj_rope`, `rope_prepare` | ○ 传递覆盖 |
| `prefill_swa.py` | `prefill_attention_swa_cp`○ | `gather_o_proj_full_weights`, `prefill_attention_swa_cp_core`, `rms_norm`, `hc_post`, `hc_pre`, `prefill_cp_token_allgather_step` | ○ 传递覆盖 |
| `prefill_swa.py` | `prefill_attention_swa_cp_test`[jit]○ | `prefill_attention_swa_cp` | ○ 传递覆盖 |
| `prefill_swa.py` | `l3_prefill_attention_swa_cp`[host]⊕ | `prefill_attention_swa_cp_test` | ⊕ 独立 golden `golden_prefill_attention_swa_cp` |
| `qkv_proj_rope.py` | `q_proj_q`○ | `q_proj_q_dequant`, `q_proj_q_matmul` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `q_proj_rope`○ | `q_proj_q`, `q_proj_qr` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `qkv_proj_rope`○ | `kv_proj_rope`, `q_proj_rope`, `rope_prepare` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `qkv_proj_rope_test`[jit]✗ | `qkv_proj_rope` | ✗ 未覆盖 |
| `qkv_proj_rope.py` | `q_kv_split_test`[jit]✗ | `kv_proj_rope`, `q_proj_rope`, `rope_prepare` | ✗ 未覆盖 |
| `rmsnorm.py` | `_rms_norm_full_tile`○ | `rms_norm_apply`, `rms_norm_inverse` | ○ 传递覆盖 |
| `rmsnorm.py` | `rms_norm`○ | `_rms_norm_full_tile`, `_rms_norm_tail_tile` | ○ 传递覆盖 |
| `rmsnorm.py` | `rms_norm_test`[jit]⊕ | `rms_norm` | ⊕ 独立 golden `golden_rms_norm_test` |

## deepseek_v4_pro

Ascend A5 上的 DeepSeek V4-Pro（可选 Flash preset），Hybrid MXFP8-MXFP4 量化。

共 101 个算子：基本算子 40 个，非基本算子 61 个。golden 覆盖 90/101（`⊕` 35、`◉` 15、`○` 40）；perf-only 1、未覆盖 10。其中**有独立 golden 函数的基本算子 15 个**（见下方同名小节）。

顶层入口（未被其它算子调用，共 37 个）：

- `decode_attention_csa.py`：`attention_csa_test`
- `decode_attention_hca.py`：`attention_hca_test`
- `decode_attention_swa.py`：`attention_swa_test`
- `decode_compressor_ratio128.py`：`compressor_test`
- `decode_compressor_ratio4.py`：`compressor_test`
- `decode_fwd.py`：`l3_decode_fwd`
- `decode_indexer.py`：`indexer_test`
- `decode_indexer_compressor.py`：`compressor_test`
- `decode_layer.py`：`l3_decode_layer`
- `decode_mtp.py`：`l3_mtp_decode_layer`
- `decode_sparse_attn.py`：`sparse_attn_test`
- `decode_sparse_attn_hca.py`：`sparse_attn_test`
- `decode_sparse_attn_swa.py`：`sparse_attn_test`
- `expert_routed.py`：`expert_routed_test`
- `expert_shared.py`：`expert_shared_test`
- `gate.py`：`gate_test`
- `hc_head.py`：`hc_head_test`
- `hc_post.py`：`hc_post_test`
- `hc_pre.py`：`hc_pre_test`
- `input_pack.py`：`pack_x_hc_test`
- `lm_head.py`：`lm_head_test`, `l3_lm_head`
- `moe.py`：`l3_moe`
- `mtp_projection.py`：`mtp_projection_test`
- `prefill_attention_csa.py`：`prefill_attention_csa_test`
- `prefill_attention_hca.py`：`prefill_attention_hca_test`
- `prefill_attention_swa.py`：`prefill_attention_swa_test`
- `prefill_compressor_ratio128.py`：`prefill_compressor_ratio128_test`
- `prefill_compressor_ratio4.py`：`prefill_compressor_ratio4_test`
- `prefill_fwd.py`：`l3_prefill_fwd`
- `prefill_indexer.py`：`prefill_indexer_test`
- `prefill_indexer_compressor.py`：`prefill_indexer_compressor_test`
- `prefill_layer.py`：`l3_prefill_layer`
- `prefill_mtp.py`：`l3_mtp_prefill_fwd`
- `prefill_sparse_attn.py`：`prefill_sparse_attn_test`
- `qkv_proj_rope.py`：`qkv_proj_rope_test`
- `rmsnorm.py`：`rms_norm_test`

### 基本算子（40 个）

算子名后第二组标记为 golden 状态（图例见文首）。

| 文件 | 基本算子 |
| --- | --- |
| `decode_compressor_ratio128.py` | `compressor_ratio128`◉ |
| `decode_compressor_ratio4.py` | `compressor_ratio4`◉ |
| `decode_indexer.py` | `project_index_query`*（`make_mxfp8_projection_from_quantized(output_width=IDX_N_HEADS * IDX_HEAD_D, width=Q_LORA)→project`）○ |
| `decode_indexer_compressor.py` | `indexer_compressor`◉ |
| `decode_sparse_attn.py` | `project_o_a`*（`make_mxfp8_projection(output_width=O_LORA, width=O_GROUP_IN)→project`）○, `project_o_b`*（`make_mxfp8_projection(output_width=D, width=O_GROUPS * O_LORA)→project`）○ |
| `decode_sparse_attn_hca.py` | `project_o_a`*（`make_mxfp8_projection(output_width=O_LORA, width=O_GROUP_IN)→project`）✗, `project_o_b`*（`make_mxfp8_projection(output_width=D, width=O_GROUPS * O_LORA)→project`）✗ |
| `decode_sparse_attn_swa.py` | `project_o_a`*（`make_mxfp8_projection(output_width=O_LORA, width=O_GROUP_IN)→project`）✗, `project_o_b`*（`make_mxfp8_projection(output_width=D, width=O_GROUPS * O_LORA)→project`）✗ |
| `expert_routed.py` | `expert_routed`◉ |
| `expert_shared.py` | `expert_shared`◉ |
| `gate.py` | `gate`◉ |
| `hc_head.py` | `hc_head`◉ |
| `hc_post.py` | `hc_post`◉, `hc_post_prefill`○ |
| `hc_pre.py` | `hc_pre`*（`_bind_hc_pre()→_bind_hc_pre`）◉, `_hc_pre_syncall`✗, `_hc_pre_separate`✗ |
| `input_pack.py` | `pack_x_hc`◉ |
| `lm_head.py` | `lm_head`○, `greedy_sample`○ |
| `moe.py` | `dispatch`○, `shared_routed`[incore]○ |
| `mtp_projection.py` | `mtp_projection`◉ |
| `prefill_compressor_ratio128.py` | `prefill_compressor_ratio128`◉ |
| `prefill_compressor_ratio4.py` | `prefill_compressor_ratio4`◉ |
| `prefill_fwd.py` | `prefill_attention_lead`*（`_bind_prefill_attention_lead()→_bind_prefill_attention_lead`）○ |
| `prefill_indexer_compressor.py` | `prefill_indexer_compressor`◉ |
| `prefill_sparse_attn.py` | `project_o_a`*（`make_mxfp8_projection(output_width=O_LORA, width=O_GROUP_IN)→project`）✗, `project_o_b`*（`make_mxfp8_projection(output_width=D, width=O_GROUPS * O_LORA)→project`）✗ |
| `qkv_proj_rope.py` | `normalize_rope_kv`*（`_make_norm_rope(heads=1, weighted=True)→normalize_rotate`）○, `normalize_rope_q`*（`_make_norm_rope(heads=H, weighted=False)→normalize_rotate`）○, `project_kv`*（`make_mxfp8_projection(output_width=HEAD_DIM, width=D)→project`）○, `project_qa`*（`make_mxfp8_projection(output_width=Q_LORA, width=D)→project`）○, `project_qb`*（`make_mxfp8_projection_from_quantized(output_width=H * HEAD_DIM, width=Q_LORA)→project`）○, `_normalize_qr`○, `_quantize_qr`○, `materialize_rope_rows`○ |
| `rmsnorm.py` | `rms_norm`◉ |

### 有独立 golden 函数的基本算子（15 个）

这些叶子算子由专属 torch 参考对比：`⊕` 表示该算子本身就是被测入口，`◉` 表示它经由唯一调用它的测试入口对比。

| 文件 | 基本算子 | golden 函数 | 接线 |
| --- | --- | --- | --- |
| `decode_compressor_ratio128.py` | `compressor_ratio128` | `golden_compressor` | 经 `compressor_test` |
| `decode_compressor_ratio4.py` | `compressor_ratio4` | `golden_compressor` | 经 `compressor_test` |
| `decode_indexer_compressor.py` | `indexer_compressor` | `golden_compressor` | 经 `compressor_test` |
| `expert_routed.py` | `expert_routed` | `golden_expert_routed` | 经 `expert_routed_test` |
| `expert_shared.py` | `expert_shared` | `golden_expert_shared` | 经 `expert_shared_test` |
| `gate.py` | `gate` | `golden_gate_core` | 经 `gate_test` |
| `hc_head.py` | `hc_head` | `golden_hc_head` | 经 `hc_head_test` |
| `hc_post.py` | `hc_post` | `golden_hc_post` | 经 `hc_post_test` |
| `hc_pre.py` | `hc_pre` | `golden_hc_pre` | 经 `hc_pre_test` |
| `input_pack.py` | `pack_x_hc` | `golden_pack_x_hc` | 经 `pack_x_hc_test` |
| `mtp_projection.py` | `mtp_projection` | `golden_mtp_projection` | 经 `mtp_projection_test` |
| `prefill_compressor_ratio128.py` | `prefill_compressor_ratio128` | `golden_prefill_compressor_ratio128` | 经 `prefill_compressor_ratio128_test` |
| `prefill_compressor_ratio4.py` | `prefill_compressor_ratio4` | `golden_prefill_compressor_ratio4` | 经 `prefill_compressor_ratio4_test` |
| `prefill_indexer_compressor.py` | `prefill_indexer_compressor` | `golden_prefill_indexer_compressor` | 经 `prefill_indexer_compressor_test` |
| `rmsnorm.py` | `rms_norm` | `golden_rms_norm_test` | 经 `rms_norm_test` |

### 非基本算子（61 个）

| 文件 | 算子 | 调用的算子 | golden |
| --- | --- | --- | --- |
| `decode_attention_csa.py` | `attention_csa`○ | `hc_pre`, `compressor_ratio4`, `indexer`, `sparse_attn`, `hc_post`, `qkv_proj_rope`, `rms_norm` | ○ 传递覆盖 |
| `decode_attention_csa.py` | `attention_csa_test`[jit]⊕ | `attention_csa` | ⊕ 独立 golden `golden_attention_csa` |
| `decode_attention_hca.py` | `attention_hca`○ | `hc_pre`, `compressor_ratio128`, `sparse_attn_hca`, `hc_post`, `qkv_proj_rope`, `rms_norm` | ○ 传递覆盖 |
| `decode_attention_hca.py` | `attention_hca_test`[jit]⊕ | `attention_hca` | ⊕ 独立 golden `golden_attention_hca` |
| `decode_attention_swa.py` | `attention_swa`○ | `hc_pre`, `sparse_attn_swa`, `hc_post`, `qkv_proj_rope`, `rms_norm` | ○ 传递覆盖 |
| `decode_attention_swa.py` | `attention_swa_test`[jit]⊕ | `attention_swa` | ⊕ 独立 golden `golden_attention_swa` |
| `decode_compressor_ratio128.py` | `compressor_test`[jit]⊕ | `compressor_ratio128` | ⊕ 独立 golden `golden_compressor` |
| `decode_compressor_ratio4.py` | `compressor_test`[jit]⊕ | `compressor_ratio4` | ⊕ 独立 golden `golden_compressor` |
| `decode_fwd.py` | `decode_fwd`[jit]✗ | `attention_csa`, `attention_hca`, `attention_swa`, `hc_head`, `pack_x_hc`, `lm_head_with_sampling`, `moe`, `rms_norm` | ✗ 未覆盖 |
| `decode_fwd.py` | `l3_decode_fwd`[host]P | `decode_fwd` | P 运行但 golden_fn=None |
| `decode_indexer.py` | `indexer`○ | `project_index_query`, `indexer_compressor` | ○ 传递覆盖 |
| `decode_indexer.py` | `indexer_test`[jit]⊕ | `indexer` | ⊕ 独立 golden `golden_indexer` |
| `decode_indexer_compressor.py` | `compressor_test`[jit]⊕ | `indexer_compressor` | ⊕ 独立 golden `golden_compressor` |
| `decode_layer.py` | `decode_layer`[jit]○ | `attention_csa`, `attention_hca`, `attention_swa`, `moe` | ○ 传递覆盖 |
| `decode_layer.py` | `l3_decode_layer`[host]⊕ | `decode_layer` | ⊕ 独立 golden `golden_decode_layer_auto` |
| `decode_mtp.py` | `mtp_decode_layer`[jit]○ | `attention_swa`, `hc_head`, `moe`, `mtp_projection`, `rms_norm` | ○ 传递覆盖 |
| `decode_mtp.py` | `l3_mtp_decode_layer`[host]⊕ | `mtp_decode_layer` | ⊕ 独立 golden `golden_mtp_decode_layer` |
| `decode_sparse_attn.py` | `sparse_attn`○ | `project_o_b（decode_sparse_attn.py）`, `project_o_a（decode_sparse_attn.py）` | ○ 传递覆盖 |
| `decode_sparse_attn.py` | `sparse_attn_test`[jit]⊕ | `sparse_attn` | ⊕ 独立 golden `golden_sparse_attn` |
| `decode_sparse_attn_hca.py` | `sparse_attn_hca`○ | `project_o_b（decode_sparse_attn.py）`, `project_o_a（decode_sparse_attn.py）` | ○ 传递覆盖 |
| `decode_sparse_attn_hca.py` | `sparse_attn_test`[jit]⊕ | `sparse_attn_hca` | ⊕ 独立 golden `golden_sparse_attn` |
| `decode_sparse_attn_swa.py` | `sparse_attn_swa`○ | `project_o_b（decode_sparse_attn.py）`, `project_o_a（decode_sparse_attn.py）` | ○ 传递覆盖 |
| `decode_sparse_attn_swa.py` | `sparse_attn_test`[jit]⊕ | `sparse_attn_swa` | ⊕ 独立 golden `golden_sparse_attn` |
| `expert_routed.py` | `expert_routed_test`[jit]⊕ | `expert_routed` | ⊕ 独立 golden `golden_expert_routed` |
| `expert_shared.py` | `expert_shared_test`[jit]⊕ | `expert_shared` | ⊕ 独立 golden `golden_expert_shared` |
| `gate.py` | `gate_test`[jit]⊕ | `gate` | ⊕ 独立 golden `golden_gate_core` |
| `hc_head.py` | `hc_head_test`[jit]⊕ | `hc_head` | ⊕ 独立 golden `golden_hc_head` |
| `hc_post.py` | `hc_post_test`[jit]⊕ | `hc_post` | ⊕ 独立 golden `golden_hc_post` |
| `hc_pre.py` | `hc_pre_test`[jit]⊕ | `hc_pre` | ⊕ 独立 golden `golden_hc_pre` |
| `input_pack.py` | `pack_x_hc_test`[jit]⊕ | `pack_x_hc` | ⊕ 独立 golden `golden_pack_x_hc` |
| `lm_head.py` | `lm_head_test`[jit]✗ | `lm_head` | ✗ 未覆盖 |
| `lm_head.py` | `lm_head_with_sampling`○ | `greedy_sample`, `lm_head` | ○ 传递覆盖 |
| `lm_head.py` | `lm_head_with_sampling_test`[jit]○ | `lm_head_with_sampling` | ○ 传递覆盖 |
| `lm_head.py` | `l3_lm_head`[host]⊕ | `lm_head_with_sampling_test` | ⊕ 独立 golden `golden_lm_head` |
| `moe.py` | `combine`○ | `shared_routed` | ○ 传递覆盖 |
| `moe.py` | `moe`○ | `hc_pre`, `expert_routed`, `expert_shared`, `gate`, `hc_post`, `combine`, `dispatch` | ○ 传递覆盖 |
| `moe.py` | `moe_test`[jit]○ | `moe` | ○ 传递覆盖 |
| `moe.py` | `l3_moe`[host]⊕ | `moe_test` | ⊕ 独立 golden `golden_moe` |
| `mtp_projection.py` | `mtp_projection_test`[jit]⊕ | `mtp_projection` | ⊕ 独立 golden `golden_mtp_projection` |
| `prefill_attention_csa.py` | `prefill_attention_csa`○ | `hc_pre`, `hc_post_prefill`, `prefill_compressor_ratio4`, `prefill_indexer`, `prefill_sparse_attn`, `materialize_rope_rows`, `qkv_proj_rope`, `rms_norm` | ○ 传递覆盖 |
| `prefill_attention_csa.py` | `prefill_attention_csa_test`[jit]⊕ | `prefill_attention_csa` | ⊕ 独立 golden `golden_prefill_attention_csa` |
| `prefill_attention_hca.py` | `prefill_attention_hca`○ | `hc_pre`, `hc_post_prefill`, `prefill_compressor_ratio128`, `prefill_sparse_attn`, `materialize_rope_rows`, `qkv_proj_rope`, `rms_norm` | ○ 传递覆盖 |
| `prefill_attention_hca.py` | `prefill_attention_hca_test`[jit]⊕ | `prefill_attention_hca` | ⊕ 独立 golden `golden_prefill_attention_hca` |
| `prefill_attention_swa.py` | `prefill_attention_swa`○ | `hc_pre`, `hc_post_prefill`, `prefill_sparse_attn`, `materialize_rope_rows`, `qkv_proj_rope`, `rms_norm` | ○ 传递覆盖 |
| `prefill_attention_swa.py` | `prefill_attention_swa_test`[jit]⊕ | `prefill_attention_swa` | ⊕ 独立 golden `golden_prefill_attention_swa` |
| `prefill_compressor_ratio128.py` | `prefill_compressor_ratio128_test`[jit]⊕ | `prefill_compressor_ratio128` | ⊕ 独立 golden `golden_prefill_compressor_ratio128` |
| `prefill_compressor_ratio4.py` | `prefill_compressor_ratio4_test`[jit]⊕ | `prefill_compressor_ratio4` | ⊕ 独立 golden `golden_prefill_compressor_ratio4` |
| `prefill_fwd.py` | `prefill_fwd`[jit]○ | `prefill_attention_lead`, `hc_head`, `pack_x_hc`, `moe`, `prefill_attention_csa`, `prefill_attention_hca`, `rms_norm` | ○ 传递覆盖 |
| `prefill_fwd.py` | `l3_prefill_fwd`[host]⊕ | `lm_head_with_sampling_test`, `prefill_fwd` | ⊕ 独立 golden `golden_prefill_fwd` |
| `prefill_indexer.py` | `prefill_indexer`○ | `project_index_query`, `prefill_indexer_compressor` | ○ 传递覆盖 |
| `prefill_indexer.py` | `prefill_indexer_test`[jit]⊕ | `prefill_indexer` | ⊕ 独立 golden `golden_prefill_indexer` |
| `prefill_indexer_compressor.py` | `prefill_indexer_compressor_test`[jit]⊕ | `prefill_indexer_compressor` | ⊕ 独立 golden `golden_prefill_indexer_compressor` |
| `prefill_layer.py` | `prefill_layer_core`[jit]○ | `moe`, `prefill_attention_csa`, `prefill_attention_hca`, `prefill_attention_swa` | ○ 传递覆盖 |
| `prefill_layer.py` | `l3_prefill_layer`[host]⊕ | `prefill_layer_core` | ⊕ 独立 golden `golden_prefill_layer` |
| `prefill_mtp.py` | `mtp_prefill_fwd`[jit]○ | `hc_head`, `moe`, `mtp_projection`, `prefill_attention_swa`, `rms_norm` | ○ 传递覆盖 |
| `prefill_mtp.py` | `l3_mtp_prefill_fwd`[host]⊕ | `mtp_prefill_fwd` | ⊕ 独立 golden `golden_mtp_prefill_fwd` |
| `prefill_sparse_attn.py` | `prefill_sparse_attn`○ | `project_o_b（decode_sparse_attn.py）`, `project_o_a（decode_sparse_attn.py）` | ○ 传递覆盖 |
| `prefill_sparse_attn.py` | `prefill_sparse_attn_test`[jit]⊕ | `prefill_sparse_attn` | ⊕ 独立 golden `golden_prefill_sparse_attn` |
| `qkv_proj_rope.py` | `qkv_proj_rope`○ | `normalize_rope_kv`, `normalize_rope_q`, `project_kv`, `project_qa`, `project_qb`, `_normalize_qr`, `_quantize_qr` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `qkv_proj_rope_test`[jit]⊕ | `qkv_proj_rope` | ⊕ 独立 golden `golden_qkv_proj_rope` |
| `rmsnorm.py` | `rms_norm_test`[jit]⊕ | `rms_norm` | ⊕ 独立 golden `golden_rms_norm_test` |

## deepseek_v4_1_flash

DeepSeek V4.1 Flash TP4/DP2/EP8 脚手架：低比特 SWA/C2A/C1A cache、层次 indexer、mHC 与 MoE。该模型大量使用「工厂函数 + jit 闭包」风格组装算子。

共 230 个算子：基本算子 75 个，非基本算子 155 个。golden 覆盖 201/230（`⊕` 38、`◉` 6、`○` 157）；未覆盖 29。其中**有独立 golden 函数的基本算子 6 个**（见下方同名小节）。

顶层入口（未被其它算子调用，共 33 个）：

- `decode_attn_c1a_full.py`：`decode_attn_c1a_full_test`
- `decode_attn_c1a_reindex.py`：`decode_attn_c1a_reindex_test`
- `decode_attn_c1a_reuse.py`：`decode_attn_c1a_reuse_test`
- `decode_c1a_full.py`：`decode_c1a_full_test`, `decode_c1a_full_sharded_test`
- `decode_c1a_reindex.py`：`decode_c1a_reindex_test`, `decode_c1a_reindex_sharded_test`
- `decode_c1a_reuse.py`：`decode_c1a_reuse_test`, `decode_c1a_reuse_sharded_test`
- `decode_c2a_full.py`：`decode_c2a_full_rank`, `decode_c2a_full_rank_sharded`
- `decode_c2a_reuse.py`：`decode_c2a_reuse_rank`, `decode_c2a_reuse_rank_sharded`
- `decode_swa.py`：`decode_swa_rank`, `decode_swa_rank_sharded`
- `engram.py`：`engram_test`, `engram_tp_group`
- `ep_transport.py`：`l3_dispatch`, `l3_combine`
- `expert_routed.py`：`expert_routed_test`
- `expert_shared.py`：`expert_shared_test`
- `gate.py`：`gate_test`
- `hc_mixes.py`：`mhc_mixes_test`
- `hc_post.py`：`mhc_post_test`
- `hc_pre.py`：`mhc_pre_test`
- `moe.py`：`l3_moe`
- `prefill_attn_c1a_full.py`：`l3_prefill_attn_c1a_full_test`
- `prefill_attn_c1a_reindex.py`：`l3_prefill_attn_c1a_reindex_test`
- `prefill_attn_c1a_reuse.py`：`l3_prefill_attn_c1a_reuse_test`
- `prefill_c1a_full.py`：`l3_prefill_c1a_full_test`
- `prefill_c1a_reindex.py`：`l3_prefill_c1a_reindex_test`
- `prefill_c1a_reuse.py`：`l3_prefill_c1a_reuse_test`
- `rmsnorm.py`：`rms_norm_test`

### 基本算子（75 个）

算子名后第二组标记为 golden 状态（图例见文首）。

| 文件 | 基本算子 |
| --- | --- |
| `attention_tp.py` | `decode_tp_output_all_reduce`*（`make_decode_tp_output_reduce()→decode_tp_output_reduce`）○, `decode_tp_output_reduce_scatter`*（`make_decode_tp_output_reduce(scatter=True)→decode_tp_output_reduce`）✗, `prefill_tp_output_all_reduce`○, `decode_tp_input_all_gather`○ |
| `compressor.py` | `compressor_pair`○, `compressor_pool`○, `compressor_state_write`○ |
| `decode_attn_c1a_full.py` | `c1a_reduce`*（`make_c1a_reduce()→c1a_reduce`）○, `c1a_reduce_scatter`*（`make_c1a_reduce(scatter=True)→c1a_reduce`）○, `project_index_query`*（`make_mx_projection_with_deps(output_width=INDEX_H * INDEX_DIM, width=Q_LORA)→project`）○, `project_kv`*（`make_mx_projection_with_deps(output_width=HEAD_DIM, width=D)→project`）○, `project_ob`*（`make_mx_projection_with_deps(output_dtype=pl.FP32, output_width=D, width=LOCAL_O_WIDTH)→project`）○, `project_qa`*（`make_mx_projection_with_deps(output_width=Q_LORA, width=D)→project`）○, `project_qb`*（`make_mx_projection_with_deps(output_width=LOCAL_H * HEAD_DIM, width=Q_LORA)→project`）○, `publish_compressed`*（`make_fp4_publish(cache_dim=CMP_BLOCKS_DYN, group=16, scale_dtype=pl.FP8E4M3FN, width=HEAD_DIM)→publish`）○, `publish_index`*（`make_fp4_publish(cache_dim=INDEX_BLOCKS_DYN, group=32, scale_dtype=pl.FP8E8M0, width=INDEX_DIM)→publish`）○, `score_full`*（`make_index_scores(use_candidates=False)→score`）○, `score_reindex`*（`make_index_scores(use_candidates=True)→score`）○, `publish_window`○, `attend_combined`○, `c1a_previous_epoch`○, `gather_combined`○, `quantize_index_query`○, `decode_index_keys`○, `select_index_topk`○ |
| `decode_attn_c2a_full.py` | `project_index_q`*（`make_mx_projection(output_width=INDEX_H * INDEX_DIM, width=Q_LORA)→project`）○, `rotate_index_key`*（`make_wide_rope(heads=1, width=INDEX_DIM)→rotate`）○, `rotate_index_query`*（`make_wide_rope(heads=INDEX_H, width=INDEX_DIM)→rotate`）○, `compressor_project`○, `index_key_project`○, `indexer_weights`○, `permute_index_query`○, `publish_compressed`○, `publish_index_key`○, `index_select`○, `gather_sparse`○, `attend_sparse`○ |
| `decode_attn_swa.py` | `publish_window`○, `gather_window`○, `attend_window`○ |
| `decode_common.py` | `zero_bf16_padding`○, `slab_owner`○ |
| `engram.py` | `engram_gate`○ |
| `ep_transport.py` | `dispatch`○, `combine`○ |
| `expert_routed.py` | `expert_routed`◉ |
| `expert_shared.py` | `expert_shared`◉ |
| `gate.py` | `gate_normalized`○ |
| `hc_mixes.py` | `mhc_mixes`◉ |
| `hc_post.py` | `mhc_post`◉ |
| `hc_pre.py` | `mhc_pre`◉ |
| `hierarchical_sparse_indexer.py` | `_merge_candidate_pairs`○, `_sort_candidate_leaf`○, `_sort_candidate_short_leaf`○ |
| `o_proj.py` | `_project_ob`*（`make_mx_projection(name_hint='attention_o_b', output_dtype=pl.FP32, output_width=D, width=LOCAL_O_WIDTH)→project`）○, `_grouped_output_block`○ |
| `prefill_attn_swa.py` | `prefill_publish_window`○, `prefill_gather_window`○, `prefill_attend_window`○ |
| `prefill_c1a_common.py` | `publish_window`○, `publish_compressed_cache`○, `publish_index_cache`○, `attend_sparse_cache`○ |
| `prefill_c1a_indexer.py` | `project_index_query`*（`make_mx_projection(output_width=INDEX_H * INDEX_DIM, width=Q_LORA)→project`）✗, `project_index_weights`✗, `_merge_topk_pairs`✗, `_sort_topk_leaf`✗ |
| `prefill_layer.py` | `widen_to_fp32`○, `_local_token_count`○ |
| `qkv_proj_rope.py` | `_project_kv`*（`make_mx_projection(name_hint='attention_kv', output_width=HEAD_DIM, width=D)→project`）○, `_project_qa`*（`make_mx_projection(name_hint='attention_q_a', output_width=Q_LORA, width=D)→project`）○, `_project_qb`*（`make_mx_projection(name_hint='attention_q_b', output_width=LOCAL_H * HEAD_DIM, width=Q_LORA)→project`）○, `_prefill_project_qa`○ |
| `rmsnorm.py` | `rms_norm`◉ |
| `rope_tables.py` | `materialize_rope_rows`✗ |

### 有独立 golden 函数的基本算子（6 个）

这些叶子算子由专属 torch 参考对比：`⊕` 表示该算子本身就是被测入口，`◉` 表示它经由唯一调用它的测试入口对比。

| 文件 | 基本算子 | golden 函数 | 接线 |
| --- | --- | --- | --- |
| `expert_routed.py` | `expert_routed` | `golden_expert_routed` | 经 `expert_routed_test` |
| `expert_shared.py` | `expert_shared` | `golden_expert_shared` | 经 `expert_shared_test` |
| `hc_mixes.py` | `mhc_mixes` | `golden_mhc_mixes_case` | 经 `mhc_mixes_test` |
| `hc_post.py` | `mhc_post` | `golden_mhc_post_case` | 经 `mhc_post_test` |
| `hc_pre.py` | `mhc_pre` | `golden_mhc_pre_case` | 经 `mhc_pre_test` |
| `rmsnorm.py` | `rms_norm` | `golden_rms_norm_case` | 经 `rms_norm_test` |

### 非基本算子（155 个）

| 文件 | 算子 | 调用的算子 | golden |
| --- | --- | --- | --- |
| `compressor.py` | `compressor_ratio2`○ | `compressor_pair`, `compressor_pool`, `compressor_state_write` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `c1a_finish`*（`make_c1a_finish()→c1a_finish`）○ | `c1a_reduce`, `o_proj_with_deps`, `attend_combined`, `gather_combined` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `c1a_finish_sharded`*（`make_c1a_finish()→c1a_finish`）○ | `c1a_reduce_scatter`, `o_proj_with_deps`, `attend_combined`, `gather_combined` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `decode_attn_c1a_full`*（`make_decode_attn_c1a_full()→decode_attn_c1a_full`）○ | `project_compressor`, `project_index_key（decode_attn_c1a_full.py）`, `normalize_compressor`, `normalize_index_key（decode_attn_c1a_full.py）`, `rotate_compressor`, `rotate_index_key（decode_attn_c1a_full.py）`, `c1a_finish`, `publish_compressed（decode_attn_c1a_full.py）`, `publish_index`, `score_full`, `c1a_index`, `c1a_prepare`, `c1a_previous_epoch`, `select_index_topk`, `hierarchical_sparse_indexer` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `decode_attn_c1a_full_sharded`*（`make_decode_attn_c1a_full()→decode_attn_c1a_full`）○ | `project_compressor`, `project_index_key（decode_attn_c1a_full.py）`, `normalize_compressor`, `normalize_index_key（decode_attn_c1a_full.py）`, `rotate_compressor`, `rotate_index_key（decode_attn_c1a_full.py）`, `c1a_finish_sharded`, `publish_compressed（decode_attn_c1a_full.py）`, `publish_index`, `score_full`, `c1a_index`, `c1a_prepare`, `c1a_previous_epoch`, `select_index_topk`, `hierarchical_sparse_indexer` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `normalize_compressor`*（`make_norm_with_deps(name_hint='c1a_compressor_rmsnorm', width=HEAD_DIM)→normalize`）○ | `_make_norm_block(width=HEAD_DIM)→normalize_block` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `normalize_index_key`*（`make_norm_with_deps(name_hint='c1a_index_key_rmsnorm', width=INDEX_DIM)→normalize`）○ | `_make_norm_block(width=INDEX_DIM)→normalize_block` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `normalize_kv`*（`make_norm_with_deps(name_hint='c1a_kv_rmsnorm', width=HEAD_DIM)→normalize`）○ | `_make_norm_block(width=HEAD_DIM)→normalize_block` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `normalize_q`*（`make_norm_with_deps(name_hint='c1a_q_rmsnorm', width=Q_LORA)→normalize`）○ | `_make_norm_block(width=Q_LORA)→normalize_block` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `o_proj_with_deps`*（`make_o_proj_with_deps()→o_proj_with_deps`）○ | `rotate_output`, `project_ob`, `grouped_output_with_deps` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `project_compressor`*（`make_bf16_projection_with_deps(name_hint='c1a_compressor_projecti, output_width=HEAD_DIM, width=D)→project`）○ | `_make_bf16_projection_block(output_width=HEAD_DIM, width=D)→project_block` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `project_index_key`*（`make_bf16_projection_with_deps(name_hint='c1a_index_key_projectio, output_width=INDEX_DIM, width=HEAD_DIM)→project`）○ | `_make_bf16_projection_block(output_width=INDEX_DIM, width=HEAD_DIM)→project_block` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `project_index_weights`*（`make_bf16_projection_with_deps(name_hint='c1a_index_weights_proje, output_width=INDEX_H, width=D)→project`）○ | `_make_bf16_projection_block(output_width=INDEX_H, width=D)→project_block` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `qkv_proj_rope_with_deps`*（`make_qkv_proj_rope_with_deps()→qkv_proj_rope_with_deps`）○ | `normalize_kv`, `normalize_q`, `rotate_kv`, `rotate_q`, `project_kv`, `project_qb`, `project_qa` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `rotate_compressor`*（`make_rope_with_deps(heads=1, name_hint='c1a_compressor_rope')→rotate`）○ | `_make_rope_block(heads=1)→rotate_block` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `rotate_index_key`*（`make_rope_with_deps(head_dim=INDEX_DIM, heads=1, name_hint='c1a_index_key_rope')→rotate`）○ | `_make_rope_block(head_dim=INDEX_DIM, heads=1)→rotate_block` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `rotate_index_query`*（`make_rope_with_deps(head_dim=INDEX_DIM, heads=INDEX_H, name_hint='c1a_index_query_rope')→rotate`）○ | `_make_rope_block(head_dim=INDEX_DIM, heads=INDEX_H)→rotate_block` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `rotate_kv`*（`make_rope_with_deps(heads=1, name_hint='c1a_kv_rope')→rotate`）○ | `_make_rope_block(heads=1)→rotate_block` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `rotate_output`*（`make_rope_with_deps(heads=LOCAL_H, inverse=True, name_hint='c1a_o_rope')→rotate`）○ | `_make_rope_block(heads=LOCAL_H, inverse=True)→rotate_block` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `rotate_q`*（`make_rope_with_deps(heads=LOCAL_H, name_hint='c1a_q_rope')→rotate`）○ | `_make_rope_block(heads=LOCAL_H)→rotate_block` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `c1a_prepare`○ | `qkv_proj_rope_with_deps`, `publish_window（decode_attn_c1a_full.py）` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `c1a_index`○ | `project_index_weights（decode_attn_c1a_full.py）`, `rotate_index_query（decode_attn_c1a_full.py）`, `project_index_query（decode_attn_c1a_full.py）`, `decode_index_keys`, `quantize_index_query` | ○ 传递覆盖 |
| `decode_attn_c1a_full.py` | `decode_attn_c1a_full_test`[jit]⊕ | `decode_attn_c1a_full` | ⊕ 独立 golden `golden_c1a` |
| `decode_attn_c1a_reindex.py` | `decode_attn_c1a_reindex`*（`make_decode_attn_c1a_reindex()→decode_attn_c1a_reindex`）○ | `c1a_finish`, `score_reindex`, `c1a_index`, `c1a_prepare`, `c1a_previous_epoch`, `select_index_topk` | ○ 传递覆盖 |
| `decode_attn_c1a_reindex.py` | `decode_attn_c1a_reindex_sharded`*（`make_decode_attn_c1a_reindex()→decode_attn_c1a_reindex`）○ | `c1a_finish_sharded`, `score_reindex`, `c1a_index`, `c1a_prepare`, `c1a_previous_epoch`, `select_index_topk` | ○ 传递覆盖 |
| `decode_attn_c1a_reindex.py` | `decode_attn_c1a_reindex_test`[jit]⊕ | `decode_attn_c1a_reindex` | ⊕ 独立 golden `golden_c1a` |
| `decode_attn_c1a_reuse.py` | `decode_attn_c1a_reuse`*（`make_decode_attn_c1a_reuse()→decode_attn_c1a_reuse`）○ | `c1a_finish`, `c1a_prepare`, `c1a_previous_epoch` | ○ 传递覆盖 |
| `decode_attn_c1a_reuse.py` | `decode_attn_c1a_reuse_sharded`*（`make_decode_attn_c1a_reuse()→decode_attn_c1a_reuse`）○ | `c1a_finish_sharded`, `c1a_prepare`, `c1a_previous_epoch` | ○ 传递覆盖 |
| `decode_attn_c1a_reuse.py` | `decode_attn_c1a_reuse_test`[jit]⊕ | `decode_attn_c1a_reuse` | ⊕ 独立 golden `golden_c1a` |
| `decode_attn_c2a_full.py` | `decode_attn_c2a_full`*（`make_decode_attn_c2a_full()→decode_attn_c2a_full`）⊕ | `decode_tp_output_all_reduce`, `c2a_full_partial` | ⊕ 独立 golden `make_golden(...)` |
| `decode_attn_c2a_full.py` | `decode_attn_c2a_full_sharded`*（`make_decode_attn_c2a_full()→decode_attn_c2a_full`）✗ | `decode_tp_output_reduce_scatter`, `c2a_full_partial` | ✗ 未覆盖 |
| `decode_attn_c2a_full.py` | `rotate_compressed`*（`make_rope(heads=1)→rotate`）○ | `_make_rope_block(heads=1)→rotate_block` | ○ 传递覆盖 |
| `decode_attn_c2a_full.py` | `c2a_full_partial`○ | `project_index_q`, `rotate_compressed（decode_attn_c2a_full.py）`, `rotate_index_key（decode_attn_c2a_full.py）`, `rotate_index_query（decode_attn_c2a_full.py）`, `o_proj`, `compressor_ratio2`, `attend_sparse`, `compressor_project`, `gather_sparse`, `index_key_project`, `index_select`, `indexer_weights`, `permute_index_query`, `publish_compressed（decode_attn_c2a_full.py）`, `publish_index_key`, `publish_window（decode_attn_swa.py）`, `qkv_proj_rope` | ○ 传递覆盖 |
| `decode_attn_c2a_reuse.py` | `decode_attn_c2a_reuse`*（`make_decode_attn_c2a_reuse()→decode_attn_c2a_reuse`）⊕ | `decode_tp_output_all_reduce`, `c2a_reuse_partial` | ⊕ 独立 golden `golden_c2a_reuse` |
| `decode_attn_c2a_reuse.py` | `decode_attn_c2a_reuse_sharded`*（`make_decode_attn_c2a_reuse()→decode_attn_c2a_reuse`）✗ | `decode_tp_output_reduce_scatter`, `c2a_reuse_partial` | ✗ 未覆盖 |
| `decode_attn_c2a_reuse.py` | `c2a_reuse_partial`○ | `o_proj`, `attend_sparse`, `gather_sparse`, `publish_window（decode_attn_swa.py）`, `qkv_proj_rope` | ○ 传递覆盖 |
| `decode_attn_swa.py` | `decode_attn_swa`*（`make_decode_attn_swa()→decode_attn_swa`）⊕ | `decode_tp_output_all_reduce`, `decode_swa_partial` | ⊕ 独立 golden `golden_swa` |
| `decode_attn_swa.py` | `decode_attn_swa_sharded`*（`make_decode_attn_swa()→decode_attn_swa`）✗ | `decode_tp_output_reduce_scatter`, `decode_swa_partial` | ✗ 未覆盖 |
| `decode_attn_swa.py` | `decode_swa_partial`○ | `o_proj`, `attend_window`, `gather_window`, `publish_window（decode_attn_swa.py）`, `qkv_proj_rope` | ○ 传递覆盖 |
| `decode_c1a_full.py` | `decode_c1a_full`○ | `decode_attn_c1a_full`, `mhc_pre_norm`, `mhc_mixes`, `mhc_post` | ○ 传递覆盖 |
| `decode_c1a_full.py` | `decode_c1a_full_sharded`○ | `decode_attn_c1a_full_sharded`, `decode_tp_input_all_gather`, `mhc_pre_norm`, `slab_owner`, `mhc_mixes`, `mhc_post` | ○ 传递覆盖 |
| `decode_c1a_full.py` | `decode_c1a_full_test`[jit]⊕ | `decode_c1a_full` | ⊕ 独立 golden `golden_fn` |
| `decode_c1a_full.py` | `decode_c1a_full_sharded_test`[jit]⊕ | `decode_c1a_full_sharded` | ⊕ 独立 golden `golden_fn` |
| `decode_c1a_reindex.py` | `decode_c1a_reindex`○ | `decode_attn_c1a_reindex`, `mhc_pre_norm`, `mhc_mixes`, `mhc_post` | ○ 传递覆盖 |
| `decode_c1a_reindex.py` | `decode_c1a_reindex_sharded`○ | `decode_attn_c1a_reindex_sharded`, `decode_tp_input_all_gather`, `mhc_pre_norm`, `slab_owner`, `mhc_mixes`, `mhc_post` | ○ 传递覆盖 |
| `decode_c1a_reindex.py` | `decode_c1a_reindex_test`[jit]⊕ | `decode_c1a_reindex` | ⊕ 独立 golden `golden_fn` |
| `decode_c1a_reindex.py` | `decode_c1a_reindex_sharded_test`[jit]⊕ | `decode_c1a_reindex_sharded` | ⊕ 独立 golden `golden_fn` |
| `decode_c1a_reuse.py` | `decode_c1a_reuse`○ | `decode_attn_c1a_reuse`, `mhc_pre_norm`, `mhc_mixes`, `mhc_post` | ○ 传递覆盖 |
| `decode_c1a_reuse.py` | `decode_c1a_reuse_sharded`○ | `decode_attn_c1a_reuse_sharded`, `decode_tp_input_all_gather`, `mhc_pre_norm`, `slab_owner`, `mhc_mixes`, `mhc_post` | ○ 传递覆盖 |
| `decode_c1a_reuse.py` | `decode_c1a_reuse_test`[jit]⊕ | `decode_c1a_reuse` | ⊕ 独立 golden `golden_fn` |
| `decode_c1a_reuse.py` | `decode_c1a_reuse_sharded_test`[jit]⊕ | `decode_c1a_reuse_sharded` | ⊕ 独立 golden `golden_fn` |
| `decode_c2a_full.py` | `decode_c2a_full`✗ | `decode_attn_c2a_full`, `attention_pre`, `mhc_post` | ✗ 未覆盖 |
| `decode_c2a_full.py` | `decode_c2a_full_sharded`✗ | `decode_attn_c2a_full_sharded`, `decode_tp_input_all_gather`, `attention_pre`, `slab_owner`, `mhc_post` | ✗ 未覆盖 |
| `decode_c2a_full.py` | `decode_c2a_full_rank`[jit]✗ | `decode_c2a_full` | ✗ 未覆盖 |
| `decode_c2a_full.py` | `decode_c2a_full_rank_sharded`[jit]✗ | `decode_c2a_full_sharded` | ✗ 未覆盖 |
| `decode_c2a_reuse.py` | `decode_c2a_reuse`✗ | `decode_attn_c2a_reuse`, `attention_pre`, `mhc_post` | ✗ 未覆盖 |
| `decode_c2a_reuse.py` | `decode_c2a_reuse_sharded`✗ | `decode_attn_c2a_reuse_sharded`, `decode_tp_input_all_gather`, `attention_pre`, `slab_owner`, `mhc_post` | ✗ 未覆盖 |
| `decode_c2a_reuse.py` | `decode_c2a_reuse_rank`[jit]✗ | `decode_c2a_reuse` | ✗ 未覆盖 |
| `decode_c2a_reuse.py` | `decode_c2a_reuse_rank_sharded`[jit]✗ | `decode_c2a_reuse_sharded` | ✗ 未覆盖 |
| `decode_common.py` | `mhc_pre_norm`○ | `zero_bf16_padding`, `mhc_pre`, `rms_norm` | ○ 传递覆盖 |
| `decode_common.py` | `attention_pre`✗ | `mhc_pre_norm`, `mhc_mixes` | ✗ 未覆盖 |
| `decode_swa.py` | `decode_swa`✗ | `decode_attn_swa`, `attention_pre`, `mhc_post` | ✗ 未覆盖 |
| `decode_swa.py` | `decode_swa_sharded`✗ | `decode_attn_swa_sharded`, `decode_tp_input_all_gather`, `attention_pre`, `slab_owner`, `mhc_post` | ✗ 未覆盖 |
| `decode_swa.py` | `decode_swa_rank`[jit]✗ | `decode_swa` | ✗ 未覆盖 |
| `decode_swa.py` | `decode_swa_rank_sharded`[jit]✗ | `decode_swa_sharded` | ✗ 未覆盖 |
| `engram.py` | `engram`○ | `engram_gate` | ○ 传递覆盖 |
| `engram.py` | `engram_test`[jit]⊕ | `engram` | ⊕ 独立 golden `golden_engram_case` |
| `engram.py` | `engram_tp`○ | `engram_gate` | ○ 传递覆盖 |
| `engram.py` | `engram_tp_rank`[jit]○ | `engram_tp` | ○ 传递覆盖 |
| `engram.py` | `engram_tp_group`[host]⊕ | `engram_tp_rank` | ⊕ 独立 golden `golden_engram_tp_case` |
| `ep_transport.py` | `dispatch_test`[jit]○ | `dispatch` | ○ 传递覆盖 |
| `ep_transport.py` | `l3_dispatch`[host]⊕ | `dispatch_test` | ⊕ 独立 golden `golden_dispatch` |
| `ep_transport.py` | `combine_test`[jit]○ | `combine` | ○ 传递覆盖 |
| `ep_transport.py` | `l3_combine`[host]⊕ | `combine_test` | ⊕ 独立 golden `golden_combine` |
| `expert_routed.py` | `expert_routed_test`[jit]⊕ | `expert_routed` | ⊕ 独立 golden `golden_expert_routed` |
| `expert_shared.py` | `expert_shared_test`[jit]⊕ | `expert_shared` | ⊕ 独立 golden `golden_expert_shared` |
| `gate.py` | `gate`○ | `gate_normalized`, `rms_norm` | ○ 传递覆盖 |
| `gate.py` | `gate_test`[jit]⊕ | `gate` | ⊕ 独立 golden `golden_gate_core` |
| `hc_mixes.py` | `mhc_mixes_test`[jit]⊕ | `mhc_mixes` | ⊕ 独立 golden `golden_mhc_mixes_case` |
| `hc_post.py` | `mhc_post_test`[jit]⊕ | `mhc_post` | ⊕ 独立 golden `golden_mhc_post_case` |
| `hc_pre.py` | `mhc_pre_test`[jit]⊕ | `mhc_pre` | ⊕ 独立 golden `golden_mhc_pre_case` |
| `hierarchical_sparse_indexer.py` | `_merge_candidate_level`○ | `_merge_candidate_pairs` | ○ 传递覆盖 |
| `hierarchical_sparse_indexer.py` | `_hierarchical_sparse_indexer`○ | `_merge_candidate_level`, `_sort_candidate_leaf`, `_sort_candidate_short_leaf` | ○ 传递覆盖 |
| `hierarchical_sparse_indexer.py` | `hierarchical_sparse_indexer`○ | `_hierarchical_sparse_indexer` | ○ 传递覆盖 |
| `moe.py` | `_moe_core`○ | `combine`, `dispatch`, `expert_routed`, `expert_shared`, `gate_normalized` | ○ 传递覆盖 |
| `moe.py` | `moe`○ | `mhc_mixes`, `mhc_post`, `mhc_pre`, `_moe_core`, `rms_norm` | ○ 传递覆盖 |
| `moe.py` | `moe_test`[jit]○ | `moe` | ○ 传递覆盖 |
| `moe.py` | `l3_moe`[host]⊕ | `moe_test` | ⊕ 独立 golden `golden_moe` |
| `o_proj.py` | `_prefill_rotate_output`*（`make_rope(heads=LOCAL_H, inverse=True, max_workers=_PREFILL_WORKERS, name_hint='prefill_attention_o_rop)→rotate`）○ | `_make_rope_block(heads=LOCAL_H, inverse=True)→rotate_block` | ○ 传递覆盖 |
| `o_proj.py` | `_rotate_output`*（`make_rope(heads=LOCAL_H, inverse=True, name_hint='attention_o_rope')→rotate`）○ | `_make_rope_block(heads=LOCAL_H, inverse=True)→rotate_block` | ○ 传递覆盖 |
| `o_proj.py` | `o_proj`*（`_make_o_proj()→o_proj`）○ | `_project_ob`, `_rotate_output`, `grouped_output` | ○ 传递覆盖 |
| `o_proj.py` | `prefill_o_proj`*（`_make_o_proj()→o_proj`）○ | `_project_ob`, `_prefill_rotate_output`, `grouped_output` | ○ 传递覆盖 |
| `o_proj.py` | `grouped_output`○ | `_grouped_output_block` | ○ 传递覆盖 |
| `o_proj.py` | `grouped_output_with_deps`○ | `_grouped_output_block` | ○ 传递覆盖 |
| `prefill_attn_c1a_full.py` | `normalize_compressed`*（`make_norm(width=C.HEAD_DIM)→normalize`）○ | `_make_norm_block(width=C.HEAD_DIM)→normalize_block` | ○ 传递覆盖 |
| `prefill_attn_c1a_full.py` | `normalize_index_key`*（`make_norm(width=C.INDEX_DIM)→normalize`）○ | `_make_norm_block(width=C.INDEX_DIM)→normalize_block` | ○ 传递覆盖 |
| `prefill_attn_c1a_full.py` | `paged_indexer`*（`make_paged_indexer()→paged_indexer`）○ | `_hierarchical_sparse_indexer` | ○ 传递覆盖 |
| `prefill_attn_c1a_full.py` | `paged_indexer_direct`*（`make_paged_indexer(direct_topk=True)→paged_indexer`）✗ | `_hierarchical_sparse_indexer` | ✗ 未覆盖 |
| `prefill_attn_c1a_full.py` | `prefill_attn_c1a_full`*（`make_prefill_attn_c1a_full()→prefill_attn_c1a_full_impl`）○ | `project_compressed`, `project_index_key（prefill_attn_c1a_full.py）`, `normalize_compressed`, `normalize_index_key（prefill_attn_c1a_full.py）`, `rotate_compressed（decode_attn_c2a_full.py）`, `rotate_index_key（prefill_attn_c1a_full.py）`, `paged_indexer（prefill_attn_c1a_full.py）`, `q_proj_qr`, `prefill_tp_output_all_reduce`, `prefill_c1a_partial`, `publish_compressed_cache`, `publish_index_cache` | ○ 传递覆盖 |
| `prefill_attn_c1a_full.py` | `prefill_attn_c1a_full_watch`*（`make_prefill_attn_c1a_full()→prefill_attn_c1a_full_impl`）✗ | `project_compressed`, `project_index_key（prefill_attn_c1a_full.py）`, `normalize_compressed`, `normalize_index_key（prefill_attn_c1a_full.py）`, `rotate_compressed（decode_attn_c2a_full.py）`, `rotate_index_key（prefill_attn_c1a_full.py）`, `paged_indexer_direct（prefill_attn_c1a_full.py）`, `q_proj_qr`, `prefill_tp_output_all_reduce`, `prefill_c1a_partial`, `publish_compressed_cache`, `publish_index_cache` | ✗ 未覆盖 |
| `prefill_attn_c1a_full.py` | `project_compressed`*（`make_bf16_projection(output_width=C.HEAD_DIM, width=C.D)→project`）○ | `_make_bf16_projection_block(output_width=C.HEAD_DIM, width=C.D)→project_block` | ○ 传递覆盖 |
| `prefill_attn_c1a_full.py` | `project_index_key`*（`make_bf16_projection(output_width=C.INDEX_DIM, width=C.HEAD_DIM)→project`）○ | `_make_bf16_projection_block(output_width=C.INDEX_DIM, width=C.HEAD_DIM)→project_block` | ○ 传递覆盖 |
| `prefill_attn_c1a_full.py` | `rotate_compressed`*（`make_rope(heads=1)→rotate`）✗ | `_make_rope_block(heads=1)→rotate_block` | ✗ 未覆盖 |
| `prefill_attn_c1a_full.py` | `rotate_index_key`*（`make_rope(head_dim=C.INDEX_DIM, heads=1, rope_dim=C.ROPE_DIM)→rotate`）○ | `_make_rope_block(head_dim=C.INDEX_DIM, heads=1, rope_dim=C.ROPE_DIM)→rotate_block` | ○ 传递覆盖 |
| `prefill_attn_c1a_full.py` | `prefill_attn_c1a_full_test`[jit]○ | `prefill_attn_c1a_full` | ○ 传递覆盖 |
| `prefill_attn_c1a_full.py` | `l3_prefill_attn_c1a_full_test`[host]⊕ | `prefill_attn_c1a_full_test` | ⊕ 独立 golden `golden_prefill_attn_c1a_full_case` |
| `prefill_attn_c1a_reindex.py` | `paged_indexer`*（`make_paged_indexer(use_candidates=True)→paged_indexer`）○ | `_hierarchical_sparse_indexer` | ○ 传递覆盖 |
| `prefill_attn_c1a_reindex.py` | `paged_indexer_direct`*（`make_paged_indexer(direct_topk=True, use_candidates=True)→paged_indexer`）✗ | `_hierarchical_sparse_indexer` | ✗ 未覆盖 |
| `prefill_attn_c1a_reindex.py` | `prefill_attn_c1a_reindex`*（`make_prefill_attn_c1a_reindex()→prefill_attn_c1a_reindex_impl`）○ | `paged_indexer（prefill_attn_c1a_reindex.py）`, `q_proj_qr`, `prefill_tp_output_all_reduce`, `prefill_c1a_partial` | ○ 传递覆盖 |
| `prefill_attn_c1a_reindex.py` | `prefill_attn_c1a_reindex_watch`*（`make_prefill_attn_c1a_reindex()→prefill_attn_c1a_reindex_impl`）✗ | `paged_indexer_direct（prefill_attn_c1a_reindex.py）`, `q_proj_qr`, `prefill_tp_output_all_reduce`, `prefill_c1a_partial` | ✗ 未覆盖 |
| `prefill_attn_c1a_reindex.py` | `prefill_attn_c1a_reindex_test`[jit]○ | `prefill_attn_c1a_reindex` | ○ 传递覆盖 |
| `prefill_attn_c1a_reindex.py` | `l3_prefill_attn_c1a_reindex_test`[host]⊕ | `prefill_attn_c1a_reindex_test` | ⊕ 独立 golden `golden_prefill_attn_c1a_reindex_case` |
| `prefill_attn_c1a_reuse.py` | `prefill_attn_c1a_reuse`○ | `q_proj_qr`, `prefill_tp_output_all_reduce`, `prefill_c1a_partial` | ○ 传递覆盖 |
| `prefill_attn_c1a_reuse.py` | `prefill_attn_c1a_reuse_test`[jit]○ | `prefill_attn_c1a_reuse` | ○ 传递覆盖 |
| `prefill_attn_c1a_reuse.py` | `l3_prefill_attn_c1a_reuse_test`[host]⊕ | `prefill_attn_c1a_reuse_test` | ⊕ 独立 golden `golden_prefill_attn_c1a_reuse_case` |
| `prefill_attn_c2a_full.py` | `prefill_attn_c2a_full`⊕ | `prefill_tp_output_all_reduce`, `c2a_full_partial` | ⊕ 独立 golden `make_golden(...)` |
| `prefill_attn_c2a_reuse.py` | `prefill_attn_c2a_reuse`⊕ | `prefill_tp_output_all_reduce`, `c2a_reuse_partial` | ⊕ 独立 golden `golden_c2a_reuse` |
| `prefill_attn_swa.py` | `prefill_attn_swa`⊕ | `prefill_o_proj`, `prefill_q_proj_qr`, `prefill_q_proj_rope`, `prefill_tp_output_all_reduce`, `prefill_attend_window`, `prefill_gather_window`, `prefill_publish_window`, `prefill_kv_proj_rope` | ⊕ 独立 golden `golden_swa` |
| `prefill_c1a_common.py` | `prefill_c1a_partial`○ | `o_proj`, `kv_proj_rope`, `q_proj_rope`, `attend_sparse_cache`, `publish_window（prefill_c1a_common.py）` | ○ 传递覆盖 |
| `prefill_c1a_full.py` | `prefill_c1a_full`○ | `mhc_mixes`, `mhc_post`, `mhc_pre`, `rms_norm` | ○ 传递覆盖 |
| `prefill_c1a_full.py` | `prefill_c1a_full_test`[jit]○ | `prefill_c1a_full` | ○ 传递覆盖 |
| `prefill_c1a_full.py` | `l3_prefill_c1a_full_test`[host]⊕ | `prefill_c1a_full_test` | ⊕ 独立 golden `golden_prefill_c1a_full_case` |
| `prefill_c1a_indexer.py` | `rotate_index_query`*（`make_rope(head_dim=INDEX_DIM, heads=INDEX_H, rope_dim=ROPE_DIM)→rotate`）✗ | `_make_rope_block(head_dim=INDEX_DIM, heads=INDEX_H, rope_dim=ROPE_DIM)→rotate_block` | ✗ 未覆盖 |
| `prefill_c1a_indexer.py` | `_merge_topk_level`✗ | `_merge_topk_pairs` | ✗ 未覆盖 |
| `prefill_c1a_reindex.py` | `prefill_c1a_reindex`○ | `mhc_mixes`, `mhc_post`, `mhc_pre`, `rms_norm` | ○ 传递覆盖 |
| `prefill_c1a_reindex.py` | `prefill_c1a_reindex_test`[jit]○ | `prefill_c1a_reindex` | ○ 传递覆盖 |
| `prefill_c1a_reindex.py` | `l3_prefill_c1a_reindex_test`[host]⊕ | `prefill_c1a_reindex_test` | ⊕ 独立 golden `golden_prefill_c1a_reindex_case` |
| `prefill_c1a_reuse.py` | `prefill_c1a_reuse`○ | `mhc_mixes`, `mhc_post`, `mhc_pre`, `rms_norm` | ○ 传递覆盖 |
| `prefill_c1a_reuse.py` | `prefill_c1a_reuse_test`[jit]○ | `prefill_c1a_reuse` | ○ 传递覆盖 |
| `prefill_c1a_reuse.py` | `l3_prefill_c1a_reuse_test`[host]⊕ | `prefill_c1a_reuse_test` | ⊕ 独立 golden `golden_prefill_c1a_reuse_case` |
| `prefill_c2a_full.py` | `attention_hc_pre`○ | `mhc_mixes`, `mhc_pre`, `rms_norm` | ○ 传递覆盖 |
| `prefill_c2a_full.py` | `prefill_c2a_full`⊕ | `mhc_post`, `prefill_attn_c2a_full`, `attention_hc_pre` | ⊕ 独立 golden `make_golden(...)` |
| `prefill_c2a_reuse.py` | `prefill_c2a_reuse`⊕ | `mhc_post`, `prefill_attn_c2a_reuse`, `attention_hc_pre` | ⊕ 独立 golden `make_golden(...)` |
| `prefill_layer.py` | `moe_hc_pre`○ | `mhc_mixes`, `mhc_pre` | ○ 传递覆盖 |
| `prefill_layer.py` | `_compact_local_tokens`○ | `_local_token_count` | ○ 传递覆盖 |
| `prefill_layer.py` | `_scatter_local_tokens`○ | `_local_token_count` | ○ 传递覆盖 |
| `prefill_layer.py` | `prefill_moe_sublayer`⊕ | `_moe_core`, `_compact_local_tokens`, `_local_token_count`, `_scatter_local_tokens`, `moe_hc_pre`, `rms_norm` | ⊕ 独立 golden `make_layer_golden(...)` |
| `prefill_layer.py` | `prefill_ffn_restore`⊕ | `prefill_tp_output_all_reduce`, `mhc_post`, `widen_to_fp32` | ⊕ 独立 golden `make_layer_golden(...)` |
| `prefill_swa.py` | `prefill_swa`⊕ | `mhc_mixes`, `mhc_post`, `mhc_pre`, `prefill_attn_swa`, `rms_norm` | ⊕ 独立 golden `golden_prefill_swa_case` |
| `qkv_proj_rope.py` | `_normalize_kv`*（`make_norm(name_hint='attention_kv_norm', width=HEAD_DIM)→normalize`）○ | `_make_norm_block(width=HEAD_DIM)→normalize_block` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `_normalize_q`*（`make_norm(name_hint='attention_q_norm', width=Q_LORA)→normalize`）○ | `_make_norm_block(width=Q_LORA)→normalize_block` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `_prefill_normalize_kv`*（`make_norm(max_workers=_PREFILL_WORKERS, name_hint='prefill_attention_kv_no, width=HEAD_DIM)→normalize`）○ | `_make_norm_block(width=HEAD_DIM)→normalize_block` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `_prefill_normalize_q`*（`make_norm(max_workers=_PREFILL_WORKERS, name_hint='prefill_attention_q_nor, width=Q_LORA)→normalize`）○ | `_make_norm_block(width=Q_LORA)→normalize_block` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `_prefill_rotate_kv`*（`make_rope(heads=1, max_workers=_PREFILL_WORKERS, name_hint='prefill_attention_kv_ro)→rotate`）○ | `_make_rope_block(heads=1)→rotate_block` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `_prefill_rotate_q`*（`make_rope(heads=LOCAL_H, max_workers=_PREFILL_WORKERS, name_hint='prefill_attention_q_rop)→rotate`）○ | `_make_rope_block(heads=LOCAL_H)→rotate_block` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `_rotate_kv`*（`make_rope(heads=1, name_hint='attention_kv_rope')→rotate`）○ | `_make_rope_block(heads=1)→rotate_block` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `_rotate_q`*（`make_rope(heads=LOCAL_H, name_hint='attention_q_rope')→rotate`）○ | `_make_rope_block(heads=LOCAL_H)→rotate_block` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `kv_proj_rope`*（`_make_kv_proj_rope()→kv_proj_rope`）○ | `_project_kv`, `_normalize_kv`, `_rotate_kv` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `prefill_q_proj_qr`*（`_make_q_proj_qr()→q_proj_qr`）○ | `_prefill_normalize_q`, `_prefill_project_qa` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `prefill_q_proj_rope`*（`_make_q_proj_rope()→q_proj_rope`）○ | `_project_qb`, `_prefill_rotate_q` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `q_proj_qr`*（`_make_q_proj_qr()→q_proj_qr`）○ | `_project_qa`, `_normalize_q` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `q_proj_rope`*（`_make_q_proj_rope()→q_proj_rope`）○ | `_project_qb`, `_rotate_q` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `prefill_kv_proj_rope`○ | `_project_kv`, `_prefill_normalize_kv`, `_prefill_rotate_kv` | ○ 传递覆盖 |
| `qkv_proj_rope.py` | `qkv_proj_rope`○ | `kv_proj_rope`, `q_proj_qr`, `q_proj_rope` | ○ 传递覆盖 |
| `rmsnorm.py` | `rms_norm_test`[jit]⊕ | `rms_norm` | ⊕ 独立 golden `golden_rms_norm_case` |

## glm5_3_flash

GLM-5.3-Flash（a2a3，TP16/EP16，W8A8 INT8）：KDA + NoPE-MLA 混合骨干与 kpool DSA indexer（staging 阶段，仅配置、golden 与算子 ABI）。

共 58 个算子：基本算子 44 个，非基本算子 14 个。golden 覆盖 26/58（`⊕` 13、`◉` 12、`○` 1）；未覆盖 32。其中**有独立 golden 函数的基本算子 12 个**（见下方同名小节）。

顶层入口（未被其它算子调用，共 13 个）：

- `decode_kda.py`：`decode_kda_test`
- `decode_sparse_attn.py`：`decode_sparse_attn_test`
- `kda_conv.py`：`kda_conv_prefill_test`, `kda_conv_decode_test`
- `kda_output.py`：`kda_output_test`
- `kda_projection.py`：`kda_projection_test`
- `mla_cache.py`：`mla_cache_write_test`
- `mla_epilog.py`：`mla_epilog_prefill_test`, `mla_epilog_decode_test`
- `mla_prolog.py`：`mla_prolog_test`, `mla_absorb_query_test`
- `prefill_kda.py`：`prefill_kda_test`
- `prefill_sparse_attn.py`：`prefill_sparse_attn_test`

### 基本算子（44 个）

算子名后第二组标记为 golden 状态（图例见文首）。

| 文件 | 基本算子 |
| --- | --- |
| `attention.py` | `vision_attention`✗, `vision_axial_rope_table`✗ |
| `attention_tp.py` | `tp_all_reduce`✗ |
| `decode_indexer.py` | `indexer_proj`✗, `indexer_kpool`✗, `indexer_score`✗, `indexer_topk`✗, `indexer_expand`✗ |
| `decode_kda.py` | `decode_kda`◉ |
| `decode_sparse_attn.py` | `decode_sparse_attn`◉ |
| `dense_mlp.py` | `dense_mlp`✗ |
| `expert_routed.py` | `expert_routed`✗ |
| `expert_shared.py` | `expert_shared`✗ |
| `fusion.py` | `vision_fusion`✗ |
| `gate.py` | `moe_gate`✗ |
| `indexer_cache.py` | `indexer_cache_write`✗ |
| `kda_conv.py` | `kda_conv_prefill`◉, `kda_conv_decode`◉ |
| `kda_output.py` | `kda_output`◉ |
| `kda_projection.py` | `kda_projection`◉ |
| `lm_head.py` | `lm_head`✗ |
| `lookup_embedding.py` | `lookup_embedding`✗ |
| `merger.py` | `vision_merger`✗ |
| `mhc.py` | `mhc_mixes`✗, `mhc_pre`✗, `mhc_post`✗, `mhc_head`✗ |
| `mla_cache.py` | `mla_cache_write`◉ |
| `mla_epilog.py` | `mla_epilog_prefill`◉, `mla_epilog_decode`◉ |
| `mla_prolog.py` | `mla_prolog`◉, `absorb_query`◉ |
| `mlp.py` | `vision_mlp`✗ |
| `mtp_projection.py` | `mtp_projection`✗ |
| `patch_embed.py` | `vision_patch_embed`✗ |
| `prefill_indexer.py` | `indexer_proj`✗, `indexer_kpool`✗, `indexer_score`✗, `indexer_topk`✗, `indexer_expand`✗ |
| `prefill_kda.py` | `prefill_kda`◉ |
| `rmsnorm.py` | `rmsnorm`✗, `rmsnorm_quant`✗, `add_rmsnorm`✗ |

### 有独立 golden 函数的基本算子（12 个）

这些叶子算子由专属 torch 参考对比：`⊕` 表示该算子本身就是被测入口，`◉` 表示它经由唯一调用它的测试入口对比。

| 文件 | 基本算子 | golden 函数 | 接线 |
| --- | --- | --- | --- |
| `decode_kda.py` | `decode_kda` | `_golden_tensors` | 经 `decode_kda_test` |
| `decode_sparse_attn.py` | `decode_sparse_attn` | `golden_decode_sparse_attn_case` | 经 `decode_sparse_attn_test` |
| `kda_conv.py` | `kda_conv_decode` | `_golden_decode_tensors` | 经 `kda_conv_decode_test` |
| `kda_conv.py` | `kda_conv_prefill` | `_golden_decode_tensors` | 经 `kda_conv_prefill_test` |
| `kda_output.py` | `kda_output` | `golden_kda_output_tensors` | 经 `kda_output_test` |
| `kda_projection.py` | `kda_projection` | `golden_kda_projection_tensors` | 经 `kda_projection_test` |
| `mla_cache.py` | `mla_cache_write` | `golden_mla_cache_case` | 经 `mla_cache_write_test` |
| `mla_epilog.py` | `mla_epilog_decode` | `golden_mla_epilog_decode_case` | 经 `mla_epilog_decode_test` |
| `mla_epilog.py` | `mla_epilog_prefill` | `golden_mla_epilog_prefill_case` | 经 `mla_epilog_prefill_test` |
| `mla_prolog.py` | `absorb_query` | `golden_mla_absorb_case` | 经 `mla_absorb_query_test` |
| `mla_prolog.py` | `mla_prolog` | `golden_mla_prolog_case` | 经 `mla_prolog_test` |
| `prefill_kda.py` | `prefill_kda` | `_golden_tensors` | 经 `prefill_kda_test` |

### 非基本算子（14 个）

| 文件 | 算子 | 调用的算子 | golden |
| --- | --- | --- | --- |
| `decode_kda.py` | `decode_kda_test`[jit]⊕ | `decode_kda` | ⊕ 独立 golden `_golden_tensors` |
| `decode_sparse_attn.py` | `decode_sparse_attn_test`[jit]⊕ | `decode_sparse_attn` | ⊕ 独立 golden `golden_decode_sparse_attn_case` |
| `kda_conv.py` | `kda_conv_prefill_test`[jit]⊕ | `kda_conv_prefill` | ⊕ 独立 golden `_golden_decode_tensors` |
| `kda_conv.py` | `kda_conv_decode_test`[jit]⊕ | `kda_conv_decode` | ⊕ 独立 golden `_golden_decode_tensors` |
| `kda_output.py` | `kda_output_test`[jit]⊕ | `kda_output` | ⊕ 独立 golden `golden_kda_output_tensors` |
| `kda_projection.py` | `kda_projection_test`[jit]⊕ | `kda_projection` | ⊕ 独立 golden `golden_kda_projection_tensors` |
| `mla_cache.py` | `mla_cache_write_test`[jit]⊕ | `mla_cache_write` | ⊕ 独立 golden `golden_mla_cache_case` |
| `mla_epilog.py` | `mla_epilog_prefill_test`[jit]⊕ | `mla_epilog_prefill` | ⊕ 独立 golden `golden_mla_epilog_prefill_case` |
| `mla_epilog.py` | `mla_epilog_decode_test`[jit]⊕ | `mla_epilog_decode` | ⊕ 独立 golden `golden_mla_epilog_decode_case` |
| `mla_prolog.py` | `mla_prolog_test`[jit]⊕ | `mla_prolog` | ⊕ 独立 golden `golden_mla_prolog_case` |
| `mla_prolog.py` | `mla_absorb_query_test`[jit]⊕ | `absorb_query` | ⊕ 独立 golden `golden_mla_absorb_case` |
| `prefill_kda.py` | `prefill_kda_test`[jit]⊕ | `prefill_kda` | ⊕ 独立 golden `_golden_tensors` |
| `prefill_sparse_attn.py` | `prefill_sparse_attn`○ | `absorb_query` | ○ 传递覆盖 |
| `prefill_sparse_attn.py` | `prefill_sparse_attn_test`[jit]⊕ | `prefill_sparse_attn` | ⊕ 独立 golden `golden_prefill_sparse_attn_case` |

## golden 校验状态说明

### 缺口的性质

- **端到端融合前向为 perf-only（`P`）**：`deepseek_v4_flash_mtp`：`l3_decode_fwd_mtp`、`l3_prefill_fwd`；`deepseek_v4_flash_dspark`：`l3_decode_fwd`、`l3_decode_fwd_dspark`；`deepseek_v4_pro`：`l3_decode_fwd`。这些整层/整网前向入口以 `golden_fn=None` 运行，只测性能与可编译性，它们独占的内部阶段因此也没有 golden；本仓库的精度承诺建立在「单层 / 单算子均有 golden」的分层校验上。
- **数据回放入口（`⊖`）**：`deepseek_v4_flash_mtp`：`l3_decode_fwd`。此类入口以 `golden_data=<目录>` 回放此前生成的期望值，库内已无对应的 golden 函数，因此**不计入**「有独立 golden 函数的基本算子」清单。
- **脚本内参考链**：`qwen3_14b` 的 `decode_fwd.py --validate-fwd` 不走 golden harness，而在脚本内与 torch host 参考链做 argmax + logits 对比；该形式无法由 `run(...)` 接线静态识别，相关算子在表中仍标 `✗`。
- **完全没有 golden 接线的文件**：`qwen3_14b` 的 A8W8 变体 `prefill_fwd_a8w8.py`、`decode_layer_a8w8.py`。
- **staging 阶段模型**：`glm5_3_flash` 仅落地配置、golden 与算子 ABI，视觉子包（`patch_embed.py`、`merger.py`）、indexer 五件套、`mhc.py`、`rmsnorm.py`、`dense_mlp.py`、experts、`lm_head.py`、`lookup_embedding.py`、`mtp_projection.py` 等尚未接 golden。
- **变体与影子副本**：`deepseek_v4_1_flash` 的 sharded/rank 变体、`*_watch` 调试入口等复用主实现的 golden 链路或不单独校验，因此成组地停在 `○`/`✗`。
- **工具算子**：`qwen3_14b` 的 `rope_qkv_regen` 是 codegen 工具，无需数值 golden。
- **extern 算子**：`qwen3_14b` 的 `paged_attention_rope_cce` 等经 CANN 图内嵌调用，pl 层没有独立接线。

### 分析限制

- 状态由静态 AST 分析得出，不做运行时验证。3 处 `run(...)` 接线未能静态解析（`deepseek_v4_1_flash/decode_common.py:583`、`deepseek_v4_flash_dspark/qkv_proj_rope.py:1271`、`deepseek_v4_flash_mtp/gate.py:433`），个别算子的状态可能因此误报。
- `◉`（专属入口直连）与经工厂函数体扫描得到的接线属于**推断**而非字面接线：前者取「`⊕` 入口唯一调用的基本算子」这一约定，后者出现在 `run_c1a(mode, kernel_factory, golden_fn)` 之类的封装上。
- `X` / `X_test` 同体孪生（`X = pl.jit.inline(_X)`、`X_test = pl.jit(_X)`）按「其一被校验即视为已校验」处理：报告里把 golden 归属到非 `_test` 的规范名（`⊕`，并在「接线」列注明实际入口），`_test` 名标 `T`。
- `○` 表示「进入某条 golden 对比链路」，不等价于该算子的每条分支都被独立断言。
- golden 函数标签为 `make_xxx(...)` 的形式表示 golden 由工厂函数现场构造，同样算作有独立 golden 函数；标签 `golden_fn`、`_golden_tensors` 一类是文件内定义的局部参考函数。
- 数据文件 `build_output/*.json` 与生成脚本均在 gitignore 范围内，重新生成命令：依次运行 `build_output/golden_annotation/` 下的 `analyze_ops.py`、`golden_map.py`、`make_status.py`、`gen_doc.py`（需 conda 环境 `pypto` 的 Python 3.11）。
