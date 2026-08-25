# profile_db — 面向 Agent 的 Profiling/Feedback 数据库

查询优先、数据可弃的 profiling 工作台（定位见 DESIGN.md 1.2）：

- profiling 数据强时效性：每轮算子修改后旧数据大多失去价值，仅余对比之用；
- 因此**查询能力是重心**，存储做薄：默认 link 归档（不复制工件），
  支持 prune 主动清理，整个 `.pfdb/` 可删除后从 `build_output` 重建；
- 对 agent 提供分层查询（整体 → 稀疏段 → 原因 → 算子 → 依赖），
  事实恒带证据状态与字节预算；视觉模态走语义结构化 + 按需渲染。

顶层设计见 [DESIGN.md](./DESIGN.md)（架构、schema、口径、T0–T10 里程碑）。

## 当前状态

- ✅ T0 骨架：schema 迁移（v1 建表 + 0002 修正）、事实 DSL v2、连接与写锁、
  合成工件生成器、CLI `init`、CI/lint 接线。
- ✅ T1 摄取：L2 泳道族（records/deps/name_map/merged 清单）——时钟域合并
  复用上游 `read_perf_data`，link 默认存档 + sha256，事务幂等；真实
  Qwen3Decode 捕获验收通过（266 任务 / 706 物理行 / 2546 边，与转换器直出
  零容差对拍）。
- ✅ T2 摄取：文本类证据——`report/perf_hints.log`（编译器文本原样）与
  `report/memory_after_AllocateMemoryAddr.txt`（缓冲占用），
  `dfx_outputs/pmu.csv` 动态列名长表化；缺省文件合法（表空、查询层报
  unavailable）；`--bench` / `--bench-log` 注册基准数据（重摄取时保留合并）；
  环境元数据（runtime_cfg）路径脱敏。
- ✅ T3 衍生层：入库即衍生（`derived/` 纯函数，随 ingest 事务落表）——
  `time_band` 5µs 密度带与 sparse/drain_tail 判定、`idle_gap` 核级空闲段
  四类确定性分类（证据标注）、`cpm_path` observed/static 双路径（与上游
  `critical_path` 逐条件同构，真实捕获对拍零差）、stall 四段分解与
  early-dispatch 四态证明；迁移 `0003` 补行级 dispatch/receive/finish。
- ✅ T4 分层查询引擎：注册表（每条查询强绑定 owner question + pydantic
  参数单一同源）+ Z0–Z4 共 17 条查询（runs_list/overview/inventory、
  density/sparse_regions、region/why_sparse/core、task/deps/subgraph、
  why_late/why_long/rows/scheduler/early_dispatch/pmu）；事实 DSL v2 输出、
  字节预算 `TRUNCATED` 显式收尾、unavailable 语义、多 rank 守门、无原始
  JSON 泄漏检查器；金质题库 20 题快照（含 6.4 全会话）+ 真实捕获锚点。
- ⬜ T5+ 见 DESIGN.md 第 11 节。

## 安装与使用

```bash
# conda pypto 环境内
pip install -e ./profile_db --no-build-isolation

# 初始化数据库（默认 <cwd>/.pfdb/profile.duckdb；PFDB_PATH 可覆盖）
pfdb init

# 摄取一次采集（profiling 命令跑完后 build_output/ 下的 dfx_outputs 目录）
pfdb ingest build_output/Qwen3Decode_<ts>/dfx_outputs --platform a2a3 --device 0
# 重复 ingest 幂等（按 records 文件 sha256 识别同一 run，行数不变）；
# 需要归档副本时加 --copy（默认 link 只记路径 + sha256）

# 查看版本
pfdb --version
```

## 测试与检查

```bash
pytest profile_db/tests -v
python -m ruff check profile_db tools --config ruff.toml
PYTHONPATH=profile_db/src lint-imports
```

运行时数据（`.pfdb/`）已加入 git 忽略，不入库。