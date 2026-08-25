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
- ✅ T5 接口：公开 Python API（`profile_db.api.ProfileDB` + `Result` +
  `format_result`）与 CLI `list` / `query` 子命令（参数由注册表 pydantic
  模型自动生成，`facts/json/markdown` 三种输出）；CLI 与 API 输出逐字节
  一致（测试兜底）；`ProfileDB.memory()` 内存模式与磁盘模式对拍。
- ✅ T6 可视化渲染：`render/` 子包（只依赖 schema 表）——R0 whole /
  R1 window（依赖箭头）/ R2 task（就绪线）/ R3 core（空闲段着色），
  确定性样式常量 + Agg 后端；缓存 `<kind>-<params_key>.png` + 同名
  `.manifest.json`（SHA-256、尺寸、µs/px、图例、生成版本），字节预算超限
  自动降采样、总量上限 LRU 逐出；`ProfileDB.render(...)` 与 CLI
  `pfdb render` 均返回 `IMAGE` fact + `ImageRef`，同参数重复渲染 SHA-256
  逐字节一致，空窗/无边任务/未知目标不崩。
- ⬜ T7+ 见 DESIGN.md 第 11 节。

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

### 查询

```bash
# 列出可用 run（多 rank 库需 --rank 消歧）
pfdb list

# 顶线指标与拓扑（run id 由 ingest / list 给出）
pfdb query overview --run-id 1

# 密度带：整体哪里密集哪里稀疏
pfdb query density --run-id 1 --engine aiv --bands 20

# 稀疏带归因、单算子、依赖、微观测时
pfdb query why_sparse --run-id 1 --band 9 --engine aiv
pfdb query task --run-id 1 --task-id 4294967298
pfdb query deps --run-id 1 --task-id 4294967298 --direction in
pfdb query why_late --run-id 1 --task-id 4294967298

# 输出格式：facts（默认 DSL）/ json / markdown；字节预算 --budget
pfdb query overview --run-id 1 --format markdown
```

### 渲染（R0–R3）

```bash
# 全图总览（全部核 × 全时间轴）
pfdb render whole --run 1

# 时间窗（附依赖箭头与窗口边界）
pfdb render window --run 1 --t0 100 --t1 200

# 单算子邻域（本任务高亮 + 生产者/消费者 + 就绪线）
pfdb render task --run 1 --task-id 4294967298

# 单核时间轴（空闲段着色）
pfdb render core --run 1 --core 5

# 图像与清单落在 <db>/.pfdb/render/（可用 --render-dir 覆盖）
```

Python API（CLI 与它走同一引擎，`facts` 输出逐字节一致）：

```python
from profile_db.api import ProfileDB, format_result

db = ProfileDB()                    # 或 ProfileDB.memory() 纯内存工作集
r = db.query("overview", run_id=1)  # -> Result(facts, images, truncated)
print(format_result(r, "facts"))

img = db.render("whole", 1)         # -> Result(IMAGE fact, ImageRef, ...)
print(img.images[0].path)           # PNG 路径（供多模态模型读取）
db.close()
```

## 测试与检查

```bash
pytest profile_db/tests -v
python -m ruff check profile_db tools --config ruff.toml
PYTHONPATH=profile_db/src lint-imports
```

运行时数据（`.pfdb/`）已加入 git 忽略，不入库。