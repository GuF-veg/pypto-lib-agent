# profile_db — 面向 Agent 的 Profiling/Feedback 数据库

查询优先、数据可弃的 profiling 工作台（定位见 DESIGN.md 1.2）：

- profiling 数据强时效性：每轮算子修改后旧数据大多失去价值，仅余对比之用；
- 因此**查询能力是重心**，存储做薄：默认 link 归档（不复制工件），
  支持 prune 主动清理，整个 `.pfdb/` 可删除后从 `build_output` 重建；
- 对 agent 提供分层查询（整体 → 稀疏段 → 原因 → 算子 → 依赖），
  事实恒带证据状态与字节预算；视觉模态走语义结构化 + 按需渲染。

顶层设计见 [DESIGN.md](./DESIGN.md)（架构、schema、口径、T0–T10 里程碑）。

## 当前状态

- ✅ T0 骨架：schema v1 迁移、事实 DSL v2、连接与写锁、合成工件生成器、
  CLI `init`、CI/lint 接线。
- ⬜ T1+ 见 DESIGN.md 第 11 节。

## 安装与使用

```bash
# conda pypto 环境内
pip install -e ./profile_db --no-build-isolation

# 初始化数据库（默认 <cwd>/.pfdb/profile.duckdb；PFDB_PATH 可覆盖）
pfdb init

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