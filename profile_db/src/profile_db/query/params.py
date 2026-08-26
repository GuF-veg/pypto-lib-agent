# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Query parameter models — the single source the CLI/MCP entries are
generated from (DESIGN.md 6.6/9).

Every registered query owns exactly one pydantic model here, so tool
schemas, CLI arguments, and validation stay one definition. Models are
deliberately thin: semantic guards that need database context (run
existence, multi-rank refusal, window ordering) live in the handlers and
raise ``QueryError``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PfdbParams(BaseModel):
    """Shared base: forbid unknown keys so typos surface loudly."""

    model_config = ConfigDict(extra="forbid")


class RunsListParams(PfdbParams):
    rank: str | None = Field(default=None, description="rank_label filter")


class RunIdParams(PfdbParams):
    run_id: int


class OverviewParams(RunIdParams):
    pass


class InventoryParams(RunIdParams):
    pass


class DensityParams(RunIdParams):
    engine: str | None = Field(default=None, description="restrict to one engine")
    bands: int = Field(default=20, ge=1, le=1000, description="display buckets")


class SparseRegionsParams(RunIdParams):
    engine: str | None = None
    top_k: int = Field(default=5, ge=1, le=100)


class WhySparseParams(RunIdParams):
    band: int = Field(ge=0, description="display bucket index from density")
    engine: str | None = None


class RegionParams(RunIdParams):
    t0_us: float
    t1_us: float
    family: str | None = None
    core: int | None = None


class CoreParams(RunIdParams):
    core: int = Field(ge=0)


class TaskIdParams(RunIdParams):
    task_id: str


class TaskParams(TaskIdParams):
    pass


class DepsParams(TaskIdParams):
    direction: str = Field(default="out", pattern="^(in|out|all)$")


class SubgraphParams(TaskIdParams):
    depth: int = Field(default=2, ge=1, le=6)
    max_nodes: int = Field(default=24, ge=1, le=200)


class WhyLateParams(TaskIdParams):
    pass


class WhyLongParams(TaskIdParams):
    pass


class RowsParams(TaskIdParams):
    pass


class SchedulerParams(TaskIdParams):
    pass


class EarlyDispatchParams(TaskIdParams):
    pass


class PmuParams(TaskIdParams):
    pass


class CriticalPathParams(RunIdParams):
    kind: str = Field(default="observed", pattern="^(observed|static)$")


class PerfHintsParams(RunIdParams):
    pass


class MemoryParams(RunIdParams):
    pass