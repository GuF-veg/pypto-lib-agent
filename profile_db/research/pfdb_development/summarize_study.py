# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Export small, source-hashed study outcomes without transcripts or local paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REFERENCE_CHECKS = {"valid_evidence_ids", "evidence_query_coverage"}


def summarize(root: Path) -> dict:
    source = root / "results.json"
    results = json.loads(source.read_text())
    if not isinstance(results, list):
        raise ValueError("expected an offline runner result list")
    usage = json.loads((root / "usage.json").read_text()) if (root / "usage.json").exists() else []
    usage_by_attempt = {r["attempt"]: r["sessions"] for r in usage}
    conditions = {}
    attempts = []
    for row in results:
        checks = row.get("score", {}).get("checks", {})
        fields = {k: v for k, v in checks.items() if k not in REFERENCE_CHECKS}
        completed = row["returncode"] == 0 and not row["timed_out"] and not row["errors"]
        name = f"{row['case_id']}-{row['condition']}-{row['replicate']}"
        item = dict(case_id=row["case_id"], condition=row["condition"], replicate=row["replicate"],
                    completed=completed, fields_passed=bool(fields) and all(fields.values()),
                    fields=fields, reference_checks={k: checks.get(k) for k in sorted(REFERENCE_CHECKS)},
                    rubric_passed=row.get("score", {}).get("passed", False),
                    error_types=[r.get("type", "unknown") for r in row["errors"]],
                    prompt_bytes=row["prompt_bytes"], prompt_sha256=row["prompt_sha256"], wall_s=row["wall_s"],
                    usage=usage_by_attempt.get(name, []))
        attempts.append(item)
        tally = conditions.setdefault(row["condition"], dict(attempts=0, completed=0, fields_passed=0,
                                                            rubric_passed=0, prompt_bytes=0))
        for key in ("completed", "fields_passed", "rubric_passed", "prompt_bytes"):
            tally[key] += item[key]
        tally["attempts"] += 1
    return dict(study_id=root.name, result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                protocol_sha256=hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest(),
                model=results[0]["model"] if results else None, conditions=conditions, attempts=attempts,
                scoring_scope="exact fields and existing fact IDs/query coverage only; free claims require review",
                independence="replicate labels, not reproducible serving seeds; fixed task-set exploration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite a published study summary")
    args.output.write_text(json.dumps([summarize(root) for root in args.experiments], indent=2) + "\n")
