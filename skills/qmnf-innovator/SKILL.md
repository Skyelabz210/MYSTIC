---
name: qmnf-innovator
description: Analyze, explore, design, test, or develop QMNF/NINE65/MYSTIC innovations and related systems. Trigger whenever work enters advanced mathematical constructs or applied sciences (research, modeling, proofs, algorithms, cryptography, quantum, chaos, exact arithmetic) to bootstrap knowledge that challenges intractability assumptions and supports selecting proper constructs at the right time, plus requests for innovation audits, mapping claims to code/docs, resource indexes, gap analyses, design proposals, or validation plans across this repo.
---

# QMNF Innovator

## Overview

Enable rapid, evidence-based innovation work for QMNF/NINE65/MYSTIC, with clear mappings from claims to code, docs, and tests using ASCII-only outputs.

## Workflow

### 1. Scope the request

Clarify the deliverable and scope (analysis, design, test plan, resource index, or code changes). Ask only for missing essentials such as target component or expected output format.

### 2. Load canonical references first

Open these in priority order as needed:
- INNOVATION_RESOURCE_INDEX.md
- NINE65_CODEX_REFERENCE.md
- skills/qmnf-innovations/references/innovation-catalog.md
- qmnf_system_summary.md
- nine65_v2_complete/INDEX.md
- ENHANCED_GAP_ANALYSIS_WITH_NINE65.md
- GAP_RESOLUTION_REPORT.md
- mathematical_gap_analysis.md

### 3. Verify implementations with targeted searches

Use `rg` for fast verification. If no code is found, mark as docs-only or TBD.

Common searches from prior exploration:
```bash
rg -n "order_finding|order finding|K-Oracle|k_oracle" -S nine65_v2_complete
rg -n "gso|GSO" -S nine65_v2_complete
ls nine65_v2_complete/src/quantum
ls nine65_v2_complete/src/ahop
ls *.py
```

Open supporting math when needed:
```bash
sed -n '1,200p' mathematical_gap_analysis.md
```

### 4. Map innovations to evidence

For each innovation, record:
- Status: implemented, docs-only, or TBD
- Code entry points (file paths)
- Evidence docs or tests (if available)

Avoid unverified claims. If evidence is missing, say so explicitly.

### 5. Build or update resource indexes

When asked to create or update `INNOVATION_RESOURCE_INDEX.md`:
- Keep ASCII-only formatting.
- Organize sections for canonical catalogs, innovation list, implementation entry points,
  system integration points, and evidence/validation.
- Prefer concise bullets; include "TBD" where code is not found.

### 6. Design or test proposals

Provide a concise design outline and minimal, high-signal test ideas.
Only add tests when the request or scope calls for it.
