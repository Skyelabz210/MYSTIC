---
name: qmnf-innovations
description: Rigorous gap analysis and innovation audit for QMNF/NINE65/MYSTIC or related exact-arithmetic, FHE, and quantum-substrate systems. Use when asked to verify that novel implementations exist and are correct, map innovations to code/tests, design validation plans, or identify expansion opportunities in this ecosystem.
---

# QMNF Innovations

## Overview

Perform evidence-based audits that map claimed innovations to implementations, validate mathematical and engineering rigor, and surface gaps or expansion opportunities across QMNF/NINE65/MYSTIC-style systems.

## Core Capabilities

- Build an innovation inventory and coverage matrix.
- Verify implementation evidence (code, tests, benchmarks) for each innovation.
- Produce rigorous gap analyses across architecture, mathematics, implementation, validation, security, and operations.
- Propose expansion or inclusion opportunities aligned with system invariants.

## Workflow

### 1. Scope and constraints

- Confirm target repo(s), output format, and whether to write report files.
- Read `AGENTS.md` in the repo root if present and follow local rules.
- Identify project summaries, design docs, and prior gap analyses.

### 2. Build the innovation inventory

- Load `references/innovation-catalog.md` for canonical innovations.
- Extract additional innovations from repo docs and code comments.
- Maintain a working list grouped by domain (arithmetic, FHE, entropy, quantum, infrastructure).

### 3. Map innovations to evidence

- Locate implementation paths with `rg` for innovation names and core primitives.
- Record evidence: file paths, functions, tests, benchmarks, and outputs.
- Mark status as Implemented, Partial, or Missing with clear justification.

### 4. Evaluate correctness and risk

- Check boundary conditions, error handling, conditioning, and invariants.
- Identify TODOs, stubs, or placeholder logic; call out unverified claims.
- Assess performance claims only when evidence exists; otherwise flag as unvalidated.

### 5. Validate with a testing plan

- Enumerate existing tests and datasets; note coverage holes.
- Propose minimal high-signal tests for each critical innovation.
- Distinguish synthetic validation from real-world scenarios.

### 6. Identify expansion and inclusion opportunities

- Propose additions that extend the architecture without violating invariants.
- Highlight missing integration seams or adjacent research directions.
- Separate speculative research from actionable engineering.

### 7. Produce deliverables

- Write the default deliverables for a rigorous audit:
  - `gap_analysis.md`
  - `mathematical_gap_analysis.md`
  - `critical_gaps_summary.md`
- Use templates in `references/report-templates.md` unless the user requests another format.
- Include an Innovation Coverage Matrix (summary + detailed appendix).

## Evidence Standards

- Cite concrete artifacts: file paths, functions, tests, and benchmark outputs.
- Avoid assumptions; label unknowns explicitly.
- Prefer reproducible steps and minimal test plans over broad speculation.

## Output Rules

- Keep reports concise but rigorous; prioritize high-severity gaps first.
- Include a prioritized action list with clear success metrics.
- Note any access limitations (missing repos, no test data, no build).

## Resources

- `references/innovation-catalog.md` - canonical innovation list and deep dives.
- `references/report-templates.md` - report templates and matrix format.
