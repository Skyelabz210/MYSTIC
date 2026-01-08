# QMNF Innovation Audit Report Templates

Use these templates as a starting point. Trim sections that do not apply.

## Template: Gap Analysis (general)

Title: <project> Gap Analysis

1. Scope and Assumptions
- Repos and paths reviewed
- Data sources and test data used
- Known constraints (time, access, build failures)

2. Innovation Coverage Matrix (summary)
- Provide a condensed table; full matrix can be appended.

3. Architecture and Integration Gaps
- Missing components, integration seams, or brittle coupling
- Dependency risks and version drift

4. Mathematical and Algorithmic Gaps
- Correctness risks
- Precision/conditioning risks
- Proof obligations or missing derivations

5. Implementation Gaps
- Unimplemented stubs, TODOs, placeholder logic
- Error handling gaps
- Performance hot spots

6. Validation and Testing Gaps
- Missing unit/integration/perf/regression tests
- Missing real-world datasets or historical scenarios

7. Security and Integrity Gaps
- PRNG validation gaps
- Data integrity gaps
- Attack surface notes

8. Operational Gaps
- Scaling/latency/throughput constraints
- Observability and monitoring gaps

9. Maintainability and Tech Debt
- Config hard-coding, readability, missing docs
- Module boundaries and interfaces

10. Priority Ranking
- High / Medium / Low with rationale

## Template: Mathematical Gap Analysis

Title: <project> Mathematical Gap Analysis

1. Mathematical Foundations
- Formal definitions, invariants, and constraints
- Missing proofs or formal guarantees

2. Stability and Conditioning
- Numerical/finite-field stability concerns
- Condition number or error propagation checks

3. Model and Attractor Dynamics
- Basin definitions and boundary behavior
- Multi-attractor coexistence handling

4. Algorithmic Complexity
- Time and space complexity gaps
- Bottlenecks without scaling plans

5. Validation of Mathematical Claims
- Tests that confirm or refute claims
- Required benchmark comparisons

## Template: Critical Gaps Summary

Title: Critical Gaps Summary

- Top critical blockers (short list)
- Immediate actions (1-2 weeks)
- Short-term actions (1 month)
- Medium-term actions (2-3 months)
- Success metrics for closure

## Template: Innovation Coverage Matrix (detailed)

Columns:
- Innovation
- Claim summary
- Implementation locations (files/functions)
- Evidence (tests/benchmarks/plots)
- Status: Implemented | Partial | Missing
- Risks / Failure modes
- Recommended next steps

## Template: Expansion and Inclusion Opportunities

- Adjacent capabilities that fit system invariants
- Missing features implied by design (gaps between modules)
- Research directions with clear validation path
