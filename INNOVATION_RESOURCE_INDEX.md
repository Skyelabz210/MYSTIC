# Innovation Resource Index (MYSTIC / QMNF / NINE65)

Purpose: quick access to the core innovations and where they live in docs, code,
and data. The references describe approaches that challenge standard assumptions
of computational infeasibility (e.g., exact arithmetic at scale, bootstrap-free
depth, constant-time sign detection).

## Start here (canonical catalogs)
- NINE65_CODEX_REFERENCE.md - canonical innovation list and deep dives
- skills/qmnf-innovations/references/innovation-catalog.md - audit catalog
- qmnf_system_summary.md - MYSTIC system summary and innovation impact
- nine65_v2_complete/INDEX.md - code map, tests, and key innovations
- ENHANCED_GAP_ANALYSIS_WITH_NINE65.md - gap-to-innovation mapping
- GAP_RESOLUTION_REPORT.md - resolved gaps status
- mathematical_gap_analysis.md - QMNF math gap analysis for phi, attractors,
  Cayley, and entropy

## Core innovations (names for fast lookup)
1. K-Elimination
2. Order Finding
3. K-Oracle
4. Encrypted Quantum (Sparse Grover)
5. State Taxonomy
6. GSO-FHE
7. CRT Shadow Entropy
8. Exact Coefficients
9. Persistent Montgomery
10. MobiusInt
11. Cyclotomic Phase
12. Integer Softmax
13. Pade Engine
14. MQ-ReLU

## Implementation entry points (quick open)
- k_elimination.py - exact RNS division (Python entry)
- nine65_v2_complete/src/arithmetic/k_elimination.rs - exact RNS division (Rust)
- nine65_v2_complete/src/arithmetic/exact_coeff.rs - dual-track coefficients
- nine65_v2_complete/src/arithmetic/persistent_montgomery.rs - persistent
  Montgomery form
- nine65_v2_complete/src/arithmetic/mobius_int.rs - signed integer split
- nine65_v2_complete/src/arithmetic/cyclotomic_phase.rs - ring trigonometry
- nine65_v2_complete/src/arithmetic/integer_softmax.rs - exact softmax
  normalization
- nine65_v2_complete/src/arithmetic/pade_engine.rs - integer-only transcendentals
- nine65_v2_complete/src/arithmetic/mq_relu.rs - O(1) sign detection
- shadow_entropy.py - shadow entropy PRNG (Python)
- nine65_v2_complete/src/entropy/shadow.rs - shadow entropy (Rust)
- nine65_v2_complete/src/entropy/wassan_noise.rs - WASSAN entropy
- nine65_v2_complete/src/ahop/grover.rs - Grover routines
- nine65_v2_complete/src/quantum/period_grover.rs - period-Grover fusion
- nine65_v2_complete/src/quantum/teleport.rs - K-Elimination teleportation
- phi_resonance_detector.py - phi resonance detection
- fibonacci_phi_validator.py - phi accuracy validator
- cayley_transform.py - Cayley unitary transform
- cayley_transform_nxn.py - NxN Cayley transform
- lyapunov_calculator.py - Lyapunov analysis
- nine65_v2_complete/src/chaos/attractor.rs - attractor basins
- nine65_v2_complete/src/chaos/lyapunov.rs - Lyapunov analysis
- weather_attractor_basins.json - attractor basin definitions
- mystic_integration_updated.py - unified MYSTIC pipeline
- mystic_v3_integrated.py - MYSTIC v3 integration
- mystic_v3_tuned.py - tuned MYSTIC pipeline

## Evidence and validation
- docs/MYSTIC_VALIDATION_REPORT.md
- docs/MYSTIC_VALIDATION_CYCLE_REPORT.md
- docs/MYSTIC_IMPROVEMENT_PROGRESS.md
- nine65_v2_complete/docs/V2_BENCHMARK_RESULTS.md
- nine65_v2_complete/audit/PRODUCTION_REPORT.md
- nine65_v2_complete/proofs/KElimination.lean
