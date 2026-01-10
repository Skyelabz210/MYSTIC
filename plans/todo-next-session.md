# TODO - Next Session

- Fix cyclic convolution: in `nine65_v2_complete/src/arithmetic/polypoly.rs` implement true `x^N - 1` reduction for `cyclic_convolution` (distinct from negacyclic), add/adjust tests to catch sign-wrapping errors.
- Restore validation data: replace or reintroduce required files under `nine65_v2_complete/data/` (e.g., `validation_results.json` and related CSV/JSON) or update scripts (`scripts/historical_validation.py`, `scripts/validate_with_training.py`, `minimal_validation_check_fixed.py`) to new locations.
- Review external k-elimination Lean repo (`https://skyelabz210.github.io/k-elimination-lean4/`): pull contents, map proofs/artifacts to local `nine65_v2_complete/proofs/`, note any gaps or improvements to integrate.
