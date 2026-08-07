# Step 5Q Results File Map

The final release separates headline evidence from diagnostic experiments so similarly named QUBO files cannot be confused.

## Headline active-rebalance experiment

- `active_rebalance_exact_qubo_ranking.csv` - **all 462 feasible states**, ranked exactly by the executed active-rebalance QUBO used in the headline 11-qubit experiment.
- `active_rebalance_qubo_matrix.csv` - executed symmetric quadratic interaction matrix in the stored convention.
- `active_rebalance_qubo_linear.csv` - executed linear coefficient vector.
- `active_rebalance_qubit_universe.csv` - ordered 11-coordinate universe used to interpret bit positions.
- `final_qaoa_sample_probabilities.csv` - final 11-qubit sampled distribution.
- `seed_level_results.csv` - complete stored 20-seed reliability table and optimizer-time diagnostics.
- `final_qaoa_run_configuration.csv` - human-readable final-mode configuration reconciled to the executed pipeline notebook.

## Supplementary robustness diagnostics

- `active_rebalance_qubo_coefficient_sensitivity.csv` - 81 declared coefficient perturbations around the executed surrogate.
- `active_rebalance_cardinality_sensitivity.csv` - exact reduced-QUBO minima for cardinalities 4 through 8; not equal continuous-refinement evidence.
- `classical_exact_enumeration_scaling_diagnostic.csv` - environment-specific release-side CPU enumeration diagnostic; not a direct speed comparison to the original Colab QAOA runs.
- `sampling_baseline_context.csv` - analytical uniform-sampling reference values used only to prevent misinterpretation of the 75% seed-level hit statistic.

## Separate diagnostic experiments

- `classical_target_recovery_diagnostic_qubo_ranking.csv` - separate diagnostic QUBO/objective; not the headline active-rebalance ground truth.
- `classical_target_recovery_diagnostic_summary.csv` - summary of that separate target-recovery diagnostic.
- `qaoa_api_smoke_test_distribution.csv` - small Qiskit API/sampler smoke test.
- `support_reduction_diagnostics.csv` - whole-support feasibility/reduction diagnostics, not the final QAOA run configuration.

`selector_comparison.csv` retains the same selector results; bitstrings are zero-padded to the full 11-coordinate width so leading-zero states remain unambiguous.
