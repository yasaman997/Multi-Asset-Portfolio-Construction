# Step 5Q - Hybrid QAOA Active Rebalancing

The final path uses `active_rebalance` with 11 screened permissions and cardinality 6. QAOA minimizes the executed reduced active-rebalance QUBO; exact enumeration checks all 462 QUBO energies. A subset of QAOA/high-ranked supports is then continuously refined under the full Step 1 model. Therefore `exact` refers to reduced-QUBO optimality, not exhaustive exact-finance optimization across all 462 continuously refined supports.

See `results/RESULTS_FILE_MAP.md` for the authoritative headline and diagnostic result-file names.
