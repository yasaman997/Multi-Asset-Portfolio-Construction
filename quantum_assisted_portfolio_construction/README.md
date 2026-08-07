# Quantum-Assisted Multi-Asset Portfolio Construction
### A Hybrid Classical–Quantum Optimization Framework

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Qiskit](https://img.shields.io/badge/Qiskit-2.x-purple) ![Project](https://img.shields.io/badge/Project-Research%20Prototype-green)

This project develops a finance-first hybrid quantum-classical framework for multi-asset portfolio construction. It combines realistic financial modeling, classical portfolio optimization, reduced-QUBO active-support selection using the Quantum Approximate Optimization Algorithm (QAOA), continuous portfolio refinement, independent numerical validation, and an explainable portfolio decision-support layer.

The work was developed by **Yasaman Yaghoobi** as an individual project through the **WISER Quantum Summer Program**, for the **Vanguard-sponsored Multi-Asset Portfolio Construction Challenge**.

---

## 🎯 Challenge Overview

Institutional portfolio construction requires balancing multiple competing objectives simultaneously:

- expected capital appreciation;
- portfolio income;
- covariance risk;
- diversification and concentration;
- transaction costs and nonlinear market impact;
- liquidity and turnover;
- exposure guardrails; and
- scenario-based stress risk.

The objective of this project is to investigate how a hybrid quantum-classical optimization workflow can support a realistic portfolio-construction process while preserving financial feasibility, interpretability, and rigorous numerical validation.

---

## 🛠️ Proposed Solution

Rather than replacing established portfolio optimization methods, the project separates the problem according to its mathematical structure:

1. formulate the full continuous 50-asset financial problem;
2. construct and validate the portfolio data and cost inputs;
3. solve classical baseline portfolios under progressively realistic constraints;
4. calibrate investor preferences using dimensionless objective scaling;
5. identify a reduced set of candidate active coordinates using classical screening;
6. build an active-rebalance QUBO over those binary permissions;
7. use fixed-cardinality QAOA to search the reduced support space;
8. refine selected supports using the exact continuous portfolio model;
9. independently reconstruct and validate the continuous solutions; and
10. compare and communicate final portfolio candidates through a decision-support layer.

The resulting workflow combines continuous classical portfolio optimization with reduced-QUBO active-support selection in a practical hybrid architecture.

---

## ⚙️ Methods and Tools

### Financial Optimization Model

The continuous portfolio model incorporates:

- expected growth;
- income yield;
- covariance risk;
- concentration;
- linear transaction costs;
- nonlinear market impact;
- turnover;
- liquidity and trade-capacity limits;
- asset, asset-class, and factor exposure constraints;
- minimum return and income requirements; and
- scenario-warning and hard stress limits.

Continuous portfolio weights are solved classically because the financial problem is convex and can be audited directly.

### Hybrid Quantum Optimization

The quantum layer is applied to a reduced active-support selection problem rather than to the complete 50-asset continuous allocation.

The final experiment uses:

- **11 screened binary active-coordinate permissions**;
- **cardinality 6**, giving \(\binom{11}{6}=462\) feasible supports;
- **QAOA depth \(p=2\)**;
- **1,024 shots per run**;
- **60 classical optimizer iterations**;
- **20 independent seeds**;
- a fixed-cardinality Dicke initial state;
- an XY-ring mixer; and
- Qiskit Aer simulation.

For screened coordinate \(i\), the implemented active-rebalance QUBO uses

\[
q_i=-\widetilde s_i^{\,2}+0.05\,\widetilde c_i\widetilde s_i,
\]

with quadratic interaction matrix

\[
Q=
\lambda_b\widetilde t\widetilde t^{\mathsf T}
+0.04(\widetilde t\widetilde t^{\mathsf T})\odot\widetilde\Sigma
+0.02(\widetilde t\widetilde t^{\mathsf T})\odot\widetilde\Gamma,
\]

where \(\lambda_b=1\) in the final configuration.

This reduced QUBO is a **support-ranking surrogate**. The complete financial objective and all portfolio constraints are restored during downstream continuous refinement.

### Validation

The project uses multiple validation layers rather than relying on solver convergence alone.

Validation includes:

- covariance positive-semidefinite checks;
- budget and trade-accounting checks;
- complete hard-constraint audits;
- independent CVXPY reconstruction;
- Clarabel and OSQP validation solves;
- convexity verification;
- Karush-Kuhn-Tucker (KKT) residual checks;
- exact enumeration of all 462 reduced-QUBO states;
- greedy and local-search reduced-problem baselines;
- common synthetic forward-path analysis; and
- benchmark and scoring-weight sensitivity analysis.

The continuous validation establishes that the declared convex portfolio problems are solved consistently under the stated model assumptions.

### Technologies

- Python
- NumPy
- pandas
- SciPy
- CVXPY
- Clarabel
- OSQP
- Qiskit
- Qiskit Aer
- Qiskit Optimization
- Matplotlib
- Jupyter Notebook

---

## 📈 Results and Findings

The final certified shortlist is:

| Portfolio | Expected Return | Volatility | Worst Stress | Turnover | Trading Cost | Soft Warnings |
|---|---:|---:|---:|---:|---:|---:|
| Primary unrestricted classical | 5.49% | 7.86% | 12.71% | 13.09% | 0.0020% | 5 |
| Strict-warning classical | 5.48% | 7.51% | 12.17% | 23.88% | 0.0056% | 0 |
| Hybrid QAOA active-rebalance | 5.45% | 7.87% | 12.72% | 11.06% | 0.0015% | 5 |

The selected QAOA support is:

**SGOV, BIL, SHY, QQQ, VUG, MTUM**

Continuous refinement produces four material trades:

- **SGOV:** +5.53 percentage points
- **VUG:** -2.37 percentage points
- **QQQ:** -2.06 percentage points
- **MTUM:** -1.10 percentage points

BIL and SHY remain permitted coordinates but receive no material incremental trade.

Fifteen of twenty independently optimized QAOA seeds reached the exact minimum of the executed reduced QUBO. The selected best run therefore has zero reduced-QUBO energy gap.

The exactness claim is deliberately limited to

\[
z_Q^*
=
\arg\min_{z:\mathbf 1^T z=6}
E_{\mathrm{QUBO}}(z).
\]

The project does **not** claim exhaustive continuous optimization over all 462 supports, quantum speedup, or quantum computational advantage.

### Main findings

- A realistic multi-asset portfolio can be formulated and solved within a disciplined hybrid workflow.
- Continuous classical optimization remains the appropriate source of truth for final portfolio weights and financial feasibility.
- QAOA can be integrated as a reduced active-support selector within a larger finance-first optimization process.
- The selected best-of-20 QAOA run reaches the exact minimum of the executed 462-state reduced QUBO.
- Independent solver reconstruction and KKT checks provide strong validation of the continuous optimization results.
- An explainable decision-support layer can expose portfolio trade-offs, constraints, and provenance without obscuring the underlying optimization.

---

## Repository Structure

```text
quantum-assisted-portfolio-construction/
│
├── README.md
│
├── steps/
│   ├── step_01_financial_formulation/
│   ├── step_02_quantum_formulation/
│   ├── step_03_data_pipeline/
│   ├── step_04_classical_baseline/
│   ├── step_05_tunable_goals/
│   ├── step_05q_hybrid_qaoa/
│   ├── step_06_comparison/
│   ├── step_07_validation/
│   ├── step_08_presentation/
│   ├── step_09_portfolio_copilot/
│   └── step_10_risk_adjusted_scoring/
│
└── supporting_materials/
```

The `steps/` directory contains the main implementation, technical reports, notebooks, and saved results. The `supporting_materials/` directory contains selected documentation useful for interpretation and reproducibility without cluttering the main project view.

---

## 🔮 Limitations and Recommended Next Steps

This project is a research prototype rather than an investment product.

Important limitations include:

- final numerical evidence is based on a controlled synthetic financial model rather than rolling historical out-of-sample performance;
- the executed QUBO is a reduced support-ranking surrogate rather than the complete continuous financial objective;
- exhaustive continuous refinement over all 462 supports was not performed;
- the final quantum experiment uses simulation rather than production quantum hardware;
- the reduced 462-state problem is classically easy to enumerate;
- no computational quantum advantage is demonstrated;
- the 11-coordinate quantum search space is preconditioned by classical screening;
- the provisional Step 4 leaders use the full 50% turnover allowance; and
- ETF-level diversification measures do not provide full constituent-level look-through.

Recommended next steps include:

- rolling historical out-of-sample validation;
- empirical transaction-cost and liquidity calibration;
- continuous refinement across the complete reduced support space;
- turnover-limit sensitivity analysis;
- constituent-level ETF look-through and factor-overlap analysis;
- larger QAOA instances with systematic runtime and circuit-scaling benchmarks;
- noisy-simulator experiments; and
- quantum-hardware evaluation.

---

## 👥 Team Member and Contribution

**Yasaman Yaghoobi — Sole Contributor**
**Email: yaghoobi.y@northeastern.edu**

This project was completed as an individual submission. I was responsible for the full end-to-end development of the work, including the financial formulation, classical portfolio optimization, QUBO and QAOA implementation, hybrid quantum-classical integration, numerical experiments, independent validation, analysis of results, technical documentation, and presentation preparation.

---

## AI Assistance

General AI assistance was used for technical writing support, presentation preparation, and code debugging.


---

## License

This repository is provided for research and educational purposes.
