# Quantum-Assisted Multi-Asset Portfolio Construction

### Hybrid Classical–Quantum Portfolio Optimization

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-2.x-purple)
![CVXPY](https://img.shields.io/badge/CVXPY-Convex%20Optimization-orange)
![Project](https://img.shields.io/badge/Project-Research%20Prototype-green)

This project explores how quantum optimization can be integrated into a realistic multi-asset portfolio-construction workflow without replacing the classical methods that are already well suited to continuous allocation and risk control.

The work was developed by **Yasaman Yaghoobi** as an individual project through the **WISER Quantum Summer Program** for the **Vanguard-sponsored Multi-Asset Portfolio Construction Challenge**.

---

## 🎯 Challenge

Multi-asset portfolio construction requires more than maximizing expected return. A practical solution must balance return, income, risk, diversification, transaction costs, liquidity, turnover, exposure limits, and stress behavior at the same time.

The challenge addressed here was to build a **50-asset portfolio framework** that handles those financial requirements while also identifying a credible role for quantum optimization.

---

## 🛠️ Approach

The project uses a hybrid architecture with clear responsibilities for the classical and quantum components.

**Classical optimization** handles the full continuous portfolio problem, including final asset weights and financial constraints.

**Quantum optimization** is used for a reduced combinatorial decision: selecting which portfolio coordinates are allowed to participate in an active rebalance.

The workflow is:

1. formulate the full 50-asset financial optimization problem;
2. build and validate the portfolio data and cost inputs;
3. solve classical baseline portfolios;
4. calibrate investor preferences and risk controls;
5. screen a smaller set of candidate rebalance coordinates;
6. formulate the reduced active-support problem as a QUBO;
7. use QAOA to search the feasible support space;
8. return selected supports to the full continuous portfolio model;
9. independently reconstruct and validate the resulting portfolios; and
10. compare the final candidates through an explainable decision-support layer.

---

## ⚙️ Methods

### Classical portfolio model

The continuous optimization layer includes:

- expected growth and income;
- covariance risk;
- concentration;
- linear transaction costs;
- nonlinear market impact;
- turnover;
- liquidity and trade-capacity limits;
- asset, asset-class, and factor exposure constraints;
- minimum return and income requirements; and
- scenario-warning and hard stress limits.

The final portfolio weights are solved classically so that feasibility and financial constraints remain explicit and auditable.

### Hybrid quantum layer

The final quantum experiment uses:

| Setting | Value |
|---|---:|
| Candidate active coordinates | 11 |
| Coordinates selected | 6 |
| Feasible supports | 462 |
| QAOA depth | p = 2 |
| Shots per run | 1,024 |
| Independent seeds | 20 |
| Classical optimizer iterations | 60 |
| Simulator | Qiskit Aer |

The reduced binary problem has the standard QUBO form:

```math
E(z) = q^\top z + z^\top Qz,
\qquad
z \in \{0,1\}^{11},
\qquad
\sum_{i=1}^{11} z_i = 6
```

The QUBO is used as a **support-ranking surrogate**. It ranks candidate rebalance supports using information from marginal portfolio value, trading costs, covariance interactions, and market-impact interactions. Final weights and all portfolio constraints are restored in the continuous optimization stage.

<details>
<summary><strong>Technical QUBO specification</strong></summary>

For screened coordinate \(i\), the implemented linear term is:

```math
q_i =
-\widetilde{s}_i^2
+ 0.05\,\widetilde{c}_i\widetilde{s}_i
```

The quadratic interaction matrix is:

```math
Q =
\lambda_b \widetilde{t}\widetilde{t}^{\top}
+0.04(\widetilde{t}\widetilde{t}^{\top})\odot\widetilde{\Sigma}
+0.02(\widetilde{t}\widetilde{t}^{\top})\odot\widetilde{\Gamma},
\qquad
\lambda_b = 1
```

This specification describes the executed reduced active-rebalance QUBO. It is not the full continuous financial objective.

</details>

### Independent validation

The project does not rely on a single solver result. Validation includes:

- covariance positive-semidefinite checks;
- budget and trade-accounting checks;
- hard-constraint audits;
- independent CVXPY reconstruction;
- Clarabel and OSQP validation solves;
- convexity checks;
- KKT residual checks;
- exact enumeration of all 462 reduced-QUBO supports;
- greedy and local-search baselines; and
- common synthetic forward-path analysis.

---

## 📈 Results and Findings

### Final portfolio shortlist

| Portfolio | Expected Return | Volatility | Worst Stress | Turnover | Trading Cost | Soft Warnings |
|---|---:|---:|---:|---:|---:|---:|
| Primary classical | 5.49% | 7.86% | 12.71% | 13.09% | 0.0020% | 5 |
| Strict-warning classical | 5.48% | 7.51% | 12.17% | 23.88% | 0.0056% | 0 |
| Hybrid QAOA | 5.45% | 7.87% | 12.72% | 11.06% | 0.0015% | 5 |

### QAOA support

The selected support is:

**SGOV, BIL, SHY, QQQ, VUG, MTUM**

After continuous refinement, the material portfolio changes are:

| Asset | Weight Change |
|---|---:|
| SGOV | +5.53 percentage points |
| VUG | -2.37 percentage points |
| QQQ | -2.06 percentage points |
| MTUM | -1.10 percentage points |

BIL and SHY remain permitted rebalance coordinates but receive no material incremental trade.

### Quantum benchmarking

Exact classical enumeration was performed over **all 462 feasible reduced-QUBO supports**.

- **15 of 20** independently optimized QAOA seeds reached the exact reduced-QUBO minimum.
- The selected best run ranked **#1 of 462**.
- The selected run had **zero reduced-QUBO energy gap**.

This establishes exact recovery for the **executed reduced QUBO** in the selected run. It does **not** establish a quantum speedup, quantum computational advantage, or exhaustive optimality of the full continuous portfolio problem across all 462 supports.

### Main takeaway

The strongest result is not that quantum optimization replaces classical portfolio construction. Instead, the project shows a practical way to use QAOA as a **targeted combinatorial layer** inside a broader workflow where classical optimization remains responsible for final weights, financial feasibility, and risk controls.

---

## 🧰 Technologies

`Python` · `NumPy` · `pandas` · `SciPy` · `CVXPY` · `Clarabel` · `OSQP` · `Qiskit` · `Qiskit Aer` · `Qiskit Optimization` · `Matplotlib` · `Jupyter Notebook`

---

## 📁 Repository Structure

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

The `steps/` directory contains the main implementation, notebooks, reports, and saved results. `supporting_materials/` contains selected documentation for interpretation and reproducibility.

---

## 🔮 Limitations and Next Steps

The current project is a research prototype. The main limitations are the use of a controlled synthetic financial model, simulator-based QAOA, classical pre-screening of the 11-coordinate quantum search space, and the fact that the full continuous portfolio model was not reoptimized for every one of the 462 possible supports.

Future work should focus on:

- rolling historical out-of-sample testing;
- empirical transaction-cost and liquidity calibration;
- full continuous refinement across the reduced support space;
- turnover-limit sensitivity analysis;
- larger QAOA instances and systematic scaling studies;
- noisy-simulator and quantum-hardware experiments;
- matched classical-versus-quantum runtime comparisons; and
- constituent-level ETF look-through and factor-overlap analysis.

---

## 👥 Team Member and Contribution

**Yasaman Yaghoobi — Sole Contributor** 
** Email: yaghoobi.y@northeastern.edu**

This project was completed as an individual submission. I was responsible for the full end-to-end development of the work, including the financial formulation, classical portfolio optimization, QUBO and QAOA implementation, hybrid quantum-classical integration, numerical experiments, independent validation, analysis of results, technical documentation, and presentation preparation.

---

## AI Assistance

General AI assistance was used for technical writing support, presentation preparation, and code debugging.



---

## License

This repository is provided for research and educational purposes.
