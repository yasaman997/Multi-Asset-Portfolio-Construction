<div align="center">

# Quantum-Assisted Multi-Asset Portfolio Construction

### Hybrid classical–quantum optimization for realistic portfolio construction and active rebalancing

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Qiskit](https://img.shields.io/badge/Qiskit-2.x-6929C4)
![CVXPY](https://img.shields.io/badge/CVXPY-Convex%20Optimization-1F6FEB)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-2E8B57)

Developed through the **WISER Quantum Summer Program** for the **Vanguard-sponsored Multi-Asset Portfolio Construction Challenge**.

</div>

---

## Project at a Glance

| | |
|---|---|
| **Portfolio universe** | 50 assets across multiple asset classes |
| **Classical layer** | Constrained continuous portfolio optimization |
| **Quantum layer** | QAOA active-rebalance support selection |
| **Reduced quantum problem** | 11 binary coordinates; select 6 |
| **Feasible reduced supports** | 462 |
| **QAOA experiment** | 20 seeds, depth `p = 2`, 1,024 shots, 60 optimizer iterations |
| **Best QAOA result** | Exact reduced-QUBO optimum; rank 1 of 462 |
| **Seed-level exact-hit rate** | 15 of 20 seeds |
| **Independent validation** | CVXPY + Clarabel/OSQP + KKT checks + exact enumeration |
| **Evidence scope** | Controlled synthetic financial model |

> **Core idea:** use quantum optimization where the decision is naturally discrete, while keeping final portfolio weights, financial constraints, and validation in the classical optimization layer.

---

## 🎯 The Challenge

Multi-asset portfolio construction is not simply a return-maximization problem. A realistic portfolio must balance several competing objectives at once:

- expected capital appreciation and income
- covariance risk and concentration
- diversification across assets and asset classes
- transaction costs and nonlinear market impact
- liquidity and turnover
- asset, class, and factor exposure limits
- scenario-based stress risk

The challenge was to design a **50-asset portfolio-construction framework** that incorporates these practical investment considerations while also identifying a meaningful and testable role for quantum optimization.

A direct quantum encoding of every continuous portfolio weight would be unnecessarily large for a near-term experiment. The project therefore separates the continuous financial allocation problem from the discrete decision of **which portfolio coordinates should be allowed to rebalance**.

---

## 🛠️ Approach

The solution uses a hybrid classical–quantum architecture.

### 1. Build the full financial problem

The complete portfolio is modeled first, including return, income, covariance risk, concentration, implementation costs, liquidity, turnover, exposure guardrails, and stress scenarios.

### 2. Solve and understand the classical problem

A sequence of classical portfolio models is used to show how realistic constraints change the solution. The final continuous allocation remains the reference for portfolio feasibility.

### 3. Reduce the discrete rebalance decision

Classical screening identifies a smaller set of economically relevant active coordinates. In the final experiment, the quantum problem contains **11 binary decisions** and requires exactly **6 selected coordinates**, giving **462 feasible supports**.

### 4. Search the reduced problem with QAOA

QAOA is used to search the reduced active-support space. A fixed-cardinality structure keeps the search inside the six-of-eleven feasible sector.

### 5. Return to the full financial model

The QAOA result is **not treated as the final portfolio**. The selected support is returned to the continuous optimizer, where portfolio weights are refined under the original financial constraints.

### 6. Validate independently

Final continuous solutions are reconstructed with CVXPY using Clarabel/OSQP and checked through feasibility and KKT conditions. The reduced QUBO is independently benchmarked by exact enumeration of all 462 feasible states.

---

## What Makes the Approach Stand Out

The project is designed around **division of labor rather than replacement**.

- **Continuous allocation stays classical.** Portfolio weights and financial feasibility are handled by methods that are transparent and directly auditable.
- **QAOA is given a focused combinatorial role.** It selects a sparse active-rebalance support instead of attempting to solve the entire 50-asset allocation problem.
- **The quantum result is benchmarked exactly.** Because the reduced problem contains 462 feasible supports, every reduced-QUBO energy can be enumerated and compared with QAOA.
- **Validation is independent of the original optimizer.** CVXPY reconstruction, Clarabel/OSQP, and KKT checks provide a separate numerical audit.
- **The project keeps its claims narrow.** It demonstrates a hybrid architecture; it does not claim quantum speedup, quantum advantage, or historical market outperformance.

---

## ⚙️ Methods and Tools Used

### Financial modeling

The continuous model incorporates:

- expected growth and income
- covariance risk
- concentration penalties
- linear trading costs
- nonlinear market impact
- gross turnover
- liquidity and trade-capacity limits
- asset-level bounds
- asset-class and factor exposure bands
- return and income requirements
- scenario-warning and hard stress limits

### Classical optimization and validation

Classical and validation components include:

- SciPy optimization
- CVXPY reconstruction
- Clarabel and OSQP validation solves
- covariance positive-semidefinite checks
- hard-constraint and trade-accounting audits
- KKT residual checks
- greedy and local-search reduced-problem baselines
- synthetic forward-path analysis

### Quantum optimization

The final QAOA experiment uses:

| Parameter | Final setting |
|---|---:|
| Qubits / binary coordinates | 11 |
| Selected coordinates | 6 |
| Feasible supports | 462 |
| QAOA depth | `p = 2` |
| Shots | 1,024 |
| Independent seeds | 20 |
| Optimizer iterations | 60 |
| Initial state | Fixed-cardinality Dicke state |
| Mixer | XY-ring mixer |
| Backend | Qiskit Aer simulator |

The executed QUBO is a **reduced support-ranking surrogate**. It is used to rank which coordinates should be permitted to rebalance; the full financial objective and constraints are restored during continuous refinement.

### Technology stack

`Python` · `NumPy` · `pandas` · `SciPy` · `CVXPY` · `Clarabel` · `OSQP` · `Qiskit` · `Qiskit Aer` · `Qiskit Optimization` · `Matplotlib` · `Jupyter Notebook`

---

## 📈 Results and Findings

### Final shortlisted portfolios

| Portfolio | Expected Return | Volatility | Worst Stress Loss | Turnover | Trading Cost | Soft Warnings |
|---|---:|---:|---:|---:|---:|---:|
| **Primary classical** | 5.49% | 7.86% | 12.71% | 13.09% | 0.0020% | 5 |
| **Strict-warning classical** | 5.48% | 7.51% | 12.17% | 23.88% | 0.0056% | 0 |
| **Hybrid QAOA** | 5.45% | 7.87% | 12.72% | 11.06% | 0.0015% | 5 |

All three shortlisted portfolios pass the final hard-constraint audit.

The results show a clear trade-off:

- the **primary classical portfolio** has the highest expected return of the shortlist;
- the **strict-warning portfolio** gives up very little expected return while improving modeled volatility and stress behavior, at the cost of higher turnover and trading cost;
- the **QAOA portfolio** produces a sparse active-rebalance solution with the lowest turnover and modeled trading cost among the three displayed candidates.

### QAOA result

The selected six-coordinate support is:

**SGOV · BIL · SHY · QQQ · VUG · MTUM**

After continuous refinement, four coordinates receive material trades:

| Asset | Weight change |
|---|---:|
| **SGOV** | +5.53 percentage points |
| **VUG** | -2.37 percentage points |
| **QQQ** | -2.06 percentage points |
| **MTUM** | -1.10 percentage points |

BIL and SHY remain available to trade but receive no material incremental change in the refined solution.

### Quantum benchmark

The reduced QUBO was checked against complete exact enumeration.

- **15 of 20** QAOA seeds reached the exact reduced optimum.
- The selected best-of-20 QAOA result ranked **#1 of 462** feasible supports.
- Its reduced-QUBO energy gap was **zero**.
- The resulting continuous QAOA portfolio passed all hard constraints.

This result should be interpreted precisely: **the selected QAOA run recovered the exact optimum of the executed reduced QUBO**. It does not prove that QAOA is faster or better than classical methods, and it does not establish the global optimum of the full continuous portfolio problem across every possible support.

### Primary recommendation

Under the declared synthetic model, the **strict-warning classical portfolio is the strongest governance-oriented candidate** because it improves modeled downside behavior while keeping expected return close to the primary classical solution.

The **QAOA portfolio is the key hybrid research result**: it demonstrates how quantum optimization can be inserted as a targeted active-support selector while classical optimization continues to control weights, feasibility, and risk.

---

## 🔍 How to Review the Repository

| Folder | What it contains |
|---|---|
| `step_01_financial_formulation/` | Financial problem definition and mathematical formulation |
| `step_02_quantum_formulation/` | QUBO/Ising formulation and quantum-compatible reduction |
| `step_03_data_pipeline/` | Portfolio data construction and validation |
| `step_04_classical_baseline/` | Classical optimization ladder and constraint modeling |
| `step_05_tunable_goals/` | Investor-goal calibration and sensitivity analysis |
| `step_05q_hybrid_qaoa/` | Final QAOA active-support experiment and reduced-QUBO results |
| `step_06_comparison/` | Candidate comparison and shortlist |
| `step_07_validation/` | Independent CVXPY/KKT validation and exact benchmarking |
| `step_08_presentation/` | Final scientific presentation |
| `step_09_portfolio_copilot/` | Explainable portfolio decision-support prototype |
| `step_10_risk_adjusted_scoring/` | Risk-adjusted candidate scoring and robustness analysis |

Supporting documentation is kept in `supporting_materials/`.

---

## 🔮 Limitations and Recommended Next Steps

The project is a research prototype, and its conclusions should be read within that scope

### Current limitations

- The final evidence uses a **controlled synthetic financial model**, not a historical rolling out-of-sample backtest
- Proprietary institutional holdings, execution, and transaction-cost data were not available
- The QAOA experiment was performed on a **Qiskit Aer simulator**, not production quantum hardware
- Classical screening reduces the quantum search to 11 active coordinates
- The 462-state reduced problem is small enough to enumerate classically
- The full continuous portfolio was not independently reoptimized for every one of the 462 supports
- The QAOA candidate passes hard limits but retains five soft scenario-warning breaches
- ETF-level diversification does not provide full constituent-level look-through

### Recommended next steps

1. Run rolling historical and out-of-sample validation across different market regimes
2. Calibrate liquidity, market impact, and transaction-cost curves using real execution data where available
3. Reoptimize the full continuous portfolio across a larger portion—or all—of the reduced support space
4. Test larger QAOA instances and multiple cardinalities
5. Evaluate noisy simulation and available quantum hardware
6. Compare classical and quantum runtime on matched hardware and problem definitions
7. Explore warm-start QAOA, noise-aware circuits, and alternative QAOA objectives
8. Add constituent-level ETF overlap and factor-look-through analysis
9. Extend the decision-support layer to live or regularly refreshed portfolio inputs

---

## 👥 Team Member and Contribution

**Yasaman Yaghoobi — Sole Contributor**
📧 Email: yaghoobi.y@northeastern.edu
This project was completed as an individual submission. I was responsible for the end-to-end development of the work, including:

- financial problem formulation and portfolio modeling;
- data-pipeline design and validation;
- classical optimization and constraint modeling;
- QUBO/Ising formulation;
- QAOA implementation and hybrid integration;
- numerical experiments and benchmarking;
- independent CVXPY/KKT validation;
- portfolio comparison and interpretation;
- decision-support prototype development; and
- technical documentation and presentation preparation.

---

## License

This repository is provided for research and educational purposes.
