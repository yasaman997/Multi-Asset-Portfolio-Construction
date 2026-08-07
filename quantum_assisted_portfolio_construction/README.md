<div align="center">

# Quantum-Assisted Multi-Asset Portfolio Construction

### A hybrid classical–quantum framework for realistic portfolio construction and active rebalancing

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

> **Core idea:** not every part of portfolio construction needs to be quantum. The project deliberately gives quantum optimization the discrete search task it is best suited to explore, while classical optimization remains responsible for continuous weights, financial feasibility, risk controls, and independent validation.

---

## 🎯 The Challenge

Multi-asset portfolio construction is not simply a return-maximization problem. A realistic portfolio has to balance several competing objectives at once:

- expected capital appreciation and income
- covariance risk and concentration
- diversification across assets and asset classes
- transaction costs and nonlinear market impact;
- liquidity and turnover
- asset, class, and factor exposure limits
- scenario-based stress risk

The challenge was to build a **50-asset portfolio-construction framework** that incorporates these practical investment considerations while also finding a credible role for quantum optimization.

That second part matters. A direct quantum encoding of all 50 continuous portfolio weights would create a much larger binary problem and would be difficult to simulate or validate meaningfully with the available computing environment. Instead of forcing the whole portfolio into a quantum formulation, this project asks a narrower question:

> **Which part of the portfolio decision is naturally discrete, computationally interesting, and suitable for quantum search?**

The answer used here is **active-rebalance support selection**: quantum optimization chooses which screened portfolio coordinates are allowed to change, and the full financial model then determines the actual weights.

---

## 🛠️ Approach

The solution uses a hybrid classical–quantum architecture with a clear division of responsibilities.

### 1. Build the full financial problem

The complete portfolio is modeled first, including return, income, covariance risk, concentration, implementation costs, liquidity, turnover, exposure guardrails, and stress scenarios.

### 2. Solve and understand the classical problem

A sequence of classical portfolio models shows how practical constraints change the allocation. The continuous optimizer remains the reference for financially feasible weights.

### 3. Reduce the discrete rebalance decision

Classical screening identifies a smaller set of economically relevant active coordinates. In the final experiment, the quantum problem contains **11 binary decisions** and requires exactly **6 selected coordinates**, producing **462 feasible supports**.

### 4. Search the reduced problem with QAOA

QAOA searches the reduced active-support space. A fixed-cardinality structure keeps the circuit in the six-of-eleven feasible sector.

### 5. Return to the full financial model

The QAOA bitstring is **not treated as the final portfolio**. Its selected support is returned to the continuous optimizer, where portfolio weights are refined using the original financial objective and constraints.

### 6. Validate independently

Final continuous solutions are reconstructed with CVXPY using Clarabel/OSQP and checked through feasibility and KKT conditions. The reduced QUBO is independently benchmarked by exact enumeration of all 462 feasible states.

---

## Why the Hybrid Split Matters

A central research question in this project is not *“How can everything be made quantum?”* but rather:

> **Where can quantum computation contribute most naturally inside a realistic investment workflow?**

Continuous allocation, risk constraints, transaction-cost accounting, and final feasibility are already well matched to mature classical optimization methods. The more natural quantum target is the discrete combinatorial decision of which assets or coordinates should participate in a rebalance.

This architecture also reflects the hardware available during development. Larger quantum simulations became increasingly expensive in CPU time and memory, and heavier notebook workloads could make the Google Colab runtime unstable or interrupt long experiments. State-vector simulation also becomes rapidly more demanding as qubit count grows. These constraints made it important to reduce the quantum problem to a size that could still be run repeatedly, benchmarked, and independently verified.

Rather than hiding those constraints, the project uses them to motivate a practical design principle:

**use quantum resources selectively, and keep the surrounding classical workflow strong enough to audit every result.**

As quantum hardware improves—and especially if large fault-tolerant quantum systems become available—the quantum portion of this architecture could be expanded to larger support spaces, deeper circuits, and more expressive discrete portfolio decisions without depending on costly classical state-vector simulation. That would not eliminate the classical layer: data processing, continuous allocation, risk management, and validation would still remain important parts of a realistic investment system.

---

## What Makes the Approach Stand Out

- **The problem is decomposed by mathematical structure.** Continuous allocation stays classical; the discrete support decision is assigned to QAOA.
- **The quantum component has a clear financial interpretation.** A selected bit does not directly set an asset weight; it grants a coordinate permission to participate in the rebalance.
- **The quantum result is benchmarked exactly.** All 462 feasible reduced-QUBO states are enumerated, so the selected QAOA result can be ranked against the true reduced-problem optimum.
- **The final portfolio is not trusted blindly.** QAOA-selected supports are returned to the full financial model and independently reconstructed.
- **Validation is separate from the original optimizer.** CVXPY, Clarabel/OSQP, feasibility checks, and KKT conditions provide an independent numerical audit.
- **The claims remain deliberately narrow.** The project demonstrates a credible hybrid architecture; it does not claim quantum speedup, quantum advantage, or historical market outperformance.

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
- greedy and local-search reduced-problem baseline 
- synthetic forward-path analysis

### Quantum optimization

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

The executed QUBO is a **reduced support-ranking surrogate**. It ranks which coordinates should be permitted to rebalance; the full financial objective and constraints are restored during continuous refinement.

### Technology stack

`Python` · `NumPy` · `pandas` · `SciPy` · `CVXPY` · `Clarabel` · `OSQP` · `Qiskit` · `Qiskit Aer` · `Qiskit Optimization` · `Matplotlib` · `Jupyter Notebook` · `Google Colab`

---

## 📈 Results and Findings

### Final shortlisted portfolios

| Portfolio | Expected Return | Volatility | Worst Stress Loss | Turnover | Trading Cost | Soft Warnings |
|---|---:|---:|---:|---:|---:|---:|
| **Primary classical** | 5.49% | 7.86% | 12.71% | 13.09% | 0.0020% | 5 |
| **Strict-warning classical** | 5.48% | 7.51% | 12.17% | 23.88% | 0.0056% | 0 |
| **Hybrid QAOA** | 5.45% | 7.87% | 12.72% | 11.06% | 0.0015% | 5 |

All three shortlisted portfolios pass the final hard-constraint audit.

The comparison highlights a real portfolio trade-off:

- the **primary classical portfolio** has the highest expected return in the shortlist;
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

- **15 of 20** independently optimized QAOA seeds reached the exact reduced optimum.
- The selected best-of-20 QAOA result ranked **#1 of 462** feasible supports.
- Its reduced-QUBO energy gap was **zero**.
- The resulting continuous QAOA portfolio passed all hard constraints.

The correct interpretation is precise: **the selected QAOA run recovered the exact optimum of the executed reduced QUBO**.

This does **not** establish quantum speedup or quantum computational advantage, and it does not prove global optimality of the full continuous portfolio problem across every possible support.

### Main finding

The most important result is architectural rather than promotional:

> **Quantum optimization can be inserted as a targeted combinatorial layer without giving up the financial realism, auditability, and numerical controls provided by classical optimization.**

The project therefore provides a testable framework for studying how the quantum portion could grow as hardware capabilities improve.

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

This project is a research prototype, and its conclusions should be interpreted within that scope.

### Development and implementation limitations

- **Compute availability:** larger QAOA simulations and repeated validation runs increased CPU and memory demand. In Google Colab, heavier workloads could lead to slower execution, runtime instability, or interrupted sessions.
- **Classical simulation cost:** increasing the number of simulated qubits rapidly increases the resources required for state-vector simulation, limiting the size of experiments that can be explored repeatedly in a notebook environment.
- **Quantum-hardware access:** the final QAOA experiments use Qiskit Aer rather than production quantum hardware.
- **Software integration:** maintaining compatible versions and configurations across Qiskit, Aer, optimization libraries, and the notebook environment required additional debugging and reproducibility work.
- **Data access:** the final evidence uses a controlled synthetic financial model; proprietary institutional holdings, execution, and transaction-cost data were not available.
- **Reduced search space:** classical screening narrows the quantum problem to 11 active coordinates, and the resulting 462-state problem remains classically enumerable.
- **Validation scope:** the full continuous portfolio was not independently reoptimized for every one of the 462 possible supports.
- **Financial scope:** the QAOA candidate passes hard limits but retains five soft scenario-warning breaches, and ETF-level diversification does not provide full constituent-level look-through.

### Recommended next steps

1. Run rolling historical and out-of-sample validation across multiple market regimes
2. Calibrate liquidity, market impact, and transaction-cost curves using real execution data where available
3. Reoptimize the continuous financial model across a larger portion—or all—of the reduced support space
4. Test larger support spaces, multiple cardinalities, and alternative screening rules
5. Move from ideal simulation to noisy simulation and available quantum hardware
6. Study circuit depth, shot count, optimizer choice, noise, and runtime as problem size grows
7. Compare classical and quantum methods on matched hardware and matched problem definitions
8. Explore warm-start QAOA, noise-aware circuits, parameter transfer, and alternative quantum objectives
9. Add constituent-level ETF overlap and factor-look-through analysis
10. Extend the decision-support layer to live or regularly refreshed portfolio inputs

### Longer-term quantum direction

A future fault-tolerant implementation could allow the quantum search to cover substantially larger combinatorial portfolio decisions than are practical to state-vector simulate today. The most interesting research question is therefore not whether the classical portion disappears, but **how much of the discrete search can move to quantum hardware while the classical system continues to provide data preparation, continuous refinement, risk controls, and independent verification**.

That is the direction this prototype is intended to make testable.

---

## 👥 Team Member and Contribution

**Yasaman Yaghoobi — Sole Contributor**
📧 Email : yaghoobi.y@northeastern.edu

This project was completed as an individual submission. I was responsible for the end-to-end development of the work, including:

- financial problem formulation and portfolio modeling
- data-pipeline design and validation
- classical optimization and constraint modeling
- QUBO/Ising formulation
- QAOA implementation and hybrid integration
- numerical experiments and benchmarking
- independent CVXPY/KKT validation
- portfolio comparison and interpretation
- decision-support prototype development
- technical documentation and presentation preparation

---

## License

This repository is provided for research and educational purposes.
