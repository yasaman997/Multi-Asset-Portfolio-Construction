from __future__ import annotations
from dataclasses import dataclass, replace
from itertools import combinations
import inspect
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from step_05_tunable_goals_final import GoalMixConfig, GoalPreferences, GoalScales, Step5Context, exact_goal_components, solve_goal_profile
EPS = 1e-12

def as_numpy(values: Any) -> np.ndarray:
    if hasattr(values, 'to_numpy'):
        return values.to_numpy(dtype=float)
    return np.asarray(values, dtype=float)

def unit_interval(values: Any, higher_is_better: bool=True) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError('All screening values must be finite.')
    lo = float(array.min())
    hi = float(array.max())
    if hi - lo <= EPS:
        score = np.full_like(array, 0.5, dtype=float)
    else:
        score = (array - lo) / (hi - lo)
    return score if higher_is_better else 1.0 - score

@dataclass(frozen=True)
class QiskitHybridConfig:
    max_qubits: int = 14
    cardinality: int | None = None
    reps: int = 2
    shots: int = 4096
    maxiter: int = 100
    seed: int = 12345
    top_quantum_samples: int = 30
    maximum_subsets_to_refine: int = 20
    material_incumbent_weight: float = 0.015
    scenario_proxy_share: float = 0.35
    class_balance_strength: float = 0.1
    selected_minimum_weight: float = 0.0
    exact_enumeration_limit: int = 10000
    run_numpy_exact_eigensolver: bool = False
    retain_raw_qiskit_objects: bool = False
    initial_point: tuple[float, ...] | None = None
    callback_checkpoint_path: str | None = None
    callback_checkpoint_interval: int = 2
    selection_mode: str = 'active_rebalance'
    selection_objective: str = 'incumbent_marginal_utility'
    active_balance_strength: float = 1.0
    trade_materiality_floor: float = 0.0005
    trade_materiality_fraction: float = 0.01
    marginal_transfer_size: float = 0.0025
    marginal_improvement_fraction: float = 0.05
    inferred_cardinality_cap: int | None = 6
    qaoa_aggregation: float | None = 0.25
    transpiler_optimization_level: int = 2
    use_cardinality_preserving_mixer: bool = True
    exact_candidates_to_refine: int = 12
    executed_trade_threshold: float = 1e-05

    def validate(self) -> None:
        if not 4 <= self.max_qubits <= 20:
            raise ValueError('max_qubits must lie between 4 and 20.')
        if self.cardinality is not None and self.cardinality < 2:
            raise ValueError('cardinality must be at least two.')
        if self.reps < 1 or self.shots < 1 or self.maxiter < 1:
            raise ValueError('reps, shots, and maxiter must be positive.')
        if self.top_quantum_samples < 1:
            raise ValueError('top_quantum_samples must be positive.')
        if self.maximum_subsets_to_refine < 1:
            raise ValueError('maximum_subsets_to_refine must be positive.')
        if self.callback_checkpoint_interval < 1:
            raise ValueError('callback_checkpoint_interval must be positive.')
        if self.initial_point is not None:
            expected = 2 * self.reps
            if len(self.initial_point) != expected:
                raise ValueError(f'initial_point must contain {expected} values for reps={self.reps}.')
        if self.selection_mode not in {'active_rebalance', 'whole_support'}:
            raise ValueError("selection_mode must be 'active_rebalance' or 'whole_support'.")
        if self.selection_objective not in {'classical_target_recovery', 'incumbent_marginal_utility'}:
            raise ValueError("selection_objective must be 'classical_target_recovery' or 'incumbent_marginal_utility'.")
        if self.active_balance_strength < 0.0:
            raise ValueError('active_balance_strength must be nonnegative.')
        if not 0.0 <= self.scenario_proxy_share <= 1.0:
            raise ValueError('scenario_proxy_share must lie in [0, 1].')
        if self.class_balance_strength < 0.0:
            raise ValueError('class_balance_strength must be nonnegative.')
        if self.trade_materiality_floor < 0.0:
            raise ValueError('trade_materiality_floor must be nonnegative.')
        if not 0.0 <= self.trade_materiality_fraction <= 1.0:
            raise ValueError('trade_materiality_fraction must lie in [0, 1].')
        if self.marginal_transfer_size <= 0.0:
            raise ValueError('marginal_transfer_size must be positive.')
        if not 0.0 <= self.marginal_improvement_fraction <= 1.0:
            raise ValueError('marginal_improvement_fraction must lie in [0, 1].')
        if self.inferred_cardinality_cap is not None and self.inferred_cardinality_cap < 2:
            raise ValueError('inferred_cardinality_cap must be at least two.')
        if self.qaoa_aggregation is not None and (not 0.0 < self.qaoa_aggregation <= 1.0):
            raise ValueError('qaoa_aggregation must lie in (0, 1].')
        if self.transpiler_optimization_level not in {0, 1, 2, 3}:
            raise ValueError('transpiler_optimization_level must be 0, 1, 2, or 3.')
        if self.exact_candidates_to_refine < 1:
            raise ValueError('exact_candidates_to_refine must be positive.')
        if self.executed_trade_threshold < 0.0:
            raise ValueError('executed_trade_threshold must be nonnegative.')

@dataclass
class ReducedUniverse:
    tickers: list[str]
    full_indices: np.ndarray
    screening_table: pd.DataFrame
    required_class_counts: dict[str, int]
    minimum_feasible_cardinality: int

    @property
    def n_qubits(self) -> int:
        return len(self.tickers)

@dataclass
class BinarySelectionModel:
    tickers: list[str]
    Q: np.ndarray
    linear: np.ndarray
    constant: float
    cardinality: int
    target_class_counts: dict[str, int]

    @property
    def n_variables(self) -> int:
        return len(self.tickers)

    def energy(self, bits: Any) -> float:
        z = np.asarray(bits, dtype=float)
        return float(z @ self.Q @ z + self.linear @ z + self.constant)

def minimum_assets_for_weight(upper_bounds: Any, required_weight: float) -> int:
    if required_weight <= EPS:
        return 0
    caps = np.sort(np.asarray(upper_bounds, dtype=float))[::-1]
    feasible = np.flatnonzero(np.cumsum(caps) >= required_weight - 1e-12)
    if feasible.size == 0:
        raise ValueError(f'Asset caps cannot support required weight {required_weight:.4f}.')
    return int(feasible[0] + 1)

def required_class_counts(asset_classes: Any, asset_upper: Any, class_bounds: dict[str, tuple[float, float]]) -> dict[str, int]:
    labels = np.asarray(asset_classes, dtype=object)
    upper = np.asarray(asset_upper, dtype=float)
    result: dict[str, int] = {}
    for asset_class, (lower, _) in class_bounds.items():
        if lower <= EPS:
            continue
        mask = labels == asset_class
        if not mask.any():
            raise ValueError(f'No asset is available for required class {asset_class}.')
        result[asset_class] = minimum_assets_for_weight(upper[mask], float(lower))
    return result

def build_screening_table(*, context: Step5Context, preferences: GoalPreferences) -> pd.DataFrame:
    data = context.portfolio_data
    shares = preferences.shares
    w0 = as_numpy(context.current_weights)
    covariance = np.asarray(data.covariance, dtype=float)
    standalone_variance = np.diag(covariance)
    losses = np.asarray(context.scenarios.loss_matrix, dtype=float)
    scenario_weights = np.asarray(context.scenarios.weights, dtype=float)
    scenario_rms = np.sqrt(np.average(losses ** 2, axis=0, weights=scenario_weights))
    impact = np.asarray(data.impact_matrix, dtype=float)
    impact_diagonal = np.diag(impact) if impact.ndim == 2 else impact
    implementation_burden = np.asarray(data.linear_cost, dtype=float) + impact_diagonal
    growth_score = unit_interval(data.growth, True)
    income_score = unit_interval(data.income, True)
    variance_score = unit_interval(standalone_variance, False)
    stress_score = unit_interval(scenario_rms, False)
    cost_score = unit_interval(implementation_burden, False)
    incumbent_score = unit_interval(w0, True)
    screening_score = shares['growth'] * growth_score + shares['income'] * income_score + shares['drawdown'] * (0.5 * variance_score + 0.5 * stress_score) + shares['cost'] * (0.6 * cost_score + 0.4 * incumbent_score)
    return pd.DataFrame({'ticker': list(data.tickers), 'asset_class': list(data.asset_classes), 'current_weight': w0, 'growth': np.asarray(data.growth, dtype=float), 'income': np.asarray(data.income, dtype=float), 'standalone_variance': standalone_variance, 'scenario_rms_loss': scenario_rms, 'linear_cost': np.asarray(data.linear_cost, dtype=float), 'impact_diagonal': impact_diagonal, 'screening_score': screening_score}).set_index('ticker')

def reduce_universe(*, context: Step5Context, preferences: GoalPreferences, config: QiskitHybridConfig) -> ReducedUniverse:
    config.validate()
    data = context.portfolio_data
    constraints = context.constraints
    table = build_screening_table(context=context, preferences=preferences)
    labels = np.asarray(data.asset_classes, dtype=object)
    upper = np.asarray(constraints.asset_upper, dtype=float)
    class_counts = required_class_counts(labels, upper, constraints.class_bounds)
    global_count = minimum_assets_for_weight(upper, 1.0)
    minimum_cardinality = max(global_count, int(sum(class_counts.values())))
    if config.max_qubits < minimum_cardinality:
        raise ValueError(f'At least {minimum_cardinality} selected assets are needed, but max_qubits={config.max_qubits}.')
    selected: list[str] = []

    def add(ticker: str) -> None:
        if ticker not in selected and len(selected) < config.max_qubits:
            selected.append(ticker)
    for asset_class, count in class_counts.items():
        ranked = table.loc[table['asset_class'] == asset_class].sort_values('screening_score', ascending=False)
        for ticker in ranked.head(count).index:
            add(str(ticker))
    incumbent_ranked = table.loc[table['current_weight'] >= config.material_incumbent_weight].sort_values('current_weight', ascending=False)
    for ticker in incumbent_ranked.index:
        add(str(ticker))
    for ticker in table.sort_values('screening_score', ascending=False).index:
        add(str(ticker))
    if len(selected) != config.max_qubits:
        raise RuntimeError('Could not construct the requested qubit universe.')
    full_tickers = list(data.tickers)
    indices = np.asarray([full_tickers.index(ticker) for ticker in selected], dtype=int)
    selected_table = table.loc[selected].copy()
    selected_table['qubit_index'] = np.arange(len(selected))
    return ReducedUniverse(tickers=selected, full_indices=indices, screening_table=selected_table, required_class_counts=class_counts, minimum_feasible_cardinality=minimum_cardinality)

def derive_step5_coefficients(preferences: GoalPreferences, scales: GoalScales, mix: GoalMixConfig) -> dict[str, float]:
    preferences.validate()
    scales.validate()
    mix.validate()
    shares = preferences.shares
    return {'growth': shares['growth'] / scales.growth, 'income': shares['income'] / scales.income, 'variance': shares['drawdown'] * mix.variance_share_of_drawdown / scales.variance, 'scenario': (shares['drawdown'] * mix.scenario_share_of_drawdown + mix.scenario_tiebreaker) / scales.scenario_hinge, 'linear_cost': (shares['cost'] * mix.linear_cost_share + mix.execution_tiebreaker) / scales.linear_cost, 'impact': (shares['cost'] * mix.impact_cost_share + mix.execution_tiebreaker) / scales.impact_cost, 'turnover': (shares['cost'] * mix.turnover_share + mix.execution_tiebreaker) / scales.turnover, 'concentration': mix.concentration_tiebreaker / scales.concentration}

def largest_remainder_class_targets(*, labels: np.ndarray, required_counts: dict[str, int], cardinality: int) -> dict[str, int]:
    classes = sorted(set(labels.tolist()))
    available = {asset_class: int(np.sum(labels == asset_class)) for asset_class in classes}
    targets = {asset_class: min(required_counts.get(asset_class, 0), available[asset_class]) for asset_class in classes}
    remaining = cardinality - sum(targets.values())
    if remaining < 0:
        raise ValueError('Required class support exceeds selected cardinality.')
    availability = np.asarray([available[asset_class] for asset_class in classes], dtype=float)
    availability /= availability.sum()
    desired = remaining * availability
    floors = np.floor(desired).astype(int)
    for asset_class, extra in zip(classes, floors, strict=True):
        capacity = available[asset_class] - targets[asset_class]
        targets[asset_class] += min(int(extra), capacity)
    remainder = cardinality - sum(targets.values())
    fractional = desired - floors
    while remainder > 0:
        feasible = [index for index, asset_class in enumerate(classes) if targets[asset_class] < available[asset_class]]
        if not feasible:
            raise ValueError('Reduced universe cannot support the chosen cardinality.')
        best = max(feasible, key=lambda index: fractional[index])
        targets[classes[best]] += 1
        fractional[best] = -np.inf
        remainder -= 1
    return targets

def build_binary_selection_model(*, context: Step5Context, reduced: ReducedUniverse, preferences: GoalPreferences, scales: GoalScales, mix: GoalMixConfig, cardinality: int, config: QiskitHybridConfig) -> BinarySelectionModel:
    data = context.portfolio_data
    idx = reduced.full_indices
    m = reduced.n_qubits
    k = int(cardinality)
    proxy_weight = 1.0 / k
    coeff = derive_step5_coefficients(preferences, scales, mix)
    growth = np.asarray(data.growth, dtype=float)[idx]
    income = np.asarray(data.income, dtype=float)[idx]
    covariance = np.asarray(data.covariance, dtype=float)[np.ix_(idx, idx)]
    linear_cost = np.asarray(data.linear_cost, dtype=float)[idx]
    impact = np.asarray(data.impact_matrix, dtype=float)[np.ix_(idx, idx)]
    incumbent = as_numpy(context.current_weights)[idx]
    loss_matrix = np.asarray(context.scenarios.loss_matrix, dtype=float)[:, idx]
    scenario_weights = np.asarray(context.scenarios.weights, dtype=float)
    Q = coeff['variance'] * proxy_weight ** 2 * covariance + config.scenario_proxy_share * coeff['scenario'] * proxy_weight ** 2 * (loss_matrix.T @ np.diag(scenario_weights) @ loss_matrix) + coeff['impact'] * proxy_weight ** 2 * impact + coeff['concentration'] * proxy_weight ** 2 * np.eye(m)
    linear = -coeff['growth'] * proxy_weight * growth - coeff['income'] * proxy_weight * income
    constant = 0.0
    selected_delta = np.abs(proxy_weight - incumbent)
    unselected_delta = incumbent
    linear += coeff['linear_cost'] * linear_cost * (selected_delta - unselected_delta)
    constant += float(coeff['linear_cost'] * linear_cost @ unselected_delta)
    linear += coeff['turnover'] * (selected_delta - unselected_delta)
    constant += float(coeff['turnover'] * unselected_delta.sum())
    linear += -2.0 * coeff['impact'] * proxy_weight * (impact @ incumbent)
    constant += float(coeff['impact'] * incumbent @ impact @ incumbent)
    labels = np.asarray(data.asset_classes, dtype=object)[idx]
    class_targets = largest_remainder_class_targets(labels=labels, required_counts=reduced.required_class_counts, cardinality=k)
    if config.class_balance_strength > 0.0:
        magnitude = max(float(np.max(np.abs(Q))), float(np.max(np.abs(linear))), 0.001)
        penalty = config.class_balance_strength * magnitude
        for asset_class, target in class_targets.items():
            indicator = (labels == asset_class).astype(float)
            Q += penalty * np.outer(indicator, indicator)
            linear += -2.0 * penalty * target * indicator
            constant += penalty * target ** 2
    Q = 0.5 * (Q + Q.T)
    return BinarySelectionModel(tickers=list(reduced.tickers), Q=Q, linear=linear, constant=float(constant), cardinality=k, target_class_counts=class_targets)

def build_qiskit_quadratic_program(model: BinarySelectionModel) -> Any:
    from qiskit_optimization.problems import QuadraticProgram
    qp = QuadraticProgram('hybrid_portfolio_selection')
    for ticker in model.tickers:
        qp.binary_var(name=ticker)
    linear = {ticker: float(model.linear[i] + model.Q[i, i]) for i, ticker in enumerate(model.tickers)}
    quadratic: dict[tuple[str, str], float] = {}
    for i in range(model.n_variables):
        for j in range(i + 1, model.n_variables):
            coefficient = float(2.0 * model.Q[i, j])
            if abs(coefficient) > 1e-15:
                quadratic[model.tickers[i], model.tickers[j]] = coefficient
    qp.minimize(constant=model.constant, linear=linear, quadratic=quadratic)
    qp.linear_constraint(linear={ticker: 1.0 for ticker in model.tickers}, sense='==', rhs=float(model.cardinality), name='fixed_cardinality')
    return qp

def enumerate_fixed_cardinality(model: BinarySelectionModel, maximum_states: int) -> pd.DataFrame:
    number = math.comb(model.n_variables, model.cardinality)
    if number > maximum_states:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for chosen in combinations(range(model.n_variables), model.cardinality):
        bits = np.zeros(model.n_variables, dtype=int)
        bits[list(chosen)] = 1
        records.append({'bitstring': ''.join((str(int(value)) for value in bits)), 'energy': model.energy(bits), 'selected_indices': tuple(chosen), 'selected_tickers': ', '.join((model.tickers[index] for index in chosen))})
    return pd.DataFrame(records).sort_values('energy', ascending=True).reset_index(drop=True)

def build_fixed_cardinality_qaoa_components(*, n_qubits: int, cardinality: int) -> tuple[Any, Any]:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    if not 0 < cardinality < n_qubits:
        raise ValueError('cardinality must lie strictly between zero and n_qubits.')
    amplitudes = np.zeros(2 ** n_qubits, dtype=complex)
    normalization = math.sqrt(math.comb(n_qubits, cardinality))
    for chosen in combinations(range(n_qubits), cardinality):
        basis_index = sum((1 << qubit for qubit in chosen))
        amplitudes[basis_index] = 1.0 / normalization
    initial_state = QuantumCircuit(n_qubits)
    initial_state.initialize(amplitudes, range(n_qubits))
    paulis: list[str] = []
    coefficients: list[float] = []
    for left in range(n_qubits):
        right = (left + 1) % n_qubits
        for symbol in ('X', 'Y'):
            label = ['I'] * n_qubits
            label[n_qubits - 1 - left] = symbol
            label[n_qubits - 1 - right] = symbol
            paulis.append(''.join(label))
            coefficients.append(0.5)
    mixer = SparsePauliOp(paulis, coeffs=coefficients)
    return (initial_state, mixer)

def run_qiskit_qaoa(*, quadratic_program: Any, model: BinarySelectionModel, config: QiskitHybridConfig) -> dict[str, Any]:
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    try:
        from qiskit_optimization.minimum_eigensolvers import QAOA
        from qiskit_optimization.optimizers import COBYLA
        qaoa_api_family = 'qiskit_optimization'
    except ImportError:
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA
        qaoa_api_family = 'qiskit_algorithms'
    callback_rows: list[dict[str, Any]] = []
    best_callback_state: dict[str, Any] = {'mean_energy': float('inf'), 'evaluation': 0, 'parameters': None, 'metadata': None}

    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return repr(value)

    def _metadata_standard_deviation(metadata: Any) -> float:
        if not isinstance(metadata, dict):
            try:
                return float(metadata)
            except (TypeError, ValueError):
                return float('nan')
        for key in ('standard_deviation', 'stddev', 'std'):
            value = metadata.get(key)
            if np.isscalar(value) and value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        for key in ('variance', 'variance_estimate'):
            value = metadata.get(key)
            if np.isscalar(value) and value is not None:
                try:
                    return float(np.sqrt(max(float(value), 0.0)))
                except (TypeError, ValueError):
                    pass
        return float('nan')

    def callback(evaluation_count: int, parameters: np.ndarray, mean: float, metadata: dict[str, Any]) -> None:
        safe_metadata = _json_safe(metadata)
        parameter_list = np.asarray(parameters, dtype=float).tolist()
        mean_value = float(np.real(mean))
        callback_rows.append({'evaluation': int(evaluation_count), 'mean_energy': mean_value, 'standard_deviation': _metadata_standard_deviation(metadata), 'parameters': parameter_list, 'metadata': safe_metadata, 'metadata_json': json.dumps(safe_metadata, sort_keys=True)})
        if mean_value < float(best_callback_state['mean_energy']):
            best_callback_state.update({'mean_energy': mean_value, 'evaluation': int(evaluation_count), 'parameters': parameter_list, 'metadata': safe_metadata})
        if config.callback_checkpoint_path and int(evaluation_count) % int(config.callback_checkpoint_interval) == 0:
            checkpoint = Path(config.callback_checkpoint_path)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(checkpoint.suffix + '.tmp')
            temporary.write_text(json.dumps(best_callback_state, indent=2, sort_keys=True), encoding='utf-8')
            temporary.replace(checkpoint)
    aer_backend = AerSimulator(method='statevector', precision='single')
    sampler = AerSamplerV2(default_shots=config.shots, seed=config.seed, options={'backend_options': {'method': 'statevector', 'precision': 'single'}})
    qaoa_transpiler = generate_preset_pass_manager(optimization_level=config.transpiler_optimization_level, backend=aer_backend)
    if config.initial_point is None:
        initial_point = np.concatenate([np.full(config.reps, 0.5, dtype=float), np.full(config.reps, 0.5, dtype=float)])
    else:
        initial_point = np.asarray(config.initial_point, dtype=float)
    qaoa_kwargs: dict[str, Any] = {'sampler': sampler, 'optimizer': COBYLA(maxiter=config.maxiter), 'reps': config.reps, 'initial_point': initial_point, 'callback': callback}
    if config.qaoa_aggregation is not None:
        qaoa_kwargs['aggregation'] = config.qaoa_aggregation
    if config.use_cardinality_preserving_mixer:
        initial_state, mixer = build_fixed_cardinality_qaoa_components(n_qubits=model.n_variables, cardinality=model.cardinality)
        qaoa_kwargs['initial_state'] = initial_state
        qaoa_kwargs['mixer'] = mixer
        mixer_type = 'XY_ring_with_Dicke_initial_state'
    else:
        mixer_type = 'default_X_mixer'
    qaoa_signature = inspect.signature(QAOA)
    qaoa_parameters = qaoa_signature.parameters
    if 'pass_manager' in qaoa_parameters:
        qaoa_kwargs['pass_manager'] = qaoa_transpiler
        transpiler_keyword = 'pass_manager'
    elif 'transpiler' in qaoa_parameters:
        qaoa_kwargs['transpiler'] = qaoa_transpiler
        transpiler_keyword = 'transpiler'
    else:
        raise RuntimeError('The installed QAOA constructor exposes neither `pass_manager` nor `transpiler`; its circuits cannot be prepared safely for Aer SamplerV2.')
    qaoa = QAOA(**qaoa_kwargs)
    qaoa_optimizer = MinimumEigenOptimizer(qaoa)
    qaoa_result = qaoa_optimizer.solve(quadratic_program)
    exact_result = None
    if config.run_numpy_exact_eigensolver:
        from qiskit_algorithms import NumPyMinimumEigensolver
        exact_optimizer = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        exact_result = exact_optimizer.solve(quadratic_program)
    sample_records: list[dict[str, Any]] = []
    for sample in qaoa_result.samples:
        bits = np.asarray(sample.x, dtype=int)
        sample_records.append({'bitstring': ''.join((str(int(value)) for value in bits)), 'cardinality': int(bits.sum()), 'raw_solver_probability': float(sample.probability), 'reported_objective': float(sample.fval), 'economic_energy': model.energy(bits), 'status': str(sample.status), 'selected_indices': tuple(np.flatnonzero(bits).tolist()), 'selected_tickers': ', '.join((model.tickers[index] for index in np.flatnonzero(bits)))})
    samples = pd.DataFrame(sample_records)
    if samples.empty:
        raise RuntimeError('QAOA returned no interpreted samples.')
    samples = samples.loc[samples['cardinality'] == model.cardinality].copy()
    if samples.empty:
        raise RuntimeError('QAOA returned no fixed-cardinality samples. Enable the cardinality-preserving mixer or increase shots.')
    feasible_probability_mass = float(samples['raw_solver_probability'].sum())
    denominator = max(feasible_probability_mass, 1e-15)
    samples['conditional_probability'] = samples['raw_solver_probability'] / denominator
    samples['probability_is_near_uniform'] = samples['conditional_probability'].max() - samples['conditional_probability'].min() <= 1e-12
    samples = samples.sort_values(['economic_energy', 'conditional_probability'], ascending=[True, False]).head(config.top_quantum_samples).reset_index(drop=True)
    exact_bits = np.asarray(exact_result.x, dtype=int) if exact_result is not None else None
    compact_result = {'samples': samples, 'callback_history': pd.DataFrame(callback_rows), 'exact_bits': exact_bits, 'exact_energy': model.energy(exact_bits) if exact_bits is not None else np.nan, 'optimizer_time': float(getattr(qaoa_result.min_eigen_solver_result, 'optimizer_time', np.nan)), 'optimal_point': np.asarray(getattr(qaoa_result.min_eigen_solver_result, 'optimal_point', np.array([], dtype=float)), dtype=float), 'callback_checkpoint_path': config.callback_checkpoint_path, 'transpiler_keyword': transpiler_keyword, 'transpiler_optimization_level': config.transpiler_optimization_level, 'aer_method': 'statevector', 'qaoa_api_family': qaoa_api_family, 'mixer_type': mixer_type, 'aggregation': config.qaoa_aggregation, 'feasible_probability_mass': feasible_probability_mass, 'aer_precision': 'single', 'optimizer_evaluations': int(getattr(qaoa_result.min_eigen_solver_result, 'cost_function_evals', len(callback_rows)) or len(callback_rows))}
    if config.retain_raw_qiskit_objects:
        compact_result.update({'qaoa_object': qaoa, 'qaoa_result': qaoa_result, 'exact_result': exact_result})
    return compact_result

def repair_selection(*, selected_indices: Any, reduced: ReducedUniverse, context: Step5Context, cardinality: int) -> np.ndarray:
    selected = set((int(index) for index in selected_indices))
    labels = np.asarray(context.portfolio_data.asset_classes, dtype=object)[reduced.full_indices]
    scores = reduced.screening_table['screening_score'].to_numpy()
    ranked = list(np.argsort(scores)[::-1])
    for index in ranked:
        if len(selected) >= cardinality:
            break
        selected.add(int(index))
    while len(selected) > cardinality:
        outgoing = min(selected, key=lambda index: scores[index])
        selected.remove(outgoing)

    def class_count(asset_class: str) -> int:
        return sum((labels[index] == asset_class for index in selected))
    for asset_class, required in reduced.required_class_counts.items():
        while class_count(asset_class) < required:
            incoming_options = [index for index in range(reduced.n_qubits) if index not in selected and labels[index] == asset_class]
            if not incoming_options:
                raise ValueError(f'Cannot repair class coverage for {asset_class}.')
            incoming = max(incoming_options, key=lambda index: scores[index])
            removable = [index for index in selected if class_count(str(labels[index])) > reduced.required_class_counts.get(str(labels[index]), 0)]
            if not removable:
                raise ValueError('No removable asset remains during class repair.')
            outgoing = min(removable, key=lambda index: scores[index])
            selected.remove(outgoing)
            selected.add(incoming)
    bits = np.zeros(reduced.n_qubits, dtype=int)
    bits[list(selected)] = 1
    return bits

def clone_constraints_for_subset(*, constraints: Any, selected_full_mask: np.ndarray, selected_minimum_weight: float) -> Any:
    lower = np.asarray(constraints.asset_lower, dtype=float).copy()
    upper = np.asarray(constraints.asset_upper, dtype=float).copy()
    lower[~selected_full_mask] = 0.0
    upper[~selected_full_mask] = 0.0
    if selected_minimum_weight > 0.0:
        lower[selected_full_mask] = np.maximum(lower[selected_full_mask], selected_minimum_weight)
    if lower.sum() > 1.0 + 1e-10:
        raise ValueError('Selected minimum weights exceed full investment.')
    if upper.sum() < 1.0 - 1e-10:
        raise ValueError('Selected asset caps cannot support full investment.')
    return replace(constraints, asset_lower=lower, asset_upper=upper)

def project_warm_start(*, source_weights: Any, selected_full_mask: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    weights = as_numpy(source_weights).copy()
    weights[~selected_full_mask] = 0.0
    weights = np.maximum(weights, lower)
    weights = np.minimum(weights, upper)
    for _ in range(200):
        difference = 1.0 - float(weights.sum())
        if abs(difference) <= 1e-10:
            break
        if difference > 0.0:
            slack = np.maximum(upper - weights, 0.0)
        else:
            slack = np.maximum(weights - lower, 0.0)
        total_slack = float(slack.sum())
        if total_slack <= EPS:
            break
        weights += difference * slack / total_slack
        weights = np.maximum(weights, lower)
        weights = np.minimum(weights, upper)
    if abs(float(weights.sum()) - 1.0) > 1e-07:
        raise ValueError('Could not construct a feasible warm start for the subset.')
    return weights

def build_active_screening_table(*, context: Step5Context, classical_reference: dict[str, Any], preferences: GoalPreferences) -> pd.DataFrame:
    table = build_screening_table(context=context, preferences=preferences).copy()
    current = as_numpy(context.current_weights)
    reference = as_numpy(classical_reference['result'].weights)
    desired_trade = reference - current
    absolute_trade = np.abs(desired_trade)
    data = context.portfolio_data
    impact = np.asarray(data.impact_matrix, dtype=float)
    impact_diagonal = np.diag(impact) if impact.ndim == 2 else impact
    implementation_burden = np.asarray(data.linear_cost, dtype=float) + impact_diagonal
    max_trade = max(float(absolute_trade.max()), 1e-12)
    trade_materiality = absolute_trade / max_trade
    base_score = unit_interval(table['screening_score'].to_numpy(dtype=float), True)
    low_cost_score = unit_interval(implementation_burden, False)
    active_score = trade_materiality * (0.85 + 0.1 * base_score + 0.05 * low_cost_score)
    table['reference_weight'] = reference
    table['reference_trade'] = desired_trade
    table['absolute_reference_trade'] = absolute_trade
    table['marginal_direction'] = np.sign(desired_trade).astype(int)
    table['marginal_improvement'] = np.nan
    table['marginal_counterparty'] = ''
    table['selection_trade'] = desired_trade
    table['absolute_selection_trade'] = absolute_trade
    table['trade_materiality'] = trade_materiality
    table['implementation_burden'] = implementation_burden
    table['active_selection_score'] = active_score
    table['selection_objective'] = 'classical_target_recovery'
    return table

def build_marginal_utility_screening_table(*, context: Step5Context, preferences: GoalPreferences, scales: GoalScales, mix: GoalMixConfig, config: QiskitHybridConfig) -> pd.DataFrame:
    table = build_screening_table(context=context, preferences=preferences).copy()
    current = as_numpy(context.current_weights)
    lower = np.asarray(context.constraints.asset_lower, dtype=float)
    upper = np.asarray(context.constraints.asset_upper, dtype=float)
    n_assets = len(current)
    epsilon = float(config.marginal_transfer_size)
    base = normalized_step5_objective(context=context, weights=current, preferences=preferences, scales=scales, mix=mix)['normalized_objective']
    best_improvement = np.full(n_assets, -np.inf, dtype=float)
    best_direction = np.zeros(n_assets, dtype=int)
    best_counterparty = np.full(n_assets, -1, dtype=int)
    for asset in range(n_assets):
        if current[asset] + epsilon <= upper[asset] + 1e-12:
            for counterparty in range(n_assets):
                if counterparty == asset:
                    continue
                if current[counterparty] - epsilon < lower[counterparty] - 1e-12:
                    continue
                trial = current.copy()
                trial[asset] += epsilon
                trial[counterparty] -= epsilon
                value = normalized_step5_objective(context=context, weights=trial, preferences=preferences, scales=scales, mix=mix)['normalized_objective']
                improvement = float(base - value)
                if improvement > best_improvement[asset]:
                    best_improvement[asset] = improvement
                    best_direction[asset] = 1
                    best_counterparty[asset] = counterparty
        if current[asset] - epsilon >= lower[asset] - 1e-12:
            for counterparty in range(n_assets):
                if counterparty == asset:
                    continue
                if current[counterparty] + epsilon > upper[counterparty] + 1e-12:
                    continue
                trial = current.copy()
                trial[asset] -= epsilon
                trial[counterparty] += epsilon
                value = normalized_step5_objective(context=context, weights=trial, preferences=preferences, scales=scales, mix=mix)['normalized_objective']
                improvement = float(base - value)
                if improvement > best_improvement[asset]:
                    best_improvement[asset] = improvement
                    best_direction[asset] = -1
                    best_counterparty[asset] = counterparty
    finite = np.isfinite(best_improvement)
    best_improvement[~finite] = 0.0
    positive_improvement = np.maximum(best_improvement, 0.0)
    maximum_improvement = max(float(positive_improvement.max()), 1e-16)
    marginal_materiality = positive_improvement / maximum_improvement
    data = context.portfolio_data
    impact = np.asarray(data.impact_matrix, dtype=float)
    impact_diagonal = np.diag(impact) if impact.ndim == 2 else impact
    implementation_burden = np.asarray(data.linear_cost, dtype=float) + impact_diagonal
    low_cost_score = unit_interval(implementation_burden, False)
    base_score = unit_interval(table['screening_score'].to_numpy(dtype=float), True)
    active_score = marginal_materiality * (0.85 + 0.1 * base_score + 0.05 * low_cost_score)
    proxy_trade = best_direction.astype(float) * epsilon * marginal_materiality
    tickers = list(context.portfolio_data.tickers)
    counterparties = [tickers[index] if index >= 0 else '' for index in best_counterparty]
    table['reference_weight'] = np.nan
    table['reference_trade'] = np.nan
    table['absolute_reference_trade'] = np.nan
    table['marginal_direction'] = best_direction
    table['marginal_improvement'] = positive_improvement
    table['marginal_counterparty'] = counterparties
    table['selection_trade'] = proxy_trade
    table['absolute_selection_trade'] = np.abs(proxy_trade)
    table['trade_materiality'] = marginal_materiality
    table['implementation_burden'] = implementation_burden
    table['active_selection_score'] = active_score
    table['selection_objective'] = 'incumbent_marginal_utility'
    return table

def infer_active_cardinality(*, reduced: ReducedUniverse, config: QiskitHybridConfig) -> int:
    if config.cardinality is not None:
        cardinality = int(config.cardinality)
        if not 1 <= cardinality <= reduced.n_qubits:
            raise ValueError('Configured cardinality must be between 1 and the reduced-universe size.')
        return cardinality
    if config.selection_objective == 'classical_target_recovery':
        magnitude = reduced.screening_table['absolute_reference_trade'].to_numpy(dtype=float)
        maximum = max(float(np.nanmax(magnitude)), 0.0)
        threshold = max(float(config.trade_materiality_floor), float(config.trade_materiality_fraction) * maximum)
    elif config.selection_objective == 'incumbent_marginal_utility':
        magnitude = reduced.screening_table['marginal_improvement'].to_numpy(dtype=float)
        maximum = max(float(np.nanmax(magnitude)), 0.0)
        threshold = max(1e-12, float(config.marginal_improvement_fraction) * maximum)
    else:
        raise ValueError(f'Unknown selection objective: {config.selection_objective!r}')
    count = int(np.count_nonzero(np.isfinite(magnitude) & (magnitude >= threshold)))
    upper = max(2, reduced.n_qubits - 1)
    if config.inferred_cardinality_cap is not None:
        upper = min(upper, int(config.inferred_cardinality_cap))
    return int(np.clip(count, 2, upper))

def reduce_active_universe(*, context: Step5Context, classical_reference: dict[str, Any], preferences: GoalPreferences, scales: GoalScales, mix: GoalMixConfig, config: QiskitHybridConfig) -> ReducedUniverse:
    config.validate()
    if config.selection_objective == 'classical_target_recovery':
        table = build_active_screening_table(context=context, classical_reference=classical_reference, preferences=preferences)
    else:
        table = build_marginal_utility_screening_table(context=context, preferences=preferences, scales=scales, mix=mix, config=config)
    selected: list[str] = []

    def add(ticker: str) -> None:
        if ticker not in selected and len(selected) < config.max_qubits:
            selected.append(ticker)
    buys = table.loc[table['selection_trade'] > EPS].sort_values(['active_selection_score', 'absolute_selection_trade'], ascending=[False, False])
    sells = table.loc[table['selection_trade'] < -EPS].sort_values(['active_selection_score', 'absolute_selection_trade'], ascending=[False, False])
    side_slots = max(1, config.max_qubits // 3)
    for ticker in buys.head(side_slots).index:
        add(str(ticker))
    for ticker in sells.head(side_slots).index:
        add(str(ticker))
    for ticker in table.sort_values(['active_selection_score', 'absolute_selection_trade'], ascending=[False, False]).index:
        add(str(ticker))
    if len(selected) != config.max_qubits:
        raise RuntimeError('Could not construct the requested active qubit universe.')
    full_tickers = list(context.portfolio_data.tickers)
    indices = np.asarray([full_tickers.index(ticker) for ticker in selected], dtype=int)
    selected_table = table.loc[selected].copy()
    selected_table['qubit_index'] = np.arange(len(selected))
    return ReducedUniverse(tickers=selected, full_indices=indices, screening_table=selected_table, required_class_counts={}, minimum_feasible_cardinality=2)

def build_active_selection_model(*, context: Step5Context, classical_reference: dict[str, Any], reduced: ReducedUniverse, cardinality: int, config: QiskitHybridConfig) -> BinarySelectionModel:
    idx = reduced.full_indices
    k = int(cardinality)
    table = reduced.screening_table
    selection_trade = table['selection_trade'].to_numpy(dtype=float)
    trade_scale = max(float(np.max(np.abs(selection_trade))), 1e-08)
    normalized_trade = selection_trade / trade_scale
    trade_materiality = np.abs(normalized_trade)
    data = context.portfolio_data
    covariance = np.asarray(data.covariance, dtype=float)[np.ix_(idx, idx)]
    impact = np.asarray(data.impact_matrix, dtype=float)[np.ix_(idx, idx)]
    linear_cost = np.asarray(data.linear_cost, dtype=float)[idx]
    covariance_scale = max(float(np.max(np.abs(covariance))), 1e-12)
    impact_scale = max(float(np.max(np.abs(impact))), 1e-12)
    covariance_normalized = covariance / covariance_scale
    impact_normalized = impact / impact_scale
    impact_diagonal = np.diag(impact_normalized)
    cost_burden = unit_interval(linear_cost + impact_diagonal, True)
    active_score = table['active_selection_score'].to_numpy(dtype=float)
    score_scale = max(float(active_score.max()), 1e-12)
    normalized_score = active_score / score_scale
    linear = -normalized_score ** 2 + 0.05 * cost_burden * normalized_score
    Q = config.active_balance_strength * np.outer(normalized_trade, normalized_trade)
    trade_outer = np.outer(normalized_trade, normalized_trade)
    Q += 0.04 * trade_outer * covariance_normalized
    Q += 0.02 * trade_outer * impact_normalized
    Q = 0.5 * (Q + Q.T)
    labels = np.asarray(data.asset_classes, dtype=object)[idx]
    class_targets = largest_remainder_class_targets(labels=labels, required_counts={}, cardinality=k)
    return BinarySelectionModel(tickers=list(reduced.tickers), Q=Q, linear=linear, constant=0.0, cardinality=k, target_class_counts=class_targets)

def repair_active_selection(*, selected_indices: Any, reduced: ReducedUniverse, cardinality: int) -> np.ndarray:
    selected = set((int(index) for index in selected_indices))
    scores = reduced.screening_table['active_selection_score'].to_numpy(dtype=float)
    trades = reduced.screening_table['selection_trade'].to_numpy(dtype=float)
    ranked = list(np.argsort(scores)[::-1])
    for index in ranked:
        if len(selected) >= cardinality:
            break
        selected.add(int(index))
    while len(selected) > cardinality:
        outgoing = min(selected, key=lambda index: scores[index])
        selected.remove(outgoing)

    def ensure_sign(sign: int) -> None:
        if sign > 0:
            already = any((trades[index] > EPS for index in selected))
            options = [index for index in range(reduced.n_qubits) if index not in selected and trades[index] > EPS]
        else:
            already = any((trades[index] < -EPS for index in selected))
            options = [index for index in range(reduced.n_qubits) if index not in selected and trades[index] < -EPS]
        if already or not options:
            return
        incoming = max(options, key=lambda index: scores[index])
        removable = [index for index in selected if (trades[index] <= EPS if sign > 0 else trades[index] >= -EPS)]
        if not removable:
            removable = list(selected)
        outgoing = min(removable, key=lambda index: scores[index])
        selected.remove(outgoing)
        selected.add(incoming)
    ensure_sign(+1)
    ensure_sign(-1)
    bits = np.zeros(reduced.n_qubits, dtype=int)
    bits[list(selected)] = 1
    return bits

def clone_constraints_for_active_set(*, constraints: Any, current_weights: Any, active_full_mask: np.ndarray) -> Any:
    current = as_numpy(current_weights)
    lower = np.asarray(constraints.asset_lower, dtype=float).copy()
    upper = np.asarray(constraints.asset_upper, dtype=float).copy()
    inactive = ~active_full_mask
    if np.any(current[inactive] < lower[inactive] - 1e-10) or np.any(current[inactive] > upper[inactive] + 1e-10):
        raise ValueError('An inactive current weight lies outside the final asset bounds.')
    lower[inactive] = current[inactive]
    upper[inactive] = current[inactive]
    return replace(constraints, asset_lower=lower, asset_upper=upper)

def project_active_warm_start(*, source_weights: Any, current_weights: Any, active_full_mask: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    current = as_numpy(current_weights)
    target = as_numpy(source_weights).copy()
    inactive = ~active_full_mask
    target[inactive] = current[inactive]
    active_indices = np.flatnonzero(active_full_mask)
    target[active_indices] = np.clip(target[active_indices], lower[active_indices], upper[active_indices])
    required_active_sum = float(current[active_indices].sum())
    for _ in range(300):
        active_sum = float(target[active_indices].sum())
        difference = required_active_sum - active_sum
        if abs(difference) <= 1e-11:
            break
        if difference > 0.0:
            slack = np.maximum(upper[active_indices] - target[active_indices], 0.0)
        else:
            slack = np.maximum(target[active_indices] - lower[active_indices], 0.0)
        total_slack = float(slack.sum())
        if total_slack <= EPS:
            target = current.copy()
            break
        target[active_indices] += difference * slack / total_slack
        target[active_indices] = np.clip(target[active_indices], lower[active_indices], upper[active_indices])
    target[inactive] = current[inactive]
    if abs(float(target.sum()) - 1.0) > 1e-08:
        target = current.copy()
    return target

def refine_active_subset(*, context: Step5Context, classical_reference: dict[str, Any], reduced: ReducedUniverse, bits: Any, preferences: GoalPreferences, scales: GoalScales, mix: GoalMixConfig, config: QiskitHybridConfig, label: str) -> dict[str, Any]:
    bit_array = np.asarray(bits, dtype=int)
    active_full_mask = np.zeros(len(context.portfolio_data.tickers), dtype=bool)
    active_full_mask[reduced.full_indices[bit_array == 1]] = True
    active_constraints = clone_constraints_for_active_set(constraints=context.constraints, current_weights=context.current_weights, active_full_mask=active_full_mask)
    active_context = replace(context, constraints=active_constraints)
    warm_source = classical_reference['result'].weights if config.selection_objective == 'classical_target_recovery' else context.current_weights
    warm_start = project_active_warm_start(source_weights=warm_source, current_weights=context.current_weights, active_full_mask=active_full_mask, lower=np.asarray(active_constraints.asset_lower, dtype=float), upper=np.asarray(active_constraints.asset_upper, dtype=float))
    solved = solve_goal_profile(context=active_context, profile_name=label, preferences=preferences, scales=scales, warm_start=warm_start, mix=mix)
    exact = normalized_step5_objective(context=context, weights=solved['result'].weights, preferences=preferences, scales=scales, mix=mix)
    solved['hybrid_normalized_objective'] = exact['normalized_objective']
    solved['selected_tickers'] = [context.portfolio_data.tickers[index] for index in np.flatnonzero(active_full_mask)]
    solved['selected_full_mask'] = active_full_mask
    solved['selection_mode'] = 'active_rebalance'
    solved['selection_objective'] = config.selection_objective
    final_weights = as_numpy(solved['result'].weights)
    current_weights = as_numpy(context.current_weights)
    executed_mask = np.abs(final_weights - current_weights) > config.executed_trade_threshold
    solved['executed_trade_tickers'] = [context.portfolio_data.tickers[index] for index in np.flatnonzero(executed_mask)]
    solved['executed_trade_count'] = int(executed_mask.sum())
    solved['nonzero_holding_count'] = int(np.count_nonzero(final_weights > 1e-06))
    return solved

def normalized_step5_objective(*, context: Step5Context, weights: Any, preferences: GoalPreferences, scales: GoalScales, mix: GoalMixConfig) -> dict[str, float]:
    components = exact_goal_components(context.portfolio_data, weights, context.current_weights, context.scenarios)
    coefficients = derive_step5_coefficients(preferences, scales, mix)
    objective = -coefficients['growth'] * components['growth'] - coefficients['income'] * components['income'] + coefficients['variance'] * components['variance'] + coefficients['scenario'] * components['scenario_hinge'] + coefficients['linear_cost'] * components['linear_cost'] + coefficients['impact'] * components['impact_cost'] + coefficients['turnover'] * components['gross_turnover'] + coefficients['concentration'] * components['concentration']
    return {'normalized_objective': float(objective), **components}

def refine_subset(*, context: Step5Context, classical_reference: dict[str, Any], reduced: ReducedUniverse, bits: Any, preferences: GoalPreferences, scales: GoalScales, mix: GoalMixConfig, config: QiskitHybridConfig, label: str) -> dict[str, Any]:
    bit_array = np.asarray(bits, dtype=int)
    selected_full_mask = np.zeros(len(context.portfolio_data.tickers), dtype=bool)
    selected_full_mask[reduced.full_indices[bit_array == 1]] = True
    subset_constraints = clone_constraints_for_subset(constraints=context.constraints, selected_full_mask=selected_full_mask, selected_minimum_weight=config.selected_minimum_weight)
    subset_context = replace(context, constraints=subset_constraints)
    warm_start = project_warm_start(source_weights=classical_reference['result'].weights, selected_full_mask=selected_full_mask, lower=np.asarray(subset_constraints.asset_lower, dtype=float), upper=np.asarray(subset_constraints.asset_upper, dtype=float))
    solved = solve_goal_profile(context=subset_context, profile_name=label, preferences=preferences, scales=scales, warm_start=warm_start, mix=mix)
    exact = normalized_step5_objective(context=context, weights=solved['result'].weights, preferences=preferences, scales=scales, mix=mix)
    solved['hybrid_normalized_objective'] = exact['normalized_objective']
    solved['selected_tickers'] = [context.portfolio_data.tickers[index] for index in np.flatnonzero(selected_full_mask)]
    solved['selected_full_mask'] = selected_full_mask
    return solved

def run_hybrid_qiskit_pipeline(*, context: Step5Context, preferences: GoalPreferences, scales: GoalScales, mix: GoalMixConfig, classical_reference: dict[str, Any], config: QiskitHybridConfig) -> dict[str, Any]:
    config.validate()
    if config.selection_mode == 'active_rebalance':
        reduced = reduce_active_universe(context=context, classical_reference=classical_reference, preferences=preferences, scales=scales, mix=mix, config=config)
        cardinality = infer_active_cardinality(reduced=reduced, config=config) if config.cardinality is None else int(config.cardinality)
        if cardinality < 2:
            raise ValueError('Active-rebalance cardinality must be at least two.')
        if cardinality > reduced.n_qubits:
            raise ValueError('cardinality cannot exceed n_qubits.')
        model = build_active_selection_model(context=context, classical_reference=classical_reference, reduced=reduced, cardinality=cardinality, config=config)

        def repair(bits: Any) -> np.ndarray:
            return repair_active_selection(selected_indices=np.flatnonzero(np.asarray(bits, dtype=int)), reduced=reduced, cardinality=cardinality)

        def refine(bits: Any, label: str) -> dict[str, Any]:
            return refine_active_subset(context=context, classical_reference=classical_reference, reduced=reduced, bits=bits, preferences=preferences, scales=scales, mix=mix, config=config, label=label)
    else:
        reduced = reduce_universe(context=context, preferences=preferences, config=config)
        cardinality = reduced.minimum_feasible_cardinality if config.cardinality is None else int(config.cardinality)
        if cardinality < reduced.minimum_feasible_cardinality:
            raise ValueError(f'cardinality={cardinality} is below the policy-implied minimum {reduced.minimum_feasible_cardinality}.')
        if cardinality > reduced.n_qubits:
            raise ValueError('cardinality cannot exceed n_qubits.')
        model = build_binary_selection_model(context=context, reduced=reduced, preferences=preferences, scales=scales, mix=mix, cardinality=cardinality, config=config)

        def repair(bits: Any) -> np.ndarray:
            return repair_selection(selected_indices=np.flatnonzero(np.asarray(bits, dtype=int)), reduced=reduced, context=context, cardinality=cardinality)

        def refine(bits: Any, label: str) -> dict[str, Any]:
            return refine_subset(context=context, classical_reference=classical_reference, reduced=reduced, bits=bits, preferences=preferences, scales=scales, mix=mix, config=config, label=label)
    qp = build_qiskit_quadratic_program(model)
    exact_enumeration = enumerate_fixed_cardinality(model, config.exact_enumeration_limit)
    qiskit_run = run_qiskit_qaoa(quadratic_program=qp, model=model, config=config)
    if not exact_enumeration.empty:
        exact_string = str(exact_enumeration.iloc[0]['bitstring'])
        exact_bits = np.fromiter((int(value) for value in exact_string), dtype=int)
        qiskit_run['exact_bits'] = exact_bits
        qiskit_run['exact_energy'] = float(exact_enumeration.iloc[0]['energy'])
    refinement_records: list[dict[str, Any]] = []
    successful_qaoa: list[dict[str, Any]] = []
    successful_fallback: list[dict[str, Any]] = []
    seen: set[str] = set()

    def attempt_candidate(*, bits: np.ndarray, sample_rank: int, probability: float, source: str) -> None:
        repaired = repair(bits)
        bitstring = ''.join((str(int(value)) for value in repaired))
        if bitstring in seen:
            return
        seen.add(bitstring)
        try:
            solved = refine(repaired, f'Qiskit QAOA sample {sample_rank}' if source == 'qaoa' else f'Exact-enumeration feasibility fallback {sample_rank}')
            solved['selection_source'] = source
            target = successful_qaoa if source == 'qaoa' else successful_fallback
            target.append(solved)
            refinement_records.append({'sample_rank': int(sample_rank), 'selection_source': source, 'bitstring': bitstring, 'probability': float(probability), 'qubo_energy': model.energy(repaired), 'success': True, 'normalized_continuous_objective': solved['hybrid_normalized_objective'], 'expected_total_return': solved['metrics']['expected_total_return'], 'volatility': solved['metrics']['volatility'], 'worst_scenario_loss': solved['metrics']['worst_scenario_loss'], 'gross_turnover': solved['metrics']['gross_turnover'], 'total_trading_cost': solved['metrics']['total_trading_cost'], 'selected_tickers': ', '.join(solved['selected_tickers']), 'message': solved['result'].message})
        except Exception as error:
            refinement_records.append({'sample_rank': int(sample_rank), 'selection_source': source, 'bitstring': bitstring, 'probability': float(probability), 'qubo_energy': model.energy(repaired), 'success': False, 'message': f'{type(error).__name__}: {error}'})
    for sample_rank, sample in qiskit_run['samples'].iterrows():
        if len(successful_qaoa) >= config.maximum_subsets_to_refine:
            break
        bits = np.zeros(reduced.n_qubits, dtype=int)
        bits[list(sample['selected_indices'])] = 1
        attempt_candidate(bits=bits, sample_rank=int(sample_rank), probability=float(sample.get('conditional_probability', sample.get('raw_solver_probability', 0.0))), source='qaoa')
    if not exact_enumeration.empty:
        exact_limit = min(len(exact_enumeration), config.exact_candidates_to_refine)
        for exact_rank, row in exact_enumeration.head(exact_limit).iterrows():
            bits = np.fromiter((int(value) for value in str(row['bitstring'])), dtype=int)
            attempt_candidate(bits=bits, sample_rank=int(exact_rank), probability=0.0, source='exact_active_set_benchmark')
    candidate_pool = successful_qaoa if successful_qaoa else successful_fallback
    if not candidate_pool:
        failure_table = pd.DataFrame(refinement_records)
        messages = failure_table.get('message', pd.Series(dtype=str)).astype(str).head(5).tolist()
        raise RuntimeError('No candidate subset produced a feasible continuous portfolio. First failures: ' + ' | '.join(messages))
    hybrid = min(candidate_pool, key=lambda result: result['hybrid_normalized_objective'])
    source = hybrid.get('selection_source', 'qaoa')
    if config.selection_mode == 'active_rebalance':
        profile = 'Hybrid Qiskit QAOA Active Rebalance' if source == 'qaoa' else 'Hybrid Active Rebalance (Exact Active-Set Benchmark)'
    else:
        profile = 'Hybrid Qiskit QAOA' if source == 'qaoa' else 'Hybrid Support Selection (Exact Feasibility Fallback)'
    hybrid['profile'] = profile
    hybrid['result'].stage = profile
    hybrid['selection_mode'] = config.selection_mode
    exact_refinement = None
    exact_bits = qiskit_run.get('exact_bits')
    try:
        if exact_bits is None:
            raise ValueError('No exact benchmark bitstring was available.')
        exact_refinement = refine(repair(exact_bits), 'Exact reduced-QUBO benchmark')
    except Exception:
        exact_refinement = None
    exact_active_profile = None
    if exact_refinement is not None:
        exact_refinement['selection_source'] = 'exact_active_set_benchmark'
        exact_refinement['profile'] = 'Exact Active-Set Benchmark'
        exact_refinement['result'].stage = 'Exact Active-Set Benchmark'
        exact_refinement['selection_mode'] = config.selection_mode
        exact_refinement['selection_objective'] = config.selection_objective
        exact_active_profile = exact_refinement
    best_additional_exact_profile = min(successful_fallback, key=lambda result: result['hybrid_normalized_objective']) if successful_fallback else None
    best_available_profile = min([result for result in (hybrid, exact_active_profile, best_additional_exact_profile) if result is not None], key=lambda result: result['hybrid_normalized_objective'])
    return {'config': config, 'selection_mode': config.selection_mode, 'selection_objective': config.selection_objective, 'reduced_universe': reduced, 'selection_model': model, 'quadratic_program': qp, 'exact_enumeration': exact_enumeration, 'qiskit_run': qiskit_run, 'refinement_table': pd.DataFrame(refinement_records), 'hybrid_profile_result': hybrid, 'exact_reduced_qubo_refinement': exact_refinement, 'exact_active_profile_result': exact_active_profile, 'best_additional_exact_profile_result': best_additional_exact_profile, 'best_available_profile_result': best_available_profile, 'cardinality': cardinality, 'qaoa_feasible_candidate_found': bool(successful_qaoa), 'selection_source': source}

def greedy_fixed_cardinality_bits(model: BinarySelectionModel) -> np.ndarray:
    selected: list[int] = []
    n = len(model.tickers)
    while len(selected) < model.cardinality:
        best_index = None
        best_energy = np.inf
        for candidate in range(n):
            if candidate in selected:
                continue
            bits = np.zeros(n, dtype=int)
            bits[selected + [candidate]] = 1
            energy = model.energy(bits)
            if energy < best_energy:
                best_energy = energy
                best_index = candidate
        if best_index is None:
            raise RuntimeError('Greedy subset construction failed.')
        selected.append(best_index)
    bits = np.zeros(n, dtype=int)
    bits[selected] = 1
    return bits

def local_search_fixed_cardinality_bits(model: BinarySelectionModel, initial_bits: Any | None=None) -> np.ndarray:
    bits = greedy_fixed_cardinality_bits(model) if initial_bits is None else np.asarray(initial_bits, dtype=int).copy()
    if int(bits.sum()) != model.cardinality:
        raise ValueError('initial_bits has the wrong cardinality.')
    improved = True
    while improved:
        improved = False
        current_energy = model.energy(bits)
        selected = np.flatnonzero(bits == 1)
        unselected = np.flatnonzero(bits == 0)
        best_swap = None
        best_energy = current_energy
        for outgoing in selected:
            for incoming in unselected:
                trial = bits.copy()
                trial[outgoing] = 0
                trial[incoming] = 1
                energy = model.energy(trial)
                if energy < best_energy - 1e-12:
                    best_energy = energy
                    best_swap = (outgoing, incoming)
        if best_swap is not None:
            bits[best_swap[0]] = 0
            bits[best_swap[1]] = 1
            improved = True
    return bits

def random_fixed_cardinality_candidates(*, n_qubits: int, cardinality: int, number_of_samples: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    total = math.comb(n_qubits, cardinality)
    target = min(int(number_of_samples), total)
    seen: set[tuple[int, ...]] = set()
    candidates: list[np.ndarray] = []
    while len(candidates) < target:
        chosen = tuple(sorted((int(value) for value in rng.choice(n_qubits, size=cardinality, replace=False))))
        if chosen in seen:
            continue
        seen.add(chosen)
        bits = np.zeros(n_qubits, dtype=int)
        bits[list(chosen)] = 1
        candidates.append(bits)
    return candidates

def evaluate_classical_subset_baselines(*, context: Step5Context, classical_reference: dict[str, Any], reduced: ReducedUniverse, model: BinarySelectionModel, preferences: GoalPreferences, scales: GoalScales, mix: GoalMixConfig, config: QiskitHybridConfig, random_budget: int=20, seed: int=12345) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    candidate_map: dict[str, np.ndarray] = {'greedy': greedy_fixed_cardinality_bits(model)}
    candidate_map['local_search'] = local_search_fixed_cardinality_bits(model, candidate_map['greedy'])
    for index, bits in enumerate(random_fixed_cardinality_candidates(n_qubits=len(model.tickers), cardinality=model.cardinality, number_of_samples=random_budget, seed=seed)):
        candidate_map[f'random_{index:03d}'] = bits
    records: list[dict[str, Any]] = []
    solved_profiles: dict[str, dict[str, Any]] = {}
    for name, bits in candidate_map.items():
        try:
            solved = refine_active_subset(context=context, classical_reference=classical_reference, reduced=reduced, bits=bits, preferences=preferences, scales=scales, mix=mix, config=config, label=f'Classical subset baseline: {name}')
            solved_profiles[name] = solved
            records.append({'selector': name, 'success': True, 'bitstring': ''.join((str(int(x)) for x in bits)), 'qubo_energy': model.energy(bits), 'normalized_objective': solved['hybrid_normalized_objective'], 'expected_total_return': solved['metrics']['expected_total_return'], 'volatility': solved['metrics']['volatility'], 'worst_scenario_loss': solved['metrics']['worst_scenario_loss'], 'gross_turnover': solved['metrics']['gross_turnover'], 'total_trading_cost': solved['metrics']['total_trading_cost'], 'selected_tickers': ', '.join(solved['selected_tickers'])})
        except Exception as error:
            records.append({'selector': name, 'success': False, 'bitstring': ''.join((str(int(x)) for x in bits)), 'qubo_energy': model.energy(bits), 'message': f'{type(error).__name__}: {error}'})
    return (pd.DataFrame(records), solved_profiles)
