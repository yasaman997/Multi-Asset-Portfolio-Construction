from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, minimize
EPS = 1e-12

def _array(values: Any) -> np.ndarray:
    if hasattr(values, 'to_numpy'):
        return values.to_numpy(dtype=float)
    return np.asarray(values, dtype=float)

@dataclass(frozen=True)
class ValidationTolerances:
    feasibility: float = 5e-06
    objective_match: float = 2e-05
    fixed_support_objective_match: float = 3e-05
    solver_weight_l1: float = 0.02
    qaoa_gap_zero: float = 1e-08
    hessian_psd: float = 1e-09
    kkt_stationarity: float = 0.0001
    kkt_complementarity: float = 1e-05
    dual_feasibility: float = 1e-07

@dataclass
class ClassicalValidationContext:
    step4: Any
    step5: Any
    hybrid: Any
    base_context: Any
    strict_context: Any | None
    preferences: Any
    scales: Any
    mix: Any
    profiles: Mapping[str, Mapping[str, Any]]
    step6_handoff: Mapping[str, Any]
    tolerances: ValidationTolerances = ValidationTolerances()

    def validate(self) -> None:
        self.base_context.portfolio_data.validate()
        self.base_context.scenarios.validate(len(self.base_context.portfolio_data.tickers))
        self.base_context.constraints.validate(self.base_context.portfolio_data)
        self.preferences.validate()
        self.scales.validate()
        self.mix.validate()
        if 'Primary unrestricted classical' not in self.profiles:
            raise ValueError('Primary classical profile is missing.')
        if 'Independent Qiskit QAOA' not in self.profiles:
            raise ValueError('Independent QAOA profile is missing.')
        if 'shortlist' not in self.step6_handoff:
            raise ValueError('The Step 6 shortlist is missing.')

@dataclass
class IndependentClassicalResult:
    label: str
    policy: str
    solver: str
    selected_solver_source: str
    success: bool
    message: str
    objective_value: float
    weights: pd.Series
    iterations: int
    optimality: float
    constr_violation: float
    kkt_stationarity_inf: float
    kkt_complementarity_inf: float
    dual_feasibility_min: float
    solver_reported_duality_gap: float
    kkt_certificate_pass: bool
    audit: pd.DataFrame
    objective_components: pd.Series
    solver_diagnostics: pd.DataFrame
    raw_result: Any
    fixed_active_tickers: tuple[str, ...] | None = None

@dataclass
class _ProblemDefinition:
    layout: dict[str, slice]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    constraint_matrix: np.ndarray
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray
    objective_data: Any
    economic_data: Any
    objective_weights: Any
    coefficients: Mapping[str, float]
    context: Any
    fixed_active_tickers: tuple[str, ...] | None

def _layout(n_assets: int, n_scenarios: int) -> dict[str, slice]:
    cursor = 0
    result = {'w': slice(cursor, cursor + n_assets)}
    cursor += n_assets
    result['p'] = slice(cursor, cursor + n_assets)
    cursor += n_assets
    result['n'] = slice(cursor, cursor + n_assets)
    cursor += n_assets
    result['h'] = slice(cursor, cursor + n_scenarios)
    cursor += n_scenarios
    result['all'] = slice(0, cursor)
    return result

def _append(rows: list[np.ndarray], lower: list[float], upper: list[float], row: np.ndarray, lb: float, ub: float) -> None:
    rows.append(np.asarray(row, dtype=float))
    lower.append(float(lb))
    upper.append(float(ub))

def build_independent_problem(*, context: ClassicalValidationContext, policy_context: Any, fixed_active_tickers: Sequence[str] | None=None) -> _ProblemDefinition:
    economic_data = policy_context.portfolio_data
    objective_data, objective_weights, coefficients = context.step5.build_goal_objective(context.step4, context.preferences, economic_data, context.scales, context.mix)
    tickers = list(economic_data.tickers)
    n_assets = len(tickers)
    scenarios = policy_context.scenarios
    n_scenarios = len(scenarios.names)
    layout = _layout(n_assets, n_scenarios)
    n_variables = layout['all'].stop
    lower_bounds = np.full(n_variables, -np.inf, dtype=float)
    upper_bounds = np.full(n_variables, np.inf, dtype=float)
    lower_bounds[layout['w']] = _array(policy_context.constraints.asset_lower)
    upper_bounds[layout['w']] = _array(policy_context.constraints.asset_upper)
    lower_bounds[layout['p']] = 0.0
    lower_bounds[layout['n']] = 0.0
    lower_bounds[layout['h']] = 0.0
    upper_bounds[layout['p']] = 1.0
    upper_bounds[layout['n']] = 1.0
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    current = _array(policy_context.current_weights)
    row = np.zeros(n_variables)
    row[layout['w']] = 1.0
    _append(rows, lower, upper, row, 1.0, 1.0)
    for index in range(n_assets):
        row = np.zeros(n_variables)
        row[layout['w'].start + index] = 1.0
        row[layout['p'].start + index] = -1.0
        row[layout['n'].start + index] = 1.0
        _append(rows, lower, upper, row, current[index], current[index])
    row = np.zeros(n_variables)
    row[layout['p']] = 1.0
    row[layout['n']] = 1.0
    _append(rows, lower, upper, row, -np.inf, policy_context.trading_config.turnover_limit_gross)
    trade_capacity = policy_context.trading_config.execution_days * policy_context.trading_config.participation_rate * _array(economic_data.adv_usd) / policy_context.trading_config.portfolio_value_usd
    for index in range(n_assets):
        row = np.zeros(n_variables)
        row[layout['p'].start + index] = 1.0
        row[layout['n'].start + index] = 1.0
        _append(rows, lower, upper, row, -np.inf, trade_capacity[index])
    asset_classes = np.asarray(economic_data.asset_classes, dtype=object)
    for class_name, (class_lower, class_upper) in policy_context.constraints.class_bounds.items():
        row = np.zeros(n_variables)
        row[layout['w']] = (asset_classes == class_name).astype(float)
        _append(rows, lower, upper, row, class_lower, class_upper)
    if economic_data.factor_loadings is not None and policy_context.constraints.factor_bounds:
        for factor, (factor_lower, factor_upper) in policy_context.constraints.factor_bounds.items():
            row = np.zeros(n_variables)
            row[layout['w']] = economic_data.factor_loadings[factor].reindex(tickers).to_numpy(dtype=float)
            _append(rows, lower, upper, row, factor_lower, factor_upper)
    if policy_context.constraints.income_floor is not None:
        row = np.zeros(n_variables)
        row[layout['w']] = _array(economic_data.income)
        _append(rows, lower, upper, row, policy_context.constraints.income_floor, np.inf)
    if policy_context.constraints.expected_total_return_floor is not None:
        row = np.zeros(n_variables)
        row[layout['w']] = _array(economic_data.total_return)
        _append(rows, lower, upper, row, policy_context.constraints.expected_total_return_floor, np.inf)
    loss_matrix = _array(scenarios.loss_matrix)
    for scenario_index in range(n_scenarios):
        row = np.zeros(n_variables)
        row[layout['w']] = loss_matrix[scenario_index]
        row[layout['h'].start + scenario_index] = -1.0
        _append(rows, lower, upper, row, -np.inf, scenarios.warning_thresholds[scenario_index])
        row = np.zeros(n_variables)
        row[layout['w']] = loss_matrix[scenario_index]
        _append(rows, lower, upper, row, -np.inf, scenarios.hard_loss_limits[scenario_index])
    fixed_tuple: tuple[str, ...] | None = None
    if fixed_active_tickers is not None:
        fixed_tuple = tuple((str(value) for value in fixed_active_tickers))
        unknown = sorted(set(fixed_tuple) - set(tickers))
        if unknown:
            raise ValueError('Unknown fixed-support tickers: ' + ', '.join(unknown))
        active = set(fixed_tuple)
        for index, ticker in enumerate(tickers):
            if ticker in active:
                continue
            row = np.zeros(n_variables)
            row[layout['w'].start + index] = 1.0
            _append(rows, lower, upper, row, current[index], current[index])
    return _ProblemDefinition(layout=layout, lower_bounds=lower_bounds, upper_bounds=upper_bounds, constraint_matrix=np.vstack(rows), constraint_lower=np.asarray(lower, dtype=float), constraint_upper=np.asarray(upper, dtype=float), objective_data=objective_data, economic_data=economic_data, objective_weights=objective_weights, coefficients=coefficients, context=policy_context, fixed_active_tickers=fixed_tuple)

def _objective_functions(problem: _ProblemDefinition):
    layout = problem.layout
    data = problem.objective_data
    context = problem.context
    scenarios = context.scenarios
    current = _array(context.current_weights)
    covariance = _array(data.covariance)
    impact = _array(data.impact_matrix)
    ow = problem.objective_weights

    def objective(x: np.ndarray) -> float:
        w = x[layout['w']]
        p = x[layout['p']]
        n = x[layout['n']]
        h = x[layout['h']]
        delta = w - current
        return float(ow.risk_penalty * (w @ covariance @ w) - ow.growth_reward * (_array(data.growth) @ w) - ow.income_reward * (_array(data.income) @ w) + ow.concentration_penalty * (w @ w) + ow.transaction_cost_penalty * (_array(data.linear_cost) @ (p + n)) + ow.market_impact_penalty * (delta @ impact @ delta) + ow.scenario_penalty * (_array(scenarios.weights) @ (h * h)))

    def gradient(x: np.ndarray) -> np.ndarray:
        w = x[layout['w']]
        h = x[layout['h']]
        delta = w - current
        grad = np.zeros_like(x)
        grad[layout['w']] = 2.0 * ow.risk_penalty * covariance @ w - ow.growth_reward * _array(data.growth) - ow.income_reward * _array(data.income) + 2.0 * ow.concentration_penalty * w + 2.0 * ow.market_impact_penalty * impact @ delta
        grad[layout['p']] = ow.transaction_cost_penalty * _array(data.linear_cost)
        grad[layout['n']] = ow.transaction_cost_penalty * _array(data.linear_cost)
        grad[layout['h']] = 2.0 * ow.scenario_penalty * _array(scenarios.weights) * h
        return grad
    n_variables = layout['all'].stop
    hessian_matrix = np.zeros((n_variables, n_variables), dtype=float)
    hessian_matrix[layout['w'], layout['w']] = 2.0 * ow.risk_penalty * covariance + 2.0 * ow.concentration_penalty * np.eye(len(data.tickers)) + 2.0 * ow.market_impact_penalty * impact
    hessian_matrix[layout['h'], layout['h']] = np.diag(2.0 * ow.scenario_penalty * _array(scenarios.weights))

    def hessian(_: np.ndarray) -> np.ndarray:
        return hessian_matrix
    return (objective, gradient, hessian, hessian_matrix)

def _initial_point(problem: _ProblemDefinition, weights: Any) -> np.ndarray:
    layout = problem.layout
    context = problem.context
    scenarios = context.scenarios
    current = _array(context.current_weights)
    w = _array(weights).copy()
    if w.shape != current.shape:
        raise ValueError('Starting weights have the wrong shape.')
    x = np.zeros(layout['all'].stop, dtype=float)
    x[layout['w']] = w
    delta = w - current
    x[layout['p']] = np.maximum(delta, 0.0)
    x[layout['n']] = np.maximum(-delta, 0.0)
    losses = _array(scenarios.loss_matrix) @ w
    x[layout['h']] = np.maximum(losses - _array(scenarios.warning_thresholds), 0.0)
    return x

def objective_components_from_weights(*, validation_context: ClassicalValidationContext, policy_context: Any, weights: Any) -> pd.Series:
    objective_data, objective_weights, _ = validation_context.step5.build_goal_objective(validation_context.step4, validation_context.preferences, policy_context.portfolio_data, validation_context.scales, validation_context.mix)
    w = _array(weights)
    current = _array(policy_context.current_weights)
    delta = w - current
    losses = _array(policy_context.scenarios.loss_matrix) @ w
    h = np.maximum(losses - _array(policy_context.scenarios.warning_thresholds), 0.0)
    p_plus_n = np.abs(delta)
    components = {'growth_reward': -float(objective_weights.growth_reward * (_array(objective_data.growth) @ w)), 'income_reward': -float(objective_weights.income_reward * (_array(objective_data.income) @ w)), 'variance_penalty': float(objective_weights.risk_penalty * (w @ _array(objective_data.covariance) @ w)), 'concentration_penalty': float(objective_weights.concentration_penalty * (w @ w)), 'linear_and_turnover_cost_penalty': float(objective_weights.transaction_cost_penalty * (_array(objective_data.linear_cost) @ p_plus_n)), 'market_impact_penalty': float(objective_weights.market_impact_penalty * (delta @ _array(objective_data.impact_matrix) @ delta)), 'scenario_hinge_penalty': float(objective_weights.scenario_penalty * (_array(policy_context.scenarios.weights) @ (h * h)))}
    components['total_objective'] = float(sum(components.values()))
    return pd.Series(components, dtype=float)

def independent_constraint_audit(*, policy_context: Any, weights: Any, fixed_active_tickers: Sequence[str] | None=None, tolerance: float=5e-06) -> pd.DataFrame:
    data = policy_context.portfolio_data
    constraints = policy_context.constraints
    trading = policy_context.trading_config
    scenarios = policy_context.scenarios
    tickers = list(data.tickers)
    w = _array(weights)
    current = _array(policy_context.current_weights)
    delta = w - current
    abs_trade = np.abs(delta)
    records: list[dict[str, Any]] = []

    def add(category: str, name: str, value: float, lower: float, upper: float) -> None:
        lower_violation = max(lower - value, 0.0) if np.isfinite(lower) else 0.0
        upper_violation = max(value - upper, 0.0) if np.isfinite(upper) else 0.0
        violation = max(lower_violation, upper_violation)
        records.append({'category': category, 'constraint': name, 'value': float(value), 'lower': float(lower), 'upper': float(upper), 'absolute_violation': float(violation), 'satisfied': bool(violation <= tolerance)})
    add('budget', 'sum_weights', float(w.sum()), 1.0, 1.0)
    for index, ticker in enumerate(tickers):
        add('asset', f'weight_{ticker}', w[index], constraints.asset_lower[index], constraints.asset_upper[index])
    asset_classes = np.asarray(data.asset_classes, dtype=object)
    for class_name, (class_lower, class_upper) in constraints.class_bounds.items():
        exposure = float(w[asset_classes == class_name].sum())
        add('asset_class', f'class_{class_name}', exposure, class_lower, class_upper)
    if data.factor_loadings is not None:
        for factor, (factor_lower, factor_upper) in constraints.factor_bounds.items():
            exposure = float(data.factor_loadings[factor].reindex(tickers).to_numpy(dtype=float) @ w)
            add('factor', f'factor_{factor}', exposure, factor_lower, factor_upper)
    if constraints.income_floor is not None:
        add('income', 'income_floor', float(_array(data.income) @ w), constraints.income_floor, np.inf)
    if constraints.expected_total_return_floor is not None:
        add('return', 'expected_total_return_floor', float(_array(data.total_return) @ w), constraints.expected_total_return_floor, np.inf)
    add('turnover', 'gross_turnover', float(abs_trade.sum()), -np.inf, trading.turnover_limit_gross)
    trade_capacity = trading.execution_days * trading.participation_rate * _array(data.adv_usd) / trading.portfolio_value_usd
    for index, ticker in enumerate(tickers):
        add('liquidity', f'trade_capacity_{ticker}', abs_trade[index], -np.inf, trade_capacity[index])
    losses = _array(scenarios.loss_matrix) @ w
    for scenario_index, scenario_name in enumerate(scenarios.names):
        add('scenario', f'scenario_{scenario_name}', losses[scenario_index], -np.inf, scenarios.hard_loss_limits[scenario_index])
    if fixed_active_tickers is not None:
        active = set((str(value) for value in fixed_active_tickers))
        for index, ticker in enumerate(tickers):
            if ticker not in active:
                add('fixed_support', f'inactive_trade_{ticker}', delta[index], 0.0, 0.0)
    return pd.DataFrame(records)

def cvxpy_solver_availability() -> pd.Series:
    try:
        import cvxpy as cp
    except Exception as exc:
        return pd.Series({'cvxpy_available': False, 'cvxpy_version': np.nan, 'installed_solvers': '', 'clarabel_available': False, 'osqp_available': False, 'error': f'{type(exc).__name__}: {exc}'}, name='value')
    installed = tuple(sorted((str(value) for value in cp.installed_solvers())))
    return pd.Series({'cvxpy_available': True, 'cvxpy_version': str(cp.__version__), 'installed_solvers': ', '.join(installed), 'clarabel_available': 'CLARABEL' in installed, 'osqp_available': 'OSQP' in installed, 'error': ''}, name='value')

def _constraint_partitions(problem: _ProblemDefinition, *, equality_tolerance: float=1e-12) -> dict[str, np.ndarray]:
    lower = problem.constraint_lower
    upper = problem.constraint_upper
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    equality = finite_lower & finite_upper & (np.abs(lower - upper) <= equality_tolerance)
    return {'equality': equality, 'lower_inequality': finite_lower & ~equality, 'upper_inequality': finite_upper & ~equality, 'non_equality': ~equality}

def _split_scipy_linear_constraints(problem: _ProblemDefinition) -> list[LinearConstraint]:
    partitions = _constraint_partitions(problem)
    constraints: list[LinearConstraint] = []
    equality = partitions['equality']
    if equality.any():
        equality_rhs = problem.constraint_lower[equality]
        constraints.append(LinearConstraint(problem.constraint_matrix[equality], equality_rhs, equality_rhs))
    non_equality = partitions['non_equality']
    if non_equality.any():
        constraints.append(LinearConstraint(problem.constraint_matrix[non_equality], problem.constraint_lower[non_equality], problem.constraint_upper[non_equality]))
    return constraints

def _full_primal_residual(problem: _ProblemDefinition, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.shape != problem.lower_bounds.shape or not np.isfinite(x).all():
        return float('inf')
    bound_lower = np.where(np.isfinite(problem.lower_bounds), np.maximum(problem.lower_bounds - x, 0.0), 0.0)
    bound_upper = np.where(np.isfinite(problem.upper_bounds), np.maximum(x - problem.upper_bounds, 0.0), 0.0)
    values = problem.constraint_matrix @ x
    row_lower = np.where(np.isfinite(problem.constraint_lower), np.maximum(problem.constraint_lower - values, 0.0), 0.0)
    row_upper = np.where(np.isfinite(problem.constraint_upper), np.maximum(values - problem.constraint_upper, 0.0), 0.0)
    return float(max(bound_lower.max(initial=0.0), bound_upper.max(initial=0.0), row_lower.max(initial=0.0), row_upper.max(initial=0.0)))

def _extract_reported_gap(extra_stats: Any) -> float:
    if extra_stats is None:
        return float('nan')
    candidates = ('gap_abs', 'duality_gap', 'dual_gap', 'gap', 'rel_gap')

    def lookup(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            for key in candidates:
                if key in value:
                    try:
                        return float(value[key])
                    except Exception:
                        pass
            for nested in value.values():
                result = lookup(nested)
                if result is not None:
                    return result
            return None
        for key in candidates:
            if hasattr(value, key):
                try:
                    return float(getattr(value, key))
                except Exception:
                    pass
        for nested_name in ('info', 'solution', 'stats'):
            if hasattr(value, nested_name):
                result = lookup(getattr(value, nested_name))
                if result is not None:
                    return result
        return None
    result = lookup(extra_stats)
    return float('nan') if result is None else float(result)

def _cvxpy_kkt_diagnostics(*, problem: _ProblemDefinition, x_value: np.ndarray, hessian_matrix: np.ndarray, linear_term: np.ndarray, dual_objects: Mapping[str, Any]) -> dict[str, float | bool]:
    x_value = np.asarray(x_value, dtype=float)
    stationarity = hessian_matrix @ x_value + linear_term
    complementarity_terms: list[np.ndarray] = []
    inequality_duals: list[np.ndarray] = []

    def dual_array(name: str) -> np.ndarray | None:
        constraint = dual_objects.get(name)
        if constraint is None or constraint.dual_value is None:
            return None
        return np.asarray(constraint.dual_value, dtype=float).reshape(-1)
    partitions = _constraint_partitions(problem)
    matrix = problem.constraint_matrix
    equality = partitions['equality']
    equality_dual = dual_array('row_equality')
    if equality.any() and equality_dual is not None:
        stationarity = stationarity + matrix[equality].T @ equality_dual
    lower_mask = partitions['lower_inequality']
    lower_dual = dual_array('row_lower')
    if lower_mask.any() and lower_dual is not None:
        stationarity = stationarity - matrix[lower_mask].T @ lower_dual
        lower_slack = matrix[lower_mask] @ x_value - problem.constraint_lower[lower_mask]
        complementarity_terms.append(lower_dual * lower_slack)
        inequality_duals.append(lower_dual)
    upper_mask = partitions['upper_inequality']
    upper_dual = dual_array('row_upper')
    if upper_mask.any() and upper_dual is not None:
        stationarity = stationarity + matrix[upper_mask].T @ upper_dual
        upper_slack = problem.constraint_upper[upper_mask] - matrix[upper_mask] @ x_value
        complementarity_terms.append(upper_dual * upper_slack)
        inequality_duals.append(upper_dual)
    finite_bound_lower = np.isfinite(problem.lower_bounds)
    bound_lower_dual = dual_array('bound_lower')
    if finite_bound_lower.any() and bound_lower_dual is not None:
        stationarity[finite_bound_lower] -= bound_lower_dual
        lower_slack = x_value[finite_bound_lower] - problem.lower_bounds[finite_bound_lower]
        complementarity_terms.append(bound_lower_dual * lower_slack)
        inequality_duals.append(bound_lower_dual)
    finite_bound_upper = np.isfinite(problem.upper_bounds)
    bound_upper_dual = dual_array('bound_upper')
    if finite_bound_upper.any() and bound_upper_dual is not None:
        stationarity[finite_bound_upper] += bound_upper_dual
        upper_slack = problem.upper_bounds[finite_bound_upper] - x_value[finite_bound_upper]
        complementarity_terms.append(bound_upper_dual * upper_slack)
        inequality_duals.append(bound_upper_dual)
    complementarity = max((float(np.abs(values).max(initial=0.0)) for values in complementarity_terms)) if complementarity_terms else 0.0
    dual_minimum = min((float(values.min(initial=0.0)) for values in inequality_duals)) if inequality_duals else 0.0
    return {'primal_residual_inf': _full_primal_residual(problem, x_value), 'stationarity_residual_inf': float(np.abs(stationarity).max(initial=0.0)), 'complementarity_residual_inf': float(complementarity), 'dual_feasibility_min': float(dual_minimum)}

def _solve_with_cvxpy(*, problem: _ProblemDefinition, x0: np.ndarray, tolerances: ValidationTolerances) -> list[dict[str, Any]]:
    try:
        import cvxpy as cp
    except Exception as exc:
        raise RuntimeError('CVXPY is required for the final independent Step 7 validation. Install cvxpy, clarabel, and osqp.') from exc
    installed = set((str(value) for value in cp.installed_solvers()))
    solver_order = [solver for solver in ('CLARABEL', 'OSQP') if solver in installed]
    if not solver_order:
        raise RuntimeError('Neither CLARABEL nor OSQP is available through CVXPY.')
    _, gradient, _, hessian_matrix = _objective_functions(problem)
    hessian_matrix = 0.5 * (hessian_matrix + hessian_matrix.T)
    linear_term = gradient(np.zeros(problem.layout['all'].stop, dtype=float))
    n_variables = problem.layout['all'].stop
    variable = cp.Variable(n_variables, name='portfolio_qp_variables')
    dual_objects: dict[str, Any] = {}
    constraints: list[Any] = []
    finite_lower = np.isfinite(problem.lower_bounds)
    if finite_lower.any():
        dual_objects['bound_lower'] = variable[finite_lower] >= problem.lower_bounds[finite_lower]
        constraints.append(dual_objects['bound_lower'])
    finite_upper = np.isfinite(problem.upper_bounds)
    if finite_upper.any():
        dual_objects['bound_upper'] = variable[finite_upper] <= problem.upper_bounds[finite_upper]
        constraints.append(dual_objects['bound_upper'])
    partitions = _constraint_partitions(problem)
    matrix = problem.constraint_matrix
    equality = partitions['equality']
    if equality.any():
        dual_objects['row_equality'] = matrix[equality] @ variable == problem.constraint_lower[equality]
        constraints.append(dual_objects['row_equality'])
    lower_mask = partitions['lower_inequality']
    if lower_mask.any():
        dual_objects['row_lower'] = matrix[lower_mask] @ variable >= problem.constraint_lower[lower_mask]
        constraints.append(dual_objects['row_lower'])
    upper_mask = partitions['upper_inequality']
    if upper_mask.any():
        dual_objects['row_upper'] = matrix[upper_mask] @ variable <= problem.constraint_upper[upper_mask]
        constraints.append(dual_objects['row_upper'])
    qp_objective = cp.Minimize(0.5 * cp.quad_form(variable, cp.psd_wrap(hessian_matrix)) + linear_term @ variable)
    cvx_problem = cp.Problem(qp_objective, constraints)
    records: list[dict[str, Any]] = []
    for solver_name in solver_order:
        variable.value = np.asarray(x0, dtype=float)
        solver_options: dict[str, Any]
        if solver_name == 'CLARABEL':
            solver_options = {'max_iter': 5000, 'tol_gap_abs': 1e-10, 'tol_gap_rel': 1e-10, 'tol_feas': 1e-10}
        else:
            solver_options = {'max_iter': 200000, 'eps_abs': 1e-09, 'eps_rel': 1e-09, 'polishing': True}
        try:
            cvx_problem.solve(solver=solver_name, warm_start=True, verbose=False, **solver_options)
            status = str(cvx_problem.status)
            x_value = None if variable.value is None else np.asarray(variable.value, dtype=float).copy()
            finite = bool(x_value is not None and x_value.shape == (n_variables,) and np.isfinite(x_value).all())
            if finite:
                kkt = _cvxpy_kkt_diagnostics(problem=problem, x_value=x_value, hessian_matrix=hessian_matrix, linear_term=linear_term, dual_objects=dual_objects)
            else:
                kkt = {'primal_residual_inf': float('inf'), 'stationarity_residual_inf': float('inf'), 'complementarity_residual_inf': float('inf'), 'dual_feasibility_min': float('-inf')}
            reported_gap = _extract_reported_gap(getattr(cvx_problem.solver_stats, 'extra_stats', None))
            kkt_pass = bool(finite and status in {str(cp.OPTIMAL), str(cp.OPTIMAL_INACCURATE)} and (kkt['primal_residual_inf'] <= tolerances.feasibility) and (kkt['stationarity_residual_inf'] <= tolerances.kkt_stationarity) and (kkt['complementarity_residual_inf'] <= tolerances.kkt_complementarity) and (kkt['dual_feasibility_min'] >= -tolerances.dual_feasibility))
            records.append({'solver_source': f'cvxpy_{solver_name.lower()}', 'role': 'independent_convex_qp_validator', 'status': status, 'finite_solution': finite, 'x': x_value, 'primal_residual_inf': float(kkt['primal_residual_inf']), 'stationarity_residual_inf': float(kkt['stationarity_residual_inf']), 'complementarity_residual_inf': float(kkt['complementarity_residual_inf']), 'dual_feasibility_min': float(kkt['dual_feasibility_min']), 'solver_reported_duality_gap': reported_gap, 'kkt_certificate_pass': kkt_pass, 'iterations': int(getattr(cvx_problem.solver_stats, 'num_iters', 0) or 0), 'solve_time_seconds': float(getattr(cvx_problem.solver_stats, 'solve_time', np.nan) or np.nan), 'raw_result': {'status': status, 'solver_name': solver_name, 'problem_value_without_constant': None if cvx_problem.value is None else float(cvx_problem.value), 'num_iters': int(getattr(cvx_problem.solver_stats, 'num_iters', 0) or 0), 'solve_time_seconds': float(getattr(cvx_problem.solver_stats, 'solve_time', np.nan) or np.nan), 'setup_time_seconds': float(getattr(cvx_problem.solver_stats, 'setup_time', np.nan) or np.nan), 'extra_stats_type': type(getattr(cvx_problem.solver_stats, 'extra_stats', None)).__name__}, 'error': ''})
        except Exception as exc:
            records.append({'solver_source': f'cvxpy_{solver_name.lower()}', 'role': 'independent_convex_qp_validator', 'status': 'ERROR', 'finite_solution': False, 'x': None, 'primal_residual_inf': float('inf'), 'stationarity_residual_inf': float('inf'), 'complementarity_residual_inf': float('inf'), 'dual_feasibility_min': float('-inf'), 'solver_reported_duality_gap': float('nan'), 'kkt_certificate_pass': False, 'iterations': 0, 'solve_time_seconds': float('nan'), 'raw_result': None, 'error': f'{type(exc).__name__}: {exc}'})
    return records

def _scipy_diagnostic_record(*, solver_source: str, role: str, result: OptimizeResult, problem: _ProblemDefinition, validation_context: ClassicalValidationContext, policy_context: Any, fixed_active_tickers: Sequence[str] | None) -> dict[str, Any]:
    x_value = np.asarray(getattr(result, 'x', np.full(problem.layout['all'].stop, np.nan)), dtype=float)
    finite = bool(x_value.shape == (problem.layout['all'].stop,) and np.isfinite(x_value).all())
    if finite:
        weights = x_value[problem.layout['w']]
        audit = independent_constraint_audit(policy_context=policy_context, weights=weights, fixed_active_tickers=fixed_active_tickers, tolerance=validation_context.tolerances.feasibility)
        objective_value = float(objective_components_from_weights(validation_context=validation_context, policy_context=policy_context, weights=weights)['total_objective'])
        feasible = bool(audit['satisfied'].all() and _full_primal_residual(problem, x_value) <= validation_context.tolerances.feasibility)
    else:
        objective_value = float('nan')
        feasible = False
    return {'solver_source': solver_source, 'role': role, 'status': str(getattr(result, 'message', '')), 'finite_solution': finite, 'feasible': feasible, 'exact_economic_objective': objective_value, 'primal_residual_inf': _full_primal_residual(problem, x_value) if finite else float('inf'), 'stationarity_residual_inf': float(getattr(result, 'optimality', np.nan)), 'complementarity_residual_inf': float('nan'), 'dual_feasibility_min': float('nan'), 'solver_reported_duality_gap': float('nan'), 'kkt_certificate_pass': False, 'iterations': int(getattr(result, 'nit', 0) or 0), 'solve_time_seconds': float('nan'), 'selected': False, 'error': ''}

def solve_independent_classical(*, validation_context: ClassicalValidationContext, policy_context: Any, label: str, policy: str, start_weights: Any, fixed_active_tickers: Sequence[str] | None=None, maxiter: int=2000) -> IndependentClassicalResult:
    problem = build_independent_problem(context=validation_context, policy_context=policy_context, fixed_active_tickers=fixed_active_tickers)
    objective, gradient, hessian, _ = _objective_functions(problem)
    x0 = _initial_point(problem, start_weights)
    scipy_constraints = _split_scipy_linear_constraints(problem)
    bounds = Bounds(problem.lower_bounds, problem.upper_bounds)
    trust_result = minimize(objective, x0, method='trust-constr', jac=gradient, hess=hessian, bounds=bounds, constraints=scipy_constraints, options={'gtol': 1e-09, 'xtol': 1e-11, 'barrier_tol': 1e-11, 'maxiter': int(maxiter), 'verbose': 0})
    trust_x = np.asarray(getattr(trust_result, 'x', x0), dtype=float)
    polish_start = trust_x if trust_x.shape == x0.shape and np.isfinite(trust_x).all() else x0.copy()
    polish_result = minimize(objective, polish_start, method='SLSQP', jac=gradient, bounds=bounds, constraints=scipy_constraints, options={'ftol': 1e-12, 'maxiter': max(3000, int(maxiter)), 'disp': False})
    scipy_records = [_scipy_diagnostic_record(solver_source='scipy_trust_constr', role='algorithmic_cross_check', result=trust_result, problem=problem, validation_context=validation_context, policy_context=policy_context, fixed_active_tickers=fixed_active_tickers), _scipy_diagnostic_record(solver_source='scipy_slsqp_split_constraints', role='numerical_cross_check_only', result=polish_result, problem=problem, validation_context=validation_context, policy_context=policy_context, fixed_active_tickers=fixed_active_tickers)]
    cvxpy_records = _solve_with_cvxpy(problem=problem, x0=x0, tolerances=validation_context.tolerances)
    eligible_cvxpy_records: list[dict[str, Any]] = []
    for record in cvxpy_records:
        candidate = record.get('x')
        if candidate is None:
            record['feasible'] = False
            record['exact_economic_objective'] = float('nan')
            record['selected'] = False
            continue
        candidate = np.asarray(candidate, dtype=float)
        weights = candidate[problem.layout['w']]
        audit = independent_constraint_audit(policy_context=policy_context, weights=weights, fixed_active_tickers=fixed_active_tickers, tolerance=validation_context.tolerances.feasibility)
        exact_objective = float(objective_components_from_weights(validation_context=validation_context, policy_context=policy_context, weights=weights)['total_objective'])
        feasible = bool(audit['satisfied'].all() and record['primal_residual_inf'] <= validation_context.tolerances.feasibility)
        record['feasible'] = feasible
        record['exact_economic_objective'] = exact_objective
        record['selected'] = False
        if feasible and bool(record['kkt_certificate_pass']) and np.isfinite(exact_objective):
            record['_audit'] = audit
            eligible_cvxpy_records.append(record)
    if not eligible_cvxpy_records:
        diagnostic_table = pd.DataFrame([{key: value for key, value in record.items() if key not in {'x', 'raw_result', '_audit'}} for record in cvxpy_records])
        raise RuntimeError('No independent CVXPY solver produced a feasible KKT-certified Step 7 solution. Diagnostics:\n' + diagnostic_table.to_string(index=False))
    selected = min(eligible_cvxpy_records, key=lambda record: record['exact_economic_objective'])
    selected['selected'] = True
    final_x = np.asarray(selected['x'], dtype=float)
    final_weights = final_x[problem.layout['w']]
    final_audit = selected['_audit']
    all_records = scipy_records + cvxpy_records
    diagnostics_rows = []
    for record in all_records:
        diagnostics_rows.append({'solver_source': record['solver_source'], 'role': record['role'], 'status': record['status'], 'finite_solution': bool(record.get('finite_solution', False)), 'feasible': bool(record.get('feasible', False)), 'exact_economic_objective': float(record.get('exact_economic_objective', np.nan)), 'primal_residual_inf': float(record.get('primal_residual_inf', np.nan)), 'stationarity_residual_inf': float(record.get('stationarity_residual_inf', np.nan)), 'complementarity_residual_inf': float(record.get('complementarity_residual_inf', np.nan)), 'dual_feasibility_min': float(record.get('dual_feasibility_min', np.nan)), 'solver_reported_duality_gap': float(record.get('solver_reported_duality_gap', np.nan)), 'kkt_certificate_pass': bool(record.get('kkt_certificate_pass', False)), 'iterations': int(record.get('iterations', 0)), 'solve_time_seconds': float(record.get('solve_time_seconds', np.nan)), 'selected': bool(record.get('selected', False)), 'error': str(record.get('error', ''))})
    solver_diagnostics = pd.DataFrame(diagnostics_rows)
    components = objective_components_from_weights(validation_context=validation_context, policy_context=policy_context, weights=final_weights)
    feasible = bool(final_audit['satisfied'].all())
    kkt_pass = bool(selected['kkt_certificate_pass'])
    combined_message = '; '.join([f"selected independent solver={selected['solver_source']}", f'trust-constr={trust_result.message}', f'split-constraint SLSQP={polish_result.message}', 'stored starting point was initialization only and was not selectable'])
    return IndependentClassicalResult(label=label, policy=policy, solver='independent CVXPY convex QP (Clarabel/OSQP)', selected_solver_source=str(selected['solver_source']), success=bool(feasible and kkt_pass and np.isfinite(components['total_objective'])), message=combined_message, objective_value=float(components['total_objective']), weights=pd.Series(final_weights, index=policy_context.portfolio_data.tickers, name=label), iterations=int(selected['iterations']), optimality=float(selected['stationarity_residual_inf']), constr_violation=float(selected['primal_residual_inf']), kkt_stationarity_inf=float(selected['stationarity_residual_inf']), kkt_complementarity_inf=float(selected['complementarity_residual_inf']), dual_feasibility_min=float(selected['dual_feasibility_min']), solver_reported_duality_gap=float(selected['solver_reported_duality_gap']), kkt_certificate_pass=kkt_pass, audit=final_audit, objective_components=components, solver_diagnostics=solver_diagnostics, raw_result=selected['raw_result'], fixed_active_tickers=None if fixed_active_tickers is None else tuple((str(value) for value in fixed_active_tickers)))

def convexity_certificate(*, validation_context: ClassicalValidationContext, policy_context: Any) -> pd.Series:
    problem = build_independent_problem(context=validation_context, policy_context=policy_context)
    _, _, _, hessian = _objective_functions(problem)
    covariance_min = float(np.linalg.eigvalsh(_array(problem.objective_data.covariance)).min())
    impact_min = float(np.linalg.eigvalsh(_array(problem.objective_data.impact_matrix)).min())
    hessian_min = float(np.linalg.eigvalsh(hessian).min())
    coefficients_nonnegative = all((float(value) >= -validation_context.tolerances.hessian_psd for key, value in problem.coefficients.items() if key not in {'growth_coefficient', 'income_coefficient'}))
    certified = bool(covariance_min >= -validation_context.tolerances.hessian_psd and impact_min >= -validation_context.tolerances.hessian_psd and (hessian_min >= -validation_context.tolerances.hessian_psd) and coefficients_nonnegative)
    return pd.Series({'covariance_minimum_eigenvalue': covariance_min, 'impact_minimum_eigenvalue': impact_min, 'objective_hessian_minimum_eigenvalue': hessian_min, 'penalty_coefficients_nonnegative': coefficients_nonnegative, 'linear_constraint_system': True, 'continuous_problem_convex': certified, 'global_optimum_interpretation': certified}, name='value')

def compare_profile_to_solution(*, validation_context: ClassicalValidationContext, policy_context: Any, profile_name: str, profile: Mapping[str, Any], independent_result: IndependentClassicalResult, validation_target: str, objective_tolerance: float | None=None) -> pd.Series:
    stored_weights = _array(profile['result'].weights)
    stored_components = objective_components_from_weights(validation_context=validation_context, policy_context=policy_context, weights=stored_weights)
    stored_audit = independent_constraint_audit(policy_context=policy_context, weights=stored_weights, fixed_active_tickers=independent_result.fixed_active_tickers, tolerance=validation_context.tolerances.feasibility)
    independent_weights = independent_result.weights.to_numpy(dtype=float)
    raw_gap = float(stored_components['total_objective'] - independent_result.objective_value)
    numerical_gap = max(raw_gap, 0.0)
    tolerance = validation_context.tolerances.objective_match if objective_tolerance is None else float(objective_tolerance)
    feasible = bool(stored_audit['satisfied'].all())
    objective_match = bool(abs(raw_gap) <= tolerance)
    if not independent_result.success:
        verdict = 'FAIL_INDEPENDENT_SOLVER'
    elif not feasible:
        verdict = 'FAIL_INFEASIBLE'
    elif objective_match:
        verdict = 'PASS_OPTIMUM_MATCH'
    else:
        verdict = 'PASS_FEASIBLE_WITH_GAP'
    economic = validation_context.step5.exact_goal_components(policy_context.portfolio_data, stored_weights, policy_context.current_weights, policy_context.scenarios)
    return pd.Series({'profile': profile_name, 'validation_target': validation_target, 'policy': independent_result.policy, 'stored_objective': float(stored_components['total_objective']), 'independent_objective': independent_result.objective_value, 'signed_objective_gap': raw_gap, 'nonnegative_objective_gap': numerical_gap, 'objective_match_tolerance': tolerance, 'weight_l1_difference': float(np.abs(stored_weights - independent_weights).sum()), 'weight_linf_difference': float(np.abs(stored_weights - independent_weights).max()), 'expected_total_return': economic['expected_total_return'], 'volatility': economic['volatility'], 'worst_scenario_loss': economic['worst_scenario_loss'], 'gross_turnover': economic['gross_turnover'], 'total_trading_cost': economic['total_trading_cost'], 'hard_constraint_pass': feasible, 'independent_solver_success': independent_result.success, 'selected_solver_source': independent_result.selected_solver_source, 'kkt_certificate_pass': independent_result.kkt_certificate_pass, 'kkt_stationarity_inf': independent_result.kkt_stationarity_inf, 'kkt_complementarity_inf': independent_result.kkt_complementarity_inf, 'dual_feasibility_min': independent_result.dual_feasibility_min, 'verdict': verdict}, name=profile_name)

def enumerate_qubo_exactly(*, model: Any, qaoa_selected_tickers: Sequence[str], exact_selected_tickers: Sequence[str] | None=None) -> tuple[pd.DataFrame, pd.Series]:
    n = int(model.n_variables)
    cardinality = int(model.cardinality)
    records: list[dict[str, Any]] = []
    for chosen in combinations(range(n), cardinality):
        bits = np.zeros(n, dtype=int)
        bits[list(chosen)] = 1
        selected = tuple((model.tickers[index] for index in chosen))
        records.append({'bitstring': ''.join((str(int(value)) for value in bits)), 'energy': float(model.energy(bits)), 'selected_tickers': selected})
    table = pd.DataFrame(records).sort_values(['energy', 'bitstring']).reset_index(drop=True)
    table['classical_rank'] = np.arange(1, len(table) + 1)
    qaoa_set = frozenset((str(value) for value in qaoa_selected_tickers))
    qaoa_matches = table['selected_tickers'].map(lambda values: frozenset(values) == qaoa_set)
    if not qaoa_matches.any():
        raise ValueError('QAOA selected set is absent from exact enumeration.')
    qaoa_row = table.loc[qaoa_matches].iloc[0]
    exact_row = table.iloc[0]
    exact_set_matches = np.nan
    if exact_selected_tickers is not None:
        exact_set_matches = bool(frozenset(exact_selected_tickers) == frozenset(exact_row['selected_tickers']))
    summary = pd.Series({'state_count': len(table), 'n_variables': n, 'cardinality': cardinality, 'exact_energy': float(exact_row['energy']), 'qaoa_energy': float(qaoa_row['energy']), 'qaoa_energy_gap': float(qaoa_row['energy'] - exact_row['energy']), 'qaoa_classical_rank': int(qaoa_row['classical_rank']), 'qaoa_exact_optimum': bool(int(qaoa_row['classical_rank']) == 1), 'reported_exact_set_matches_reenumeration': exact_set_matches, 'exact_selected_tickers': ', '.join(exact_row['selected_tickers']), 'qaoa_selected_tickers': ', '.join(qaoa_row['selected_tickers'])}, name='value')
    return (table, summary)

def build_shortlist_opportunity_cost(*, validation_context: ClassicalValidationContext, base_solution: IndependentClassicalResult) -> pd.DataFrame:
    shortlist = validation_context.step6_handoff['shortlist']
    profile_results = validation_context.profiles
    records: list[dict[str, Any]] = []
    for profile_name in shortlist.index:
        profile = profile_results[profile_name]
        weights = _array(profile['result'].weights)
        components = objective_components_from_weights(validation_context=validation_context, policy_context=validation_context.base_context, weights=weights)
        economics = validation_context.step5.exact_goal_components(validation_context.base_context.portfolio_data, weights, validation_context.base_context.current_weights, validation_context.base_context.scenarios)
        audit = independent_constraint_audit(policy_context=validation_context.base_context, weights=weights, tolerance=validation_context.tolerances.feasibility)
        records.append({'profile': profile_name, 'base_policy_objective': float(components['total_objective']), 'objective_gap_to_base_global_optimum': max(float(components['total_objective'] - base_solution.objective_value), 0.0), 'expected_total_return': economics['expected_total_return'], 'volatility': economics['volatility'], 'worst_scenario_loss': economics['worst_scenario_loss'], 'gross_turnover': economics['gross_turnover'], 'total_trading_cost': economics['total_trading_cost'], 'hard_constraint_pass': bool(audit['satisfied'].all())})
    return pd.DataFrame(records).set_index('profile').sort_values('objective_gap_to_base_global_optimum')

def summarize_independent_result(result: IndependentClassicalResult) -> pd.Series:
    return pd.Series({'policy': result.policy, 'solver': result.solver, 'selected_solver_source': result.selected_solver_source, 'success': result.success, 'message': result.message, 'objective_value': result.objective_value, 'iterations': result.iterations, 'optimality': result.optimality, 'constraint_violation': result.constr_violation, 'kkt_stationarity_inf': result.kkt_stationarity_inf, 'kkt_complementarity_inf': result.kkt_complementarity_inf, 'dual_feasibility_min': result.dual_feasibility_min, 'solver_reported_duality_gap': result.solver_reported_duality_gap, 'kkt_certificate_pass': result.kkt_certificate_pass, 'hard_constraint_pass': bool(result.audit['satisfied'].all()), 'maximum_hard_violation': float(result.audit['absolute_violation'].max()), 'fixed_support_size': np.nan if result.fixed_active_tickers is None else len(result.fixed_active_tickers)}, name=result.label)

def build_validation_verdict(*, base_comparison: pd.Series, strict_comparison: pd.Series | None, qaoa_support_comparison: pd.Series, exact_support_comparison: pd.Series, qubo_summary: pd.Series, convexity: pd.Series) -> pd.Series:
    base_pass = base_comparison['verdict'] == 'PASS_OPTIMUM_MATCH' and bool(base_comparison['hard_constraint_pass']) and bool(base_comparison['independent_solver_success']) and bool(base_comparison['kkt_certificate_pass'])
    strict_pass = True
    if strict_comparison is not None:
        strict_pass = strict_comparison['verdict'] == 'PASS_OPTIMUM_MATCH' and bool(strict_comparison['hard_constraint_pass']) and bool(strict_comparison['independent_solver_success']) and bool(strict_comparison['kkt_certificate_pass'])
    support_pass = qaoa_support_comparison['verdict'] == 'PASS_OPTIMUM_MATCH' and exact_support_comparison['verdict'] == 'PASS_OPTIMUM_MATCH' and bool(qaoa_support_comparison['independent_solver_success']) and bool(exact_support_comparison['independent_solver_success']) and bool(qaoa_support_comparison['kkt_certificate_pass']) and bool(exact_support_comparison['kkt_certificate_pass'])
    convex = bool(convexity['continuous_problem_convex'])
    exact_reproduced = bool(qubo_summary['reported_exact_set_matches_reenumeration'])
    all_core = bool(base_pass and strict_pass and support_pass and convex and exact_reproduced)
    qaoa_exact = bool(qubo_summary['qaoa_exact_optimum'])
    if not all_core:
        verdict = 'FAIL_CLASSICAL_VALIDATION'
    elif qaoa_exact:
        verdict = 'PASS_QAOA_EXACT'
    else:
        verdict = 'PASS_QAOA_FEASIBLE_NEAR_OPTIMAL'
    return pd.Series({'continuous_problem_convex': convex, 'base_classical_global_optimum_reproduced': base_pass, 'strict_policy_optimum_reproduced': strict_pass, 'qaoa_fixed_support_refinement_reproduced': qaoa_support_comparison['verdict'] == 'PASS_OPTIMUM_MATCH', 'exact_fixed_support_refinement_reproduced': exact_support_comparison['verdict'] == 'PASS_OPTIMUM_MATCH', 'reported_exact_qubo_reproduced': exact_reproduced, 'independent_cvxpy_kkt_certificates_pass': bool(base_comparison['kkt_certificate_pass'] and (True if strict_comparison is None else strict_comparison['kkt_certificate_pass']) and qaoa_support_comparison['kkt_certificate_pass'] and exact_support_comparison['kkt_certificate_pass']), 'stored_starting_point_selectable': False, 'qaoa_exact_qubo_optimum': qaoa_exact, 'qaoa_qubo_rank': int(qubo_summary['qaoa_classical_rank']), 'qaoa_qubo_energy_gap': float(qubo_summary['qaoa_energy_gap']), 'overall_validation_verdict': verdict, 'supports_quantum_advantage_claim': False, 'supports_feasible_near_optimal_qaoa_claim': bool(all_core)}, name='value')
