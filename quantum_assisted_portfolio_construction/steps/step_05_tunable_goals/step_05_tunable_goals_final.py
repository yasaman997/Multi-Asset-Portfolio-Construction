from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping
import numpy as np
import pandas as pd
from scipy.optimize import linprog
EPS = 1e-12

def as_numpy(values: Any) -> np.ndarray:
    if hasattr(values, 'to_numpy'):
        return values.to_numpy(dtype=float)
    return np.asarray(values, dtype=float)

@dataclass(frozen=True)
class GoalPreferences:
    growth: float = 55.0
    income: float = 55.0
    drawdown_control: float = 70.0
    cost_sensitivity: float = 60.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value):
                raise ValueError(f'{name} must be finite.')
            if not 0.0 <= value <= 100.0:
                raise ValueError(f'{name} must lie between 0 and 100.')

    @property
    def shares(self) -> dict[str, float]:
        self.validate()
        values = np.array([self.growth, self.income, self.drawdown_control, self.cost_sensitivity], dtype=float)
        if float(values.sum()) <= EPS:
            values = np.ones(4, dtype=float)
        values /= values.sum()
        return {'growth': float(values[0]), 'income': float(values[1]), 'drawdown': float(values[2]), 'cost': float(values[3])}

@dataclass(frozen=True)
class GoalMixConfig:
    variance_share_of_drawdown: float = 0.35
    scenario_share_of_drawdown: float = 0.65
    linear_cost_share: float = 0.3
    impact_cost_share: float = 0.25
    turnover_share: float = 0.45
    concentration_tiebreaker: float = 0.0001
    scenario_tiebreaker: float = 1e-05
    execution_tiebreaker: float = 1e-06

    def validate(self) -> None:
        drawdown_total = self.variance_share_of_drawdown + self.scenario_share_of_drawdown
        cost_total = self.linear_cost_share + self.impact_cost_share + self.turnover_share
        if not np.isclose(drawdown_total, 1.0, atol=1e-12):
            raise ValueError('Drawdown-component shares must sum to one.')
        if not np.isclose(cost_total, 1.0, atol=1e-12):
            raise ValueError('Cost-component shares must sum to one.')
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f'{name} must be finite and nonnegative.')

@dataclass(frozen=True)
class GoalScales:
    growth: float
    income: float
    variance: float
    scenario_hinge: float
    linear_cost: float
    impact_cost: float
    turnover: float
    concentration: float

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'Scale {name} must be finite and positive.')

@dataclass
class Step5Context:
    step4: Any
    portfolio_data: Any
    current_weights: Any
    stages: list[Any]
    scenarios: Any
    constraints: Any
    trading_config: Any
    daily_returns: pd.DataFrame
GOAL_MIX = GoalMixConfig()
GOAL_PRESETS: dict[str, GoalPreferences] = {'Balanced': GoalPreferences(55, 55, 70, 60), 'Growth Focus': GoalPreferences(90, 25, 35, 25), 'Income Focus': GoalPreferences(30, 95, 55, 45), 'Capital Preservation': GoalPreferences(20, 40, 100, 70), 'Low Turnover / Cost': GoalPreferences(35, 40, 60, 100)}

def patch_step4_exact_hinge_reporting(step4: Any) -> None:
    if hasattr(step4, '_step5_original_calculate_metrics'):
        return
    step4._step5_original_calculate_metrics = step4.calculate_metrics

    def calculate_metrics_with_exact_hinge(*, data: Any, weights: Any, current_weights: Any, scenarios: Any, objective_weights: Any, scenario_excess: Any) -> dict[str, float]:
        weights_array = np.asarray(weights, dtype=float)
        if scenarios is not None:
            losses = scenarios.loss_matrix @ weights_array
            scenario_excess = np.maximum(losses - scenarios.warning_thresholds, 0.0)
        return step4._step5_original_calculate_metrics(data=data, weights=weights_array, current_weights=np.asarray(current_weights, dtype=float), scenarios=scenarios, objective_weights=objective_weights, scenario_excess=np.asarray(scenario_excess, dtype=float))
    step4.calculate_metrics = calculate_metrics_with_exact_hinge

def exact_goal_components(data: Any, weights: Any, incumbent_weights: Any, scenario_set: Any) -> dict[str, float]:
    w = as_numpy(weights)
    w0 = as_numpy(incumbent_weights)
    delta = w - w0
    variance = float(w @ data.covariance @ w)
    losses = scenario_set.loss_matrix @ w
    excess = np.maximum(losses - scenario_set.warning_thresholds, 0.0)
    linear_cost = float(data.linear_cost @ np.abs(delta))
    impact_cost = float(delta @ data.impact_matrix @ delta)
    concentration = float(w @ w)
    return {'growth': float(data.growth @ w), 'income': float(data.income @ w), 'expected_total_return': float(data.total_return @ w), 'variance': variance, 'volatility': float(np.sqrt(max(variance, 0.0))), 'scenario_hinge': float(scenario_set.weights @ excess ** 2), 'worst_scenario_loss': float(losses.max()), 'linear_cost': linear_cost, 'impact_cost': impact_cost, 'total_trading_cost': linear_cost + impact_cost, 'gross_turnover': float(np.abs(delta).sum()), 'one_way_turnover': float(0.5 * np.abs(delta).sum()), 'concentration': concentration, 'effective_holdings': float(1.0 / concentration) if concentration > EPS else np.inf, 'maximum_asset_weight': float(w.max())}

def portfolio_path_metrics(daily_returns: pd.DataFrame, weights: Any, tickers: Iterable[str]) -> dict[str, float]:
    weight_series = pd.Series(as_numpy(weights), index=list(tickers), dtype=float)
    aligned = daily_returns.reindex(columns=weight_series.index).dropna(how='any')
    if len(aligned) < 20:
        return {'realized_annual_return': np.nan, 'realized_annual_volatility': np.nan, 'realized_maximum_drawdown': np.nan, 'daily_var_95': np.nan, 'daily_cvar_95': np.nan, 'path_observations': len(aligned)}
    portfolio_returns = aligned @ weight_series
    wealth = (1.0 + portfolio_returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    count = len(portfolio_returns)
    q05 = float(portfolio_returns.quantile(0.05))
    tail = portfolio_returns[portfolio_returns <= q05]
    return {'realized_annual_return': float(wealth.iloc[-1] ** (252.0 / count) - 1.0), 'realized_annual_volatility': float(portfolio_returns.std(ddof=1) * np.sqrt(252.0)), 'realized_maximum_drawdown': float(-drawdown.min()), 'daily_var_95': float(-q05), 'daily_cvar_95': float(-tail.mean()), 'path_observations': count}

def robust_component_scale(values: Iterable[float], floor: float) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError('Scale inputs must be finite and nonempty.')
    observed_range = float(np.ptp(array))
    typical_magnitude = float(np.quantile(np.abs(array), 0.75))
    return max(observed_range, 0.1 * typical_magnitude, floor)

def _exact_anchor_audit(context: Step5Context, weights: Any, tolerance: float=1e-07) -> pd.DataFrame:
    w = as_numpy(weights)
    w0 = as_numpy(context.current_weights)
    buys = np.maximum(w - w0, 0.0)
    sells = np.maximum(w0 - w, 0.0)
    return context.step4.audit_constraints(data=context.portfolio_data, weights=w, current_weights=w0, buys=buys, sells=sells, scenarios=context.scenarios, constraint_config=context.constraints, trading_config=context.trading_config, check_asset_caps=True, check_classes=True, check_factors=context.portfolio_data.factor_loadings is not None and bool(context.constraints.factor_bounds), check_income=True, check_return=True, check_trading=True, check_scenario_hard=True, tolerance=tolerance)

def build_fully_constrained_linear_program(context: Step5Context, reward_vector: Any) -> dict[str, Any]:
    data = context.portfolio_data
    config = context.constraints
    trading = context.trading_config
    n = len(data.tickers)
    w_slice = slice(0, n)
    p_slice = slice(n, 2 * n)
    n_slice = slice(2 * n, 3 * n)
    n_vars = 3 * n
    reward = np.asarray(reward_vector, dtype=float)
    if reward.shape != (n,):
        raise ValueError('reward_vector has the wrong shape.')
    objective = np.zeros(n_vars, dtype=float)
    objective[w_slice] = -reward
    cleanup = 1e-09
    normalized_cost = data.linear_cost / max(float(np.max(data.linear_cost)), EPS)
    objective[p_slice] = cleanup * (1.0 + normalized_cost)
    objective[n_slice] = cleanup * (1.0 + normalized_cost)
    eq_rows: list[np.ndarray] = []
    eq_rhs: list[float] = []
    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []
    row = np.zeros(n_vars)
    row[w_slice] = 1.0
    eq_rows.append(row)
    eq_rhs.append(1.0)
    incumbent = as_numpy(context.current_weights)
    for i in range(n):
        row = np.zeros(n_vars)
        row[w_slice.start + i] = 1.0
        row[p_slice.start + i] = -1.0
        row[n_slice.start + i] = 1.0
        eq_rows.append(row)
        eq_rhs.append(float(incumbent[i]))
    row = np.zeros(n_vars)
    row[p_slice] = 1.0
    row[n_slice] = 1.0
    ub_rows.append(row)
    ub_rhs.append(float(trading.turnover_limit_gross))
    capacity = trading.execution_days * trading.participation_rate * data.adv_usd / trading.portfolio_value_usd
    for i in range(n):
        row = np.zeros(n_vars)
        row[p_slice.start + i] = 1.0
        row[n_slice.start + i] = 1.0
        ub_rows.append(row)
        ub_rhs.append(float(capacity[i]))
    classes = np.asarray(data.asset_classes, dtype=object)
    for asset_class, (lower, upper) in config.class_bounds.items():
        exposure = (classes == asset_class).astype(float)
        row = np.zeros(n_vars)
        row[w_slice] = exposure
        ub_rows.append(row)
        ub_rhs.append(float(upper))
        row = np.zeros(n_vars)
        row[w_slice] = -exposure
        ub_rows.append(row)
        ub_rhs.append(float(-lower))
    if data.factor_loadings is not None:
        for factor, (lower, upper) in config.factor_bounds.items():
            exposure = data.factor_loadings[factor].to_numpy(dtype=float)
            row = np.zeros(n_vars)
            row[w_slice] = exposure
            ub_rows.append(row)
            ub_rhs.append(float(upper))
            row = np.zeros(n_vars)
            row[w_slice] = -exposure
            ub_rows.append(row)
            ub_rhs.append(float(-lower))
    if config.income_floor is not None:
        row = np.zeros(n_vars)
        row[w_slice] = -data.income
        ub_rows.append(row)
        ub_rhs.append(float(-config.income_floor))
    if config.expected_total_return_floor is not None:
        row = np.zeros(n_vars)
        row[w_slice] = -data.total_return
        ub_rows.append(row)
        ub_rhs.append(float(-config.expected_total_return_floor))
    for loss_vector, hard_limit in zip(context.scenarios.loss_matrix, context.scenarios.hard_loss_limits, strict=True):
        row = np.zeros(n_vars)
        row[w_slice] = loss_vector
        ub_rows.append(row)
        ub_rhs.append(float(hard_limit))
    bounds: list[tuple[float | None, float | None]] = [(float(lower), float(upper)) for lower, upper in zip(config.asset_lower, config.asset_upper, strict=True)]
    bounds.extend([(0.0, None)] * (2 * n))
    return {'objective': objective, 'A_eq': np.vstack(eq_rows), 'b_eq': np.asarray(eq_rhs, dtype=float), 'A_ub': np.vstack(ub_rows), 'b_ub': np.asarray(ub_rhs, dtype=float), 'bounds': bounds, 'w_slice': w_slice}

def solve_linear_anchor_highs(context: Step5Context, anchor_name: str, reward_vector: Any) -> np.ndarray:
    problem = build_fully_constrained_linear_program(context, reward_vector)
    result = linprog(c=problem['objective'], A_ub=problem['A_ub'], b_ub=problem['b_ub'], A_eq=problem['A_eq'], b_eq=problem['b_eq'], bounds=problem['bounds'], method='highs', options={'presolve': True, 'primal_feasibility_tolerance': 1e-09, 'dual_feasibility_tolerance': 1e-09})
    if not result.success:
        raise RuntimeError(f'{anchor_name} failed under HiGHS: status={result.status}; {result.message}')
    weights = np.asarray(result.x[problem['w_slice']], dtype=float)
    weights[np.abs(weights) < 1e-12] = 0.0
    audit = _exact_anchor_audit(context, weights)
    if not audit['satisfied'].all():
        failed = audit.loc[~audit['satisfied']]
        raise RuntimeError(f'{anchor_name} failed exact audit:\n{failed}')
    return weights

def solve_minimum_variance_anchor(context: Step5Context) -> np.ndarray:
    step4 = context.step4
    objective = step4.ObjectiveWeights(growth_reward=0.0, income_reward=0.0, risk_penalty=1.0, transaction_cost_penalty=0.0001, market_impact_penalty=0.0001, scenario_penalty=0.001, concentration_penalty=1e-07)
    warm_starts = [as_numpy(context.stages[-1].weights), as_numpy(context.current_weights)]
    solver_configs = [step4.SolverConfig(1e-11, 8000, False, 100.0, 5e-06), step4.SolverConfig(1e-10, 12000, False, 10.0, 5e-06), step4.SolverConfig(1e-09, 15000, False, 1.0, 5e-06)]
    attempts: list[dict[str, Any]] = []
    for config in solver_configs:
        for warm_start in warm_starts:
            result = step4.solve_portfolio(stage='minimum_feasible_variance', data=context.portfolio_data, current_weights=context.current_weights, objective_weights=objective, constraint_config=context.constraints, trading_config=context.trading_config, solver_config=config, scenarios=context.scenarios, include_return_reward=True, include_asset_caps=True, include_class_constraints=True, include_factor_constraints=context.portfolio_data.factor_loadings is not None and bool(context.constraints.factor_bounds), include_income_floor=True, include_return_floor=True, include_trading=True, include_scenarios=True, include_scenario_hard_limits=True, warm_start_weights=warm_start)
            attempts.append({'message': result.message, 'success': result.success, 'objective_scale': config.objective_scale})
            if result.success:
                weights = as_numpy(result.weights)
                audit = _exact_anchor_audit(context, weights)
                if audit['satisfied'].all():
                    return weights
    raise RuntimeError('Fully constrained minimum-variance anchor failed after stable retries:\n' + str(pd.DataFrame(attempts)))

def solve_fully_constrained_anchors(context: Step5Context) -> dict[str, np.ndarray]:
    patch_step4_exact_hinge_reporting(context.step4)
    return {'current_portfolio': as_numpy(context.current_weights), 'step4_scenario_aware': as_numpy(context.stages[-1].weights), 'maximum_feasible_growth': solve_linear_anchor_highs(context, 'maximum_feasible_growth', context.portfolio_data.growth), 'maximum_feasible_income': solve_linear_anchor_highs(context, 'maximum_feasible_income', context.portfolio_data.income), 'minimum_feasible_variance': solve_minimum_variance_anchor(context)}

def calibrate_goal_scales_from_feasible_set(context: Step5Context, anchor_weights: Mapping[str, Any]) -> tuple[GoalScales, pd.DataFrame]:
    records = []
    for name, weights in anchor_weights.items():
        records.append({'anchor': name, **exact_goal_components(context.portfolio_data, weights, context.current_weights, context.scenarios)})
    table = pd.DataFrame(records).set_index('anchor')
    scales = GoalScales(growth=robust_component_scale(table['growth'], 0.005), income=robust_component_scale(table['income'], 0.0025), variance=robust_component_scale(table['variance'], 1e-05), scenario_hinge=robust_component_scale(table['scenario_hinge'], 1e-06), linear_cost=robust_component_scale(table['linear_cost'], 1e-05), impact_cost=robust_component_scale(table['impact_cost'], 1e-06), turnover=robust_component_scale(table['gross_turnover'], 0.1), concentration=robust_component_scale(table['concentration'], 0.001))
    scales.validate()
    return (scales, table)

def clone_portfolio_data_with_linear_objective_cost(step4: Any, data: Any, objective_linear_cost: Any) -> Any:
    cloned = step4.PortfolioData(tickers=list(data.tickers), growth=np.asarray(data.growth, dtype=float).copy(), income=np.asarray(data.income, dtype=float).copy(), covariance=np.asarray(data.covariance, dtype=float).copy(), linear_cost=np.asarray(objective_linear_cost, dtype=float).copy(), impact_gamma=np.asarray(data.impact_matrix, dtype=float).copy(), adv_usd=np.asarray(data.adv_usd, dtype=float).copy(), asset_classes=list(data.asset_classes), descriptions=list(data.descriptions), factor_loadings=None if data.factor_loadings is None else data.factor_loadings.copy())
    cloned.validate()
    return cloned

def build_goal_objective(step4: Any, preferences: GoalPreferences, economic_data: Any, scales: GoalScales, mix: GoalMixConfig=GOAL_MIX) -> tuple[Any, Any, dict[str, float]]:
    preferences.validate()
    scales.validate()
    mix.validate()
    shares = preferences.shares
    growth_coefficient = shares['growth'] / scales.growth
    income_coefficient = shares['income'] / scales.income
    variance_coefficient = shares['drawdown'] * mix.variance_share_of_drawdown / scales.variance
    scenario_coefficient = (shares['drawdown'] * mix.scenario_share_of_drawdown + mix.scenario_tiebreaker) / scales.scenario_hinge
    linear_execution_coefficient = (shares['cost'] * mix.linear_cost_share + mix.execution_tiebreaker) / scales.linear_cost
    impact_coefficient = (shares['cost'] * mix.impact_cost_share + mix.execution_tiebreaker) / scales.impact_cost
    turnover_coefficient = (shares['cost'] * mix.turnover_share + mix.execution_tiebreaker) / scales.turnover
    objective_linear_cost = linear_execution_coefficient * economic_data.linear_cost + turnover_coefficient * np.ones(len(economic_data.tickers), dtype=float)
    objective_data = clone_portfolio_data_with_linear_objective_cost(step4, economic_data, objective_linear_cost)
    objective_weights = step4.ObjectiveWeights(growth_reward=growth_coefficient, income_reward=income_coefficient, risk_penalty=variance_coefficient, transaction_cost_penalty=1.0, market_impact_penalty=impact_coefficient, scenario_penalty=scenario_coefficient, concentration_penalty=mix.concentration_tiebreaker / scales.concentration)
    coefficients = {'growth_coefficient': growth_coefficient, 'income_coefficient': income_coefficient, 'variance_coefficient': variance_coefficient, 'scenario_coefficient': scenario_coefficient, 'linear_execution_coefficient': linear_execution_coefficient, 'impact_coefficient': impact_coefficient, 'turnover_coefficient': turnover_coefficient}
    return (objective_data, objective_weights, coefficients)

def solve_goal_profile(context: Step5Context, profile_name: str, preferences: GoalPreferences, scales: GoalScales, economic_data: Any | None=None, trading: Any | None=None, warm_start: Any | None=None, mix: GoalMixConfig=GOAL_MIX) -> dict[str, Any]:
    if economic_data is None:
        economic_data = context.portfolio_data
    if trading is None:
        trading = context.trading_config
    if warm_start is None:
        warm_start = as_numpy(context.stages[-1].weights)
    objective_data, objective_weights, coefficients = build_goal_objective(context.step4, preferences, economic_data, scales, mix)
    solver_config = context.step4.SolverConfig(ftol=1e-10, maxiter=5000, disp=False, objective_scale=1.0, feasibility_tolerance=5e-06)
    result = context.step4.solve_portfolio(stage=profile_name, data=objective_data, current_weights=context.current_weights, objective_weights=objective_weights, constraint_config=context.constraints, trading_config=trading, solver_config=solver_config, scenarios=context.scenarios, include_return_reward=True, include_asset_caps=True, include_class_constraints=True, include_factor_constraints=economic_data.factor_loadings is not None and bool(context.constraints.factor_bounds), include_income_floor=True, include_return_floor=True, include_trading=True, include_scenarios=True, include_scenario_hard_limits=True, warm_start_weights=as_numpy(warm_start))
    if not result.success or not result.constraint_audit['satisfied'].all():
        failed = result.constraint_audit.loc[~result.constraint_audit['satisfied']]
        raise RuntimeError(f'{profile_name} failed: {result.message}\nFailed constraints:\n{failed}')
    metrics = {**exact_goal_components(economic_data, result.weights, context.current_weights, context.scenarios), **portfolio_path_metrics(context.daily_returns, result.weights, economic_data.tickers)}
    return {'profile': profile_name, 'preferences': preferences, 'shares': preferences.shares, 'objective_weights': objective_weights, 'objective_coefficients': coefficients, 'result': result, 'metrics': metrics}

def run_presets(context: Step5Context, scales: GoalScales, presets: Mapping[str, GoalPreferences]=GOAL_PRESETS) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    warm_start = as_numpy(context.stages[-1].weights)
    for name, preferences in presets.items():
        solved = solve_goal_profile(context, name, preferences, scales, warm_start=warm_start)
        results[name] = solved
        warm_start = as_numpy(solved['result'].weights)
    return results

def run_one_way_sensitivity(context: Step5Context, scales: GoalScales, levels: Iterable[int]=(0, 25, 50, 75, 100)) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    goal_names = ['growth', 'income', 'drawdown_control', 'cost_sensitivity']
    for varied_goal in goal_names:
        warm_start = as_numpy(context.stages[-1].weights)
        for level in levels:
            scores = {'growth': 50.0, 'income': 50.0, 'drawdown_control': 50.0, 'cost_sensitivity': 50.0}
            scores[varied_goal] = float(level)
            preferences = GoalPreferences(**scores)
            solved = solve_goal_profile(context, f'{varied_goal}_{level}', preferences, scales, warm_start=warm_start)
            warm_start = as_numpy(solved['result'].weights)
            records.append({'varied_goal': varied_goal, 'score': level, **solved['shares'], **solved['metrics']})
    return pd.DataFrame(records)

def sensitivity_directional_checks(table: pd.DataFrame) -> pd.Series:

    def endpoints(goal: str, metric: str) -> tuple[float, float]:
        subset = table.loc[table['varied_goal'] == goal].sort_values('score')
        return (float(subset.iloc[0][metric]), float(subset.iloc[-1][metric]))
    growth_low, growth_high = endpoints('growth', 'growth')
    income_low, income_high = endpoints('income', 'income')
    stress_low, stress_high = endpoints('drawdown_control', 'worst_scenario_loss')
    mdd_low, mdd_high = endpoints('drawdown_control', 'realized_maximum_drawdown')
    cost_low, cost_high = endpoints('cost_sensitivity', 'total_trading_cost')
    turn_low, turn_high = endpoints('cost_sensitivity', 'gross_turnover')
    return pd.Series({'growth_score_increases_growth': growth_high >= growth_low - 1e-07, 'income_score_increases_income': income_high >= income_low - 1e-07, 'drawdown_score_reduces_stress_loss': stress_high <= stress_low + 1e-07, 'drawdown_score_reduces_endpoint_realized_mdd': mdd_high <= mdd_low + 1e-07, 'cost_score_strictly_reduces_trading_cost': cost_high < cost_low - 1e-07, 'cost_score_does_not_increase_turnover': turn_high <= turn_low + 1e-07, 'cost_score_strictly_reduces_turnover': turn_high < turn_low - 1e-05}, name='passed')

def evaluate_trading_cost_under_data(data: Any, weights: Any, incumbent_weights: Any) -> dict[str, float]:
    w = as_numpy(weights)
    w0 = as_numpy(incumbent_weights)
    delta = w - w0
    linear = float(data.linear_cost @ np.abs(delta))
    impact = float(delta @ data.impact_matrix @ delta)
    return {'linear_cost': linear, 'impact_cost': impact, 'total_trading_cost': linear + impact, 'gross_turnover': float(np.abs(delta).sum())}

def portfolio_data_for_cost_case(step4: Any, base_data: Any, source_dir: Any, prefix: str, cost_case: str) -> Any:
    from pathlib import Path
    source_dir = Path(source_dir)
    if cost_case == 'base':
        candidates = [source_dir / f'{prefix}_cost_estimates_base.csv', source_dir / f'{prefix}_cost_estimates.csv']
    elif cost_case == 'institutional_high_participation':
        candidates = [source_dir / f'{prefix}_cost_estimates_institutional_high_participation.csv']
    else:
        raise ValueError(f'Unknown cost case: {cost_case}')
    cost_path = next((path for path in candidates if path.exists()), None)
    if cost_path is None:
        raise FileNotFoundError(f'No cost table found for {cost_case}.')
    table = pd.read_csv(cost_path, index_col=0).reindex(base_data.tickers)
    if table.isna().any().any():
        raise ValueError(f'Missing values in {cost_path}.')
    if 'linear_cost_fraction' in table.columns:
        linear_cost = table['linear_cost_fraction'].to_numpy(dtype=float)
    else:
        linear_cost = table['linear_cost_bps'].to_numpy(dtype=float) / 10000.0
    new_data = step4.PortfolioData(tickers=list(base_data.tickers), growth=np.asarray(base_data.growth, dtype=float).copy(), income=np.asarray(base_data.income, dtype=float).copy(), covariance=np.asarray(base_data.covariance, dtype=float).copy(), linear_cost=linear_cost, impact_gamma=table['quadratic_impact_gamma'].to_numpy(dtype=float), adv_usd=table['adv_usd'].to_numpy(dtype=float), asset_classes=list(base_data.asset_classes), descriptions=list(base_data.descriptions), factor_loadings=None if base_data.factor_loadings is None else base_data.factor_loadings.copy())
    new_data.validate()
    return new_data
