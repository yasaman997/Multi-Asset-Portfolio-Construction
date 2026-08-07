from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence
import numpy as np
import pandas as pd
EPS = 1e-12

def _array(values: Any) -> np.ndarray:
    if hasattr(values, 'to_numpy'):
        return np.asarray(values.to_numpy(dtype=float), dtype=float)
    return np.asarray(values, dtype=float)

@dataclass(frozen=True)
class Step6Context:
    step4: Any
    portfolio_data: Any
    current_weights: Any
    scenarios: Any
    constraints: Any
    trading_config: Any
    daily_returns: pd.DataFrame
    hard_tolerance: float = 5e-06
    binding_tolerance: float = 1e-05
    duplicate_tolerance: float = 1e-08
    material_weight_threshold: float = 0.005
    material_trade_threshold: float = 0.001

    def validate(self) -> None:
        tickers = list(self.portfolio_data.tickers)
        n_assets = len(tickers)
        current = _array(self.current_weights)
        if current.shape != (n_assets,):
            raise ValueError('current_weights has the wrong shape.')
        if not np.isclose(current.sum(), 1.0, atol=1e-08):
            raise ValueError('current_weights must sum to one.')
        if list(self.daily_returns.columns) != tickers:
            raise ValueError('daily_returns columns must exactly match the portfolio ticker order.')
        if self.daily_returns.empty:
            raise ValueError('daily_returns must not be empty.')
        if not np.isfinite(self.daily_returns.to_numpy(dtype=float)).all():
            raise ValueError('daily_returns contains non-finite values.')
        self.portfolio_data.validate()
        self.constraints.validate(self.portfolio_data)
        self.scenarios.validate(n_assets)

def register_candidate(registry: MutableMapping[str, dict[str, Any]], *, context: Step6Context, label: str, family: str, role: str, method: str, weights: Any, source_name: str, decision_eligible: bool, independent_selection: bool, method_traceability: float, metadata: Mapping[str, Any] | None=None) -> None:
    if label in registry:
        raise ValueError(f'Candidate already registered: {label}')
    vector = _array(weights).reshape(-1)
    n_assets = len(context.portfolio_data.tickers)
    if vector.shape != (n_assets,):
        raise ValueError(f'{label}: weights have the wrong shape.')
    if not np.isfinite(vector).all():
        raise ValueError(f'{label}: weights contain non-finite values.')
    if abs(float(vector.sum()) - 1.0) > 1e-05:
        raise ValueError(f'{label}: weights do not sum to one.')
    if float(vector.min()) < -1e-06:
        raise ValueError(f'{label}: negative weight detected.')
    if not 0.0 <= float(method_traceability) <= 1.0:
        raise ValueError('method_traceability must lie in [0, 1].')
    registry[label] = {'label': label, 'family': family, 'role': role, 'method': method, 'weights': vector.copy(), 'source_name': source_name, 'decision_eligible': bool(decision_eligible), 'independent_selection': bool(independent_selection), 'method_traceability': float(method_traceability), 'metadata': dict(metadata or {})}

def _path_metrics(daily_returns: pd.DataFrame, weights: np.ndarray) -> dict[str, float]:
    path = daily_returns.to_numpy(dtype=float) @ weights
    path = np.maximum(path, -0.999999)
    wealth = np.cumprod(1.0 + path)
    running_peak = np.maximum.accumulate(wealth)
    drawdown = 1.0 - wealth / np.maximum(running_peak, EPS)
    n_days = len(path)
    annualized_return = float(wealth[-1] ** (252.0 / max(n_days, 1)) - 1.0)
    annualized_volatility = float(np.std(path, ddof=1) * np.sqrt(252.0))
    quantile_05 = float(np.quantile(path, 0.05))
    tail = path[path <= quantile_05 + EPS]
    daily_var_95 = max(-quantile_05, 0.0)
    daily_cvar_95 = max(-float(tail.mean()), 0.0) if len(tail) else daily_var_95
    return {'in_sample_annualized_return': annualized_return, 'in_sample_annualized_volatility': annualized_volatility, 'in_sample_maximum_drawdown': float(drawdown.max()), 'daily_var_95': daily_var_95, 'daily_cvar_95': daily_cvar_95}

def _policy_scale(value: float, lower: float, upper: float) -> float:
    finite = [abs(float(x)) for x in (value, lower, upper) if np.isfinite(x)]
    return max(finite + [0.01])

def _decorate_audit(audit: pd.DataFrame, *, tolerance: float, binding_tolerance: float) -> pd.DataFrame:
    frame = audit.copy()
    lower = frame['lower'].to_numpy(dtype=float)
    upper = frame['upper'].to_numpy(dtype=float)
    value = frame['value'].to_numpy(dtype=float)
    lower_violation = np.where(np.isfinite(lower), np.maximum(lower - value, 0.0), 0.0)
    upper_violation = np.where(np.isfinite(upper), np.maximum(value - upper, 0.0), 0.0)
    absolute_violation = np.maximum(lower_violation, upper_violation)
    scales = np.asarray([_policy_scale(v, lo, hi) for v, lo, hi in zip(value, lower, upper, strict=True)], dtype=float)
    frame['lower_violation'] = lower_violation
    frame['upper_violation'] = upper_violation
    frame['absolute_violation'] = absolute_violation
    frame['normalized_violation'] = absolute_violation / scales
    frame['satisfied'] = (lower_violation <= tolerance) & (upper_violation <= tolerance)
    zero_lower_bound = np.isclose(frame['lower'], 0.0)
    zero_realized_exposure = frame['value'].abs() <= binding_tolerance
    supported_nonnegativity_row = frame['category'].eq('asset') & frame['constraint'].str.startswith('weight_') | frame['category'].eq('asset_class') & frame['constraint'].str.startswith('class_')
    trivial_lower_zero = zero_lower_bound & zero_realized_exposure & supported_nonnegativity_row
    excluded_policy_rows = frame['category'].eq('budget') | frame['constraint'].eq('minimum_weight') | frame['constraint'].eq('trade_accounting_max_abs_error')
    frame['is_policy_guardrail'] = ~excluded_policy_rows
    frame['trivial_nonnegativity_bound'] = trivial_lower_zero
    lower_margin = np.where(np.isfinite(lower), (value - lower) / scales, np.inf)
    upper_margin = np.where(np.isfinite(upper), (upper - value) / scales, np.inf)
    lower_margin = np.where(trivial_lower_zero, np.inf, lower_margin)
    normalized_margin = np.minimum(lower_margin, upper_margin)
    normalized_margin = np.where(trivial_lower_zero, np.nan, normalized_margin)
    normalized_margin = np.where(frame['is_policy_guardrail'], normalized_margin, np.nan)
    frame['policy_normalized_margin'] = normalized_margin
    lower_active = np.isfinite(lower) & (np.abs(value - lower) <= binding_tolerance)
    upper_active = np.isfinite(upper) & (np.abs(upper - value) <= binding_tolerance)
    frame['mathematical_active_bound'] = lower_active | upper_active
    frame['binding_policy_guardrail'] = frame['is_policy_guardrail'] & frame['satisfied'] & ~frame['trivial_nonnegativity_bound'] & (frame['policy_normalized_margin'] <= binding_tolerance)
    return frame

def _scenario_audit(*, scenarios: Any, weights: np.ndarray, tolerance: float) -> pd.DataFrame:
    losses = _array(scenarios.loss_matrix) @ weights
    warnings = _array(scenarios.warning_thresholds)
    if hasattr(scenarios, 'hard_loss_limits'):
        hard_limits = _array(scenarios.hard_loss_limits)
    elif hasattr(scenarios, 'hard_limits'):
        hard_limits = _array(scenarios.hard_limits)
    else:
        raise AttributeError('Scenario set has no hard-limit field.')
    return pd.DataFrame({'scenario_loss': losses, 'warning_threshold': warnings, 'hard_limit': hard_limits, 'warning_headroom': warnings - losses, 'warning_excess': np.maximum(losses - warnings, 0.0), 'warning_satisfied': losses <= warnings + tolerance, 'hard_limit_headroom': hard_limits - losses, 'hard_limit_excess': np.maximum(losses - hard_limits, 0.0), 'hard_limit_satisfied': losses <= hard_limits + tolerance}, index=list(scenarios.names))

def _attribution_tables(*, context: Step6Context, weights: np.ndarray) -> dict[str, pd.DataFrame]:
    data = context.portfolio_data
    tickers = list(data.tickers)
    current = _array(context.current_weights)
    trade = weights - current
    covariance = _array(data.covariance)
    marginal_variance = covariance @ weights
    variance = float(weights @ marginal_variance)
    volatility = float(np.sqrt(max(variance, 0.0)))
    growth_contribution = weights * _array(data.growth)
    income_contribution = weights * _array(data.income)
    total_return_contribution = growth_contribution + income_contribution
    variance_contribution = weights * marginal_variance
    if volatility > EPS:
        volatility_contribution = variance_contribution / volatility
    else:
        volatility_contribution = np.zeros_like(weights)
    linear_cost_contribution = np.abs(trade) * _array(data.linear_cost)
    impact_vector = _array(data.impact_matrix) @ trade
    impact_cost_contribution = trade * impact_vector
    turnover_contribution = np.abs(trade)
    scenario_losses = _array(context.scenarios.loss_matrix) @ weights
    worst_index = int(np.argmax(scenario_losses))
    worst_scenario_vector = _array(context.scenarios.loss_matrix)[worst_index]
    worst_scenario_contribution = weights * worst_scenario_vector
    asset = pd.DataFrame({'asset_class': list(data.asset_classes), 'description': list(data.descriptions), 'weight': weights, 'current_weight': current, 'trade': trade, 'absolute_trade': np.abs(trade), 'growth_contribution': growth_contribution, 'income_contribution': income_contribution, 'expected_return_contribution': total_return_contribution, 'variance_contribution': variance_contribution, 'volatility_contribution': volatility_contribution, 'turnover_contribution': turnover_contribution, 'linear_cost_contribution': linear_cost_contribution, 'impact_cost_contribution': impact_cost_contribution, 'total_cost_contribution': linear_cost_contribution + impact_cost_contribution, 'worst_scenario_loss_contribution': worst_scenario_contribution}, index=tickers)
    numeric_columns = ['weight', 'current_weight', 'trade', 'absolute_trade', 'growth_contribution', 'income_contribution', 'expected_return_contribution', 'variance_contribution', 'volatility_contribution', 'turnover_contribution', 'linear_cost_contribution', 'impact_cost_contribution', 'total_cost_contribution', 'worst_scenario_loss_contribution']
    by_class = asset.groupby('asset_class')[numeric_columns].sum().sort_index()
    scenario_detail = pd.DataFrame(_array(context.scenarios.loss_matrix) * weights[None, :], index=list(context.scenarios.names), columns=tickers)
    return {'asset': asset, 'asset_class': by_class, 'scenario_asset': scenario_detail, 'worst_scenario_name': pd.DataFrame({'worst_scenario_name': [context.scenarios.names[worst_index]]})}

def _absolute_top_coverage(values: np.ndarray, count: int=5) -> float:
    absolute = np.abs(np.asarray(values, dtype=float))
    denominator = float(absolute.sum())
    if denominator <= EPS:
        return 1.0
    return float(np.sort(absolute)[::-1][:count].sum() / denominator)

def _evaluate_candidate(*, context: Step6Context, candidate: Mapping[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    data = context.portfolio_data
    weights = _array(candidate['weights'])
    current = _array(context.current_weights)
    trade = weights - current
    buys = np.maximum(trade, 0.0)
    sells = np.maximum(-trade, 0.0)
    growth = float(_array(data.growth) @ weights)
    income = float(_array(data.income) @ weights)
    expected_return = growth + income
    variance = float(weights @ _array(data.covariance) @ weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    gross_turnover = float(np.abs(trade).sum())
    linear_cost = float(_array(data.linear_cost) @ np.abs(trade))
    impact_cost = float(trade @ _array(data.impact_matrix) @ trade)
    total_cost = linear_cost + impact_cost
    concentration = float(weights @ weights)
    effective_holdings = 1.0 / concentration if concentration > EPS else np.inf
    scenario = _scenario_audit(scenarios=context.scenarios, weights=weights, tolerance=context.hard_tolerance)
    raw_audit = context.step4.audit_constraints(data=data, weights=weights, current_weights=current, buys=buys, sells=sells, scenarios=context.scenarios, constraint_config=context.constraints, trading_config=context.trading_config, check_asset_caps=True, check_classes=True, check_factors=True, check_income=True, check_return=True, check_trading=True, check_scenario_hard=True, tolerance=context.hard_tolerance)
    audit = _decorate_audit(raw_audit, tolerance=context.hard_tolerance, binding_tolerance=context.binding_tolerance)
    attributions = _attribution_tables(context=context, weights=weights)
    asset_attr = attributions['asset']
    policy_rows = audit.loc[audit['is_policy_guardrail']]
    policy_margins = policy_rows['policy_normalized_margin'].replace([np.inf, -np.inf], np.nan).dropna()
    if policy_margins.empty:
        headroom_min = np.nan
        headroom_10 = np.nan
        headroom_median = np.nan
    else:
        headroom_min = float(policy_margins.min())
        headroom_10 = float(policy_margins.quantile(0.1))
        headroom_median = float(policy_margins.median())
    hard_breaches = audit.loc[~audit['satisfied']]
    warning_breaches = scenario.loc[~scenario['warning_satisfied']]
    path_metrics = _path_metrics(context.daily_returns, weights)
    top5_weight_share = float(np.sort(weights)[::-1][:5].sum())
    top10_weight_share = float(np.sort(weights)[::-1][:10].sum())
    material_holdings_count = int(np.count_nonzero(weights >= context.material_weight_threshold))
    active_trade_count = int(np.count_nonzero(np.abs(trade) >= context.material_trade_threshold))
    row: dict[str, Any] = {'candidate': candidate['label'], 'family': candidate['family'], 'role': candidate['role'], 'method': candidate['method'], 'source_name': candidate['source_name'], 'decision_eligible_declared': candidate['decision_eligible'], 'independent_selection': candidate['independent_selection'], 'method_traceability': candidate['method_traceability'], 'expected_growth': growth, 'income_yield': income, 'expected_total_return': expected_return, 'variance': variance, 'volatility': volatility, 'return_to_volatility': expected_return / volatility if volatility > EPS else np.nan, 'worst_scenario_loss': float(scenario['scenario_loss'].max()), 'weighted_scenario_loss': float(_array(context.scenarios.weights) @ scenario['scenario_loss'].to_numpy(dtype=float)), 'gross_turnover': gross_turnover, 'one_way_turnover': 0.5 * gross_turnover, 'linear_transaction_cost': linear_cost, 'quadratic_impact_cost': impact_cost, 'total_trading_cost': total_cost, 'concentration_hhi': concentration, 'effective_holdings': effective_holdings, 'maximum_asset_weight': float(weights.max()), 'nonzero_holdings': int(np.count_nonzero(weights > 1e-08)), 'material_holdings_count': material_holdings_count, 'active_trade_count': active_trade_count, 'top5_weight_share': top5_weight_share, 'top10_weight_share': top10_weight_share, 'hard_guardrail_status': 'PASS' if hard_breaches.empty else 'BREACH', 'hard_breach_count': int(len(hard_breaches)), 'maximum_normalized_hard_violation': float(hard_breaches['normalized_violation'].max()) if not hard_breaches.empty else 0.0, 'binding_guardrail_count': int(audit['binding_policy_guardrail'].sum()), 'mathematical_active_bound_count': int(audit['mathematical_active_bound'].sum()), 'trivial_nonnegativity_bound_count': int(audit['trivial_nonnegativity_bound'].sum()), 'guardrail_headroom_min': headroom_min, 'guardrail_headroom_10pct': headroom_10, 'guardrail_headroom_median': headroom_median, 'warning_breach_count': int(len(warning_breaches)), 'maximum_warning_excess': float(warning_breaches['warning_excess'].max()) if not warning_breaches.empty else 0.0, 'total_warning_excess': float(scenario['warning_excess'].sum()), 'minimum_warning_headroom': float(scenario['warning_headroom'].min()), 'minimum_hard_limit_headroom': float(scenario['hard_limit_headroom'].min()), 'expected_return_top5_abs_coverage': _absolute_top_coverage(asset_attr['expected_return_contribution'].to_numpy(dtype=float)), 'volatility_top5_abs_coverage': _absolute_top_coverage(asset_attr['volatility_contribution'].to_numpy(dtype=float)), 'turnover_top5_coverage': _absolute_top_coverage(asset_attr['turnover_contribution'].to_numpy(dtype=float)), 'worst_scenario_top5_abs_coverage': _absolute_top_coverage(asset_attr['worst_scenario_loss_contribution'].to_numpy(dtype=float)), **path_metrics}
    reconciliation_errors = {'expected_return': abs(float(asset_attr['expected_return_contribution'].sum()) - expected_return), 'volatility': abs(float(asset_attr['volatility_contribution'].sum()) - volatility), 'turnover': abs(float(asset_attr['turnover_contribution'].sum()) - gross_turnover), 'linear_cost': abs(float(asset_attr['linear_cost_contribution'].sum()) - linear_cost), 'impact_cost': abs(float(asset_attr['impact_cost_contribution'].sum()) - impact_cost), 'worst_scenario': abs(float(asset_attr['worst_scenario_loss_contribution'].sum()) - row['worst_scenario_loss'])}
    row['maximum_attribution_reconciliation_error'] = max(reconciliation_errors.values())
    return (row, audit, scenario, attributions)

def compare_candidates(*, context: Step6Context, candidates: Mapping[str, Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, dict[str, pd.DataFrame]]]:
    context.validate()
    if not candidates:
        raise ValueError('At least one candidate is required.')
    rows: list[dict[str, Any]] = []
    weight_columns: dict[str, np.ndarray] = {}
    audits: dict[str, pd.DataFrame] = {}
    warnings: dict[str, pd.DataFrame] = {}
    attributions: dict[str, dict[str, pd.DataFrame]] = {}
    for label, candidate in candidates.items():
        row, audit, warning, attribution = _evaluate_candidate(context=context, candidate=candidate)
        rows.append(row)
        weight_columns[label] = _array(candidate['weights'])
        audits[label] = audit
        warnings[label] = warning
        attributions[label] = attribution
    comparison = pd.DataFrame(rows).set_index('candidate')
    weights = pd.DataFrame(weight_columns, index=list(context.portfolio_data.tickers))
    return (comparison, weights, audits, warnings, attributions)

def mark_duplicate_portfolios(*, comparison: pd.DataFrame, weights: pd.DataFrame, priority: Sequence[str] | None=None, tolerance: float=1e-08) -> pd.DataFrame:
    result = comparison.copy()
    labels = list(result.index)
    if priority is None:
        ordered = labels
    else:
        priority_order = {label: rank for rank, label in enumerate(priority)}
        ordered = sorted(labels, key=lambda label: (priority_order.get(label, len(priority_order)), labels.index(label)))
    canonical: list[str] = []
    duplicate_of: dict[str, str | None] = {}
    for label in ordered:
        vector = weights[label].to_numpy(dtype=float)
        match = None
        for existing in canonical:
            if np.max(np.abs(vector - weights[existing].to_numpy(dtype=float))) <= tolerance:
                match = existing
                break
        if match is None:
            canonical.append(label)
            duplicate_of[label] = None
        else:
            duplicate_of[label] = match
    result['duplicate_of'] = pd.Series(duplicate_of).reindex(result.index)
    result['is_unique_portfolio'] = result['duplicate_of'].isna()
    result['policy_compliant'] = result['hard_breach_count'].eq(0)
    result['eligible_for_selection'] = result['decision_eligible_declared'] & result['policy_compliant'] & result['is_unique_portfolio']
    return result

def _minmax_higher(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum <= EPS:
        return pd.Series(0.5, index=values.index, dtype=float)
    return (values - minimum) / (maximum - minimum)

def _minmax_lower(series: pd.Series) -> pd.Series:
    return 1.0 - _minmax_higher(series)
DEFAULT_DECISION_SCENARIOS: dict[str, dict[str, float]] = {'Balanced': {'expected_return_score': 0.25, 'risk_control_score': 0.25, 'implementation_score': 0.2, 'guardrail_resilience_score': 0.2, 'explainability_score': 0.1}, 'Return First': {'expected_return_score': 0.45, 'risk_control_score': 0.2, 'implementation_score': 0.1, 'guardrail_resilience_score': 0.15, 'explainability_score': 0.1}, 'Risk First': {'expected_return_score': 0.15, 'risk_control_score': 0.45, 'implementation_score': 0.1, 'guardrail_resilience_score': 0.2, 'explainability_score': 0.1}, 'Implementation First': {'expected_return_score': 0.15, 'risk_control_score': 0.15, 'implementation_score': 0.45, 'guardrail_resilience_score': 0.15, 'explainability_score': 0.1}, 'Governance First': {'expected_return_score': 0.15, 'risk_control_score': 0.2, 'implementation_score': 0.1, 'guardrail_resilience_score': 0.45, 'explainability_score': 0.1}, 'Explainability First': {'expected_return_score': 0.15, 'risk_control_score': 0.15, 'implementation_score': 0.15, 'guardrail_resilience_score': 0.15, 'explainability_score': 0.4}}

def rank_candidates(*, comparison: pd.DataFrame, decision_scenarios: Mapping[str, Mapping[str, float]] | None=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = dict(decision_scenarios or DEFAULT_DECISION_SCENARIOS)
    ranking = comparison.copy()
    eligible = ranking.loc[ranking['eligible_for_selection']].copy()
    if eligible.empty:
        raise RuntimeError('No unique, hard-compliant decision candidate is available.')
    eligible['expected_return_score'] = _minmax_higher(eligible['expected_total_return'])
    risk_components = pd.DataFrame({'volatility': _minmax_lower(eligible['volatility']), 'scenario': _minmax_lower(eligible['worst_scenario_loss']), 'drawdown': _minmax_lower(eligible['in_sample_maximum_drawdown']), 'cvar': _minmax_lower(eligible['daily_cvar_95'])})
    eligible['risk_control_score'] = 0.35 * risk_components['volatility'] + 0.35 * risk_components['scenario'] + 0.15 * risk_components['drawdown'] + 0.15 * risk_components['cvar']
    implementation_components = pd.DataFrame({'turnover': _minmax_lower(eligible['gross_turnover']), 'cost': _minmax_lower(eligible['total_trading_cost']), 'trades': _minmax_lower(eligible['active_trade_count'])})
    eligible['implementation_score'] = 0.55 * implementation_components['turnover'] + 0.35 * implementation_components['cost'] + 0.1 * implementation_components['trades']
    guardrail_components = pd.DataFrame({'policy_headroom': _minmax_higher(eligible['guardrail_headroom_10pct'].fillna(0.0)), 'hard_scenario_headroom': _minmax_higher(eligible['minimum_hard_limit_headroom']), 'warning_excess': _minmax_lower(eligible['total_warning_excess']), 'bindings': _minmax_lower(eligible['binding_guardrail_count'])})
    eligible['guardrail_resilience_score'] = 0.35 * guardrail_components['policy_headroom'] + 0.3 * guardrail_components['hard_scenario_headroom'] + 0.25 * guardrail_components['warning_excess'] + 0.1 * guardrail_components['bindings']
    explainability_components = pd.DataFrame({'method': eligible['method_traceability'], 'holdings': _minmax_lower(eligible['material_holdings_count']), 'trades': _minmax_lower(eligible['active_trade_count']), 'coverage': eligible[['expected_return_top5_abs_coverage', 'volatility_top5_abs_coverage', 'turnover_top5_coverage', 'worst_scenario_top5_abs_coverage']].mean(axis=1)})
    eligible['explainability_score'] = 0.35 * explainability_components['method'] + 0.2 * explainability_components['holdings'] + 0.2 * explainability_components['trades'] + 0.25 * explainability_components['coverage']
    score_columns = ['expected_return_score', 'risk_control_score', 'implementation_score', 'guardrail_resilience_score', 'explainability_score']
    scenario_score_columns: dict[str, str] = {}
    scenario_rank_columns: dict[str, str] = {}
    for scenario_name, weights in scenarios.items():
        missing = set(score_columns) - set(weights)
        if missing:
            raise ValueError(f'{scenario_name}: missing decision weights for {sorted(missing)}')
        total = float(sum((float(weights[key]) for key in score_columns)))
        if abs(total - 1.0) > 1e-10:
            raise ValueError(f'{scenario_name}: weights must sum to one.')
        score_column = 'decision_score__' + scenario_name.lower().replace(' ', '_')
        rank_column = 'decision_rank__' + scenario_name.lower().replace(' ', '_')
        eligible[score_column] = sum((float(weights[key]) * eligible[key] for key in score_columns))
        eligible[rank_column] = eligible[score_column].rank(ascending=False, method='min')
        scenario_score_columns[scenario_name] = score_column
        scenario_rank_columns[scenario_name] = rank_column
    score_matrix = eligible[list(scenario_score_columns.values())]
    rank_matrix = eligible[list(scenario_rank_columns.values())]
    eligible['robust_mean_score'] = score_matrix.mean(axis=1)
    eligible['robust_mean_rank'] = rank_matrix.mean(axis=1)
    eligible['rank_best'] = rank_matrix.min(axis=1)
    eligible['rank_worst'] = rank_matrix.max(axis=1)
    eligible['top_1_frequency'] = rank_matrix.eq(1.0).mean(axis=1)
    eligible['top_3_frequency'] = rank_matrix.le(3.0).mean(axis=1)
    eligible['base_decision_score'] = eligible[scenario_score_columns['Balanced']]
    eligible['base_rank'] = eligible[scenario_rank_columns['Balanced']]
    eligible['selection_rank'] = eligible['robust_mean_score'].rank(ascending=False, method='min')
    ranking_columns = score_columns + list(scenario_score_columns.values()) + list(scenario_rank_columns.values()) + ['robust_mean_score', 'robust_mean_rank', 'rank_best', 'rank_worst', 'top_1_frequency', 'top_3_frequency', 'base_decision_score', 'base_rank', 'selection_rank']
    for column in ranking_columns:
        ranking[column] = np.nan
    ranking.loc[eligible.index, ranking_columns] = eligible[ranking_columns]
    scenario_table = pd.DataFrame({scenario_name: eligible[score_column] for scenario_name, score_column in scenario_score_columns.items()})
    return (ranking, scenario_table)

def _pareto_mask(data: pd.DataFrame, *, maximize: Sequence[str], minimize: Sequence[str]) -> pd.Series:
    labels = list(data.index)
    mask = pd.Series(True, index=labels, dtype=bool)
    for label in labels:
        row = data.loc[label]
        for other_label in labels:
            if other_label == label:
                continue
            other = data.loc[other_label]
            weakly_better = True
            strictly_better = False
            for column in maximize:
                if other[column] < row[column] - EPS:
                    weakly_better = False
                    break
                if other[column] > row[column] + EPS:
                    strictly_better = True
            if not weakly_better:
                continue
            for column in minimize:
                if other[column] > row[column] + EPS:
                    weakly_better = False
                    break
                if other[column] < row[column] - EPS:
                    strictly_better = True
            if weakly_better and strictly_better:
                mask.loc[label] = False
                break
    return mask

def add_pareto_flags(comparison: pd.DataFrame) -> pd.DataFrame:
    result = comparison.copy()
    eligible = result.loc[result['eligible_for_selection']]
    for column in ['pareto_risk_return', 'pareto_return_implementation', 'pareto_governance', 'pareto_comprehensive']:
        result[column] = False
    if eligible.empty:
        return result
    result.loc[eligible.index, 'pareto_risk_return'] = _pareto_mask(eligible, maximize=['expected_total_return'], minimize=['volatility', 'worst_scenario_loss'])
    result.loc[eligible.index, 'pareto_return_implementation'] = _pareto_mask(eligible, maximize=['expected_total_return'], minimize=['gross_turnover', 'total_trading_cost'])
    result.loc[eligible.index, 'pareto_governance'] = _pareto_mask(eligible, maximize=['minimum_hard_limit_headroom', 'guardrail_headroom_10pct'], minimize=['warning_breach_count', 'total_warning_excess'])
    result.loc[eligible.index, 'pareto_comprehensive'] = _pareto_mask(eligible, maximize=['expected_total_return'], minimize=['volatility', 'worst_scenario_loss', 'gross_turnover', 'total_warning_excess'])
    return result

def build_explainability_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    columns = ['family', 'role', 'method', 'method_traceability', 'nonzero_holdings', 'material_holdings_count', 'active_trade_count', 'top5_weight_share', 'top10_weight_share', 'expected_return_top5_abs_coverage', 'volatility_top5_abs_coverage', 'turnover_top5_coverage', 'worst_scenario_top5_abs_coverage', 'binding_guardrail_count', 'maximum_attribution_reconciliation_error']
    return comparison[columns].copy()

def build_narratives(comparison: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for label, row in comparison.iterrows():
        if row['hard_breach_count'] == 0:
            governance = 'All hard guardrails pass'
        else:
            governance = f"{int(row['hard_breach_count'])} hard guardrail breach(es)"
        if row['warning_breach_count'] == 0:
            governance += '; all scenario warnings pass.'
        else:
            governance += f"; {int(row['warning_breach_count'])} soft scenario warning breach(es)."
        implementation = f"Gross turnover {row['gross_turnover']:.2%}, estimated trading cost {row['total_trading_cost']:.4%}, {int(row['active_trade_count'])} material trade(s)."
        risk = f"Expected volatility {row['volatility']:.2%}, worst modeled scenario loss {row['worst_scenario_loss']:.2%}, in-sample maximum drawdown {row['in_sample_maximum_drawdown']:.2%}."
        explainability = f"{row['method']} Method traceability {row['method_traceability']:.0%}; {int(row['material_holdings_count'])} material holdings; top-five return attribution coverage {row['expected_return_top5_abs_coverage']:.1%}."
        records.append({'candidate': label, 'return_summary': f"Model-implied expected total return {row['expected_total_return']:.2%} ({row['expected_growth']:.2%} growth + {row['income_yield']:.2%} income).", 'risk_summary': risk, 'implementation_summary': implementation, 'governance_summary': governance, 'explainability_summary': explainability})
    return pd.DataFrame(records).set_index('candidate')

def concatenate_audits(audits: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for candidate, frame in audits.items():
        copy = frame.copy()
        copy.insert(0, 'candidate', candidate)
        frames.append(copy)
    return pd.concat(frames, ignore_index=True)

def concatenate_warning_audits(warnings: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for candidate, frame in warnings.items():
        copy = frame.copy()
        copy.insert(0, 'scenario', copy.index)
        copy.insert(0, 'candidate', candidate)
        frames.append(copy.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)
