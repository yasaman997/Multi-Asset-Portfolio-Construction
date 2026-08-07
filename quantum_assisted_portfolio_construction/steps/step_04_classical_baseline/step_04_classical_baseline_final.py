from __future__ import annotations
import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, linprog, minimize
TOL = 1e-08

@dataclass(frozen=True)
class ObjectiveWeights:
    growth_reward: float = 1.0
    income_reward: float = 1.0
    risk_penalty: float = 3.0
    transaction_cost_penalty: float = 1.0
    market_impact_penalty: float = 1.0
    scenario_penalty: float = 25.0
    concentration_penalty: float = 0.05

@dataclass(frozen=True)
class TradingConfig:
    portfolio_value_usd: float = 10000000.0
    turnover_limit_gross: float = 0.5
    execution_days: float = 5.0
    participation_rate: float = 0.1

@dataclass(frozen=True)
class SolverConfig:
    ftol: float = 1e-10
    maxiter: int = 3000
    disp: bool = False
    objective_scale: float = 100.0
    feasibility_tolerance: float = 5e-06

@dataclass
class PortfolioData:
    tickers: list[str]
    growth: np.ndarray
    income: np.ndarray
    covariance: np.ndarray
    linear_cost: np.ndarray
    impact_gamma: np.ndarray
    adv_usd: np.ndarray
    asset_classes: list[str]
    descriptions: list[str]
    factor_loadings: pd.DataFrame | None = None

    def validate(self) -> None:
        n = len(self.tickers)
        if len(set(self.tickers)) != n:
            raise ValueError('Ticker labels must be unique.')
        one_dimensional = {'growth': self.growth, 'income': self.income, 'linear_cost': self.linear_cost, 'adv_usd': self.adv_usd}
        for name, array in one_dimensional.items():
            array = np.asarray(array, dtype=float)
            if array.shape != (n,):
                raise ValueError(f'{name} must have shape {(n,)}, got {array.shape}.')
            if not np.isfinite(array).all():
                raise ValueError(f'{name} contains non-finite values.')
        covariance = np.asarray(self.covariance, dtype=float)
        if covariance.shape != (n, n):
            raise ValueError(f'covariance must have shape {(n, n)}.')
        if not np.isfinite(covariance).all():
            raise ValueError('covariance contains non-finite values.')
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError('covariance must be symmetric.')
        if np.linalg.eigvalsh(covariance).min() < -1e-08:
            raise ValueError('covariance must be positive semidefinite.')
        impact = np.asarray(self.impact_gamma, dtype=float)
        if impact.shape == (n,):
            if not np.isfinite(impact).all() or (impact < 0).any():
                raise ValueError('impact diagonal must be finite and nonnegative.')
        elif impact.shape == (n, n):
            if not np.isfinite(impact).all():
                raise ValueError('impact matrix contains non-finite values.')
            if not np.allclose(impact, impact.T, atol=1e-10):
                raise ValueError('impact matrix must be symmetric.')
            if np.linalg.eigvalsh(impact).min() < -1e-08:
                raise ValueError('impact matrix must be positive semidefinite.')
        else:
            raise ValueError('impact_gamma must be a length-N diagonal or an N x N matrix.')
        if len(self.asset_classes) != n or len(self.descriptions) != n:
            raise ValueError('Metadata lengths must match the ticker count.')
        if (self.linear_cost < 0).any():
            raise ValueError('linear costs must be nonnegative.')
        if (self.adv_usd <= 0).any():
            raise ValueError('ADV values must be strictly positive.')
        if self.factor_loadings is not None:
            if list(self.factor_loadings.index) != self.tickers:
                raise ValueError('Factor-loading rows must exactly match ticker order.')
            if not np.isfinite(self.factor_loadings.to_numpy(dtype=float)).all():
                raise ValueError('Factor loadings contain non-finite values.')

    @property
    def total_return(self) -> np.ndarray:
        return self.growth + self.income

    @property
    def impact_matrix(self) -> np.ndarray:
        impact = np.asarray(self.impact_gamma, dtype=float)
        if impact.ndim == 1:
            return np.diag(impact)
        return impact

@dataclass
class ScenarioSet:
    names: list[str]
    returns: np.ndarray
    warning_thresholds: np.ndarray
    hard_loss_limits: np.ndarray
    weights: np.ndarray
    descriptions: list[str]

    def validate(self, n_assets: int) -> None:
        s = len(self.names)
        if s == 0:
            raise ValueError('At least one scenario is required.')
        if len(set(self.names)) != s:
            raise ValueError('Scenario names must be unique.')
        if len(self.descriptions) != s:
            raise ValueError('Scenario descriptions must match the scenario count.')
        returns = np.asarray(self.returns, dtype=float)
        if returns.shape != (s, n_assets):
            raise ValueError('Scenario-return matrix has the wrong shape.')
        if not np.isfinite(returns).all():
            raise ValueError('Scenario returns contain non-finite values.')
        for name, vector in {'warning_thresholds': self.warning_thresholds, 'hard_loss_limits': self.hard_loss_limits, 'weights': self.weights}.items():
            vector = np.asarray(vector, dtype=float)
            if vector.shape != (s,):
                raise ValueError(f'{name} has the wrong shape.')
            if not np.isfinite(vector).all():
                raise ValueError(f'{name} contains non-finite values.')
        if (self.weights < 0).any():
            raise ValueError('Scenario weights must be nonnegative.')
        if np.allclose(self.weights, 0.0):
            raise ValueError('At least one scenario penalty weight must be positive.')
        if (self.warning_thresholds > self.hard_loss_limits + TOL).any():
            raise ValueError('Every warning threshold must be at or below the hard loss limit.')

    @property
    def loss_matrix(self) -> np.ndarray:
        return -self.returns

@dataclass
class ConstraintConfig:
    asset_lower: np.ndarray
    asset_upper: np.ndarray
    class_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    factor_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    income_floor: float | None = None
    expected_total_return_floor: float | None = None

    def validate(self, data: PortfolioData) -> None:
        n = len(data.tickers)
        lower = np.asarray(self.asset_lower, dtype=float)
        upper = np.asarray(self.asset_upper, dtype=float)
        if lower.shape != (n,) or upper.shape != (n,):
            raise ValueError('Asset-bound vectors must match the asset count.')
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError('Asset bounds must be finite.')
        if (lower > upper + TOL).any():
            raise ValueError('Every asset lower bound must not exceed its upper bound.')
        if lower.sum() > 1.0 + TOL or upper.sum() < 1.0 - TOL:
            raise ValueError('Asset bounds cannot support a fully invested portfolio.')
        for label, bounds in {**self.class_bounds, **self.factor_bounds}.items():
            if len(bounds) != 2 or not np.isfinite(bounds).all():
                raise ValueError(f'Bounds for {label} must be two finite numbers.')
            if bounds[0] > bounds[1] + TOL:
                raise ValueError(f'Lower bound exceeds upper bound for {label}.')
        for label, value in {'income_floor': self.income_floor, 'expected_total_return_floor': self.expected_total_return_floor}.items():
            if value is not None and (not np.isfinite(value)):
                raise ValueError(f'{label} must be finite when provided.')

@dataclass
class SolveResult:
    stage: str
    success: bool
    message: str
    objective_value: float
    weights: pd.Series
    buys: pd.Series
    sells: pd.Series
    scenario_excess: pd.Series
    metrics: dict[str, float]
    constraint_audit: pd.DataFrame
    raw_result: OptimizeResult
CLASS_TARGETS = {'US Equity': 0.25, 'Developed Equity': 0.1, 'Emerging Equity': 0.05, 'Core Bonds': 0.15, 'Government Bonds': 0.1, 'Inflation Linked': 0.05, 'Credit': 0.08, 'Real Estate': 0.04, 'Precious Metals': 0.03, 'Broad Commodities': 0.03, 'Real-Asset Sectors': 0.03, 'Cash': 0.03, 'FX': 0.01, 'Alternatives': 0.03, 'Preferred': 0.02}
DEFAULT_CLASS_BOUNDS = {'US Equity': (0.15, 0.4), 'Developed Equity': (0.05, 0.2), 'Emerging Equity': (0.0, 0.12), 'Core Bonds': (0.08, 0.25), 'Government Bonds': (0.05, 0.25), 'Inflation Linked': (0.0, 0.1), 'Credit': (0.03, 0.18), 'Real Estate': (0.0, 0.1), 'Precious Metals': (0.0, 0.08), 'Broad Commodities': (0.0, 0.08), 'Real-Asset Sectors': (0.0, 0.1), 'Cash': (0.01, 0.15), 'FX': (0.0, 0.05), 'Alternatives': (0.0, 0.1), 'Preferred': (0.0, 0.05)}
SCENARIO_CLASS_RETURNS: dict[str, dict[str, float]] = {'Global equity selloff': {'US Equity': -0.25, 'Developed Equity': -0.27, 'Emerging Equity': -0.32, 'Core Bonds': 0.02, 'Government Bonds': 0.05, 'Inflation Linked': 0.01, 'Credit': -0.1, 'Real Estate': -0.24, 'Precious Metals': 0.04, 'Broad Commodities': -0.12, 'Real-Asset Sectors': -0.2, 'Cash': 0.002, 'FX': 0.01, 'Alternatives': 0.02, 'Preferred': -0.16}, 'Inflation and rate shock': {'US Equity': -0.14, 'Developed Equity': -0.13, 'Emerging Equity': -0.1, 'Core Bonds': -0.08, 'Government Bonds': -0.14, 'Inflation Linked': -0.03, 'Credit': -0.09, 'Real Estate': -0.18, 'Precious Metals': 0.08, 'Broad Commodities': 0.15, 'Real-Asset Sectors': 0.09, 'Cash': 0.003, 'FX': 0.01, 'Alternatives': 0.03, 'Preferred': -0.12}, 'Credit and liquidity crisis': {'US Equity': -0.18, 'Developed Equity': -0.2, 'Emerging Equity': -0.24, 'Core Bonds': -0.02, 'Government Bonds': 0.06, 'Inflation Linked': 0.01, 'Credit': -0.2, 'Real Estate': -0.25, 'Precious Metals': 0.02, 'Broad Commodities': -0.1, 'Real-Asset Sectors': -0.16, 'Cash': 0.002, 'FX': 0.015, 'Alternatives': -0.04, 'Preferred': -0.24}, 'Commodity supply shock': {'US Equity': -0.08, 'Developed Equity': -0.1, 'Emerging Equity': -0.06, 'Core Bonds': -0.04, 'Government Bonds': -0.06, 'Inflation Linked': 0.03, 'Credit': -0.06, 'Real Estate': -0.08, 'Precious Metals': 0.14, 'Broad Commodities': 0.22, 'Real-Asset Sectors': 0.13, 'Cash': 0.002, 'FX': 0.0, 'Alternatives': 0.04, 'Preferred': -0.07}, 'Broad deleveraging shock': {'US Equity': -0.2, 'Developed Equity': -0.22, 'Emerging Equity': -0.27, 'Core Bonds': -0.05, 'Government Bonds': -0.02, 'Inflation Linked': -0.04, 'Credit': -0.16, 'Real Estate': -0.23, 'Precious Metals': -0.08, 'Broad Commodities': -0.15, 'Real-Asset Sectors': -0.18, 'Cash': 0.001, 'FX': 0.0, 'Alternatives': -0.1, 'Preferred': -0.2}}
SCENARIO_TICKER_OVERRIDES = {'Inflation and rate shock': {'TLT': -0.22, 'IEF': -0.11, 'SHY': -0.025, 'TIP': -0.035}, 'Commodity supply shock': {'USO': 0.35, 'DBA': 0.18, 'DBC': 0.24, 'XLE': 0.18}, 'Global equity selloff': {'QQQ': -0.3, 'IWM': -0.31, 'HYG': -0.17}, 'Credit and liquidity crisis': {'HYG': -0.27, 'LQD': -0.14, 'EMB': -0.22, 'REM': -0.32}}
SCENARIO_DESCRIPTIONS = {'Global equity selloff': 'A synchronized global equity drawdown with a flight to high-quality government bonds.', 'Inflation and rate shock': 'A sharp increase in inflation expectations and discount rates that hurts duration-sensitive assets.', 'Credit and liquidity crisis': 'A widening of credit spreads and impaired market liquidity, with severe losses in lower-quality credit and real estate.', 'Commodity supply shock': 'A geopolitical or supply disruption that lifts commodities while pressuring conventional financial assets.', 'Broad deleveraging shock': 'A cross-asset liquidation in which most risky and diversifying assets decline simultaneously.'}

def nearest_psd(matrix: np.ndarray, epsilon: float=1e-10) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, epsilon, None)
    return eigenvectors * clipped @ eigenvectors.T

def load_step3_data(input_dir: str | Path, prefix: str='synthetic') -> PortfolioData:
    input_dir = Path(input_dir)
    stats_path = input_dir / f'{prefix}_asset_statistics.csv'
    covariance_path = input_dir / f'{prefix}_covariance.csv'
    costs_path = input_dir / f'{prefix}_cost_estimates.csv'
    factors_path = input_dir / f'{prefix}_factor_loadings.csv'
    stats = pd.read_csv(stats_path, index_col=0)
    covariance = pd.read_csv(covariance_path, index_col=0)
    costs = pd.read_csv(costs_path, index_col=0)
    tickers = stats.index.astype(str).tolist()
    covariance = covariance.loc[tickers, tickers]
    costs = costs.loc[tickers]
    growth_candidates = ['expected_growth_return_target', 'expected_growth_return_shrunk', 'expected_growth_return', 'expected_growth_return_realized']
    income_candidates = ['income_yield_target', 'income_yield']
    total_return_candidates = ['expected_total_return_target', 'expected_total_return_shrunk', 'expected_total_return_ticker_prior', 'expected_total_return_class_prior', 'expected_total_return_ewma_raw', 'expected_total_return_realized']
    growth_column = next((column for column in growth_candidates if column in stats.columns), None)
    income_column = next((column for column in income_candidates if column in stats.columns), None)
    if income_column is None:
        raise KeyError(f'Could not identify the income-yield column in Step 3 statistics. Available columns: {stats.columns.tolist()}')
    if growth_column is not None:
        growth_values = stats[growth_column].to_numpy(dtype=float)
    else:
        total_return_column = next((column for column in total_return_candidates if column in stats.columns), None)
        if total_return_column is None:
            raise KeyError(f'Could not identify either a growth-return column or a compatible total-return column in Step 3 statistics. Available columns: {stats.columns.tolist()}')
        growth_values = stats[total_return_column].to_numpy(dtype=float) - stats[income_column].to_numpy(dtype=float)
    income_values = stats[income_column].to_numpy(dtype=float)
    if 'linear_cost_fraction' in costs.columns:
        linear_cost = costs['linear_cost_fraction'].to_numpy(dtype=float)
    elif 'linear_cost_bps' in costs.columns:
        linear_cost = costs['linear_cost_bps'].to_numpy(dtype=float) / 10000.0
    else:
        raise KeyError('Step 3 costs must contain linear_cost_fraction or linear_cost_bps.')
    if 'quadratic_impact_gamma' not in costs.columns:
        raise KeyError('Step 3 costs must contain quadratic_impact_gamma.')
    if 'adv_usd' not in costs.columns:
        raise KeyError('Step 3 costs must contain adv_usd.')
    factor_loadings = None
    if factors_path.exists():
        factor_loadings = pd.read_csv(factors_path, index_col=0).loc[tickers]
    data = PortfolioData(tickers=tickers, growth=growth_values, income=income_values, covariance=nearest_psd(covariance.to_numpy(dtype=float)), linear_cost=linear_cost, impact_gamma=costs['quadratic_impact_gamma'].to_numpy(dtype=float), adv_usd=costs['adv_usd'].to_numpy(dtype=float), asset_classes=stats['asset_class'].astype(str).tolist(), descriptions=stats['description'].astype(str).tolist(), factor_loadings=factor_loadings)
    data.validate()
    return data

def build_strategic_current_portfolio(data: PortfolioData, class_targets: Mapping[str, float]=CLASS_TARGETS) -> np.ndarray:
    n = len(data.tickers)
    weights = np.zeros(n, dtype=float)
    vol = np.sqrt(np.clip(np.diag(data.covariance), 1e-12, None))
    classes = np.asarray(data.asset_classes, dtype=object)
    missing_classes = set(classes) - set(class_targets)
    if missing_classes:
        raise KeyError(f'Missing class targets for: {sorted(missing_classes)}')
    total_target = sum((class_targets[c] for c in sorted(set(classes))))
    if not np.isclose(total_target, 1.0, atol=1e-10):
        raise ValueError(f'Class targets for the active universe must sum to 1, got {total_target}.')
    for asset_class in sorted(set(classes)):
        indices = np.flatnonzero(classes == asset_class)
        inverse_vol = 1.0 / vol[indices]
        local = inverse_vol / inverse_vol.sum()
        weights[indices] = class_targets[asset_class] * local
    weights /= weights.sum()
    return weights

def build_default_constraint_config(data: PortfolioData, current_weights: np.ndarray, asset_cap: float=0.1) -> ConstraintConfig:
    n = len(data.tickers)
    lower = np.zeros(n)
    upper = np.full(n, asset_cap)
    for ticker in ('BIL', 'SGOV'):
        if ticker in data.tickers:
            upper[data.tickers.index(ticker)] = 0.15
    factor_bounds: dict[str, tuple[float, float]] = {}
    if data.factor_loadings is not None:
        exposures = data.factor_loadings.to_numpy(dtype=float).T @ current_weights
        tolerances = {'Equity': 0.2, 'Duration': 0.2, 'Credit': 0.15, 'Inflation': 0.15, 'USD': 0.1, 'Trend': 0.12}
        for index, factor in enumerate(data.factor_loadings.columns):
            tolerance = tolerances.get(str(factor), 0.15)
            factor_bounds[str(factor)] = (float(exposures[index] - tolerance), float(exposures[index] + tolerance))
    current_income = float(data.income @ current_weights)
    current_return = float(data.total_return @ current_weights)
    return ConstraintConfig(asset_lower=lower, asset_upper=upper, class_bounds={key: value for key, value in DEFAULT_CLASS_BOUNDS.items() if key in set(data.asset_classes)}, factor_bounds=factor_bounds, income_floor=max(0.015, current_income - 0.003), expected_total_return_floor=max(0.025, current_return - 0.01))

def build_scenarios(data: PortfolioData, current_weights: np.ndarray) -> ScenarioSet:
    classes = np.asarray(data.asset_classes, dtype=object)
    names = list(SCENARIO_CLASS_RETURNS)
    returns = np.zeros((len(names), len(data.tickers)), dtype=float)
    for s, scenario_name in enumerate(names):
        class_shocks = SCENARIO_CLASS_RETURNS[scenario_name]
        for i, (ticker, asset_class) in enumerate(zip(data.tickers, classes, strict=True)):
            returns[s, i] = class_shocks[asset_class]
            returns[s, i] = SCENARIO_TICKER_OVERRIDES.get(scenario_name, {}).get(ticker, returns[s, i])
    current_losses = -returns @ current_weights
    warning_thresholds = np.maximum(0.0, current_losses - 0.015)
    hard_limits = current_losses + 0.025
    weights = np.array([1.0, 1.0, 1.25, 0.75, 1.25], dtype=float)
    scenario_set = ScenarioSet(names=names, returns=returns, warning_thresholds=warning_thresholds, hard_loss_limits=hard_limits, weights=weights, descriptions=[SCENARIO_DESCRIPTIONS[name] for name in names])
    scenario_set.validate(len(data.tickers))
    return scenario_set

def analytical_fully_invested_mean_variance(expected_return: np.ndarray, covariance: np.ndarray, risk_penalty: float) -> np.ndarray:
    if risk_penalty <= 0:
        raise ValueError('risk_penalty must be strictly positive.')
    inverse = np.linalg.pinv(covariance)
    ones = np.ones(len(expected_return))
    denominator = float(ones @ inverse @ ones)
    multiplier = float((ones @ inverse @ expected_return - 2.0 * risk_penalty) / denominator)
    weights = inverse @ (expected_return - multiplier * ones) / (2.0 * risk_penalty)
    return weights

def analytical_global_minimum_variance(covariance: np.ndarray) -> np.ndarray:
    inverse = np.linalg.pinv(covariance)
    ones = np.ones(covariance.shape[0])
    numerator = inverse @ ones
    return numerator / float(ones @ numerator)

def _class_matrix(data: PortfolioData) -> tuple[list[str], np.ndarray]:
    names = sorted(set(data.asset_classes))
    matrix = np.zeros((len(names), len(data.tickers)), dtype=float)
    for row, name in enumerate(names):
        matrix[row, :] = np.asarray([c == name for c in data.asset_classes], dtype=float)
    return (names, matrix)

def _build_variable_layout(n_assets: int, n_scenarios: int, trading: bool, scenario: bool) -> dict[str, slice]:
    cursor = 0
    layout = {'w': slice(cursor, cursor + n_assets)}
    cursor += n_assets
    if trading:
        layout['p'] = slice(cursor, cursor + n_assets)
        cursor += n_assets
        layout['n'] = slice(cursor, cursor + n_assets)
        cursor += n_assets
    if scenario:
        layout['h'] = slice(cursor, cursor + n_scenarios)
        cursor += n_scenarios
    layout['all'] = slice(0, cursor)
    return layout

def _append_constraint(rows: list[np.ndarray], lower: list[float], upper: list[float], row: np.ndarray, lb: float, ub: float) -> None:
    rows.append(np.asarray(row, dtype=float))
    lower.append(float(lb))
    upper.append(float(ub))

def solve_portfolio(*, stage: str, data: PortfolioData, current_weights: np.ndarray, objective_weights: ObjectiveWeights, constraint_config: ConstraintConfig, trading_config: TradingConfig, solver_config: SolverConfig, scenarios: ScenarioSet | None=None, include_return_reward: bool=True, include_asset_caps: bool=True, include_class_constraints: bool=False, include_factor_constraints: bool=False, include_income_floor: bool=False, include_return_floor: bool=False, include_trading: bool=False, include_scenarios: bool=False, include_scenario_hard_limits: bool=False, warm_start_weights: np.ndarray | None=None) -> SolveResult:
    data.validate()
    constraint_config.validate(data)
    n_assets = len(data.tickers)
    current_weights = np.asarray(current_weights, dtype=float)
    if current_weights.shape != (n_assets,):
        raise ValueError('current_weights has the wrong shape.')
    if not np.isclose(current_weights.sum(), 1.0, atol=1e-08):
        raise ValueError('current_weights must sum to one.')
    if include_scenarios and scenarios is None:
        raise ValueError('A ScenarioSet is required when scenarios are enabled.')
    n_scenarios = 0 if scenarios is None else len(scenarios.names)
    if scenarios is not None:
        scenarios.validate(n_assets)
    layout = _build_variable_layout(n_assets, n_scenarios, include_trading, include_scenarios)
    n_variables = layout['all'].stop
    lower_bounds = np.full(n_variables, -np.inf)
    upper_bounds = np.full(n_variables, np.inf)
    lower_bounds[layout['w']] = constraint_config.asset_lower if include_asset_caps else 0.0
    upper_bounds[layout['w']] = constraint_config.asset_upper if include_asset_caps else 1.0
    if include_trading:
        lower_bounds[layout['p']] = 0.0
        lower_bounds[layout['n']] = 0.0
        upper_bounds[layout['p']] = 1.0
        upper_bounds[layout['n']] = 1.0
    if include_scenarios:
        lower_bounds[layout['h']] = 0.0
    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []
    row = np.zeros(n_variables)
    row[layout['w']] = 1.0
    _append_constraint(rows, lb, ub, row, 1.0, 1.0)
    if include_trading:
        for i in range(n_assets):
            row = np.zeros(n_variables)
            row[layout['w'].start + i] = 1.0
            row[layout['p'].start + i] = -1.0
            row[layout['n'].start + i] = 1.0
            _append_constraint(rows, lb, ub, row, current_weights[i], current_weights[i])
        row = np.zeros(n_variables)
        row[layout['p']] = 1.0
        row[layout['n']] = 1.0
        _append_constraint(rows, lb, ub, row, -np.inf, trading_config.turnover_limit_gross)
        trade_capacity = trading_config.execution_days * trading_config.participation_rate * data.adv_usd / trading_config.portfolio_value_usd
        for i in range(n_assets):
            row = np.zeros(n_variables)
            row[layout['p'].start + i] = 1.0
            row[layout['n'].start + i] = 1.0
            _append_constraint(rows, lb, ub, row, -np.inf, trade_capacity[i])
    if include_class_constraints:
        class_names, class_matrix = _class_matrix(data)
        for name, exposure_row in zip(class_names, class_matrix, strict=True):
            if name not in constraint_config.class_bounds:
                continue
            row = np.zeros(n_variables)
            row[layout['w']] = exposure_row
            class_lb, class_ub = constraint_config.class_bounds[name]
            _append_constraint(rows, lb, ub, row, class_lb, class_ub)
    if include_factor_constraints:
        if data.factor_loadings is None:
            raise ValueError('Factor constraints requested but no factor loadings are available.')
        for factor, (factor_lb, factor_ub) in constraint_config.factor_bounds.items():
            row = np.zeros(n_variables)
            row[layout['w']] = data.factor_loadings[factor].to_numpy(dtype=float)
            _append_constraint(rows, lb, ub, row, factor_lb, factor_ub)
    if include_income_floor and constraint_config.income_floor is not None:
        row = np.zeros(n_variables)
        row[layout['w']] = data.income
        _append_constraint(rows, lb, ub, row, constraint_config.income_floor, np.inf)
    if include_return_floor and constraint_config.expected_total_return_floor is not None:
        row = np.zeros(n_variables)
        row[layout['w']] = data.total_return
        _append_constraint(rows, lb, ub, row, constraint_config.expected_total_return_floor, np.inf)
    if include_scenarios and scenarios is not None:
        loss = scenarios.loss_matrix
        for s in range(n_scenarios):
            row = np.zeros(n_variables)
            row[layout['w']] = loss[s]
            row[layout['h'].start + s] = -1.0
            _append_constraint(rows, lb, ub, row, -np.inf, scenarios.warning_thresholds[s])
            if include_scenario_hard_limits:
                row = np.zeros(n_variables)
                row[layout['w']] = loss[s]
                _append_constraint(rows, lb, ub, row, -np.inf, scenarios.hard_loss_limits[s])
    constraint_matrix = np.vstack(rows)
    lower_vector = np.asarray(lb)
    upper_vector = np.asarray(ub)
    equality_mask = np.isfinite(lower_vector) & np.isfinite(upper_vector) & np.isclose(lower_vector, upper_vector, atol=1e-14)
    optimization_constraints: list[LinearConstraint] = []
    if equality_mask.any():
        optimization_constraints.append(LinearConstraint(constraint_matrix[equality_mask], lower_vector[equality_mask], upper_vector[equality_mask]))
    if (~equality_mask).any():
        optimization_constraints.append(LinearConstraint(constraint_matrix[~equality_mask], lower_vector[~equality_mask], upper_vector[~equality_mask]))
    variable_bounds = Bounds(lower_bounds, upper_bounds)
    covariance = data.covariance
    impact = data.impact_matrix
    ow = objective_weights
    scale = solver_config.objective_scale

    def objective(x: np.ndarray) -> float:
        w = x[layout['w']]
        value = ow.risk_penalty * float(w @ covariance @ w)
        if include_return_reward:
            value -= ow.growth_reward * float(data.growth @ w)
            value -= ow.income_reward * float(data.income @ w)
        value += ow.concentration_penalty * float(w @ w)
        if include_trading:
            p = x[layout['p']]
            n = x[layout['n']]
            value += ow.transaction_cost_penalty * float(data.linear_cost @ (p + n))
            delta = w - current_weights
            value += ow.market_impact_penalty * float(delta @ impact @ delta)
        if include_scenarios and scenarios is not None:
            h = x[layout['h']]
            value += ow.scenario_penalty * float(scenarios.weights @ (h * h))
        return scale * value

    def gradient(x: np.ndarray) -> np.ndarray:
        w = x[layout['w']]
        grad = np.zeros_like(x)
        grad_w = 2.0 * ow.risk_penalty * covariance @ w
        if include_return_reward:
            grad_w -= ow.growth_reward * data.growth
            grad_w -= ow.income_reward * data.income
        grad_w += 2.0 * ow.concentration_penalty * w
        if include_trading:
            grad_w += 2.0 * ow.market_impact_penalty * impact @ (w - current_weights)
            grad[layout['p']] = ow.transaction_cost_penalty * data.linear_cost
            grad[layout['n']] = ow.transaction_cost_penalty * data.linear_cost
        if include_scenarios and scenarios is not None:
            h = x[layout['h']]
            grad[layout['h']] = 2.0 * ow.scenario_penalty * scenarios.weights * h
        grad[layout['w']] = grad_w
        return scale * grad
    initial_w = current_weights.copy() if warm_start_weights is None else np.asarray(warm_start_weights, dtype=float)
    initial_w = np.clip(initial_w, lower_bounds[layout['w']], upper_bounds[layout['w']])
    initial_w /= initial_w.sum()
    x0 = np.zeros(n_variables)
    x0[layout['w']] = initial_w
    if include_trading:
        delta = initial_w - current_weights
        x0[layout['p']] = np.maximum(delta, 0.0)
        x0[layout['n']] = np.maximum(-delta, 0.0)
    if include_scenarios and scenarios is not None:
        loss = scenarios.loss_matrix @ initial_w
        x0[layout['h']] = np.maximum(loss - scenarios.warning_thresholds, 0.0)
    result = minimize(objective, x0, method='SLSQP', jac=gradient, bounds=variable_bounds, constraints=optimization_constraints, options={'ftol': solver_config.ftol, 'maxiter': solver_config.maxiter, 'disp': solver_config.disp})
    x = np.asarray(result.x, dtype=float)
    w = x[layout['w']]
    p = x[layout['p']] if include_trading else np.maximum(w - current_weights, 0.0)
    n = x[layout['n']] if include_trading else np.maximum(current_weights - w, 0.0)
    h = x[layout['h']] if include_scenarios else np.array([], dtype=float)
    audit = audit_constraints(data=data, weights=w, current_weights=current_weights, buys=p, sells=n, scenarios=scenarios if include_scenarios else None, constraint_config=constraint_config, trading_config=trading_config, check_asset_caps=include_asset_caps, check_classes=include_class_constraints, check_factors=include_factor_constraints, check_income=include_income_floor, check_return=include_return_floor, check_trading=include_trading, check_scenario_hard=include_scenario_hard_limits, tolerance=solver_config.feasibility_tolerance)
    metrics = calculate_metrics(data=data, weights=w, current_weights=current_weights, scenarios=scenarios, objective_weights=objective_weights, scenario_excess=h)
    audit_success = bool(audit['satisfied'].all())
    success = bool(result.success and audit_success)
    return SolveResult(stage=stage, success=success, message=str(result.message), objective_value=float(objective(x) / scale), weights=pd.Series(w, index=data.tickers, name=stage), buys=pd.Series(p, index=data.tickers, name='buy'), sells=pd.Series(n, index=data.tickers, name='sell'), scenario_excess=pd.Series(h, index=scenarios.names if include_scenarios and scenarios is not None else [], name='scenario_excess'), metrics=metrics, constraint_audit=audit, raw_result=result)

def calculate_metrics(*, data: PortfolioData, weights: np.ndarray, current_weights: np.ndarray, scenarios: ScenarioSet | None, objective_weights: ObjectiveWeights, scenario_excess: np.ndarray) -> dict[str, float]:
    expected_growth = float(data.growth @ weights)
    income = float(data.income @ weights)
    total_return = expected_growth + income
    variance = float(weights @ data.covariance @ weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    delta = weights - current_weights
    gross_turnover = float(np.abs(delta).sum())
    one_way_turnover = 0.5 * gross_turnover
    transaction_cost = float(data.linear_cost @ np.abs(delta))
    impact_cost = float(delta @ data.impact_matrix @ delta)
    concentration = float(weights @ weights)
    effective_holdings = 1.0 / concentration if concentration > 0 else np.inf
    maximum_weight = float(weights.max())
    scenario_penalty = 0.0
    worst_scenario_loss = np.nan
    if scenarios is not None:
        losses = scenarios.loss_matrix @ weights
        exact_excess = np.maximum(losses - scenarios.warning_thresholds, 0.0)
        scenario_penalty = float(objective_weights.scenario_penalty * scenarios.weights @ (exact_excess * exact_excess))
        worst_scenario_loss = float(losses.max())
        if len(scenario_excess):
            if not np.allclose(scenario_excess, exact_excess, atol=5e-05):
                if np.max(np.abs(scenario_excess - exact_excess)) > 0.0005:
                    raise RuntimeError('Scenario auxiliary variables do not match the exact hinge values.')
    return {'expected_growth': expected_growth, 'income_yield': income, 'expected_total_return': total_return, 'variance': variance, 'volatility': volatility, 'gross_turnover': gross_turnover, 'one_way_turnover': one_way_turnover, 'linear_transaction_cost': transaction_cost, 'quadratic_impact_cost': impact_cost, 'concentration_hhi': concentration, 'effective_holdings': effective_holdings, 'maximum_asset_weight': maximum_weight, 'scenario_penalty': scenario_penalty, 'worst_scenario_loss': worst_scenario_loss}

def audit_constraints(*, data: PortfolioData, weights: np.ndarray, current_weights: np.ndarray, buys: np.ndarray, sells: np.ndarray, scenarios: ScenarioSet | None, constraint_config: ConstraintConfig, trading_config: TradingConfig, check_asset_caps: bool, check_classes: bool, check_factors: bool, check_income: bool, check_return: bool, check_trading: bool, check_scenario_hard: bool, tolerance: float) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    def add(name: str, value: float, lower: float, upper: float, category: str) -> None:
        lower_ok = value >= lower - tolerance
        upper_ok = value <= upper + tolerance
        records.append({'category': category, 'constraint': name, 'value': value, 'lower': lower, 'upper': upper, 'lower_slack': value - lower if np.isfinite(lower) else np.nan, 'upper_slack': upper - value if np.isfinite(upper) else np.nan, 'satisfied': bool(lower_ok and upper_ok)})
    add('full_investment', float(weights.sum()), 1.0, 1.0, 'budget')
    add('minimum_weight', float(weights.min()), 0.0, np.inf, 'asset')
    if check_asset_caps:
        for ticker, value, lower, upper in zip(data.tickers, weights, constraint_config.asset_lower, constraint_config.asset_upper, strict=True):
            add(f'weight_{ticker}', float(value), float(lower), float(upper), 'asset')
    if check_classes:
        classes = np.asarray(data.asset_classes, dtype=object)
        for asset_class, (lower, upper) in constraint_config.class_bounds.items():
            value = float(weights[classes == asset_class].sum())
            add(f'class_{asset_class}', value, lower, upper, 'asset_class')
    if check_factors:
        if data.factor_loadings is None:
            raise ValueError('Cannot audit factors without factor loadings.')
        exposures = data.factor_loadings.to_numpy(dtype=float).T @ weights
        for factor, value in zip(data.factor_loadings.columns, exposures, strict=True):
            if factor in constraint_config.factor_bounds:
                lower, upper = constraint_config.factor_bounds[str(factor)]
                add(f'factor_{factor}', float(value), lower, upper, 'factor')
    if check_income and constraint_config.income_floor is not None:
        add('income_floor', float(data.income @ weights), constraint_config.income_floor, np.inf, 'income')
    if check_return and constraint_config.expected_total_return_floor is not None:
        add('return_floor', float(data.total_return @ weights), constraint_config.expected_total_return_floor, np.inf, 'return')
    if check_trading:
        delta = weights - current_weights
        accounting_residual = np.max(np.abs(delta - buys + sells))
        add('trade_accounting_max_abs_error', float(accounting_residual), -np.inf, tolerance, 'trading')
        add('gross_turnover', float((buys + sells).sum()), -np.inf, trading_config.turnover_limit_gross, 'trading')
        trade_capacity = trading_config.execution_days * trading_config.participation_rate * data.adv_usd / trading_config.portfolio_value_usd
        for ticker, trade, capacity in zip(data.tickers, buys + sells, trade_capacity, strict=True):
            add(f'liquidity_{ticker}', float(trade), -np.inf, float(capacity), 'liquidity')
    if scenarios is not None and check_scenario_hard:
        losses = scenarios.loss_matrix @ weights
        for name, loss, limit in zip(scenarios.names, losses, scenarios.hard_loss_limits, strict=True):
            add(f'scenario_{name}', float(loss), -np.inf, float(limit), 'scenario')
    return pd.DataFrame.from_records(records)

def build_efficient_frontier(data: PortfolioData, asset_upper: np.ndarray, points: int=15, solver_config: SolverConfig=SolverConfig()) -> pd.DataFrame:
    n = len(data.tickers)
    bounds_list = [(0.0, float(cap)) for cap in asset_upper]
    max_return_result = linprog(-data.total_return, A_eq=np.ones((1, n)), b_eq=np.array([1.0]), bounds=bounds_list, method='highs')
    if not max_return_result.success:
        raise RuntimeError(f'Could not calculate the maximum feasible return: {max_return_result.message}')

    def gmv_objective(w: np.ndarray) -> float:
        return solver_config.objective_scale * float(w @ data.covariance @ w)

    def gmv_gradient(w: np.ndarray) -> np.ndarray:
        return solver_config.objective_scale * 2.0 * data.covariance @ w
    gmv_result = minimize(gmv_objective, np.full(n, 1.0 / n), method='SLSQP', jac=gmv_gradient, bounds=Bounds(np.zeros(n), asset_upper), constraints=[LinearConstraint(np.ones((1, n)), np.array([1.0]), np.array([1.0]))], options={'ftol': solver_config.ftol, 'maxiter': solver_config.maxiter, 'disp': False})
    if not gmv_result.success:
        raise RuntimeError(f'Could not calculate the capped minimum-variance portfolio: {gmv_result.message}')
    minimum_frontier_return = float(data.total_return @ gmv_result.x)
    maximum_feasible_return = float(data.total_return @ max_return_result.x)
    if maximum_feasible_return <= minimum_frontier_return + 1e-10:
        targets = np.array([minimum_frontier_return])
    else:
        targets = np.linspace(minimum_frontier_return, minimum_frontier_return + 0.98 * (maximum_feasible_return - minimum_frontier_return), points)
    records: list[dict[str, float]] = []
    warm = gmv_result.x.copy()
    for target in targets:

        def objective(w: np.ndarray) -> float:
            return solver_config.objective_scale * float(w @ data.covariance @ w)

        def gradient(w: np.ndarray) -> np.ndarray:
            return solver_config.objective_scale * 2.0 * data.covariance @ w
        constraints = [LinearConstraint(np.ones((1, n)), np.array([1.0]), np.array([1.0])), LinearConstraint(data.total_return.reshape(1, -1), np.array([target]), np.array([np.inf]))]
        result = minimize(objective, warm, method='SLSQP', jac=gradient, bounds=Bounds(np.zeros(n), asset_upper), constraints=constraints, options={'ftol': solver_config.ftol, 'maxiter': solver_config.maxiter, 'disp': False})
        if result.success:
            warm = result.x
            variance = float(result.x @ data.covariance @ result.x)
            records.append({'target_return': target, 'achieved_return': float(data.total_return @ result.x), 'volatility': float(np.sqrt(max(variance, 0.0))), 'variance': variance, 'success': True})
        else:
            records.append({'target_return': target, 'achieved_return': np.nan, 'volatility': np.nan, 'variance': np.nan, 'success': False})
    return pd.DataFrame.from_records(records)

def run_constraint_ladder(data: PortfolioData, current_weights: np.ndarray, objective_weights: ObjectiveWeights=ObjectiveWeights(), trading_config: TradingConfig=TradingConfig(), solver_config: SolverConfig=SolverConfig()) -> tuple[list[SolveResult], ScenarioSet, ConstraintConfig]:
    constraints = build_default_constraint_config(data, current_weights)
    constraints.validate(data)
    scenarios = build_scenarios(data, current_weights)
    scenarios.validate(len(data.tickers))
    stages: list[SolveResult] = []
    pure_mean_variance_weights = ObjectiveWeights(growth_reward=objective_weights.growth_reward, income_reward=objective_weights.income_reward, risk_penalty=objective_weights.risk_penalty, transaction_cost_penalty=0.0, market_impact_penalty=0.0, scenario_penalty=0.0, concentration_penalty=0.0)
    stages.append(solve_portfolio(stage='01_minimum_variance', data=data, current_weights=current_weights, objective_weights=ObjectiveWeights(growth_reward=0.0, income_reward=0.0, risk_penalty=1.0, transaction_cost_penalty=0.0, market_impact_penalty=0.0, scenario_penalty=0.0, concentration_penalty=0.0), constraint_config=constraints, trading_config=trading_config, solver_config=solver_config, scenarios=scenarios, include_return_reward=False, include_asset_caps=False))
    stages.append(solve_portfolio(stage='02_mean_variance', data=data, current_weights=current_weights, objective_weights=pure_mean_variance_weights, constraint_config=constraints, trading_config=trading_config, solver_config=solver_config, scenarios=scenarios, include_asset_caps=False, warm_start_weights=stages[-1].weights.to_numpy()))
    stages.append(solve_portfolio(stage='03_asset_caps', data=data, current_weights=current_weights, objective_weights=pure_mean_variance_weights, constraint_config=constraints, trading_config=trading_config, solver_config=solver_config, scenarios=scenarios, include_asset_caps=True, warm_start_weights=current_weights))
    stages.append(solve_portfolio(stage='04_guardrails', data=data, current_weights=current_weights, objective_weights=objective_weights, constraint_config=constraints, trading_config=trading_config, solver_config=solver_config, scenarios=scenarios, include_asset_caps=True, include_class_constraints=True, include_factor_constraints=data.factor_loadings is not None, include_income_floor=True, include_return_floor=True, warm_start_weights=current_weights))
    stages.append(solve_portfolio(stage='05_trading_costs', data=data, current_weights=current_weights, objective_weights=objective_weights, constraint_config=constraints, trading_config=trading_config, solver_config=solver_config, scenarios=scenarios, include_asset_caps=True, include_class_constraints=True, include_factor_constraints=data.factor_loadings is not None, include_income_floor=True, include_return_floor=True, include_trading=True, warm_start_weights=current_weights))
    stages.append(solve_portfolio(stage='06_scenario_aware', data=data, current_weights=current_weights, objective_weights=objective_weights, constraint_config=constraints, trading_config=trading_config, solver_config=solver_config, scenarios=scenarios, include_asset_caps=True, include_class_constraints=True, include_factor_constraints=data.factor_loadings is not None, include_income_floor=True, include_return_floor=True, include_trading=True, include_scenarios=True, include_scenario_hard_limits=True, warm_start_weights=stages[-1].weights.to_numpy()))
    return (stages, scenarios, constraints)

def save_results(output_dir: str | Path, data: PortfolioData, current_weights: np.ndarray, stages: Iterable[SolveResult], scenarios: ScenarioSet, constraints: ConstraintConfig, objective_weights: ObjectiveWeights, trading_config: TradingConfig, solver_config: SolverConfig) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = list(stages)
    weights = pd.DataFrame({'current': current_weights}, index=data.tickers)
    metrics_records = []
    all_audits = []
    trade_records = []
    for result in stages:
        weights[result.stage] = result.weights
        metrics_records.append({'stage': result.stage, 'success': result.success, **result.metrics})
        audit = result.constraint_audit.copy()
        audit.insert(0, 'stage', result.stage)
        all_audits.append(audit)
        trade = pd.DataFrame({'ticker': data.tickers, 'stage': result.stage, 'weight': result.weights.to_numpy(), 'buy': result.buys.to_numpy(), 'sell': result.sells.to_numpy(), 'net_trade': result.weights.to_numpy() - current_weights})
        trade_records.append(trade)
    weights.to_csv(output_dir / 'stage_weights.csv')
    pd.DataFrame(metrics_records).set_index('stage').to_csv(output_dir / 'stage_metrics.csv')
    pd.concat(all_audits, ignore_index=True).to_csv(output_dir / 'constraint_audit.csv', index=False)
    pd.concat(trade_records, ignore_index=True).to_csv(output_dir / 'stage_trades.csv', index=False)
    scenario_frame = pd.DataFrame(scenarios.returns, index=scenarios.names, columns=data.tickers)
    scenario_frame.to_csv(output_dir / 'scenario_returns.csv')
    pd.DataFrame({'description': scenarios.descriptions, 'warning_threshold': scenarios.warning_thresholds, 'hard_loss_limit': scenarios.hard_loss_limits, 'penalty_weight': scenarios.weights}, index=scenarios.names).to_csv(output_dir / 'scenario_definitions.csv')
    final_weights = stages[-1].weights.to_numpy()
    losses = scenarios.loss_matrix @ final_weights
    excess = np.maximum(losses - scenarios.warning_thresholds, 0.0)
    pd.DataFrame({'portfolio_loss': losses, 'warning_threshold': scenarios.warning_thresholds, 'hard_loss_limit': scenarios.hard_loss_limits, 'exact_excess': excess, 'hard_limit_satisfied': losses <= scenarios.hard_loss_limits + solver_config.feasibility_tolerance}, index=scenarios.names).to_csv(output_dir / 'final_scenario_audit.csv')
    class_names, class_matrix = _class_matrix(data)
    class_exposure = pd.DataFrame(index=class_names)
    for name, vector in weights.items():
        class_exposure[name] = class_matrix @ vector.to_numpy()
    class_exposure.to_csv(output_dir / 'class_exposures.csv')
    if data.factor_loadings is not None:
        factor_exposure = pd.DataFrame(index=data.factor_loadings.columns)
        for name, vector in weights.items():
            factor_exposure[name] = data.factor_loadings.to_numpy(dtype=float).T @ vector.to_numpy()
        factor_exposure.to_csv(output_dir / 'factor_exposures.csv')
    frontier = build_efficient_frontier(data, constraints.asset_upper, points=15, solver_config=solver_config)
    frontier.to_csv(output_dir / 'efficient_frontier.csv', index=False)
    configuration = {'objective_weights': asdict(objective_weights), 'trading_config': asdict(trading_config), 'solver_config': asdict(solver_config), 'constraint_config': {'asset_lower': constraints.asset_lower.tolist(), 'asset_upper': constraints.asset_upper.tolist(), 'class_bounds': constraints.class_bounds, 'factor_bounds': constraints.factor_bounds, 'income_floor': constraints.income_floor, 'expected_total_return_floor': constraints.expected_total_return_floor}}
    (output_dir / 'step4_configuration.json').write_text(json.dumps(configuration, indent=2))

def build_summary(stages: Iterable[SolveResult]) -> str:
    lines = []
    for result in stages:
        m = result.metrics
        lines.append(f"{result.stage}: success={result.success}, return={m['expected_total_return']:.3%}, vol={m['volatility']:.3%}, gross_turnover={m['gross_turnover']:.3%}, worst_scenario={m['worst_scenario_loss']:.3%}")
    return '\n'.join(lines)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Step 4 classical portfolio optimizer.')
    parser.add_argument('--input-dir', type=Path, required=True, help='Directory containing Step 3 CSV outputs.')
    parser.add_argument('--prefix', default='synthetic', help='Step 3 file prefix.')
    parser.add_argument('--output', type=Path, default=Path('step4_outputs'))
    parser.add_argument('--risk-penalty', type=float, default=3.0)
    parser.add_argument('--scenario-penalty', type=float, default=25.0)
    parser.add_argument('--turnover-limit', type=float, default=0.5)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    data = load_step3_data(args.input_dir, args.prefix)
    current_weights = build_strategic_current_portfolio(data)
    objective_weights = ObjectiveWeights(risk_penalty=args.risk_penalty, scenario_penalty=args.scenario_penalty)
    trading_config = TradingConfig(turnover_limit_gross=args.turnover_limit)
    solver_config = SolverConfig()
    stages, scenarios, constraints = run_constraint_ladder(data, current_weights, objective_weights, trading_config, solver_config)
    save_results(args.output, data, current_weights, stages, scenarios, constraints, objective_weights, trading_config, solver_config)
    print(build_summary(stages))
    if not all((result.success for result in stages)):
        failed = [result.stage for result in stages if not result.success]
        raise SystemExit(f'One or more stages failed: {failed}')
if __name__ == '__main__':
    main()
