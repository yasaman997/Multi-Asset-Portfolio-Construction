from __future__ import annotations
import argparse
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf
TRADING_DAYS = 252

@dataclass(frozen=True)
class PipelineConfig:
    start: str = '2021-01-01'
    end: str | None = None
    interval: str = '1d'
    annualization: int = TRADING_DAYS
    min_observations: int = 756
    min_common_observations: int = 500
    winsor_tail_probability: float = 0.001
    ewma_halflife_days: float = 126.0
    empirical_return_weight: float = 0.35
    require_full_universe: bool = True
    download_retries: int = 3
    download_timeout_seconds: int = 60
    portfolio_value_usd: float = 10000000.0
    execution_days: int = 3
    reference_trade_weight: float = 0.01
    commission_bps: float = 0.2
    half_spread_floor_bps: float = 0.25
    high_low_to_spread_fraction: float = 0.1
    impact_eta: float = 0.75
    adv_window_days: int = 60
    institutional_portfolio_value_usd: float = 250000000.0
    institutional_execution_days: int = 2
    institutional_reference_trade_weight: float = 0.02
    institutional_impact_eta: float = 0.9
    institutional_adv_multiplier: float = 1.0
    synthetic_days: int = 1260
    synthetic_df: float = 7.0
    synthetic_seed: int = 20260802
    synthetic_idio_vol_floor_abs: float = 0.001
    synthetic_idio_vol_floor_fraction: float = 0.1
    synthetic_factor_variance_cap: float = 0.9

@dataclass
class MarketDataBundle:
    adj_close: pd.DataFrame
    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    dividends: pd.DataFrame
    invalid_tickers: list[str]

@dataclass
class EmpiricalEstimates:
    raw_daily_returns: pd.DataFrame
    robust_daily_returns: pd.DataFrame
    common_returns: pd.DataFrame
    asset_stats: pd.DataFrame
    covariance: pd.DataFrame
    correlation: pd.DataFrame
    class_returns: pd.DataFrame
    class_stats: pd.DataFrame
    class_correlation: pd.DataFrame
    cost_table: pd.DataFrame
    cost_sensitivity_tables: dict[str, pd.DataFrame]

@dataclass
class SyntheticEstimates:
    daily_returns: pd.DataFrame
    prices: pd.DataFrame
    asset_stats: pd.DataFrame
    covariance: pd.DataFrame
    correlation: pd.DataFrame
    class_returns: pd.DataFrame
    class_stats: pd.DataFrame
    class_correlation: pd.DataFrame
    cost_table: pd.DataFrame
    cost_sensitivity_tables: dict[str, pd.DataFrame]
    factor_returns: pd.DataFrame
    factor_loadings: pd.DataFrame

def build_asset_universe() -> pd.DataFrame:
    records: list[tuple[str, str, str]] = [('SPY', 'US Equity', 'Broad US large-cap equity'), ('QQQ', 'US Equity', 'US technology/growth equity'), ('IWM', 'US Equity', 'US small-cap equity'), ('DIA', 'US Equity', 'US blue-chip equity'), ('VTV', 'US Equity', 'US value equity'), ('VUG', 'US Equity', 'US growth equity'), ('USMV', 'US Equity', 'US minimum-volatility equity'), ('MTUM', 'US Equity', 'US momentum equity'), ('QUAL', 'US Equity', 'US quality equity'), ('SCHD', 'US Equity', 'US dividend equity'), ('EFA', 'Developed Equity', 'Developed ex-US equity'), ('VGK', 'Developed Equity', 'European equity'), ('EWJ', 'Developed Equity', 'Japanese equity'), ('EWC', 'Developed Equity', 'Canadian equity'), ('EWA', 'Developed Equity', 'Australian equity'), ('EEM', 'Emerging Equity', 'Broad emerging-market equity'), ('VWO', 'Emerging Equity', 'Broad emerging-market equity'), ('INDA', 'Emerging Equity', 'Indian equity'), ('MCHI', 'Emerging Equity', 'Chinese equity'), ('EWZ', 'Emerging Equity', 'Brazilian equity'), ('AGG', 'Core Bonds', 'US aggregate bonds'), ('BND', 'Core Bonds', 'US total bond market'), ('BNDX', 'Core Bonds', 'International investment-grade bonds'), ('BWX', 'Core Bonds', 'International government bonds'), ('TLT', 'Government Bonds', 'Long-duration US Treasuries'), ('IEF', 'Government Bonds', 'Intermediate US Treasuries'), ('SHY', 'Government Bonds', 'Short US Treasuries'), ('TIP', 'Inflation Linked', 'US inflation-linked Treasuries'), ('LQD', 'Credit', 'Investment-grade corporate bonds'), ('HYG', 'Credit', 'High-yield corporate bonds'), ('EMB', 'Credit', 'USD emerging-market debt'), ('MUB', 'Credit', 'US municipal bonds'), ('VNQ', 'Real Estate', 'US listed real estate'), ('REET', 'Real Estate', 'Global listed real estate'), ('REM', 'Real Estate', 'US mortgage real estate'), ('GLD', 'Precious Metals', 'Gold'), ('SLV', 'Precious Metals', 'Silver'), ('DBC', 'Broad Commodities', 'Diversified commodities'), ('USO', 'Broad Commodities', 'Crude-oil exposure'), ('DBA', 'Broad Commodities', 'Agricultural commodities'), ('XLE', 'Real-Asset Sectors', 'US energy equity sector'), ('XLB', 'Real-Asset Sectors', 'US materials equity sector'), ('XLU', 'Real-Asset Sectors', 'US utilities equity sector'), ('BIL', 'Cash', 'Treasury bills'), ('SGOV', 'Cash', 'Short Treasury bills'), ('UUP', 'FX', 'US-dollar exposure'), ('FXE', 'FX', 'Euro exposure'), ('DBMF', 'Alternatives', 'Managed-futures strategy'), ('KMLM', 'Alternatives', 'Managed-futures strategy'), ('PFF', 'Preferred', 'Preferred securities')]
    universe = pd.DataFrame(records, columns=['ticker', 'asset_class', 'description'])
    universe['ticker'] = universe['ticker'].str.upper()
    if len(universe) != 50 or universe['ticker'].duplicated().any():
        raise AssertionError('The universe must contain exactly 50 unique tickers.')
    return universe.set_index('ticker', drop=False)

def _nearest_psd(matrix: NDArray[np.float64], epsilon: float=1e-10) -> NDArray[np.float64]:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, epsilon, None)
    return eigenvectors * clipped @ eigenvectors.T

def covariance_to_correlation(covariance: pd.DataFrame) -> pd.DataFrame:
    cov = covariance.to_numpy(dtype=float)
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denominator = np.outer(std, std)
    corr = np.divide(cov, denominator, out=np.zeros_like(cov), where=denominator > 0)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    return pd.DataFrame(corr, index=covariance.index, columns=covariance.columns)

def _winsorize_columns(data: pd.DataFrame, tail_probability: float) -> pd.DataFrame:
    if tail_probability <= 0.0:
        return data.copy()
    if not 0.0 < tail_probability < 0.5:
        raise ValueError('winsor_tail_probability must lie in [0, 0.5).')
    lower = data.quantile(tail_probability)
    upper = data.quantile(1.0 - tail_probability)
    return data.clip(lower=lower, upper=upper, axis=1)

def _ewma_mean(returns: pd.DataFrame, halflife_days: float) -> pd.Series:
    if returns.empty:
        raise ValueError('Returns are empty.')
    if halflife_days <= 0:
        raise ValueError('halflife_days must be positive.')
    ages = np.arange(len(returns) - 1, -1, -1, dtype=float)
    weights = np.power(0.5, ages / halflife_days)
    weights /= weights.sum()
    values = returns.to_numpy(dtype=float)
    mask = np.isfinite(values)
    weighted_sum = np.nansum(values * weights[:, None], axis=0)
    available_weight = np.sum(mask * weights[:, None], axis=0)
    means = np.divide(weighted_sum, available_weight, out=np.full(values.shape[1], np.nan), where=available_weight > 0)
    return pd.Series(means, index=returns.columns)

def _cagr_from_adjusted_prices(adjusted_close: pd.DataFrame) -> pd.Series:
    output: dict[str, float] = {}
    for ticker in adjusted_close.columns:
        series = adjusted_close[ticker].dropna().astype(float)
        if len(series) < 2 or series.iloc[0] <= 0 or series.iloc[-1] <= 0:
            output[ticker] = np.nan
            continue
        elapsed_years = (series.index[-1] - series.index[0]).days / 365.25
        if elapsed_years <= 0:
            output[ticker] = np.nan
            continue
        output[ticker] = float((series.iloc[-1] / series.iloc[0]) ** (1.0 / elapsed_years) - 1.0)
    return pd.Series(output, name='cagr_raw_adjusted_price')

def _common_sample(returns: pd.DataFrame, minimum: int) -> pd.DataFrame:
    common = returns.dropna(how='any')
    if len(common) < minimum:
        raise ValueError(f'Only {len(common)} complete observations remain; at least {minimum} are required.')
    return common

def _last_valid_value(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(lambda column: column.dropna().iloc[-1] if column.notna().any() else np.nan)

def _roll_spread_fraction(close: pd.Series) -> float:
    prices = close.dropna().astype(float)
    changes = prices.diff().dropna()
    if len(changes) < 30:
        return float('nan')
    covariance = float(np.cov(changes.iloc[1:], changes.iloc[:-1], ddof=1)[0, 1])
    spread_dollars = 2.0 * math.sqrt(max(-covariance, 0.0))
    reference_price = float(prices.median())
    if not np.isfinite(reference_price) or reference_price <= 0:
        return float('nan')
    return spread_dollars / reference_price

def _extract_download_field(raw: pd.DataFrame, field: str, tickers: Sequence[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=[ticker.upper() for ticker in tickers], dtype=float)
    result: pd.DataFrame | pd.Series
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)
        if field in level0:
            result = raw[field]
        elif field in level1:
            result = raw.xs(field, axis=1, level=1)
        else:
            return pd.DataFrame(index=raw.index, columns=tickers, dtype=float)
    else:
        if field not in raw.columns:
            return pd.DataFrame(index=raw.index, columns=tickers, dtype=float)
        result = raw[field]
    if isinstance(result, pd.Series):
        if len(tickers) != 1:
            raise ValueError(f'Unexpected one-dimensional result for {field!r}.')
        result = result.to_frame(name=tickers[0])
    result = result.copy()
    result.columns = [str(column).upper() for column in result.columns]
    return result.reindex(columns=[ticker.upper() for ticker in tickers]).sort_index()

def _merge_field(base: pd.DataFrame, update: pd.DataFrame) -> pd.DataFrame:
    index = base.index.union(update.index)
    columns = base.columns.union(update.columns)
    result = base.reindex(index=index, columns=columns)
    update_aligned = update.reindex(index=index, columns=columns)
    return result.combine_first(update_aligned).sort_index()

def build_synthetic_priors() -> pd.DataFrame:
    records = [('US Equity', 0.055, 0.015, 0.18, 1.0, 1000.0), ('Developed Equity', 0.045, 0.025, 0.19, 2.0, 300.0), ('Emerging Equity', 0.055, 0.025, 0.25, 4.0, 100.0), ('Core Bonds', 0.01, 0.03, 0.07, 1.5, 200.0), ('Government Bonds', 0.005, 0.025, 0.09, 1.0, 400.0), ('Inflation Linked', 0.01, 0.025, 0.08, 1.5, 150.0), ('Credit', 0.015, 0.04, 0.1, 2.5, 150.0), ('Real Estate', 0.035, 0.04, 0.22, 3.0, 80.0), ('Precious Metals', 0.035, 0.0, 0.2, 2.5, 200.0), ('Broad Commodities', 0.035, 0.0, 0.24, 5.0, 60.0), ('Real-Asset Sectors', 0.045, 0.025, 0.23, 2.0, 250.0), ('Cash', 0.005, 0.035, 0.008, 0.5, 500.0), ('FX', 0.01, 0.0, 0.1, 3.0, 100.0), ('Alternatives', 0.035, 0.015, 0.13, 5.0, 20.0), ('Preferred', 0.02, 0.05, 0.14, 3.0, 80.0)]
    return pd.DataFrame(records, columns=['asset_class', 'annual_growth', 'income_yield', 'annual_volatility', 'linear_cost_bps', 'adv_usd_millions']).set_index('asset_class')

def build_ticker_prior_overlays() -> pd.DataFrame:
    records = [('QQQ', 0.006, -0.008, 1.12, 1.08, 1.5), ('VUG', 0.004, -0.007, 1.05, 1.03, 1.2), ('IWM', 0.003, 0.002, 1.2, 1.1, 0.75), ('USMV', -0.004, 0.002, 0.72, 0.78, 0.8), ('SCHD', -0.003, 0.018, 0.88, 0.88, 0.9), ('MTUM', 0.003, -0.002, 1.08, 1.05, 0.8), ('QUAL', 0.0, 0.002, 0.9, 0.9, 0.9)]
    return pd.DataFrame(records, columns=['ticker', 'growth_shift', 'income_shift', 'volatility_multiplier', 'factor_loading_multiplier', 'adv_multiplier']).set_index('ticker')

def build_ticker_prior_table(universe: pd.DataFrame, priors: pd.DataFrame | None=None, overlays: pd.DataFrame | None=None) -> pd.DataFrame:
    priors = priors if priors is not None else build_synthetic_priors()
    overlays = overlays if overlays is not None else build_ticker_prior_overlays()
    rows: list[dict[str, float | str | bool]] = []
    for ticker, metadata in universe.iterrows():
        asset_class = str(metadata['asset_class'])
        prior = priors.loc[asset_class]
        if ticker in overlays.index:
            overlay = overlays.loc[ticker]
            applied = True
        else:
            overlay = pd.Series({'growth_shift': 0.0, 'income_shift': 0.0, 'volatility_multiplier': 1.0, 'factor_loading_multiplier': 1.0, 'adv_multiplier': 1.0})
            applied = False
        growth = float(prior['annual_growth'] + overlay['growth_shift'])
        income = max(float(prior['income_yield'] + overlay['income_shift']), 0.0)
        rows.append({'ticker': ticker, 'asset_class': asset_class, 'ticker_overlay_applied': applied, 'growth_shift': float(overlay['growth_shift']), 'income_shift': float(overlay['income_shift']), 'volatility_multiplier': float(overlay['volatility_multiplier']), 'factor_loading_multiplier': float(overlay['factor_loading_multiplier']), 'adv_multiplier': float(overlay['adv_multiplier']), 'annual_growth_prior': growth, 'income_yield_prior': income, 'expected_total_return_prior': growth + income, 'annual_volatility_prior': float(prior['annual_volatility'] * overlay['volatility_multiplier']), 'linear_cost_bps_prior': float(prior['linear_cost_bps']), 'adv_usd_prior': float(prior['adv_usd_millions'] * 1000000.0 * overlay['adv_multiplier'])})
    return pd.DataFrame(rows).set_index('ticker').reindex(universe.index)

def _factor_specification() -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    factors = ['Equity', 'Duration', 'Credit', 'Inflation', 'USD', 'Trend']
    factor_vol = pd.Series([0.16, 0.08, 0.1, 0.15, 0.09, 0.1], index=factors)
    factor_corr = pd.DataFrame([[1.0, -0.2, 0.55, 0.2, -0.2, 0.05], [-0.2, 1.0, -0.1, -0.25, 0.1, 0.0], [0.55, -0.1, 1.0, 0.15, -0.1, 0.05], [0.2, -0.25, 0.15, 1.0, -0.25, 0.1], [-0.2, 0.1, -0.1, -0.25, 1.0, 0.0], [0.05, 0.0, 0.05, 0.1, 0.0, 1.0]], index=factors, columns=factors)
    factor_cov = pd.DataFrame(np.outer(factor_vol, factor_vol) * factor_corr, index=factors, columns=factors)
    loadings = pd.DataFrame({'US Equity': [1.0, 0.0, 0.1, 0.05, -0.05, 0.05], 'Developed Equity': [0.9, 0.0, 0.1, 0.05, -0.2, 0.05], 'Emerging Equity': [1.1, 0.0, 0.2, 0.1, -0.2, 0.05], 'Core Bonds': [0.0, 0.7, 0.1, -0.1, 0.0, 0.0], 'Government Bonds': [0.0, 1.0, 0.0, -0.2, 0.0, 0.0], 'Inflation Linked': [0.0, 0.65, 0.0, 0.45, 0.0, 0.0], 'Credit': [0.2, 0.3, 0.8, -0.05, 0.0, 0.0], 'Real Estate': [0.75, -0.25, 0.25, 0.25, -0.05, 0.0], 'Precious Metals': [0.05, 0.0, 0.0, 0.8, -0.4, 0.1], 'Broad Commodities': [0.2, 0.0, 0.1, 1.0, -0.3, 0.1], 'Real-Asset Sectors': [0.75, -0.05, 0.25, 0.45, -0.1, 0.0], 'Cash': [0.0, 0.05, 0.0, 0.0, 0.0, 0.0], 'FX': [0.0, 0.0, 0.0, 0.05, 1.0, 0.0], 'Alternatives': [0.2, 0.0, 0.1, 0.15, 0.0, 1.0], 'Preferred': [0.45, 0.2, 0.45, -0.05, 0.0, 0.0]}, index=factors).T
    return (factors, factor_cov, loadings)

def _multivariate_student_t(rng: np.random.Generator, target_covariance: NDArray[np.float64], df: float, n_samples: int) -> NDArray[np.float64]:
    if df <= 2:
        raise ValueError('Student-t degrees of freedom must exceed two.')
    gaussian_covariance = target_covariance * (df - 2.0) / df
    z = rng.multivariate_normal(np.zeros(target_covariance.shape[0]), gaussian_covariance, n_samples)
    u = rng.chisquare(df, size=n_samples)
    return z / np.sqrt(u[:, None] / df)

def _aggregate_class_returns(returns: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    class_returns: dict[str, pd.Series] = {}
    metadata = universe.reindex(returns.columns)
    for asset_class, members in metadata.groupby('asset_class'):
        columns = members.index.intersection(returns.columns)
        class_returns[str(asset_class)] = returns.loc[:, columns].mean(axis=1, skipna=True)
    return pd.DataFrame(class_returns, index=returns.index).sort_index()

def _build_cost_table_from_volatility(annual_volatility: pd.Series, linear_cost_bps: pd.Series, adv_usd: pd.Series, config: PipelineConfig, *, scenario_name: str='base', portfolio_value_usd: float | None=None, execution_days: int | None=None, reference_trade_weight: float | None=None, impact_eta: float | None=None, adv_multiplier: float=1.0) -> pd.DataFrame:
    portfolio_value = config.portfolio_value_usd if portfolio_value_usd is None else float(portfolio_value_usd)
    days = config.execution_days if execution_days is None else int(execution_days)
    trade_weight = config.reference_trade_weight if reference_trade_weight is None else float(reference_trade_weight)
    eta = config.impact_eta if impact_eta is None else float(impact_eta)
    if portfolio_value <= 0 or days <= 0 or trade_weight <= 0 or (eta < 0) or (adv_multiplier <= 0):
        raise ValueError('Cost-scenario parameters must be economically valid.')
    effective_adv = adv_usd.astype(float) * float(adv_multiplier)
    trade_dollars = portfolio_value * trade_weight
    participation = trade_dollars / (days * effective_adv)
    daily_volatility = annual_volatility / math.sqrt(config.annualization)
    impact_rate = eta * daily_volatility * np.sqrt(participation)
    gamma_diag = impact_rate / trade_weight
    return pd.DataFrame({'cost_scenario': scenario_name, 'portfolio_value_usd': portfolio_value, 'execution_days': days, 'reference_trade_weight': trade_weight, 'impact_eta_assumption': eta, 'adv_multiplier_assumption': adv_multiplier, 'linear_cost_bps': linear_cost_bps, 'adv_usd': effective_adv, 'daily_volatility_for_impact': daily_volatility, 'reference_participation': participation, 'sqrt_impact_bps': impact_rate * 10000.0, 'all_in_reference_cost_bps': linear_cost_bps + impact_rate * 10000.0, 'quadratic_impact_gamma': gamma_diag}, index=annual_volatility.index)

def build_cost_sensitivity_tables(annual_volatility: pd.Series, linear_cost_bps: pd.Series, adv_usd: pd.Series, config: PipelineConfig | None=None) -> dict[str, pd.DataFrame]:
    config = config or PipelineConfig()
    base = _build_cost_table_from_volatility(annual_volatility, linear_cost_bps, adv_usd, config, scenario_name='base')
    institutional = _build_cost_table_from_volatility(annual_volatility, linear_cost_bps, adv_usd, config, scenario_name='institutional_high_participation', portfolio_value_usd=config.institutional_portfolio_value_usd, execution_days=config.institutional_execution_days, reference_trade_weight=config.institutional_reference_trade_weight, impact_eta=config.institutional_impact_eta, adv_multiplier=config.institutional_adv_multiplier)
    return {'base': base, 'institutional_high_participation': institutional}

def validate_cost_sensitivity_tables(tables: Mapping[str, pd.DataFrame]) -> dict[str, bool | float]:
    required = {'base', 'institutional_high_participation'}
    names_ok = required.issubset(tables)
    if not names_ok:
        return {'cost_sensitivity_cases_present': False, 'cost_sensitivity_finite': False, 'institutional_participation_higher': False, 'institutional_impact_higher': False, 'cost_sensitivity_all_checks_pass': False}
    base = tables['base']
    inst = tables['institutional_high_participation'].reindex(base.index)
    columns = ['reference_participation', 'sqrt_impact_bps', 'all_in_reference_cost_bps', 'quadratic_impact_gamma']
    finite = bool(np.isfinite(base[columns].to_numpy(dtype=float)).all() and np.isfinite(inst[columns].to_numpy(dtype=float)).all())
    participation_higher = bool(finite and (inst['reference_participation'] > base['reference_participation']).all())
    impact_higher = bool(finite and (inst['sqrt_impact_bps'] > base['sqrt_impact_bps']).all())
    ratio = float((inst['reference_participation'] / base['reference_participation']).median()) if finite else float('nan')
    return {'cost_sensitivity_cases_present': names_ok, 'cost_sensitivity_finite': finite, 'institutional_participation_higher': participation_higher, 'institutional_impact_higher': impact_higher, 'median_participation_ratio_institutional_to_base': ratio, 'cost_sensitivity_all_checks_pass': bool(names_ok and finite and participation_higher and impact_higher)}

def simulate_synthetic_data(universe: pd.DataFrame, config: PipelineConfig | None=None, priors: pd.DataFrame | None=None) -> SyntheticEstimates:
    config = config or PipelineConfig()
    priors = priors if priors is not None else build_synthetic_priors()
    overlays = build_ticker_prior_overlays()
    ticker_priors = build_ticker_prior_table(universe, priors, overlays)
    missing_classes = sorted(set(universe['asset_class']) - set(priors.index))
    if missing_classes:
        raise ValueError(f'Synthetic priors missing classes: {missing_classes}')
    rng = np.random.default_rng(config.synthetic_seed)
    factors, factor_cov_annual, class_loadings = _factor_specification()
    factor_shocks = _multivariate_student_t(rng, factor_cov_annual.to_numpy() / config.annualization, config.synthetic_df, config.synthetic_days)
    tickers = universe.index.tolist()
    n_assets = len(tickers)
    loadings_array = np.zeros((n_assets, len(factors)))
    growth_target = np.zeros(n_assets)
    income_target = np.zeros(n_assets)
    total_target = np.zeros(n_assets)
    target_vol = np.zeros(n_assets)
    idio_daily_vol = np.zeros(n_assets)
    linear_cost_bps = np.zeros(n_assets)
    adv_usd = np.zeros(n_assets)
    class_tilts: dict[str, dict[str, float]] = {}
    for asset_class, members in universe.groupby('asset_class'):
        draws = rng.normal(0.0, 0.008, size=len(members))
        draws -= draws.mean()
        class_tilts[str(asset_class)] = dict(zip(members.index, draws, strict=True))
    factor_cov_array = factor_cov_annual.to_numpy()
    for i, ticker in enumerate(tickers):
        asset_class = str(universe.loc[ticker, 'asset_class'])
        prior = priors.loc[asset_class]
        ticker_prior = ticker_priors.loc[ticker]
        income_target[i] = max(float(ticker_prior['income_yield_prior'] + rng.normal(0.0, 0.0015)), 0.0)
        growth_target[i] = float(ticker_prior['annual_growth_prior'] + class_tilts[asset_class][ticker])
        total_target[i] = growth_target[i] + income_target[i]
        loading = class_loadings.loc[asset_class].to_numpy(dtype=float) * float(ticker_prior['factor_loading_multiplier']) + rng.normal(0.0, 0.03, size=len(factors))
        if ticker == 'FXE':
            loading[factors.index('USD')] = -1.0
        elif ticker == 'UUP':
            loading[factors.index('USD')] = 1.0
        asset_target_vol = float(ticker_prior['annual_volatility_prior'] * np.exp(rng.normal(0.0, 0.04)))
        factor_variance = float(loading @ factor_cov_array @ loading)
        cap = config.synthetic_factor_variance_cap * asset_target_vol ** 2
        if factor_variance > cap and factor_variance > 0:
            loading *= math.sqrt(cap / factor_variance)
            factor_variance = float(loading @ factor_cov_array @ loading)
        floor = max(config.synthetic_idio_vol_floor_abs, config.synthetic_idio_vol_floor_fraction * asset_target_vol)
        idiosyncratic_variance = max(asset_target_vol ** 2 - factor_variance, floor ** 2)
        loadings_array[i] = loading
        target_vol[i] = asset_target_vol
        idio_daily_vol[i] = math.sqrt(idiosyncratic_variance / config.annualization)
        linear_cost_bps[i] = float(prior['linear_cost_bps'] * np.exp(rng.normal(0.0, 0.2)))
        adv_usd[i] = float(ticker_prior['adv_usd_prior'] * np.exp(rng.normal(0.0, 0.3)))
    idiosyncratic = rng.standard_t(config.synthetic_df, size=(config.synthetic_days, n_assets))
    idiosyncratic *= math.sqrt((config.synthetic_df - 2.0) / config.synthetic_df)
    idiosyncratic *= idio_daily_vol
    returns_array = total_target[None, :] / config.annualization + factor_shocks @ loadings_array.T + idiosyncratic
    returns_array = np.clip(returns_array, -0.95, None)
    dates = pd.bdate_range('2000-01-03', periods=config.synthetic_days)
    returns = pd.DataFrame(returns_array, index=dates, columns=tickers)
    prices = 100.0 * (1.0 + returns).cumprod()
    covariance = pd.DataFrame(_nearest_psd(returns.cov().to_numpy() * config.annualization), index=tickers, columns=tickers)
    correlation = covariance_to_correlation(covariance)
    optimizer_volatility = pd.Series(np.sqrt(np.diag(covariance.to_numpy())), index=tickers, name='annual_volatility_optimizer')
    raw_volatility = returns.std(ddof=1) * math.sqrt(config.annualization)
    total_realized = returns.mean() * config.annualization
    growth_realized = total_realized - pd.Series(income_target, index=tickers)
    cost_sensitivity_tables = build_cost_sensitivity_tables(optimizer_volatility, pd.Series(linear_cost_bps, index=tickers), pd.Series(adv_usd, index=tickers), config)
    cost_table = cost_sensitivity_tables['base']
    asset_stats = pd.DataFrame({'expected_growth_return_target': growth_target, 'income_yield_target': income_target, 'expected_total_return_target': total_target, 'expected_growth_return_realized': growth_realized, 'expected_total_return_realized': total_realized, 'annual_volatility_target': target_vol, 'annual_volatility_raw': raw_volatility, 'annual_volatility_optimizer': optimizer_volatility, 'observations': returns.notna().sum(), 'ticker_overlay_applied': ticker_priors['ticker_overlay_applied'].to_numpy(), 'growth_shift_overlay': ticker_priors['growth_shift'].to_numpy(), 'income_shift_overlay': ticker_priors['income_shift'].to_numpy(), 'volatility_multiplier_overlay': ticker_priors['volatility_multiplier'].to_numpy(), 'factor_loading_multiplier_overlay': ticker_priors['factor_loading_multiplier'].to_numpy(), 'adv_multiplier_overlay': ticker_priors['adv_multiplier'].to_numpy()}, index=tickers).join(cost_table[['linear_cost_bps', 'quadratic_impact_gamma', 'adv_usd']])
    asset_stats = asset_stats.join(universe[['asset_class', 'description']])
    class_returns = _aggregate_class_returns(returns, universe)
    class_covariance = pd.DataFrame(_nearest_psd(class_returns.cov().to_numpy() * config.annualization), index=class_returns.columns, columns=class_returns.columns)
    class_optimizer_vol = pd.Series(np.sqrt(np.diag(class_covariance.to_numpy())), index=class_covariance.index)
    grouped = asset_stats.groupby('asset_class')
    class_stats = pd.DataFrame({'expected_total_return_target': grouped['expected_total_return_target'].mean(), 'expected_total_return_realized': class_returns.mean() * config.annualization, 'annual_volatility_raw': class_returns.std(ddof=1) * math.sqrt(config.annualization), 'annual_volatility_optimizer': class_optimizer_vol, 'mean_income_yield_target': grouped['income_yield_target'].mean(), 'mean_linear_cost_bps': grouped['linear_cost_bps'].mean(), 'constituents': grouped.size()})
    return SyntheticEstimates(daily_returns=returns, prices=prices, asset_stats=asset_stats, covariance=covariance, correlation=correlation, class_returns=class_returns, class_stats=class_stats.sort_index(), class_correlation=covariance_to_correlation(class_covariance), cost_table=cost_table, cost_sensitivity_tables=cost_sensitivity_tables, factor_returns=pd.DataFrame(factor_shocks, index=dates, columns=factors), factor_loadings=pd.DataFrame(loadings_array, index=tickers, columns=factors))

def _download_yfinance_once(yf: object, tickers: Sequence[str], config: PipelineConfig) -> pd.DataFrame:
    return yf.download(tickers=list(tickers), start=config.start, end=config.end, interval=config.interval, actions=True, threads=True, group_by='column', auto_adjust=False, repair=True, keepna=False, progress=False, timeout=config.download_timeout_seconds, multi_level_index=True)

def download_yfinance_bundle(universe: pd.DataFrame, config: PipelineConfig | None=None) -> MarketDataBundle:
    config = config or PipelineConfig()
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError('Install yfinance before running the historical branch.') from exc
    tickers = universe.index.tolist()
    raw: pd.DataFrame | None = None
    last_error: Exception | None = None
    for attempt in range(config.download_retries):
        try:
            raw = _download_yfinance_once(yf, tickers, config)
            if raw is not None and (not raw.empty):
                break
        except Exception as exc:
            last_error = exc
        time.sleep(2.0 * (attempt + 1))
    if raw is None or raw.empty:
        raise RuntimeError(f'yfinance batch download failed: {last_error}')
    fields = {'adj_close': _extract_download_field(raw, 'Adj Close', tickers), 'close': _extract_download_field(raw, 'Close', tickers), 'high': _extract_download_field(raw, 'High', tickers), 'low': _extract_download_field(raw, 'Low', tickers), 'volume': _extract_download_field(raw, 'Volume', tickers), 'dividends': _extract_download_field(raw, 'Dividends', tickers)}
    if fields['adj_close'].notna().sum().sum() == 0:
        adjusted_raw = yf.download(tickers=tickers, start=config.start, end=config.end, interval=config.interval, actions=False, threads=True, group_by='column', auto_adjust=True, repair=True, keepna=False, progress=False, timeout=config.download_timeout_seconds, multi_level_index=True)
        fields['adj_close'] = _extract_download_field(adjusted_raw, 'Close', tickers)
    counts = fields['adj_close'].notna().sum()
    weak = counts[counts < config.min_observations].index.tolist()
    for ticker in weak:
        single_raw: pd.DataFrame | None = None
        for attempt in range(config.download_retries):
            try:
                single_raw = _download_yfinance_once(yf, [ticker], config)
                if single_raw is not None and (not single_raw.empty):
                    break
            except Exception:
                pass
            time.sleep(2.0 * (attempt + 1))
        if single_raw is None or single_raw.empty:
            continue
        for name, field in [('adj_close', 'Adj Close'), ('close', 'Close'), ('high', 'High'), ('low', 'Low'), ('volume', 'Volume'), ('dividends', 'Dividends')]:
            extracted = _extract_download_field(single_raw, field, [ticker])
            fields[name] = _merge_field(fields[name], extracted)
    counts = fields['adj_close'].notna().sum()
    invalid = counts[counts < config.min_observations].index.tolist()
    if invalid and config.require_full_universe:
        raise RuntimeError('The following tickers failed the minimum-history rule after retries: ' + ', '.join(invalid))
    valid = [ticker for ticker in tickers if ticker not in invalid]
    if invalid:
        warnings.warn('Removed insufficient-history tickers: ' + ', '.join(invalid), stacklevel=2)
    for name in fields:
        fields[name] = fields[name].reindex(columns=valid).sort_index()
    return MarketDataBundle(invalid_tickers=invalid, **fields)

def prepare_historical_returns(bundle: MarketDataBundle, config: PipelineConfig | None=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or PipelineConfig()
    raw_returns = bundle.adj_close.pct_change(fill_method=None)
    raw_returns = raw_returns.replace([np.inf, -np.inf], np.nan)
    counts = raw_returns.notna().sum()
    valid = counts[counts >= config.min_observations].index
    raw_returns = raw_returns.loc[:, valid]
    if config.require_full_universe and len(valid) != 50:
        missing = sorted(set(bundle.adj_close.columns) - set(valid))
        raise RuntimeError(f'Historical returns do not retain all 50 assets: {missing}')
    robust_returns = _winsorize_columns(raw_returns, config.winsor_tail_probability)
    common_returns = _common_sample(robust_returns, config.min_common_observations)
    return (raw_returns, robust_returns, common_returns)

def estimate_income_yield(dividends: pd.DataFrame, close: pd.DataFrame, as_of: pd.Timestamp | None=None) -> pd.Series:
    if close.empty:
        raise ValueError('Raw close prices are required.')
    as_of = pd.Timestamp(close.index.max()) if as_of is None else pd.Timestamp(as_of)
    cutoff = as_of - pd.Timedelta(days=365)
    trailing_dividends = dividends.loc[(dividends.index > cutoff) & (dividends.index <= as_of)].sum(min_count=1)
    latest_close = _last_valid_value(close.loc[:as_of])
    income = trailing_dividends.divide(latest_close.where(latest_close > 0))
    return income.fillna(0.0).clip(lower=0.0).rename('income_yield')

def estimate_historical_cost_table(bundle: MarketDataBundle, robust_returns: pd.DataFrame, optimizer_volatility: pd.Series, config: PipelineConfig | None=None) -> pd.DataFrame:
    config = config or PipelineConfig()
    rows: list[dict[str, float | str]] = []
    commission_fraction = config.commission_bps / 10000.0
    half_floor_fraction = config.half_spread_floor_bps / 10000.0
    for ticker in robust_returns.columns:
        close = bundle.close[ticker].dropna()
        high = bundle.high[ticker]
        low = bundle.low[ticker]
        volume = bundle.volume[ticker]
        aligned_close = bundle.close[ticker]
        high_low = ((high - low) / aligned_close.where(aligned_close > 0)).replace([np.inf, -np.inf], np.nan)
        median_range = float(high_low.median(skipna=True))
        median_range = median_range if np.isfinite(median_range) else 0.0
        roll = _roll_spread_fraction(close)
        roll = roll if np.isfinite(roll) else 0.0
        range_spread = config.high_low_to_spread_fraction * max(median_range, 0.0)
        full_spread = max(roll, range_spread, 2.0 * half_floor_fraction)
        half_spread = 0.5 * full_spread
        linear_rate = half_spread + commission_fraction
        dollar_volume = (aligned_close * volume).replace([np.inf, -np.inf], np.nan).dropna()
        recent_dollar_volume = dollar_volume.tail(config.adv_window_days)
        adv = float(recent_dollar_volume.median()) if not recent_dollar_volume.empty else np.nan
        annual_vol = float(optimizer_volatility.loc[ticker])
        daily_vol = annual_vol / math.sqrt(config.annualization)
        trade_dollars = config.portfolio_value_usd * config.reference_trade_weight
        denominator = config.execution_days * adv if np.isfinite(adv) else np.nan
        participation = trade_dollars / denominator if np.isfinite(denominator) and denominator > 0 else np.nan
        impact_rate = config.impact_eta * daily_vol * math.sqrt(participation) if np.isfinite(participation) and participation >= 0 else np.nan
        gamma_diag = impact_rate / config.reference_trade_weight if np.isfinite(impact_rate) else np.nan
        rows.append({'ticker': ticker, 'cost_data_type': 'proxy_from_OHLCV_plus_assumptions', 'roll_full_spread_bps': roll * 10000.0, 'median_high_low_range_bps': median_range * 10000.0, 'range_to_spread_fraction_assumption': config.high_low_to_spread_fraction, 'full_spread_proxy_bps': full_spread * 10000.0, 'half_spread_proxy_bps': half_spread * 10000.0, 'commission_bps_assumption': config.commission_bps, 'linear_cost_bps': linear_rate * 10000.0, 'adv_usd': adv, 'annual_volatility_for_impact': annual_vol, 'reference_trade_weight': config.reference_trade_weight, 'impact_eta_assumption': config.impact_eta, 'reference_participation': participation, 'sqrt_impact_bps': impact_rate * 10000.0 if np.isfinite(impact_rate) else np.nan, 'all_in_reference_cost_bps': (linear_rate + impact_rate) * 10000.0 if np.isfinite(impact_rate) else np.nan, 'quadratic_impact_gamma': gamma_diag})
    return pd.DataFrame(rows).set_index('ticker')

def estimate_historical_statistics(universe: pd.DataFrame, bundle: MarketDataBundle, config: PipelineConfig | None=None, priors: pd.DataFrame | None=None) -> EmpiricalEstimates:
    config = config or PipelineConfig()
    priors = priors if priors is not None else build_synthetic_priors()
    raw_returns, robust_returns, common_returns = prepare_historical_returns(bundle, config)
    ewma_total_raw = _ewma_mean(robust_returns, config.ewma_halflife_days) * config.annualization
    class_prior_total = universe.loc[robust_returns.columns, 'asset_class'].map((priors['annual_growth'] + priors['income_yield']).to_dict())
    ticker_prior_table = build_ticker_prior_table(universe, priors).loc[robust_returns.columns]
    ticker_prior_total = ticker_prior_table['expected_total_return_prior']
    expected_total = config.empirical_return_weight * ewma_total_raw + (1.0 - config.empirical_return_weight) * ticker_prior_total
    income_yield = estimate_income_yield(bundle.dividends.reindex(columns=robust_returns.columns), bundle.close.reindex(columns=robust_returns.columns))
    expected_growth = expected_total - income_yield
    cagr = _cagr_from_adjusted_prices(bundle.adj_close.reindex(columns=robust_returns.columns))
    covariance_array = _nearest_psd(LedoitWolf(assume_centered=False).fit(common_returns.to_numpy(dtype=float)).covariance_ * config.annualization)
    covariance = pd.DataFrame(covariance_array, index=common_returns.columns, columns=common_returns.columns)
    correlation = covariance_to_correlation(covariance)
    optimizer_volatility = pd.Series(np.sqrt(np.diag(covariance_array)), index=covariance.index, name='annual_volatility_optimizer')
    raw_volatility = raw_returns.std(ddof=1) * math.sqrt(config.annualization)
    cost_table = estimate_historical_cost_table(bundle, robust_returns, optimizer_volatility, config)
    cost_sensitivity_tables = build_cost_sensitivity_tables(optimizer_volatility, cost_table['linear_cost_bps'], cost_table['adv_usd'], config)
    cost_sensitivity_tables['base'] = cost_table.copy()
    asset_stats = pd.concat([ewma_total_raw.rename('expected_total_return_ewma_raw'), class_prior_total.rename('expected_total_return_class_prior'), ticker_prior_total.rename('expected_total_return_ticker_prior'), expected_total.rename('expected_total_return_shrunk'), expected_growth.rename('expected_growth_return'), income_yield, cagr, raw_volatility.rename('annual_volatility_raw'), optimizer_volatility, raw_returns.notna().sum().rename('observations'), cost_table[['linear_cost_bps', 'quadratic_impact_gamma', 'adv_usd']]], axis=1).join(universe[['asset_class', 'description']])
    class_returns = _aggregate_class_returns(robust_returns, universe)
    class_common = _common_sample(class_returns, min(config.min_common_observations, len(class_returns)))
    class_covariance = pd.DataFrame(_nearest_psd(LedoitWolf(assume_centered=False).fit(class_common.to_numpy()).covariance_ * config.annualization), index=class_common.columns, columns=class_common.columns)
    class_optimizer_vol = pd.Series(np.sqrt(np.diag(class_covariance.to_numpy())), index=class_covariance.index)
    grouped = asset_stats.groupby('asset_class')
    class_stats = pd.DataFrame({'expected_total_return_shrunk': grouped['expected_total_return_shrunk'].mean(), 'annual_volatility_raw': class_returns.std(ddof=1) * math.sqrt(config.annualization), 'annual_volatility_optimizer': class_optimizer_vol, 'mean_income_yield': grouped['income_yield'].mean(), 'mean_linear_cost_bps': grouped['linear_cost_bps'].mean(), 'constituents': grouped.size()})
    return EmpiricalEstimates(raw_daily_returns=raw_returns, robust_daily_returns=robust_returns, common_returns=common_returns, asset_stats=asset_stats, covariance=covariance, correlation=correlation, class_returns=class_returns, class_stats=class_stats.sort_index(), class_correlation=covariance_to_correlation(class_covariance), cost_table=cost_table, cost_sensitivity_tables=cost_sensitivity_tables)

def build_optimizer_inputs(empirical: EmpiricalEstimates, synthetic: SyntheticEstimates, empirical_weight: float=0.7) -> dict[str, pd.DataFrame | pd.Series]:
    if not 0.0 <= empirical_weight <= 1.0:
        raise ValueError('empirical_weight must lie in [0, 1].')
    common = [ticker for ticker in empirical.covariance.index if ticker in synthetic.covariance.index]
    if not common:
        raise ValueError('No common tickers.')
    w = empirical_weight
    growth = w * empirical.asset_stats.loc[common, 'expected_growth_return'] + (1.0 - w) * synthetic.asset_stats.loc[common, 'expected_growth_return_target']
    income = w * empirical.asset_stats.loc[common, 'income_yield'] + (1.0 - w) * synthetic.asset_stats.loc[common, 'income_yield_target']
    covariance = pd.DataFrame(_nearest_psd(w * empirical.covariance.loc[common, common].to_numpy() + (1.0 - w) * synthetic.covariance.loc[common, common].to_numpy()), index=common, columns=common)
    linear_cost = (w * empirical.cost_table.loc[common, 'linear_cost_bps'] + (1.0 - w) * synthetic.cost_table.loc[common, 'linear_cost_bps']) / 10000.0
    gamma_diag = w * empirical.cost_table.loc[common, 'quadratic_impact_gamma'] + (1.0 - w) * synthetic.cost_table.loc[common, 'quadratic_impact_gamma']
    gamma = pd.DataFrame(np.diag(gamma_diag), index=common, columns=common)
    return {'growth_vector_g': growth.rename('g'), 'income_vector_d': income.rename('d'), 'covariance_Sigma': covariance, 'correlation': covariance_to_correlation(covariance), 'linear_cost_vector_c': linear_cost.rename('c'), 'impact_matrix_Gamma': gamma}

def validate_estimates(covariance: pd.DataFrame, correlation: pd.DataFrame, cost_table: pd.DataFrame, asset_stats: pd.DataFrame | None=None, expected_assets: int | None=None, tolerance: float=1e-08) -> dict[str, float | bool | int]:
    cov = covariance.to_numpy(dtype=float)
    corr = correlation.to_numpy(dtype=float)
    cost_columns = ['linear_cost_bps', 'quadratic_impact_gamma', 'adv_usd']
    cost_columns_present = all((column in cost_table.columns for column in cost_columns))
    cost_values = cost_table[cost_columns].to_numpy(dtype=float) if cost_columns_present else np.array([[np.nan]])
    covariance_finite = bool(np.isfinite(cov).all())
    correlation_finite = bool(np.isfinite(corr).all())
    costs_finite = bool(np.isfinite(cost_values).all())
    symmetry_error = float(np.max(np.abs(cov - cov.T))) if covariance_finite else float('inf')
    minimum_eigenvalue = float(np.linalg.eigvalsh(0.5 * (cov + cov.T)).min()) if covariance_finite else float('-inf')
    diagonal_error = float(np.max(np.abs(np.diag(corr) - 1.0))) if correlation_finite else float('inf')
    correlation_bound = float(np.max(np.abs(corr))) if correlation_finite else float('inf')
    costs_nonnegative = bool(costs_finite and (cost_table['linear_cost_bps'] >= 0).all() and (cost_table['quadratic_impact_gamma'] >= 0).all() and (cost_table['adv_usd'] > 0).all())
    labels_match = bool(covariance.index.equals(covariance.columns) and correlation.index.equals(correlation.columns) and covariance.index.equals(correlation.index) and covariance.index.equals(cost_table.index))
    asset_count = len(covariance)
    expected_asset_count_ok = expected_assets is None or asset_count == expected_assets
    stats_finite = True
    optimizer_volatility_consistent = True
    if asset_stats is not None:
        numerical = asset_stats.select_dtypes(include=[np.number])
        stats_finite = bool(np.isfinite(numerical.to_numpy(dtype=float)).all())
        if 'annual_volatility_optimizer' in asset_stats.columns:
            expected_vol = np.sqrt(np.diag(cov))
            actual_vol = asset_stats.loc[covariance.index, 'annual_volatility_optimizer'].to_numpy()
            optimizer_volatility_consistent = bool(np.allclose(actual_vol, expected_vol, rtol=1e-10, atol=1e-12))
    checks: dict[str, float | bool | int] = {'asset_count': asset_count, 'expected_asset_count_ok': expected_asset_count_ok, 'matrix_labels_match': labels_match, 'covariance_finite': covariance_finite, 'correlation_finite': correlation_finite, 'cost_columns_present': cost_columns_present, 'costs_finite': costs_finite, 'covariance_symmetric': symmetry_error <= tolerance, 'covariance_symmetry_error': symmetry_error, 'covariance_psd': minimum_eigenvalue >= -tolerance, 'minimum_covariance_eigenvalue': minimum_eigenvalue, 'correlation_unit_diagonal': diagonal_error <= tolerance, 'correlation_diagonal_error': diagonal_error, 'correlation_within_bounds': correlation_bound <= 1.0 + tolerance, 'maximum_absolute_correlation': correlation_bound, 'costs_nonnegative': costs_nonnegative, 'asset_stats_finite': stats_finite, 'optimizer_volatility_consistent': optimizer_volatility_consistent}
    boolean_checks = [value for value in checks.values() if isinstance(value, bool)]
    checks['all_checks_pass'] = bool(all(boolean_checks))
    return checks

def save_synthetic_outputs(estimates: SyntheticEstimates, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    estimates.daily_returns.to_csv(output / 'synthetic_daily_returns.csv')
    estimates.prices.to_csv(output / 'synthetic_prices.csv')
    estimates.asset_stats.to_csv(output / 'synthetic_asset_statistics.csv')
    estimates.covariance.to_csv(output / 'synthetic_covariance.csv')
    estimates.correlation.to_csv(output / 'synthetic_correlation.csv')
    estimates.class_returns.to_csv(output / 'synthetic_asset_class_returns.csv')
    estimates.class_stats.to_csv(output / 'synthetic_asset_class_statistics.csv')
    estimates.class_correlation.to_csv(output / 'synthetic_asset_class_correlation.csv')
    estimates.cost_table.to_csv(output / 'synthetic_cost_estimates.csv')
    combined_cost_cases = []
    for name, table in estimates.cost_sensitivity_tables.items():
        table.to_csv(output / f'synthetic_cost_estimates_{name}.csv')
        combined_cost_cases.append(table.assign(cost_scenario=name))
    pd.concat(combined_cost_cases).to_csv(output / 'synthetic_transaction_cost_sensitivity.csv')
    estimates.factor_returns.to_csv(output / 'synthetic_factor_returns.csv')
    estimates.factor_loadings.to_csv(output / 'synthetic_factor_loadings.csv')

def save_empirical_outputs(estimates: EmpiricalEstimates, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    estimates.raw_daily_returns.to_csv(output / 'yfinance_daily_returns_raw.csv')
    estimates.robust_daily_returns.to_csv(output / 'yfinance_daily_returns_robust.csv')
    estimates.common_returns.to_csv(output / 'yfinance_common_returns_for_covariance.csv')
    estimates.asset_stats.to_csv(output / 'yfinance_asset_statistics.csv')
    estimates.covariance.to_csv(output / 'yfinance_covariance.csv')
    estimates.correlation.to_csv(output / 'yfinance_correlation.csv')
    estimates.class_returns.to_csv(output / 'yfinance_asset_class_returns.csv')
    estimates.class_stats.to_csv(output / 'yfinance_asset_class_statistics.csv')
    estimates.class_correlation.to_csv(output / 'yfinance_asset_class_correlation.csv')
    estimates.cost_table.to_csv(output / 'yfinance_cost_estimates.csv')
    combined_cost_cases = []
    for name, table in estimates.cost_sensitivity_tables.items():
        table.to_csv(output / f'yfinance_cost_estimates_{name}.csv')
        combined_cost_cases.append(table.assign(cost_scenario=name))
    pd.concat(combined_cost_cases).to_csv(output / 'yfinance_transaction_cost_sensitivity.csv')

def save_optimizer_inputs(inputs: Mapping[str, pd.DataFrame | pd.Series], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, value in inputs.items():
        value.to_csv(output / f'{name}.csv')

def run_synthetic_branch(output_directory: str | Path, config: PipelineConfig | None=None) -> dict[str, object]:
    config = config or PipelineConfig()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    universe = build_asset_universe()
    priors = build_synthetic_priors()
    overlays = build_ticker_prior_overlays()
    estimates = simulate_synthetic_data(universe, config, priors)
    checks = validate_estimates(estimates.covariance, estimates.correlation, estimates.cost_table, estimates.asset_stats, expected_assets=50)
    sensitivity_checks = validate_cost_sensitivity_tables(estimates.cost_sensitivity_tables)
    checks.update(sensitivity_checks)
    checks['all_checks_pass'] = bool(checks['all_checks_pass'] and sensitivity_checks['cost_sensitivity_all_checks_pass'])
    if not checks['all_checks_pass']:
        raise RuntimeError(f'Synthetic validation failed: {checks}')
    universe.to_csv(output / 'asset_universe_50.csv', index=False)
    priors.to_csv(output / 'synthetic_asset_class_priors.csv')
    overlays.to_csv(output / 'synthetic_ticker_prior_overlays.csv')
    build_ticker_prior_table(universe, priors, overlays).to_csv(output / 'synthetic_ticker_prior_table.csv')
    save_synthetic_outputs(estimates, output / 'synthetic')
    (output / 'synthetic_validation.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    (output / 'pipeline_config.json').write_text(json.dumps(asdict(config), indent=2), encoding='utf-8')
    return {'universe': universe, 'synthetic': estimates, 'validation': checks}

def run_yfinance_branch(output_directory: str | Path, config: PipelineConfig | None=None) -> dict[str, object]:
    config = config or PipelineConfig()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    universe = build_asset_universe()
    priors = build_synthetic_priors()
    bundle = download_yfinance_bundle(universe, config)
    estimates = estimate_historical_statistics(universe, bundle, config, priors)
    checks = validate_estimates(estimates.covariance, estimates.correlation, estimates.cost_table, estimates.asset_stats, expected_assets=50 if config.require_full_universe else None)
    sensitivity_checks = validate_cost_sensitivity_tables(estimates.cost_sensitivity_tables)
    checks.update(sensitivity_checks)
    checks['all_checks_pass'] = bool(checks['all_checks_pass'] and sensitivity_checks['cost_sensitivity_all_checks_pass'])
    if not checks['all_checks_pass']:
        raise RuntimeError(f'yfinance validation failed: {checks}')
    universe.to_csv(output / 'asset_universe_50.csv', index=False)
    build_ticker_prior_overlays().to_csv(output / 'yfinance_ticker_prior_overlays.csv')
    build_ticker_prior_table(universe, priors).to_csv(output / 'yfinance_ticker_prior_table.csv')
    save_empirical_outputs(estimates, output / 'yfinance')
    (output / 'yfinance_validation.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    (output / 'pipeline_config.json').write_text(json.dumps(asdict(config), indent=2), encoding='utf-8')
    return {'universe': universe, 'bundle': bundle, 'empirical': estimates, 'validation': checks}

def run_full_pipeline(output_directory: str | Path, config: PipelineConfig | None=None, empirical_weight: float=0.7) -> dict[str, object]:
    config = config or PipelineConfig()
    output = Path(output_directory)
    synthetic_result = run_synthetic_branch(output, config)
    yfinance_result = run_yfinance_branch(output, config)
    optimizer_inputs = build_optimizer_inputs(yfinance_result['empirical'], synthetic_result['synthetic'], empirical_weight)
    save_optimizer_inputs(optimizer_inputs, output / 'optimizer_inputs')
    return {**synthetic_result, **yfinance_result, 'optimizer_inputs': optimizer_inputs}

def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['synthetic', 'yfinance', 'both'], default='both')
    parser.add_argument('--output', default='step3_validated_outputs')
    parser.add_argument('--start', default='2021-01-01')
    parser.add_argument('--end', default=None)
    parser.add_argument('--empirical-weight', type=float, default=0.7)
    parser.add_argument('--synthetic-days', type=int, default=1260)
    parser.add_argument('--synthetic-seed', type=int, default=20260802)
    parser.add_argument('--portfolio-value-usd', type=float, default=10000000.0)
    parser.add_argument('--execution-days', type=int, default=3)
    parser.add_argument('--reference-trade-weight', type=float, default=0.01)
    parser.add_argument('--impact-eta', type=float, default=0.75)
    parser.add_argument('--institutional-portfolio-value-usd', type=float, default=250000000.0)
    parser.add_argument('--institutional-execution-days', type=int, default=2)
    parser.add_argument('--institutional-reference-trade-weight', type=float, default=0.02)
    parser.add_argument('--institutional-impact-eta', type=float, default=0.9)
    return parser.parse_args()

def main() -> None:
    args = _parse_arguments()
    config = PipelineConfig(start=args.start, end=args.end, synthetic_days=args.synthetic_days, synthetic_seed=args.synthetic_seed, portfolio_value_usd=args.portfolio_value_usd, execution_days=args.execution_days, reference_trade_weight=args.reference_trade_weight, impact_eta=args.impact_eta, institutional_portfolio_value_usd=args.institutional_portfolio_value_usd, institutional_execution_days=args.institutional_execution_days, institutional_reference_trade_weight=args.institutional_reference_trade_weight, institutional_impact_eta=args.institutional_impact_eta)
    if args.mode == 'synthetic':
        result = run_synthetic_branch(args.output, config)
    elif args.mode == 'yfinance':
        result = run_yfinance_branch(args.output, config)
    else:
        result = run_full_pipeline(args.output, config, args.empirical_weight)
    print(f'Saved outputs to {Path(args.output).resolve()}')
    if 'validation' in result:
        print(json.dumps(result['validation'], indent=2))
if __name__ == '__main__':
    main()
