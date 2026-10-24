"""Tail analysis of degree and strength distributions (Clauset, Shalizi & Newman, 2009).

For a positive sample the procedure is:

1. choose ``xmin`` by minimising the Kolmogorov-Smirnov distance between the
   empirical tail and the fitted power law (candidate ``xmin`` values are the
   observed values leaving at least ``min_tail`` points, thinned log-uniformly);
2. estimate the exponent by maximum likelihood (discrete: Hurwitz-zeta
   normalisation; continuous: closed form);
3. goodness of fit by the semi-parametric bootstrap (``p >= 0.1`` = plausible);
4. compare the power law on the same tail with log-normal, exponential and
   power law with exponential cutoff: Vuong's normalised log-likelihood ratio for
   non-nested pairs, the chi-square(1) likelihood-ratio test for the nested cutoff.

The result is a description of tail behaviour. A "scale-free" claim is never
produced; the interpretation field uses "heavy-tailed" language only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize, special, stats
from scipy.integrate import quad

ALPHA_GRID = np.arange(1.005, 8.0, 0.0025)


@dataclass
class PowerLawFit:
    xmin: float
    alpha: float
    ks: float
    n_tail: int
    discrete: bool


def _candidates(values: np.ndarray, min_tail: int, max_candidates: int) -> np.ndarray:
    unique = np.unique(values)
    sorted_values = np.sort(values)
    tail_sizes = len(values) - np.searchsorted(sorted_values, unique, side="left")
    unique = unique[tail_sizes >= min_tail]
    if len(unique) > max_candidates:
        positions = np.unique(np.round(np.geomspace(1, len(unique), max_candidates)).astype(int) - 1)
        unique = unique[positions]
    return unique


def _discrete_alpha(tail: np.ndarray, xmin: float, refine: bool) -> float:
    n, log_sum = len(tail), float(np.log(tail).sum())
    loglik = -ALPHA_GRID * log_sum - n * np.log(special.zeta(ALPHA_GRID, xmin))
    best = float(ALPHA_GRID[int(np.nanargmax(loglik))])
    if not refine:
        return best
    result = optimize.minimize_scalar(
        lambda a: a * log_sum + n * math.log(special.zeta(a, xmin)),
        bounds=(max(1.0001, best - 0.01), best + 0.01), method="bounded")
    return float(result.x)


def _ks_discrete(tail: np.ndarray, xmin: float, alpha: float) -> float:
    unique, counts = np.unique(tail, return_counts=True)
    empirical = np.cumsum(counts) / len(tail)
    model = 1.0 - special.zeta(alpha, unique + 1) / special.zeta(alpha, xmin)
    return float(np.max(np.abs(empirical - model)))


def _ks_continuous(tail: np.ndarray, xmin: float, alpha: float) -> float:
    ordered = np.sort(tail)
    n = len(ordered)
    model = 1.0 - (ordered / xmin) ** (1.0 - alpha)
    upper = np.arange(1, n + 1) / n
    lower = np.arange(0, n) / n
    return float(max(np.max(np.abs(upper - model)), np.max(np.abs(model - lower))))


def fit_power_law(values: np.ndarray, *, discrete: bool, min_tail: int, max_candidates: int,
                  refine: bool = True) -> PowerLawFit | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if discrete:
        values = values[values >= 1]
    if len(values) < min_tail:
        return None
    best: PowerLawFit | None = None
    for xmin in _candidates(values, min_tail, max_candidates):
        tail = values[values >= xmin]
        if discrete:
            alpha = _discrete_alpha(tail, xmin, refine=False)
            ks = _ks_discrete(tail, xmin, alpha)
        else:
            log_ratio = np.log(tail / xmin).sum()
            if log_ratio <= 0:
                continue
            alpha = 1.0 + len(tail) / log_ratio
            ks = _ks_continuous(tail, xmin, alpha)
        if best is None or ks < best.ks:
            best = PowerLawFit(float(xmin), float(alpha), ks, int(len(tail)), discrete)
    if best is not None and discrete and refine:
        tail = values[values >= best.xmin]
        best.alpha = _discrete_alpha(tail, best.xmin, refine=True)
        best.ks = _ks_discrete(tail, best.xmin, best.alpha)
    return best


def _sample_power_law(rng: np.random.Generator, size: int, xmin: float, alpha: float,
                      discrete: bool) -> np.ndarray:
    r = rng.random(size)
    if discrete:
        # Clauset et al. (2009), eq. D.6: accurate discrete approximation.
        return np.floor((xmin - 0.5) * (1.0 - r) ** (-1.0 / (alpha - 1.0)) + 0.5)
    return xmin * (1.0 - r) ** (-1.0 / (alpha - 1.0))


def bootstrap_gof(values: np.ndarray, fit: PowerLawFit, *, n_boot: int, seed: int,
                  min_tail: int, max_candidates: int) -> float:
    """Semi-parametric bootstrap p-value: share of synthetic KS distances >= observed."""
    if n_boot <= 0:
        return math.nan
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values >= (1 if fit.discrete else 0)) & (values > 0)]
    body = values[values < fit.xmin]
    n, p_tail = len(values), fit.n_tail / len(values)
    # Synthetic fits use the unrefined grid estimator, so the observed distance must too.
    observed = fit_power_law(values, discrete=fit.discrete, min_tail=min_tail,
                             max_candidates=max_candidates, refine=False)
    threshold = observed.ks if observed is not None else fit.ks
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_boot):
        n_tail = rng.binomial(n, p_tail)
        synthetic = _sample_power_law(rng, n_tail, fit.xmin, fit.alpha, fit.discrete)
        if n - n_tail and len(body):
            synthetic = np.concatenate([synthetic, rng.choice(body, n - n_tail, replace=True)])
        refit = fit_power_law(synthetic, discrete=fit.discrete, min_tail=min_tail,
                              max_candidates=max_candidates, refine=False)
        if refit is not None and refit.ks >= threshold:
            exceed += 1
    return exceed / n_boot


# ----- per-observation log-likelihoods on the tail x >= xmin ------------------------------

def _ll_power_law(tail: np.ndarray, xmin: float, alpha: float, discrete: bool) -> np.ndarray:
    if discrete:
        return -alpha * np.log(tail) - math.log(special.zeta(alpha, xmin))
    return math.log(alpha - 1.0) - math.log(xmin) - alpha * np.log(tail / xmin)


def _ll_exponential(tail: np.ndarray, xmin: float, discrete: bool) -> tuple[np.ndarray, dict]:
    excess = float(np.mean(tail) - xmin)
    if excess <= 0:
        raise ValueError("degenerate tail for exponential fit")
    if discrete:
        lam = math.log1p(1.0 / excess)
        return math.log(-math.expm1(-lam)) - lam * (tail - xmin), {"lambda": lam}
    lam = 1.0 / excess
    return math.log(lam) - lam * (tail - xmin), {"lambda": lam}


def _log_diff_sf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """log(sf(a) - sf(b)) for a < b, stable in the far tail."""
    la, lb = stats.norm.logsf(a), stats.norm.logsf(b)
    return la + np.log1p(-np.exp(np.minimum(lb - la, 0.0)) + 1e-300)


def _ll_lognormal(tail: np.ndarray, xmin: float, discrete: bool) -> tuple[np.ndarray, dict]:
    logs = np.log(tail)

    def per_point(mu: float, sigma: float) -> np.ndarray:
        if discrete:
            lower = (np.log(np.maximum(tail - 0.5, 1e-12)) - mu) / sigma
            upper = (np.log(tail + 0.5) - mu) / sigma
            norm = stats.norm.logsf((math.log(max(xmin - 0.5, 1e-12)) - mu) / sigma)
            return _log_diff_sf(lower, upper) - norm
        z = (logs - mu) / sigma
        norm = stats.norm.logsf((math.log(xmin) - mu) / sigma)
        return stats.norm.logpdf(z) - math.log(sigma) - logs - norm

    def objective(params: np.ndarray) -> float:
        value = -per_point(params[0], math.exp(params[1])).sum()
        return value if np.isfinite(value) else 1e300

    start = np.array([logs.mean(), math.log(max(logs.std(), 0.1))])
    best = None
    for shift in (0.0, -2.0, -5.0):
        result = optimize.minimize(objective, start + np.array([shift, 0.0]), method="Nelder-Mead",
                                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
        if best is None or result.fun < best.fun:
            best = result
    mu, sigma = float(best.x[0]), float(math.exp(best.x[1]))
    return per_point(mu, sigma), {"mu": mu, "sigma": sigma}


def _ll_cutoff(tail: np.ndarray, xmin: float, discrete: bool, alpha0: float) -> tuple[np.ndarray, dict]:
    """Power law with exponential cutoff p(x) ~ x^-alpha exp(-lambda x), on scaled data u = x/xmin."""
    scaled = tail / xmin
    # Integer support k >= xmin, truncated where exp(-lambda x) is negligible (cap documented).
    cap = int(min(max(100 * tail.max(), 1e5), 2e6))
    support = np.arange(int(xmin), int(xmin) + cap, dtype=float) / xmin if discrete else None
    log_support = np.log(support) if discrete else None

    def log_norm(alpha: float, lam: float) -> float:
        if discrete:
            horizon = int(min(math.ceil(45.0 * xmin / lam) + 1, cap))
            terms = -alpha * log_support[:horizon] - lam * (support[:horizon] - 1.0)
            peak = terms.max()
            total = math.exp(peak) * np.exp(terms - peak).sum()
            return math.log(total)
        value, _ = quad(lambda t: math.exp((1 - alpha) * t - lam * (math.exp(t) - 1.0)), 0, np.inf,
                        limit=200)
        return math.log(value)

    def per_point(alpha: float, lam: float) -> np.ndarray:
        return -alpha * np.log(scaled) - lam * (scaled - 1.0) - log_norm(alpha, lam) - (
            0.0 if discrete else math.log(xmin))

    def objective(params: np.ndarray) -> float:
        try:
            value = -per_point(params[0], math.exp(params[1])).sum()
        except (OverflowError, ValueError, ZeroDivisionError):
            return 1e300
        return value if np.isfinite(value) else 1e300

    start_lambda = math.log(max(1.0 / max(scaled.mean(), 1.0), 1e-6))
    best = None
    for start in ([alpha0, start_lambda], [max(alpha0 - 0.5, 0.0), start_lambda - 3.0],
                  [alpha0, -10.0]):
        result = optimize.minimize(objective, np.array(start), method="Nelder-Mead",
                                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
        if best is None or result.fun < best.fun:
            best = result
    alpha, lam = float(best.x[0]), float(math.exp(best.x[1]))
    return per_point(alpha, lam), {"alpha": alpha, "lambda_scaled": lam, "lambda": lam / xmin}


def _vuong(ll_a: np.ndarray, ll_b: np.ndarray) -> tuple[float, float]:
    diff = ll_a - ll_b
    n, sd = len(diff), float(diff.std())
    ratio = float(diff.sum())
    if sd == 0:
        return ratio, math.nan
    return ratio / (math.sqrt(n) * sd), float(special.erfc(abs(ratio) / (math.sqrt(2 * n) * sd)))


def analyse_tail(values: np.ndarray, *, discrete: bool, min_tail: int, max_candidates: int,
                 n_boot: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    result: dict[str, Any] = {"n_positive": int(len(positive)), "n_zero_or_missing": int(len(values) - len(positive)),
                              "discrete": discrete}
    fit = fit_power_law(positive, discrete=discrete, min_tail=min_tail, max_candidates=max_candidates)
    if fit is None:
        result["interpretation"] = f"not assessed: fewer than {min_tail} positive values"
        return result
    tail = positive[positive >= fit.xmin]
    result.update({"xmin": fit.xmin, "alpha": fit.alpha,
                   "alpha_se": (fit.alpha - 1.0) / math.sqrt(fit.n_tail),
                   "n_tail": fit.n_tail, "tail_share": fit.n_tail / len(positive), "ks": fit.ks})
    result["gof_p"] = bootstrap_gof(positive, fit, n_boot=n_boot, seed=seed, min_tail=min_tail,
                                    max_candidates=max_candidates)
    result["gof_bootstrap"] = n_boot
    ll_pl = _ll_power_law(tail, fit.xmin, fit.alpha, discrete)
    favoured = []
    for name, builder in (("lognormal", lambda: _ll_lognormal(tail, fit.xmin, discrete)),
                          ("exponential", lambda: _ll_exponential(tail, fit.xmin, discrete))):
        try:
            ll_alt, params = builder()
            ratio, p_value = _vuong(ll_pl, ll_alt)
            result.update({f"lr_vs_{name}": ratio, f"p_vs_{name}": p_value,
                           **{f"{name}_{k}": v for k, v in params.items()}})
            if p_value < 0.1 and ratio < 0:
                favoured.append(name)
        except (ValueError, FloatingPointError) as exc:
            result[f"lr_vs_{name}"] = math.nan
            result[f"error_vs_{name}"] = str(exc)
    try:
        ll_cut, params = _ll_cutoff(tail, fit.xmin, discrete, fit.alpha)
        statistic = 2.0 * float((ll_cut - ll_pl).sum())
        result.update({"lr_vs_cutoff": float((ll_pl - ll_cut).sum()),
                       "p_vs_cutoff": float(stats.chi2.sf(max(statistic, 0.0), df=1)),
                       **{f"cutoff_{k}": v for k, v in params.items()}})
        if result["p_vs_cutoff"] < 0.1 and result["lr_vs_cutoff"] < 0:
            favoured.append("power law with cutoff")
    except (ValueError, FloatingPointError, OverflowError) as exc:
        result["lr_vs_cutoff"] = math.nan
        result["error_vs_cutoff"] = str(exc)
    result["interpretation"] = _interpret(result.get("gof_p", math.nan), favoured, result)
    return result


def _interpret(gof_p: float, favoured: list[str], result: dict[str, Any]) -> str:
    disfavoured = [name for name in ("lognormal", "exponential")
                   if result.get(f"p_vs_{name}", 1.0) < 0.1 and result.get(f"lr_vs_{name}", 0.0) > 0]
    if math.isfinite(gof_p) and gof_p < 0.1:
        head = "power law rejected by bootstrap GOF"
    else:
        head = "power law not rejected" if math.isfinite(gof_p) else "power law GOF not tested"
    if favoured:
        return f"{head}; heavy-tailed, better described by {', '.join(favoured)}"
    if disfavoured:
        return f"{head}; preferred over {', '.join(disfavoured)}; other alternatives indistinguishable"
    return f"{head}; alternatives statistically indistinguishable (heavy-tailed evidence only)"


def ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical complementary CDF P(X >= x) at each distinct positive value."""
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values) & (values > 0)]
    unique, first = np.unique(values, return_index=True)
    return unique, 1.0 - first / len(values)


def model_ccdf(model: str, x: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    """P(X >= x | X >= xmin) of a fitted tail model, for overlaying on empirical CCDFs.

    ``fit`` is one row of ``analyse_tail`` output; ``model`` is one of
    ``power_law``, ``lognormal``, ``exponential`` or ``cutoff``.
    """
    x = np.asarray(x, dtype=float)
    xmin, discrete = float(fit["xmin"]), bool(fit["discrete"])
    if model == "power_law":
        alpha = float(fit["alpha"])
        if discrete:
            return special.zeta(alpha, x) / special.zeta(alpha, xmin)
        return (x / xmin) ** (1.0 - alpha)
    if model == "exponential":
        return np.exp(-float(fit["exponential_lambda"]) * (x - xmin))
    if model == "lognormal":
        mu, sigma = float(fit["lognormal_mu"]), float(fit["lognormal_sigma"])
        shift = 0.5 if discrete else 0.0
        top = stats.norm.logsf((np.log(np.maximum(x - shift, 1e-12)) - mu) / sigma)
        bottom = stats.norm.logsf((math.log(max(xmin - shift, 1e-12)) - mu) / sigma)
        return np.exp(top - bottom)
    if model == "cutoff":
        alpha, lam = float(fit["cutoff_alpha"]), float(fit["cutoff_lambda_scaled"])
        if discrete:
            horizon = int(min(math.ceil(45.0 * xmin / lam) + 1, 2e6, max(10 * x.max(), 1e5)))
            k = np.arange(int(xmin), int(xmin) + horizon, dtype=float)
            logp = -alpha * np.log(k / xmin) - lam * (k / xmin - 1.0)
            pmf = np.exp(logp - logp.max())
            tail = np.cumsum(pmf[::-1])[::-1]
            index = np.clip(np.round(x).astype(int) - int(xmin), 0, len(k) - 1)
            return tail[index] / tail[0]

        def integral(lower: float) -> float:
            value, _ = quad(lambda t: math.exp((1 - alpha) * t - lam * (math.exp(t) - 1.0)),
                            math.log(lower), np.inf, limit=200)
            return value
        total = integral(1.0)
        return np.array([integral(v / xmin) / total for v in x])
    raise ValueError(f"unknown model {model!r}")
