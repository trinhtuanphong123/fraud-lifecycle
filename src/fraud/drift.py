"""Drift detection without labels.

Effect sizes come from information theory: PSI is the symmetric KL divergence
(J-divergence) between binned distributions, and JS is the Jensen-Shannon
divergence. Significance comes from their large-sample behaviour: for two
samples of sizes n and m from the same distribution,
n_eff * PSI ~ chi2(k - 1) with n_eff = n*m / (n+m), because to second order
PSI equals the chi-square homogeneity statistic divided by n_eff.

With ~20k transactions a week almost every difference is "significant", so an
alarm needs both a significant test (after FDR control across features) and
an effect size above a threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ------------------------------------------------------------- statistics


def psi(reference, current, n_bins: int = 10, eps: float = 1e-4) -> tuple[float, int]:
    """PSI between two numeric samples on quantile bins of the reference; NaN ignored.

    Returns (psi, number of bins). Ties can merge quantiles, so a column with
    few distinct values gets fewer bins.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return np.nan, 0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
    k = len(edges) + 1
    p = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=k) / len(ref)
    q = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=k) / len(cur)
    p, q = p + eps, q + eps
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p))), k


def psi_pvalue(value: float, n: int, m: int, k: int) -> float:
    """Asymptotic p-value of PSI under 'same distribution': n_eff * PSI ~ chi2(k - 1)."""
    if k < 2 or np.isnan(value):
        return 1.0
    return float(stats.chi2.sf(value * n * m / (n + m), df=k - 1))


def categorical_shift(reference, current, top: int = 50) -> dict:
    """JS divergence (nats), chi-square p-value and share of unseen values for a categorical column.

    Categories are the `top` most frequent reference values; the rest are
    pooled into OTHER, and values never seen in the reference into NEW.
    """
    ref = pd.Series(reference).dropna().astype(str)
    cur = pd.Series(current).dropna().astype(str)
    if len(ref) == 0 or len(cur) == 0:
        return {"js": np.nan, "pvalue": 1.0, "new_share": np.nan}
    keep = set(ref.value_counts().index[:top])
    seen = set(ref.unique())

    def label(s):
        return np.where(s.isin(keep), s, np.where(s.isin(seen), "OTHER", "NEW"))

    table = pd.crosstab(
        np.r_[np.zeros(len(ref)), np.ones(len(cur))], np.r_[label(ref), label(cur)]
    ).to_numpy()
    p, q = table[0] / table[0].sum(), table[1] / table[1].sum()
    mix = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.nansum(np.where(p > 0, p * np.log(p / mix), 0.0))
        kl_q = np.nansum(np.where(q > 0, q * np.log(q / mix), 0.0))
    pvalue = stats.chi2_contingency(table)[1] if table.shape[1] > 1 else 1.0
    return {"js": float((kl_p + kl_q) / 2), "pvalue": float(pvalue), "new_share": float((~cur.isin(seen)).mean())}


def ks_pvalue(reference, current) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 1.0
    return float(stats.ks_2samp(ref, cur, method="asymp").pvalue)


def null_rate_pvalue(ref_missing: int, n: int, cur_missing: int, m: int) -> float:
    """Two-proportion z-test for a change in missing rate."""
    pooled = (ref_missing + cur_missing) / (n + m)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n + 1 / m))
    if se == 0:
        return 1.0
    z = (cur_missing / m - ref_missing / n) / se
    return float(2 * stats.norm.sf(abs(z)))


def benjamini_hochberg(pvalues, alpha: float = 0.05) -> np.ndarray:
    """Reject flags controlling the false discovery rate at `alpha`."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    passed = np.nonzero(ranked <= alpha)[0]
    reject = np.zeros(len(p), dtype=bool)
    if len(passed):
        reject[order[: passed.max() + 1]] = True
    return reject


def domain_classifier(reference: pd.DataFrame, current: pd.DataFrame, max_rows: int = 20_000, seed: int = 0) -> dict:
    """Train a model to tell current rows from reference rows (balanced, 2-fold out-of-fold).

    With balanced classes, the best achievable cross-entropy is ln 2 - JS, so
    ln 2 minus the out-of-fold log loss is a lower-bound estimate of the
    multivariate JS divergence between the two periods.
    """
    import lightgbm as lgb

    n = min(len(reference), len(current), max_rows)
    x = pd.concat([reference.sample(n, random_state=seed), current.sample(n, random_state=seed)], ignore_index=True)
    # Categoricals with different category sets concatenate to object; restore a shared category set.
    for col in x.select_dtypes(exclude=["number", "category", "bool"]).columns:
        x[col] = x[col].astype(str).where(x[col].notna()).astype("category")
    y = np.r_[np.zeros(n), np.ones(n)]
    oof = np.zeros(len(y))
    for train_idx, test_idx in StratifiedKFold(2, shuffle=True, random_state=seed).split(x, y):
        model = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.1, random_state=seed, verbose=-1)
        model.fit(x.iloc[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(x.iloc[test_idx])[:, 1]
    ce = log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))
    return {"auc": float(roc_auc_score(y, oof)), "js_estimate": float(max(0.0, np.log(2) - ce))}


# --------------------------------------------------------- weekly monitor


def monitor_week(reference: pd.DataFrame, current: pd.DataFrame, numeric: list[str], categorical: list[str],
                 block_columns: dict[str, str], cfg: dict, run_domain: bool = True, seed: int = 0,
                 run_ks: bool = True, domain_columns: list[str] | None = None) -> dict:
    """All candidate monitors on one (reference, current) pair. Returns one row of alarm counts and effect sizes.

    `domain_columns` restricts the domain classifier (e.g. to columns without a time trend).
    """
    alpha = cfg["fdr"]["alpha"]
    n, m = len(reference), len(current)

    rows = []
    for col in numeric:
        value, k = psi(reference[col], current[col], cfg["psi"]["n_bins"], cfg["psi"]["epsilon"])
        n_obs, m_obs = reference[col].notna().sum(), current[col].notna().sum()
        rows.append({"col": col, "psi": value, "psi_p": psi_pvalue(value, n_obs, m_obs, k),
                     "ks_p": ks_pvalue(reference[col], current[col]) if run_ks else np.nan})
    num = pd.DataFrame(rows).set_index("col")
    num["psi_flag"] = benjamini_hochberg(num["psi_p"], alpha) & (num["psi"] > cfg["psi"]["effect_size_threshold"])

    cat = pd.DataFrame({c: categorical_shift(reference[c], current[c], cfg["categorical"]["top_categories"]) for c in categorical}).T
    cat["flag"] = benjamini_hochberg(cat["pvalue"].astype(float), alpha) & (cat["js"].astype(float) > cfg["categorical"]["js_effect_threshold"])

    blocks = pd.DataFrame({
        name: {
            "shift": current[col].isna().mean() - reference[col].isna().mean(),
            "p": null_rate_pvalue(reference[col].isna().sum(), n, current[col].isna().sum(), m),
        }
        for name, col in block_columns.items()
    }).T
    blocks["flag"] = benjamini_hochberg(blocks["p"], alpha) & (blocks["shift"].abs() > cfg["null_rate"]["effect_threshold"])

    row = {
        "ks_flagged": int((num["ks_p"] < 0.05).sum()),
        "psi_fixed_flagged": int((num["psi"] > cfg["psi"]["fixed_threshold"]).sum()),
        "psi_test_flagged": int(num["psi_flag"].sum()),
        "top_psi": ", ".join(num["psi"].nlargest(3).index),
        "cat_flagged": int(cat["flag"].sum()),
        "cat_max_js": float(cat["js"].astype(float).max()),
        "cat_top": cat["js"].astype(float).idxmax(),
        "max_new_share": float(cat["new_share"].astype(float).max()),
        "blocks_flagged": int(blocks["flag"].sum()),
        "max_block_shift": float(blocks["shift"].abs().max()),
    }
    if run_domain:
        cols = domain_columns or numeric + categorical
        row |= {f"domain_{k}": v for k, v in domain_classifier(reference[cols], current[cols],
                                                                cfg["domain_classifier"]["max_rows"], seed).items()}
    return row


def trend_columns(df: pd.DataFrame, columns: list[str], time_col: str = "TransactionDT",
                  threshold: float = 0.3, sample: int = 50_000, seed: int = 0) -> list[str]:
    """Numeric columns whose rank correlation with time exceeds `threshold` in absolute value.

    Time deltas (days since the card was first used, ...) move with the
    calendar by construction, so a monitor that sees them always finds drift.
    """
    s = df.sample(min(sample, len(df)), random_state=seed)
    rho = s[columns].corrwith(s[time_col], method="spearman")
    return rho[rho.abs() > threshold].index.tolist()


# ---------------------------------------------------------- injected drift

SCENARIOS = ("pipeline", "covariate", "new_category", "prior", "concept")


def inject_drift(week: pd.DataFrame, scenario: str, intensity: float, rng: np.random.Generator,
                 block: list[str] | None = None, category_col: str = "id_31") -> pd.DataFrame:
    """A copy of one week with a known, synthetic change.

    pipeline      a block of columns goes missing on `intensity` of the rows where it was present
    covariate     TransactionAmt and C1 multiplied by (1 + intensity)
    new_category  `intensity` of the non-missing `category_col` values become a never-seen value
    prior         fraud rows duplicated so the fraud rate grows by a factor (1 + intensity)
    concept       labels flipped on `intensity` of the rows; features unchanged, so no
                  label-free monitor can see it (included to show that limit)
    """
    out = week.copy()
    if scenario == "pipeline":
        present = out[block[0]].notna().to_numpy()
        hit = present & (rng.random(len(out)) < intensity)
        out.loc[hit, block] = np.nan
    elif scenario == "covariate":
        for col in ("TransactionAmt", "C1"):
            out[col] = (out[col].astype(float) * (1 + intensity)).astype(out[col].dtype)
    elif scenario == "new_category":
        s = out[category_col].astype(object)
        hit = s.notna().to_numpy() & (rng.random(len(out)) < intensity)
        s[hit] = "injected browser 1.0"
        out[category_col] = s.astype("category")
    elif scenario == "prior":
        fraud = out[out["isFraud"] == 1]
        extra = fraud.sample(int(round(intensity * len(fraud))), replace=True, random_state=int(rng.integers(1 << 31)))
        out = pd.concat([out, extra], ignore_index=True)
    elif scenario == "concept":
        hit = rng.random(len(out)) < intensity
        out.loc[hit, "isFraud"] = 1 - out.loc[hit, "isFraud"]
    else:
        raise ValueError(scenario)
    return out


# ------------------------------------------------------------ alarm rules

ALARM_STATS = {
    "new_values": "max_new_share",        # share of never-seen categorical values
    "psi_columns": "psi_test_flagged",    # numeric columns significant after FDR with PSI above the floor
    "block_missing": "max_block_shift",   # largest change in a V block's missing rate
    "domain": "domain_auc",               # domain classifier on columns without a time trend
}


def apply_alarm_rules(stats: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """One boolean column per rule (statistic strictly above its threshold), plus `any`."""
    alarms = pd.DataFrame({rule: stats[col].astype(float).abs() > thresholds[rule]
                           for rule, col in ALARM_STATS.items() if rule in thresholds}, index=stats.index)
    alarms["any"] = alarms.any(axis=1)
    return alarms


def choose_thresholds(clean: pd.DataFrame, max_false_alarm: float = 0.1, margin: float = 1.1) -> dict[str, float]:
    """Per rule, the most sensitive threshold whose alarm rate on clean weeks is at most `max_false_alarm`.

    The threshold is the (1 - max_false_alarm) quantile of the statistic on
    clean weeks, times a small safety `margin`. Sensitivity to real changes is
    then measured, not optimised, on injected drift.
    """
    return {rule: float(np.quantile(clean[col].astype(float).abs(), 1 - max_false_alarm) * margin)
            for rule, col in ALARM_STATS.items() if col in clean}


def rolling_monitor(df: pd.DataFrame, numeric: list[str], categorical: list[str], block_columns: dict[str, str],
                    cfg: dict, weeks=None, run_domain: bool = True) -> pd.DataFrame:
    """Monitor each week against the `reference_weeks` weeks just before it."""
    window = cfg["reference_weeks"]
    weeks = sorted(df["week"].unique())[window:] if weeks is None else weeks
    rows = {}
    for w in weeks:
        reference = df[(df["week"] >= w - window) & (df["week"] < w)]
        current = df[df["week"] == w]
        rows[w] = monitor_week(reference, current, numeric, categorical, block_columns, cfg, run_domain, seed=int(w))
    return pd.DataFrame(rows).T.rename_axis("week")
