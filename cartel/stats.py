"""Statistics used by the analysis. Unit of analysis is the EPISODE (agents inside an episode are not independent), so every CI for an
agent-level quantity is a cluster bootstrap over episodes. numpy + scipy only."""
import numpy as np
from scipy import stats as st

RNG_SEED = 12345


def _rng(seed=None):
    return np.random.default_rng(RNG_SEED if seed is None else seed)


def mean_ci(x, B=2000, alpha=0.05, seed=None):
    """Mean with a percentile bootstrap CI. Returns (mean, lo, hi, n)."""
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    if len(x) == 1:
        return float(x[0]), np.nan, np.nan, 1
    r = _rng(seed)
    bs = x[r.integers(0, len(x), (B, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2)), len(x)


def diff_test(a, b, B=4000, seed=None, alternative="greater"):
    """Difference in means a - b: bootstrap CI and a permutation p-value.
    alternative: 'greater' (a > b), 'less' (a < b) or 'two-sided'. Returns dict(est, lo, hi, p, n_a, n_b)."""
    a = np.asarray([v for v in a if v is not None and not np.isnan(v)], float)
    b = np.asarray([v for v in b if v is not None and not np.isnan(v)], float)
    if len(a) < 2 or len(b) < 2:
        return dict(est=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n_a=len(a), n_b=len(b))
    r = _rng(seed)
    est = a.mean() - b.mean()
    bs = a[r.integers(0, len(a), (B, len(a)))].mean(1) - b[r.integers(0, len(b), (B, len(b)))].mean(1)
    pool = np.concatenate([a, b])
    perm = np.array([r.permutation(pool) for _ in range(B)])
    null = perm[:, :len(a)].mean(1) - perm[:, len(a):].mean(1)
    if alternative == "greater":
        p = (np.sum(null >= est) + 1) / (B + 1)
    elif alternative == "less":
        p = (np.sum(null <= est) + 1) / (B + 1)
    else:
        p = (np.sum(np.abs(null) >= abs(est)) + 1) / (B + 1)
    return dict(est=float(est), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), p=float(p), n_a=len(a), n_b=len(b))


def cluster_bootstrap(groups, stat, B=1000, seed=None):
    """groups: list of per-episode data objects. stat: function(list_of_groups) -> float. Resamples whole episodes.
    Returns (estimate, lo, hi, boot_array)."""
    est = stat(groups)
    r = _rng(seed)
    n = len(groups)
    bs = []
    for _ in range(B):
        idx = r.integers(0, n, n)
        try:
            v = stat([groups[i] for i in idx])
        except Exception:                                   # noqa: BLE001  (degenerate resample)
            v = np.nan
        bs.append(v)
    bs = np.array(bs, float)
    ok = bs[~np.isnan(bs)]
    if len(ok) < B * 0.5:
        return float(est), np.nan, np.nan, bs
    return float(est), float(np.quantile(ok, .025)), float(np.quantile(ok, .975)), bs


def boot_p(bs, null=0.0, alternative="greater"):
    """One-sided p from a bootstrap distribution (share of resamples on the wrong side of the null). A coarse but honest summary."""
    bs = np.asarray(bs, float)
    bs = bs[~np.isnan(bs)]
    if len(bs) == 0:
        return np.nan
    if alternative == "greater":
        return float((np.sum(bs <= null) + 1) / (len(bs) + 1))
    if alternative == "less":
        return float((np.sum(bs >= null) + 1) / (len(bs) + 1))
    return float(min(1.0, 2 * min(np.sum(bs <= null) + 1, np.sum(bs >= null) + 1) / (len(bs) + 1)))


def holm(pvals):
    """Holm step-down adjusted p-values (NaN entries stay NaN and do not count)."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    m = len(idx)
    order = idx[np.argsort(p[idx])]
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        out[i] = running
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return ph, max(0.0, c - h), min(1.0, c + h)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 4 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return np.nan, np.nan
    r, p = st.spearmanr(x[ok], y[ok])
    return float(r), float(p)


def cohen_kappa(a, b):
    """Cohen's kappa between two label lists, plus the confusion matrix (labels sorted)."""
    labs = sorted(set(a) | set(b))
    ix = {l: i for i, l in enumerate(labs)}
    cm = np.zeros((len(labs), len(labs)), int)
    for x, y in zip(a, b):
        cm[ix[x], ix[y]] += 1
    n = cm.sum()
    if n == 0:
        return np.nan, labs, cm
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return (float((po - pe) / (1 - pe)) if pe < 1 else np.nan), labs, cm


def km(durations, events):
    """Kaplan-Meier with Greenwood 95% CI. Returns arrays (t, S, lo, hi) as step points starting at t=0."""
    d, e = np.asarray(durations, float), np.asarray(events, bool)
    ts = np.unique(d[e])
    S, var, out = 1.0, 0.0, [(0.0, 1.0, 1.0, 1.0)]
    for t in ts:
        n_risk = np.sum(d >= t)
        k = np.sum((d == t) & e)
        if n_risk == 0:
            continue
        S *= 1 - k / n_risk
        if n_risk > k:
            var += k / (n_risk * (n_risk - k))
        se = S * np.sqrt(var)
        out.append((float(t), S, max(0.0, S - 1.96 * se), min(1.0, S + 1.96 * se)))
    a = np.array(out)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def logrank(durations, events, groups):
    """k-group log-rank test. Returns (chi2, df, p)."""
    d, e, g = np.asarray(durations, float), np.asarray(events, bool), np.asarray(groups)
    labs = list(dict.fromkeys(g))
    if len(labs) < 2:
        return np.nan, 0, np.nan
    O = np.zeros(len(labs)); E = np.zeros(len(labs)); V = np.zeros((len(labs), len(labs)))
    for t in np.unique(d[e]):
        at = d >= t
        n = at.sum()
        dt = ((d == t) & e).sum()
        if n <= 1:
            continue
        nj = np.array([(at & (g == l)).sum() for l in labs], float)
        dj = np.array([((d == t) & e & (g == l)).sum() for l in labs], float)
        O += dj
        E += dt * nj / n
        fac = dt * (n - dt) / (n * n * (n - 1))
        V += fac * (np.diag(nj * n) - np.outer(nj, nj))
    k = len(labs) - 1
    diff = (O - E)[:k]
    try:
        chi2 = float(diff @ np.linalg.solve(V[:k, :k], diff))
    except np.linalg.LinAlgError:
        return np.nan, k, np.nan
    return chi2, k, float(st.chi2.sf(chi2, k))


def logistic(x, y):
    """Logistic regression y ~ a + b x by IRLS. Returns (a, b, se_a, se_b, cov)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ w))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - p))
        w += step
        if np.abs(step).max() < 1e-10:
            break
    cov = np.linalg.inv(H)
    return float(w[0]), float(w[1]), float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1])), cov


def welch_p(a, b, alternative="greater"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(st.ttest_ind(a, b, equal_var=False, alternative=alternative).pvalue)
