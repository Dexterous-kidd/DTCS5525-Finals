"""Ordered-probit judge models and their normal-approximation flip-rate predictions (RQ3).

M0 (pre-specified):  y = beta*m + gamma*s + eps,  eps ~ N(0,1);  theta = [beta, gamma, c1, log(c2-c1)]
M1 (exploratory):    adds a battle random intercept u ~ N(0, sigma_u^2) shared by all calls on a battle (persistent
                     item-level preferences);  theta = [beta, gamma, c1, log(c2-c1), log(sigma_u)]
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(15)
_GH_W = _GH_W / np.sqrt(2 * np.pi)


def _unpack(theta):
    beta, gamma, c1, ld = theta[:4]
    return beta, gamma, c1, c1 + np.exp(ld)


def category_probs(theta, m, s, shift=0.0):
    """P(B wins), P(tie), P(A wins) in the canonical frame. s = +1 if canonical A is shown first, -1 otherwise."""
    beta, gamma, c1, c2 = _unpack(theta)
    mu = beta * np.asarray(m) + gamma * np.asarray(s) + shift
    pB = norm.cdf(c1 - mu); pBT = norm.cdf(c2 - mu)
    return pB, pBT - pB, 1 - pBT


def _grad_ok(r, tol=0.05):
    return bool(np.max(np.abs(r.jac)) < tol)


def fit_probit(m, s, y):
    """M0. m: signed pair margin, s: +-1 position indicator, y: 0 (B wins), 1 (tie), 2 (A wins). Maximum likelihood."""
    m, s, y = map(np.asarray, (m, s, y))
    def nll(theta):
        pB, pT, pA = category_probs(theta, m, s)
        p = np.where(y == 0, pB, np.where(y == 1, pT, pA))
        return -np.sum(np.log(np.clip(p, 1e-12, 1)))
    best = None
    for start in ([0.0, 0.0, -0.3, np.log(0.6)], [3.0, 0.2, -0.5, np.log(1.0)], [-1.0, 0.0, -1.0, np.log(0.3)]):
        r = minimize(nll, np.array(start), method="BFGS")
        if best is None or r.fun < best.fun:
            best = r
    beta, gamma, c1, c2 = _unpack(best.x)
    return dict(beta=float(beta), gamma=float(gamma), c1=float(c1), c2=float(c2), theta=best.x.tolist(), nll=float(best.fun),
               n=int(len(y)), converged=_grad_ok(best), grad_max=float(np.max(np.abs(best.jac))))


def fit_probit_re(m, s, y, unit):
    """M1. unit: array of battle ids (calls on the same battle share the random intercept). Gauss-Hermite marginal likelihood."""
    m, s, y = map(np.asarray, (m, s, y)); _, codes = np.unique(unit, return_inverse=True); codes = codes.ravel(); nu = int(codes.max()) + 1
    def nll(theta):
        sig = np.exp(theta[4]); logp = np.empty((len(y), len(_GH_X)))
        for q, x in enumerate(_GH_X):
            pB, pT, pA = category_probs(theta, m, s, shift=sig * x)
            p = np.where(y == 0, pB, np.where(y == 1, pT, pA))
            logp[:, q] = np.log(np.clip(p, 1e-12, 1))
        per_unit = np.stack([np.bincount(codes, weights=logp[:, q], minlength=nu) for q in range(len(_GH_X))], 1)
        return -np.sum(logsumexp(per_unit + np.log(_GH_W)[None, :], axis=1))
    best = None
    for start in ([2.0, 0.2, -0.4, np.log(0.7), np.log(0.5)], [4.0, 0.3, -0.5, np.log(1.0), np.log(1.0)]):
        r = minimize(nll, np.array(start), method="BFGS")
        if best is None or r.fun < best.fun:
            best = r
    beta, gamma, c1, c2 = _unpack(best.x)
    return dict(beta=float(beta), gamma=float(gamma), c1=float(c1), c2=float(c2), sigma_u=float(np.exp(best.x[4])), theta=best.x.tolist(),
               nll=float(best.fun), n=int(len(y)), converged=_grad_ok(best, 0.5), grad_max=float(np.max(np.abs(best.jac))))


def _moments(p):                       # p = (pB, pT, pA) -> mean, var of score in {0, .5, 1}
    pB, pT, pA = p
    mu = 0.5 * pT + pA
    return mu, (0.25 * pT + pA) - mu ** 2


def _cond_moments(proto, theta, m, f_ds, shift):
    """Mean/variance of one battle score for a protocol, conditional on a battle effect `shift` (independent calls given shift)."""
    pAB = category_probs(theta, m, +1, shift); pBA = category_probs(theta, m, -1, shift)
    if proto in ("P0", "P0r"):
        f = f_ds if proto == "P0" else 0.5
        return _moments(tuple(np.array(pAB) * (1 - f) + np.array(pBA) * f))
    if proto in ("P1", "P1x"):
        pa = pAB[2] * pBA[2]; pb = pAB[0] * pBA[0]
        return _moments((pb, 1 - pa - pb, pa))
    if proto == "P2":
        res = [(p[0] ** 2, 1 - p[0] ** 2 - p[2] ** 2, p[2] ** 2) for p in (pAB, pBA)]
        return _moments(tuple(0.5 * (a + b) for a, b in zip(*res)))
    raise ValueError(proto)


def _judge_moments(proto, theta, m, f_ds):
    """Returns (mean, within-battle variance, between-battle variance of the conditional mean) of a battle score."""
    if len(theta) == 5:                                    # integrate the battle random intercept
        sig = np.exp(theta[4]); mus, vs = [], []
        for x in _GH_X:
            mu, v = _cond_moments(proto, theta, m, f_ds, sig * x); mus.append(mu); vs.append(v)
        mus, vs = np.array(mus), np.array(vs); mean = float(np.sum(_GH_W * mus))
        return mean, float(np.sum(_GH_W * vs)), float(np.sum(_GH_W * mus ** 2) - mean ** 2)
    mu, v = _cond_moments(proto, theta, m, f_ds, 0.0)
    return float(mu), float(v), 0.0


def score_moments(proto, thetas, m, f_ds=0.5):
    if proto == "P3":                                      # alternating judges: average of the judges' P0r moments
        mv = np.array([_judge_moments("P0r", t, m, 0.5) for t in thetas.values()])
        return tuple(float(x) for x in mv.mean(0))
    return _judge_moments(proto, next(iter(thetas.values())), m, f_ds)


def predict_cell(proto, thetas, m, n, zstar, k, f_ds=0.5, m_pool=None):
    """Predicted decisive flip rate and coverage.
    Both replicates use the same k battles drawn from a pool of m_pool battles, so the subset mean mu_set is shared:
    mu_set ~ N(mean, tau2 * (1/m_pool + (m_pool-k)/(k*(m_pool-1))));  given mu_set each replicate mean is N(mu_set, within/k)
    and the decision threshold uses the total score variance, as the empirical standard error does."""
    mean, within, tau2 = score_moments(proto, thetas, m, f_ds)
    sd_w = np.sqrt(max(within, 1e-12) / k); sd_th = np.sqrt(max(within + tau2, 1e-12) / k)
    if m_pool and m_pool > 1 and tau2 > 0:
        t2 = tau2 * (1.0 / m_pool + max(m_pool - k, 0) / (k * (m_pool - 1)))
    else:
        t2 = tau2 / k if tau2 > 0 else 0.0
    nodes, wts = (_GH_X, _GH_W) if t2 > 0 else (np.array([0.0]), np.array([1.0]))
    num = den = cov = 0.0
    for x, wq in zip(nodes, wts):
        mu = mean + np.sqrt(t2) * x
        if zstar == 0:
            piA = 1 - norm.cdf((0.5 - mu) / sd_w); piB = norm.cdf((0.5 - mu) / sd_w)
        else:
            piA = 1 - norm.cdf((0.5 + zstar * sd_th - mu) / sd_w); piB = norm.cdf((0.5 - zstar * sd_th - mu) / sd_w)
        num += wq * 2 * piA * piB; den += wq * (piA + piB) ** 2; cov += wq * (piA + piB)
    return (float(num / den) if den > 1e-12 else float("nan")), float(cov)
