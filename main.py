from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional, Dict, Any
import numpy as np
import mosek.fusion as mf

  
@dataclass
class Horizon:
    T: int
    delta_t_hours: float
  

@dataclass
class GeneratorParams:
    G_max: float
    alpha: float = 0.0

    beta: float = 0.0
    const: float = 0.0
    ramp_rate: float = 1e6


@dataclass
class ESSParams:
    s_min: float
    s_max: float
    s_init: float
    eta_ch: float
    eta_dis: float
    p_ch_max: float
    p_dis_max: float
    lambda_cycle: float = 0.0
    s_target: Optional[float] = None


@dataclass
class DROParams:
    rho: float = 5.0


@dataclass
class EVParams:
    dummy: float = 0.0


@dataclass
class ShortTermWeights:
    dummy: float = 0.0


@dataclass
class PlanResult:
    g: np.ndarray
    p_ch: np.ndarray
    p_dis: np.ndarray
    s: np.ndarray
    x_ev: np.ndarray
    l_shed: np.ndarray


@dataclass
class DistributionStats:
    mu_ev_kwh: Optional[np.ndarray] = None


def _harmonize(
    base_scen_kw: List[np.ndarray],
    pv_scen_kw: List[np.ndarray],
    ev_arrival_scen_kwh: List[np.ndarray],
    T: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    N = min(len(base_scen_kw), len(pv_scen_kw), len(ev_arrival_scen_kwh))
    base_scen_kw = [np.asarray(v, float)[:T] for v in base_scen_kw[:N]]
    pv_scen_kw = [np.asarray(v, float)[:T] for v in pv_scen_kw[:N]]
    ev_arrival_scen_kwh = [np.asarray(v, float)[:T] for v in ev_arrival_scen_kwh[:N]]
    return base_scen_kw, pv_scen_kw, ev_arrival_scen_kwh


def _feature_matrix(
    base_scen: List[np.ndarray],
    pv_scen: List[np.ndarray],
    w_base: float = 1.0,
    w_pv: float = 1.0,
) -> np.ndarray:
    N = len(base_scen)
    T = len(base_scen[0])
    F = np.zeros((N, 2 * T), dtype=float)
    for n in range(N):
        F[n, :T] = w_base * base_scen[n]
        F[n, T:] = w_pv * pv_scen[n]
    return F


def _distance_matrix(F: np.ndarray, metric: Literal["l1", "l2"] = "l1") -> np.ndarray:
    if metric == "l1":
        return np.sum(np.abs(F[:, None, :] - F[None, :, :]), axis=2)
    diff = F[:, None, :] - F[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _normalize_distance(D: np.ndarray, dist_scale: float = 1.0) -> Tuple[np.ndarray, float]:
    mask = D > 0
    if not np.any(mask):
        return D, 1.0
    med = np.median(D[mask])
    if med <= 0:
        med = 1.0
    Dn = D / (med * max(1e-9, float(dist_scale)))
    return Dn, med


def solve_wdro_wass_wc(
    horizon: Horizon,
    gen: GeneratorParams,
    ess: ESSParams,
    dro: DROParams,
    base_scen_kw: List[np.ndarray],
    pv_scen_kw: List[np.ndarray],
    ev_arrival_scen_kwh: List[np.ndarray],
    x_ev_cap_kw: float,
    kappa_surp: float = 5.0,
    kappa_ev_unserved: float = 3.0,
    kappa_import: float = 400.0,
    wass_metric: Literal["l1", "l2"] = "l1",
    stats: Optional[DistributionStats] = None,
    dist_scale: float = 1.0,
    w_base: float = 1.0,
    w_pv: float = 1.0,
    return_diag: bool = False,
    force_ev_complete: bool = False,
    ev_terminal_tol: float = 0.0,
) -> PlanResult | Tuple[PlanResult, Dict[str, Any]]:
    T, dt = horizon.T, float(horizon.delta_t_hours)
    base_scen_kw, pv_scen_kw, _ = _harmonize(base_scen_kw, pv_scen_kw, ev_arrival_scen_kwh, T)
    N = len(base_scen_kw)
    if N == 0:
        plan = PlanResult(
            g=np.zeros(T),
            p_ch=np.zeros(T),
            p_dis=np.zeros(T),
            s=np.full(T, ess.s_init),
            x_ev=np.zeros(T),
            l_shed=np.zeros(T),
        )
        return (plan, {"theta": 0.0, "D_median": 0.0, "rho": float(getattr(dro, "rho", 0.0))}) if return_diag else plan
    rho = float(getattr(dro, "rho", 5.0))
    if stats is not None and getattr(stats, "mu_ev_kwh", None) is not None:
        a_mu_kWh = np.asarray(stats.mu_ev_kwh, dtype=float)
        if len(a_mu_kWh) != T:
            a_mu_kWh = np.resize(a_mu_kWh, T)
    else:
        if len(ev_arrival_scen_kwh) > 0:
            a_mu_kWh = np.mean(np.vstack([np.asarray(v, float)[:T] for v in ev_arrival_scen_kwh]), axis=0)
        else:
            a_mu_kWh = np.zeros(T, dtype=float)
    F = _feature_matrix(base_scen_kw, pv_scen_kw, w_base=float(w_base), w_pv=float(w_pv))
    D = _distance_matrix(F, wass_metric)
    Dn, D_med = _normalize_distance(D, dist_scale=float(dist_scale))
    M = mf.Model("wdro_wass_wc")
    try:
        g = M.variable("g", T, mf.Domain.inRange(0.0, 1.0))
        pch = M.variable("pch", T, mf.Domain.greaterThan(0.0))
        pds = M.variable("pds", T, mf.Domain.greaterThan(0.0))
        s = M.variable("s", T, mf.Domain.inRange(ess.s_min, ess.s_max))
        xev = M.variable("xev", T, mf.Domain.inRange(0.0, x_ev_cap_kw))
        E = M.variable("E", T, mf.Domain.greaterThan(0.0))
        yimp = M.variable("yimp", T, mf.Domain.greaterThan(0.0))
        a = M.variable("a", [N], mf.Domain.unbounded())
        theta = M.variable("theta", 1, mf.Domain.greaterThan(0.0))
        eta_ch = float(ess.eta_ch)
        eta_dis = float(ess.eta_dis)
        for t in range(T):
            inflow = mf.Expr.mul(dt * eta_ch, pch.index(t))
            outflow = mf.Expr.mul(dt / max(1e-9, eta_dis), pds.index(t))
            if t == 0:
                M.constraint(
                    mf.Expr.sub(s.index(t), mf.Expr.add(ess.s_init, mf.Expr.sub(inflow, outflow))),
                    mf.Domain.equalsTo(0.0),
                )
            else:
                M.constraint(
                    mf.Expr.sub(s.index(t), mf.Expr.add(s.index(t - 1), mf.Expr.sub(inflow, outflow))),
                    mf.Domain.equalsTo(0.0),
                )
        rr = float(getattr(gen, "ramp_rate", 1.0))
        if T >= 2 and rr < 1e6:
            for t in range(1, T):
                M.constraint(mf.Expr.sub(g.index(t), g.index(t - 1)), mf.Domain.lessThan(rr))
                M.constraint(mf.Expr.sub(g.index(t - 1), g.index(t)), mf.Domain.lessThan(rr))
        for t in range(T):
            inflow = mf.Expr.mul(dt * eta_ch, pch.index(t))
            outflow = mf.Expr.mul(dt / max(1e-9, eta_dis), pds.index(t))
            if t == 0:
                M.constraint(
                    mf.Expr.sub(s.index(t), mf.Expr.add(ess.s_init, mf.Expr.sub(inflow, outflow))),
                    mf.Domain.equalsTo(0.0),
                )
            else:
                M.constraint(
                    mf.Expr.sub(s.index(t), mf.Expr.add(s.index(t - 1), mf.Expr.sub(inflow, outflow))),
                    mf.Domain.equalsTo(0.0),
                )
        for t in range(T):
            rhs = mf.Expr.sub(
                mf.Expr.add(E.index(t - 1), float(a_mu_kWh[t])) if t > 0 else mf.Expr.constTerm(float(a_mu_kWh[t])),
                mf.Expr.mul(dt, xev.index(t)),
            )
            M.constraint(mf.Expr.sub(E.index(t), rhs), mf.Domain.equalsTo(0.0))
        if force_ev_complete:
            if ev_terminal_tol <= 0.0:
                M.constraint(E.index(T - 1), mf.Domain.equalsTo(0.0))
            else:
                M.constraint(E.index(T - 1), mf.Domain.lessThan(float(ev_terminal_tol)))
        surp = M.variable("surplus", [N, T], mf.Domain.greaterThan(0.0))
        Gmax = float(gen.G_max)
        for m in range(N):
            for t in range(T):
                demand_mt = mf.Expr.add(float(base_scen_kw[m][t]), mf.Expr.add(pch.index(t), xev.index(t)))
                supply_mt = mf.Expr.add(
                    mf.Expr.mul(Gmax, g.index(t)),
                    mf.Expr.add(float(pv_scen_kw[m][t]), pds.index(t)),
                )
                M.constraint(
                    mf.Expr.sub(demand_mt, mf.Expr.add(supply_mt, yimp.index(t))),
                    mf.Domain.lessThan(0.0),
                )
                M.constraint(
                    mf.Expr.sub(supply_mt, demand_mt),
                    mf.Domain.lessThan(surp.index(m, t)),
                )
        Lm = [
            mf.Expr.mul(kappa_surp * dt, mf.Expr.sum(surp.slice([m, 0], [m + 1, T])))
            for m in range(N)
        ]
        for n in range(N):
            for m in range(N):
                M.constraint(
                    mf.Expr.sub(
                        a.index(n),
                        mf.Expr.sub(Lm[m], mf.Expr.mul(theta.index(0), float(Dn[n, m]))),
                    ),
                    mf.Domain.greaterThan(0.0),
                )
        alpha = float(getattr(gen, "alpha", 0.0))
        beta = float(getattr(gen, "beta", 0.0))
        gen_cost = mf.Expr.add(
            mf.Expr.mul(alpha * (Gmax**2) * dt, mf.Expr.sum(mf.Expr.mul(g, g))),
            mf.Expr.mul(beta * Gmax * dt, mf.Expr.sum(g)),
        )
        gen_const = float(getattr(gen, "const", 0.0)) * T * dt
        lam = float(getattr(ess, "lambda_cycle", 0.0))
        ess_cycle = mf.Expr.mul(lam * dt, mf.Expr.add(mf.Expr.sum(pch), mf.Expr.sum(pds)))
        ev_term_pen = mf.Expr.mul(float(kappa_ev_unserved), E.index(T - 1))
        imp_cost = mf.Expr.mul(float(kappa_import * dt), mf.Expr.sum(yimp))
        dual_part = mf.Expr.add(mf.Expr.mul(1.0 / N, mf.Expr.sum(a)), mf.Expr.mul(float(rho), theta.index(0)))
        obj = mf.Expr.add(
            [gen_cost, mf.Expr.constTerm(gen_const), ess_cycle, dual_part, ev_term_pen, imp_cost]
        )
        M.objective(mf.ObjectiveSense.Minimize, obj)
        M.solve()
        g_lv = np.asarray(g.level())
        pch_lv = np.asarray(pch.level())
        pds_lv = np.asarray(pds.level())
        s_lv = np.asarray(s.level())
        xev_lv = np.asarray(xev.level())
        plan = PlanResult(
            g=g_lv,
            p_ch=pch_lv,
            p_dis=pds_lv,
            s=s_lv,
            x_ev=xev_lv,
            l_shed=np.zeros(T),
        )
        if return_diag:
            diag = {
                "theta": float(theta.level()[0]),
                "D_median": float(D_med),
                "rho": rho,
            }
            return plan, diag
        return plan
    finally:
        M.dispose()


@dataclass
class DriftConfig:
    k_soc: float = 0.5
    max_soc_step_ratio: float = 0.25


class ShortTermControllerQP:
    def __init__(
        self,
        gen: GeneratorParams,
        ess: ESSParams,
        evp: Optional[EVParams],
        weights: ShortTermWeights,
        dt_hours: float,
        drift_cfg: Optional[DriftConfig] = None,
    ):
        self.gen = gen
        self.ess = ess
        self.evp = evp
        self.w = weights
        self.dt = float(dt_hours)
        self.cfg = drift_cfg or DriftConfig()
        self.s_target = ess.s_target if ess.s_target is not None else 0.5 * (ess.s_min + ess.s_max)
        self._soc_span = max(1e-6, float(ess.s_max - ess.s_min))

    def _desired_ess_power_from_drift(self, s_now: float) -> float:
        delta_s_des = self.cfg.k_soc * (self.s_target - s_now)
        max_step = self.cfg.max_soc_step_ratio * self._soc_span
        delta_s_des = float(np.clip(delta_s_des, -max_step, max_step))
        p_des = delta_s_des / self.dt
        return p_des

    def _split_to_charge_discharge(self, p_des: float) -> Tuple[float, float]:
        if p_des >= 0.0:
            pch = min(p_des / max(1e-6, self.ess.eta_ch), float(self.ess.p_ch_max))
            pds = 0.0
        else:
            pds = min((-p_des) * max(1e-6, self.ess.eta_dis), float(self.ess.p_dis_max))
            pch = 0.0
        return float(pch), float(pds)

    def step(
        self,
        t: int,
        base_now_kw: float,
        pv_now_kw: float,
        s_now_kWh: float,
        g_norm_hint: float,
    ) -> Tuple[float, float, float, float, float]:
        p_des = self._desired_ess_power_from_drift(s_now_kWh)
        pch, pds = self._split_to_charge_discharge(p_des)
        inflow = self.ess.eta_ch * pch * self.dt
        outflow = (pds / max(1e-6, self.ess.eta_dis)) * self.dt
        s_next = float(np.clip(s_now_kWh + inflow - outflow, self.ess.s_min, self.ess.s_max))
        g_kw = float(self.gen.G_max) * float(np.clip(g_norm_hint, 0.0, 1.0))
        supply_kw = float(pv_now_kw) + g_kw + pds
        demand_kw = float(base_now_kw) + pch
        short_kw = max(0.0, demand_kw - supply_kw)
        surplus_kw = max(0.0, supply_kw - demand_kw)
        return float(pch), float(pds), s_next, float(short_kw), float(surplus_kw)

    def act(
        self,
        t: int,
        base_now_kw: float,
        pv_now_kw: float,
        s_now_kWh: float,
        g_norm_hint: float,
    ) -> Tuple[float, float, float, float, float]:
        return self.step(t, base_now_kw, pv_now_kw, s_now_kWh, g_norm_hint)
