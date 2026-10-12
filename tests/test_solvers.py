"""Check solvers against the analytic solution of dx/dt = -x, and their order of convergence."""
import numpy as np
import pytest

import tops.solvers as dps_sol

T_END = 1.0


def run_ode(solver_cls, dt):
    sol = solver_cls(lambda t, x: -x, 0, np.array([1.0]), T_END, dt=dt)
    while sol.t < T_END - 1e-12:
        sol.step()
    return sol


def run_dae(solver_cls, dt):
    # dx/dt = -v with algebraic equation v = x, which is equivalent to dx/dt = -x
    sol = solver_cls(lambda t, x, v: -v, lambda t, x, v: x.copy(), 0, np.array([1.0]), np.array([1.0]), T_END, dt=dt)
    while sol.t < T_END - 1e-12:
        sol.step()
    return sol


def error(sol):
    return abs(sol.x[0] - np.exp(-sol.t))


@pytest.mark.parametrize('solver_cls, run, order', [
    (dps_sol.Euler, run_ode, 1),
    (dps_sol.ModifiedEuler, run_ode, 2),
    (dps_sol.SimpleRK4, run_ode, 4),
    (dps_sol.EulerDAE, run_dae, 1),
    (dps_sol.ModifiedEulerDAE, run_dae, 2),
])
def test_order_of_convergence(solver_cls, run, order):
    dt = 0.02
    err_coarse = error(run(solver_cls, dt))
    err_fine = error(run(solver_cls, dt / 2))
    observed_order = np.log2(err_coarse / err_fine)
    assert observed_order == pytest.approx(order, abs=0.2)


def test_dae_solver_keeps_algebraic_variables_updated():
    sol = run_dae(dps_sol.ModifiedEulerDAE, 0.01)
    np.testing.assert_allclose(sol.v, sol.x)


def test_max_step_overrides_dt():
    sol = dps_sol.Euler(lambda t, x: -x, 0, np.array([1.0]), dt=0.1, max_step=0.01)
    assert sol.dt == 0.01


def test_solver_stops_at_t_end():
    sol = dps_sol.Euler(lambda t, x: -x, 0, np.array([1.0]), T_END, dt=0.1)
    while sol.t < T_END:
        sol.step()
    t_final = sol.t
    sol.step()
    assert sol.t == t_final
