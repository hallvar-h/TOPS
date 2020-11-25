import dynpssimpy.dynamic as dps
import dynpssimpy.real_time_sim as dps_rts
import importlib
import pyqtgraph as pg
import dynpssimpy.gui as dps_rts_gui
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import dynpssimpy.utility_functions as dps_uf




if __name__ == '__main__':

    [importlib.reload(module) for module in [dps, dps_rts, dps_rts_gui]]
    import ps_models.n44 as model_data
    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.use_numba = True
    ps.use_sparse = True
    ps.power_flow()
    ps.init_dyn_sim()

    ps.ode_fun(0, ps.x0)
    ps.build_y_bus_red(ps.buses['name'])
    log = defaultdict(list)
    rts = dps_rts.RealTimeSimulator(ps, dt=1e-3, solver=dps_uf.ModifiedEuler, log_fun=lambda x: dps_rts.logger(x, log))
    rts.sol.n_it = 0
    rts.start()

    time.sleep(10)
    rts.stop()

    fig, ax = plt.subplots(1)
    ax.plot(log['dt_loop'])
    ax.plot(log['dt_ideal'], zorder=10)
    ax.plot(log['dt_sim'])

    plt.show()
