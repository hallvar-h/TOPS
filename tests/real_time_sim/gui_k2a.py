from PyQt5 import QtWidgets
import sys
import pyqtgraph as pg
import dynpssimpy.dynamic as dps
import importlib
from pyqtconsole.console import PythonConsole
import dynpssimpy.real_time_sim as dps_rts
import dynpssimpy.gui as gui
import dynpssimpy.utility_functions as dps_uf


def main(rts):
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    phasor_plot = gui.PhasorPlot(rts, update_freq=30)
    ts_plot = gui.TimeSeriesPlot(rts, ['speed', 'angle'], update_freq=30)  # , 'speed', 'e_q_t', 'e_d_t', 'e_q_st', 'e_d_st'])
    # stats_plot = gui.SimulationStatsPlot(rts, update_freq=30)

    # Add Control Widgets
    line_outage_ctrl = gui.LineOutageWidget(rts)
    gen_ctrl = gui.GenCtrlWidget(rts)

    console = PythonConsole()
    console.push_local_ns('rts', rts)
    console.push_local_ns('ts_plot', ts_plot)
    console.push_local_ns('phasor_plot', phasor_plot)
    console.push_local_ns('line_outage_ctrl', line_outage_ctrl)
    console.push_local_ns('gen_ctrl', gen_ctrl)
    console.show()
    console.eval_in_thread()

    app.exec_()

    return app


if __name__ == '__main__':

    [importlib.reload(module) for module in [dps, dps_rts, gui]]

    import ps_models.k2a as model_data
    model = model_data.load()

    importlib.reload(dps)
    ps = dps.PowerSystemModel(model=model)
    # ps.use_numba = True

    ps.power_flow()
    ps.init_dyn_sim()
    ps.build_y_bus_red(ps.buses['name'])
    ps.ode_fun(0, ps.x0)

    rts = dps_rts.RealTimeSimulator(ps, dt=10e-3, speed=1, solver=dps_uf.ModifiedEuler)
    rts.sol.n_it = 0
    rts.ode_fun(0, ps.x0)

    rts.start()

    app = main(rts)
    rts.stop()
