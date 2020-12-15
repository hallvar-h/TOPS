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
    phasor_plot = gui.PhasorPlotFast(rts, update_freq=30)
    ts_plot = gui.TimeSeriesPlotFast(rts, ['speed', 'angle'], update_freq=30)  # , 'speed', 'e_q_t', 'e_d_t', 'e_q_st', 'e_d_st'])
    # stats_plot = gui.SimulationStatsPlot(rts, update_freq=30)

    # Add Control Widgets
    line_outage_ctrl = gui.LineOutageWidget(rts)
    excitation_ctrl = gui.GenCtrlWidget(rts)

    console = PythonConsole()
    console.push_local_ns('rts', rts)
    console.push_local_ns('ts_plot', ts_plot)
    console.push_local_ns('phasor_plot', phasor_plot)
    console.push_local_ns('line_outage_ctrl', line_outage_ctrl)
    console.push_local_ns('excitation_ctrl', excitation_ctrl)
    console.show()
    console.eval_in_thread()

    app.exec_()

    return app


if __name__ == '__main__':

    [importlib.reload(module) for module in [dps, dps_rts, gui]]

    import ps_models.n44 as model_data
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)

    # Add controls for all generators (not specified in model)
    model['gov'] = {'TGOV1':
        [['name', 'gen', 'R', 'D_t', 'V_min', 'V_max', 'T_1', 'T_2', 'T_3']] +
        [['GOV'+str(i), gen_name, 0.05, 0, 0, 1, 0.2, 1, 2] for i, gen_name in enumerate(ps.generators['name'])]
    }

    model['avr'] = {'SEXS':
        [['name', 'gen', 'K', 'T_a', 'T_b', 'T_e', 'E_min', 'E_max']] +
        [['AVR'+str(i), gen_name, 100, 2.0, 10.0, 0.5, -3, 3] for i, gen_name in enumerate(ps.generators['name'])]
    }

    model['pss'] = {'STAB1':
        [['name', 'gen', 'K', 'T', 'T_1', 'T_2', 'T_3', 'T_4', 'H_lim']] +
        [['PSS'+str(i), gen_name, 50, 10.0, 0.5, 0.5, 0.05, 0.05, 0.03] for i, gen_name in enumerate(ps.generators['name'])]
    }

    # Add generator controls?

    importlib.reload(dps)
    ps = dps.PowerSystemModel(model=model)
    ps.use_numba = False

    ps.power_flow()
    ps.init_dyn_sim()
    ps.build_y_bus_red(ps.buses['name'])
    ps.ode_fun(0, ps.x0)

    rts = dps_rts.RealTimeSimulator(ps, dt=5e-3, speed=1, solver=dps_uf.ModifiedEuler)
    rts.sol.n_it = 0
    rts.ode_fun(0, ps.x0)

    rts.start()

    app = main(rts)
    rts.stop()
