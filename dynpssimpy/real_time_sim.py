from PyQt5 import QtWidgets
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import time
import threading
from scipy.integrate import RK23, RK45, BDF, trapz
sys.path.append(r'C:/Users/lokal_hallvhau/Dropbox/Python/DynPSSimPy/')
import dynpssimpy.dynamic as dps
import importlib
from pyqtconsole.console import PythonConsole
import pandas as pd
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import dynpssimpy.utility_functions as dps_uf


class RealTimeSimulator(threading.Thread):
    def __init__(self, ps, dt=5e-3, speed=1, solver=dps_uf.SimpleRK4, log_fun=[], ode_fun=[]):
        threading.Thread.__init__(self)
        self.daemon = True

        self.t_end = 1000
        self.dt = dt
        self.adjust_time = False
        self.speed = speed
        self.running = True
        self.log_fun = log_fun
        self.log = callable(self.log_fun)

        self.dt_sim = 0
        self.t_world = 0
        self.dt_loop = 0
        self.dt_err = 0
        self.dt_ideal = self.dt / self.speed

        self.ps = ps


        if callable(ode_fun):
            self.ode_fun = ode_fun
        else:
            self.ode_fun = self.ps.ode_fun

        # self.sol = RK23(self.ps.ode_fun, 0, self.ps.x0, self.t_end, max_step=self.dt, first_step=self.dt)
        self.sol = solver(self.ode_fun, 0, self.ps.x0, self.t_end, max_step=self.dt, first_step=self.dt)
        # self.sol = RK45(self.ps.ode_fun, 0, self.ps.x0, self.t_end, max_step=self.dt, first_step=self.dt)
        # self.sol = BDF(self.ps.ode_fun, 0, self.ps.x0, self.t_end, max_step=self.dt, first_step=self.dt)
        # self.ode_fun(0, self.ps.x0)

        self.new_data_cv = threading.Condition()  # condition variable used to both lock and to notify threads
        self.new_data_ready = False

    def run(self):
        self.exit_flag = threading.Event()

        t_start_sim = time.time()
        # t_adj = 0
        t_prev = time.time()

        while self.running:  # and t < self.t_end:

            # Simulate next step
            t_sim_0 = time.time()
            if self.speed > 0:
                with self.new_data_cv:
                    self.sol.step()
                    self.new_data_ready = True
                    self.new_data_cv.notify()

            self.dt_sim = time.time() - t_sim_0
            self.dt_loop = time.time() - t_prev

            t_prev = time.time()
            self.t_world += self.dt_loop*self.speed
            self.dt_err = self.sol.t - self.t_world
            if self.dt_err > 0:
                time.sleep(self.dt_err / self.speed)
            elif self.dt_err < 0:
                print('Overflow! {:.2f} ms.'.format(1000*self.dt_err))
                # if self.adjust_time:
                #     t_adj -= t_err

            self.dt_ideal = self.dt / self.speed

            if self.log:
                self.log_fun(self)

        return

    def stop(self):
        self.running = False
        if not self.is_alive():
            print('RTS thread stopped.')

    def set_speed(self, speed):
        self.speed = speed




def logger(rts, log, attributes=['dt_loop', 'dt_sim', 'dt_ideal']):
    for attr in attributes:
        log[attr].append(getattr(rts, attr))


if __name__ == '__main__':
    importlib.reload(dps)

    # import ps_models.n44 as model_data
    import ps_models.ieee39 as model_data
    # import ps_models.sm_ib as model_data

    model = model_data.load()

    importlib.reload(dps)
    ps = dps.PowerSystemModel(model=model)
    ps.use_numba = True
    ps.use_sparse = True

    # The below code simply adds generator controls (gov, avr, pss) to all generators. Used for N44 (since these are not defined in the model).
    # ps.gov['TGOV1'] = np.recarray(
    #     data=[
    #         ['GOV' + str(i), name, 0.05, 0.02, 0, 1, 0.5, 1, 2]
    #         for i, name in enumerate(ps.generators['name'])
    #     ],
    #     dtype=[
    #         'name', 'gen', 'R', 'D_t', 'V_min', 'V_max', 'T_1', 'T_2', 'T_3',
    #     ],
    # )
    #
    # ps.avr['SEXS'] = pd.DataFrame(
    #     columns=[
    #         'name', 'gen', 'K', 'T_a', 'T_b', 'T_e', 'E_min', 'E_max',
    #     ],
    #     data=[
    #         ['AVR' + str(i), name, 100, 10.0, 10.0, 0.01, -3, 3]
    #         for i, name in enumerate(ps.generators['name'])
    #     ])
    #
    # ps.pss['STAB1'] = pd.DataFrame(
    #     columns=[
    #         'name', 'gen', 'K', 'T', 'T_1', 'T_2', 'T_3', 'T_4', 'H_lim',
    #     ],
    #     data=[
    #         ['PSS' + str(i), name, 50, 10.0, 0.5, 0.5, 0.5, 0.05, 0.03]
    #         for i, name in enumerate(ps.generators['name'])
    #     ])


    ps.power_flow()
    ps.init_dyn_sim()
    ps.build_y_bus_red()  # ps.buses['name'])
    ps.x0[ps.angle_idx][0] += 1e-3
    ps.ode_fun(0, ps.x0)
    log = defaultdict(list)
    rts = RealTimeSimulator(ps, dt=5e-3, speed=1, solver=dps_uf.SimpleRK4, log_fun=lambda x: logger(x, log))
    time.sleep(2)
    rts.start()

    # print(rts.is_alive())
    from threading import Thread
    # app, main = main(rts)

    time.sleep(10)
    rts.stop()

    fig, ax = plt.subplots(1)
    ax.plot(log['dt_loop'])
    ax.plot(log['dt_ideal'])
    ax.plot(log['dt_sim'])
    plt.show()
    # #
    # np.savez(r'C:\Users\lokal_hallvhau\Dropbox\Python\Plotting for Presentations\2020 - DynPSSimPy\rtsim\data',
    #          dt_loop=np.array(log['dt_loop']),
    #          dt_ideal=np.array(log['dt_ideal']),
    #          dt_sim=np.array(log['dt_sim']),
    #          )