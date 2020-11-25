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
    def __init__(self, ps, dt=5e-3, speed=1, solver=dps_uf.ModifiedEuler, log_fun=[], ode_fun=[]):
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

        self.sol = solver(self.ode_fun, 0, self.ps.x0, self.t_end, max_step=self.dt, first_step=self.dt)

        self.new_data_cv = threading.Condition()  # condition variable used to both lock and to notify threads
        self.new_data_ready = False
        self.x = self.sol.y
        self.t = self.sol.t

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

    ps.power_flow()
    ps.init_dyn_sim()
    ps.build_y_bus_red()  # ps.buses['name'])
    ps.ode_fun(0, ps.x0)
    log = defaultdict(list)
    rts = RealTimeSimulator(ps, dt=5e-3, speed=1, solver=dps_uf.ModifiedEuler, log_fun=lambda x: logger(x, log))
    time.sleep(2)
    rts.start()

    time.sleep(10)
    rts.stop()

    fig, ax = plt.subplots(1)
    ax.plot(log['dt_loop'])
    ax.plot(log['dt_ideal'])
    ax.plot(log['dt_sim'])
    plt.show()