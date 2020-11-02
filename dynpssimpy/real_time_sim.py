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


class RK4_simple:
    def __init__(self, f, t0, x0, t_end, dt=5e-3, **kwargs):
        self.f = f
        self.t = t0
        self.x = x0
        self.t_end = t_end
        self.dt = dt

        for key, value in kwargs.items():
            if key == 'max_step':
                self.dt = value

    def step(self):
        f = self.f
        x = self.x
        t = self.t
        dt = self.dt

        if t < self.t_end:
            k_1 = f(t, x)
            k_2 = f(t + dt / 2, x + (dt / 2) * k_1)
            k_3 = f(t + dt / 2, x + (dt / 2) * k_2)
            k_4 = f(t + dt, x + dt * k_3)

            self.x = x + (dt / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            self.t = t + dt
        else:
            print('End of simulation time reached.')


class RealTimeSimulator(threading.Thread):
    def __init__(self, ps, dt=5e-3, speed=1, solver=RK4_simple):
        threading.Thread.__init__(self)
        self.daemon = True

        self.t_end = 1000
        self.dt = dt
        self.adjust_time = False
        self.speed = speed
        self.running = True

        self.ps = ps
        # self.sol = RK23(self.ps.ode_fun, 0, self.ps.x0, self.t_end, max_step=self.dt, first_step=self.dt)
        self.sol = solver(self.ps.ode_fun, 0, self.ps.x0, self.t_end, max_step=self.dt, first_step=self.dt)
        # self.sol = RK45(self.ps.ode_fun, 0, self.ps.x0, self.t_end, max_step=self.dt, first_step=self.dt)
        # self.sol = BDF(self.ps.ode_fun, 0, self.ps.x0, self.t_end, max_step=self.dt, first_step=self.dt)
        self.ps.ode_fun(0, self.ps.x0)

        self.new_data_cv = threading.Condition()  # condition variable used to both lock and to notify threads
        self.new_data_ready = False

    def run(self):
        self.exit_flag = threading.Event()

        t_start_sim = time.time()
        # t_adj = 0
        t_prev = time.time()
        t_world = 0
        t_err = 0

        while self.running:  # and t < self.t_end:

            # Simulate next step
            t_sim = time.time()
            if self.speed > 0:
                with self.new_data_cv:
                    self.sol.step()
                    self.new_data_ready = True
                    self.new_data_cv.notify()

            t_sim = time.time() - t_sim

            dt = time.time() - t_prev
            t_prev = time.time()
            t_world += dt*self.speed
            t_err = self.sol.t - t_world
            if t_err > 0:
                time.sleep(t_err/self.speed)
            elif t_err < 0:
                print('Overflow! {:.2f} ms.'.format(1000*t_err))
                # if self.adjust_time:
                #     t_adj -= t_err

        return

    def stop(self):
        self.running = False
        if not self.is_alive():
            print('RTS thread stopped.')

    def set_speed(self, speed):
        self.speed = speed







if __name__ == '__main__':
    importlib.reload(dps)

    # import ps_models.n44 as model_data
    import ps_models.n44 as model_data
    # import ps_models.sm_ib as model_data

    model = model_data.load()

    importlib.reload(dps)
    ps = dps.PowerSystemModel(model=model)

    # The below code simply adds generator controls (gov, avr, pss) to all generators. Used for N44 (since these are not defined in the model).
    # ps.gov['TGOV1'] = pd.DataFrame(
    #     columns=[
    #         'name', 'gen', 'R', 'D_t', 'V_min', 'V_max', 'T_1', 'T_2', 'T_3',
    #     ],
    #     data=[
    #         ['GOV' + str(i), name, 0.05, 0.02, 0, 1, 0.5, 1, 2]
    #         for i, name in enumerate(ps.generators['name'].tolist())
    #     ])
    #
    # ps.avr['SEXS'] = pd.DataFrame(
    #     columns=[
    #         'name', 'gen', 'K', 'T_a', 'T_b', 'T_e', 'E_min', 'E_max',
    #     ],
    #     data=[
    #         ['AVR' + str(i), name, 100, 10.0, 10.0, 0.01, -3, 3]
    #         for i, name in enumerate(ps.generators['name'].tolist())
    #     ])
    #
    # ps.pss['STAB1'] = pd.DataFrame(
    #     columns=[
    #         'name', 'gen', 'K', 'T', 'T_1', 'T_2', 'T_3', 'T_4', 'H_lim',
    #     ],
    #     data=[
    #         ['PSS' + str(i), name, 50, 10.0, 0.5, 0.5, 0.5, 0.05, 0.03]
    #         for i, name in enumerate(ps.generators['name'].tolist())
    #     ])


    ps.power_flow()
    ps.init_dyn_sim()
    ps.build_y_bus_red(ps.buses['name'])
    ps.x0[ps.angle_idx][0] += 1e-3
    rts = RealTimeSimulator(ps, dt=5e-3, speed=1, solver=RK4_simple)
    rts.start()

    # print(rts.is_alive())
    from threading import Thread
    # app, main = main(rts)

    time.sleep(10)
    rts.stop()
