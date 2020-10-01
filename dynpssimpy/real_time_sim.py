from PyQt5 import QtWidgets
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import time
import threading
from scipy.integrate import RK23
sys.path.append(r'C:/Users/lokal_hallvhau/Dropbox/Python/DynPSSimPy/')
import dynpssimpy.dynamic as dps
import importlib
from pyqtconsole.console import PythonConsole


def map_idx_fun(x, y):
    if len(x) > 0:
        index = np.argsort(x)
        sorted_x = x[index]
        sorted_index = np.searchsorted(sorted_x, y)

        yindex = np.take(index, sorted_index, mode="clip")
        mask = x[yindex] != y

        yindex[mask] = -1
        return yindex  # np.ma.array(yindex, mask=mask)
    else:
        return np.zeros(0)


class TimeSeriesKeeper:
    def __init__(self):
        pass


class RealTimeSimulator(threading.Thread):
    def __init__(self, ps):
        threading.Thread.__init__(self)
        self.daemon = True

        self.running = True
        self.t_end = 1000
        self.dt = 20e-3
        self.adjust_time = False
        self.speed = 1

        self.ps = ps
        self.sol = RK23(self.ps.ode_fun, 0, ps.x0, self.t_end, max_step=self.dt)

    def run(self):
        t_start_sim = time.time()
        t_adj = 0
        t_prev = time.time()
        t_world = 0
        while self.running:  # and t < self.t_end:

            # Simulate next step
            if self.speed > 0:
                self.sol.step()

            dt = time.time() - t_prev
            t_prev = time.time()
            t_world += dt*self.speed

            # t_world = time.time() - t_start_sim - t_adj
            # t_world = t_world + real_time_passed*self.speed
            # real_time_passed = time.time() - t_prev

            t_err = self.sol.t - t_world
            if t_err > 0:
                time.sleep(t_err/self.speed)
            elif t_err < 0:
                print('Overflow! {:.2f} ms'.format(1000*t_err))
                if self.adjust_time:
                    t_adj -= t_err

        return

    def stop(self):
        self.running = False
        if not self.is_alive():
            print('RTS thread stopped.')

    def set_speed(self, speed):
        self.speed = speed


class LineOutageWidget(QtWidgets.QWidget):
    def __init__(self, rts, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rts = rts
        self.ps = rts.ps
        self.dt = self.rts.dt

        # dt = self.dt
        n_samples = 500
        dt = 20e-3

        self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255)

        # Controls
        self.ctrlWidget = QtWidgets.QWidget()
        self.ctrlWidget.setWindowTitle('Lines')

        # self.graphWidget_ctrl = pg.GraphicsLayoutWidget(show=True, title="Controls")
        layout_box = QtWidgets.QVBoxLayout()
        self.check_boxes = []
        for i, line in self.ps.lines.iterrows():
            check_box = QtWidgets.QCheckBox(line['name'])
            check_box.setChecked(True)
            check_box.stateChanged.connect(self.updateLines)
            check_box.setAccessibleName(line['name'])

            layout_box.addWidget(check_box)
            # layout_box.addSpacing(15)
            layout_box.addSpacing(0)


        self.ctrlWidget.setLayout(layout_box)
        self.ctrlWidget.show()

    def updateLines(self):
        if self.sender().isChecked():
            action = 'connect'
        else:
            action = 'disconnect'
        self.ps.network_event('lines', self.sender().accessibleName(), action)


class GenCtrlWidget(QtWidgets.QWidget):
    def __init__(self, rts, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rts = rts
        self.ps = rts.ps
        self.dt = self.rts.dt

        # dt = self.dt
        n_samples = 500
        dt = 20e-3

        self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255)

        # Controls
        self.ctrlWidget = QtWidgets.QWidget()
        self.ctrlWidget.setWindowTitle('Generator Controls')


        # self.graphWidget_ctrl = pg.GraphicsLayoutWidget(show=True, title="Controls")
        # layout_box = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QGridLayout()
        self.sliders = []
        avr_to_gen_idx = self.ps.avr_mdls['SEXS'].gen_idx if 'SEXS' in rts.ps.avr_mdls else np.zeros(0)
        gov_to_gen_idx = self.ps.gov_mdls['TGOV1'].gen_idx if 'TGOV1' in rts.ps.gov_mdls else np.zeros(0)
        pss_to_gen_idx = self.ps.pss_mdls['STAB1'].gen_idx if 'STAB1' in rts.ps.pss_mdls else np.zeros(0)
        self.gen_to_avr_idx = map_idx_fun(avr_to_gen_idx, np.arange(self.ps.n_gen))
        self.gen_to_gov_idx = map_idx_fun(gov_to_gen_idx, np.arange(self.ps.n_gen))
        self.gen_to_pss_idx = map_idx_fun(pss_to_gen_idx, np.arange(self.ps.n_gen))
        for i, gen in self.ps.generators.iterrows():
            # y_0 = 1/np.conj(abs(self.ps.v_0[idx_bus]) ** 2 / s_load)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.ctrlWidget)
            slider.setMinimum(0)
            slider.setMaximum(300)
            slider.valueChanged.connect(lambda: self.updateExcitation())
            slider.setAccessibleName(str(i))
            slider.setValue(100*self.ps.e_q[i])
            self.sliders.append(slider)
            layout.addWidget(slider, i, 0)
            for j, (ctrl, idx) in enumerate(zip(['AVR', 'GOV', 'PSS'], [avr_to_gen_idx, gov_to_gen_idx, pss_to_gen_idx])):
                if i in idx:
                    button = QtWidgets.QPushButton(ctrl)
                    # button.setAccessibleName(str(i))
                    button.setCheckable(True)
                    button.setChecked(True)
                    layout.addWidget(button, i, j+1)
                    button.clicked.connect(lambda state, args_=(ctrl, i): self.updateActivation(args_[0], args_[1]))


            # layout_box.addWidget(layout_box_H)
            # layout.addSpacing(15)

        self.ctrlWidget.setLayout(layout)
        self.ctrlWidget.show()

    def updateExcitation(self):
        # print(int(self.sender().accessibleName()), self.sender().value())
        self.ps.e_q[int(self.sender().accessibleName())] = self.sender().value()/100

    def updateActivation(self, ctrl, gen):
        gen_idx = gen
        if ctrl == 'AVR':
            idx = self.gen_to_avr_idx[gen_idx]
            self.ps.avr_mdls['SEXS'].active[idx] = self.sender().isChecked()
            # print(ctrl, gen, self.sender().isChecked())

        if ctrl == 'GOV':
            idx = self.gen_to_gov_idx[gen_idx]
            self.ps.gov_mdls['TGOV1'].active[idx] = self.sender().isChecked()
            # print(ctrl, gen, self.sender().isChecked())


        if ctrl == 'PSS':
            idx = self.gen_to_pss_idx[gen_idx]
            self.ps.pss_mdls['STAB1'].active[idx] = self.sender().isChecked()
            # print(ctrl, gen, self.sender().isChecked())


class LivePlotter(QtWidgets.QMainWindow):
    def __init__(self, rts, plots=['angle', 'speed'], *args, **kwargs):
        super(LivePlotter, self).__init__(*args, **kwargs)

        self.rts = rts
        self.ps = rts.ps
        self.dt = self.rts.dt

        self.graphWidget = pg.GraphicsLayoutWidget(show=True, title="Live plot")
        # self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        dt = self.dt
        n_samples = 500
        dt = 20e-3

        self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255)

        # Phasor diagram
        self.graphWidget_ph = pg.GraphicsLayoutWidget(show=True, title="Phasors")
        self.phasor_0 = np.array([0, 1, 0.9, 1, 0.9, 1]) + 1j * np.array([0, 0, -0.1, 0, 0.1, 0])
        plot_win_ph = self.graphWidget_ph.addPlot(title='Phasors')
        plot_win_ph.setAspectLocked(True)

        phasors = self.ps.e[:, None]*self.phasor_0
        self.pl_ph = []
        for i, phasor in enumerate(phasors):
            self.pl_ph.append(plot_win_ph.plot(phasor.real, phasor.imag, pen=self.colors(i)))
        plot_win_ph.enableAutoRange('xy', False)


        self.plots = plots
        self.ts_keeper = TimeSeriesKeeper()
        self.ts_keeper.time = np.arange(-n_samples*dt, 0, dt)
        # self.y = np.zeros_like(self.ts_keeper.time)
        self.pl = {}


        for plot in self.plots:
            graphWidget = self.graphWidget.addPlot(title=plot)
            # p_1 = self.addPlot(title="Updating plot 1")
            n_series = len(getattr(self.ps, plot))
            setattr(self.ts_keeper, plot, np.zeros((n_samples, n_series)))

            pl_tmp = []
            for i in range(n_series):
                pl_tmp.append(graphWidget.plot(self.ts_keeper.time, np.zeros(n_samples), pen=self.colors(i)))

            self.pl[plot] = pl_tmp
            self.graphWidget.nextRow()


        # self.pl_1 = self.graphWidget.plot(self.ts_keeper.time, self.y)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(20)

    def update(self):
        if not np.isclose(self.ts_keeper.time[-1], self.ps.time):
            # print(np.isclose(self.ts_keeper.time[-1], self.ps.time))
            # print(self.ts_keeper.time[-1]-  self.ps.time)
            self.ts_keeper.time = np.append(self.ts_keeper.time[1:], self.ps.time)

            # Phasors:
            phasors = self.ps.e[:, None] * self.phasor_0
            for i, (pl_ph, phasor) in enumerate(zip(self.pl_ph, phasors)):
                pl_ph.setData(phasor.real, phasor.imag)

            for plot in self.plots:
                old_data = getattr(self.ts_keeper, plot)[1:, :]
                new_data = getattr(self.ps, plot)
                setattr(self.ts_keeper, plot, np.vstack([old_data, new_data]))
                plot_data = getattr(self.ts_keeper, plot)
                for i, pl in enumerate(self.pl[plot]):
                    pl.setData(self.ts_keeper.time, plot_data[:, i])


def main(rts):
    app = QtWidgets.QApplication(sys.argv)
    live_plot = LivePlotter(rts, ['angle', 'speed'])

    # Add Control Widgets
    line_outage_ctrl = LineOutageWidget(rts)
    excitation_ctrl = GenCtrlWidget(rts)

    # console = PythonConsole()
    console = PythonConsole()
    console.push_local_ns('rts', rts)
    console.push_local_ns('main', live_plot)
    console.push_local_ns('line_outage_ctrl', line_outage_ctrl)
    console.push_local_ns('excitation_ctrl', excitation_ctrl)
    console.show()
    console.eval_in_thread()
    live_plot.show()
    app.exec_()

    return app
    # sys.exit(app.exec_())


if __name__ == '__main__':

    import ps_models.k2a as model_data
    model = model_data.load()

    importlib.reload(dps)
    ps = dps.PowerSystemModel(model=model)
    # ps.pss = {}
    ps.avr['SEXS']['T_e'] = 0.25
    ps.avr['SEXS']['T_a'] = 5
    ps.gov['TGOV1']['T_1'] = 0.5
    ps.gov['TGOV1']['T_2'] = 1
    ps.gov['TGOV1']['T_3'] = 2

    ps.power_flow()
    ps.init_dyn_sim()
    ps.x0[ps.angle_idx][0] += 1e-3
    rts = RealTimeSimulator(ps)
    rts.start()

    from threading import Thread
    app = main(rts)
    rts.stop()