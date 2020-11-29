from PyQt5 import QtWidgets
import sys
import pyqtgraph as pg
import dynpssimpy.dynamic as dps
import importlib
from pyqtconsole.console import PythonConsole
import dynpssimpy.real_time_sim as dps_rts
import dynpssimpy.gui as gui
import dynpssimpy.utility_functions as dps_uf

import networkx as nx
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore

import matplotlib.pyplot as plt


class GridPlot(QtWidgets.QWidget):
    def __init__(self, rts, update_freq=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rts = rts
        self.ps = rts.ps
        self.dt = self.rts.dt

        G = nx.MultiGraph()
        G.add_nodes_from(ps.buses['name'])
        G.add_edges_from(ps.lines[['from_bus', 'to_bus']])
        G.add_edges_from(ps.transformers[['from_bus', 'to_bus']])

        # nx.draw(G)

        pos = nx.spring_layout(G)
        x = np.zeros(ps.n_bus)
        y = np.zeros(ps.n_bus)
        for i, key in enumerate(pos.keys()):
            x[i], y[i] = pos[key]

        #plt.scatter(x, y)

        n_edges = len(ps.lines) + len(ps.transformers)
        x_edge = np.zeros(2*n_edges)
        y_edge = np.zeros(2*n_edges)
        connect = np.zeros(2*n_edges, dtype=bool)
        i = 0
        for branches in [ps.lines, ps.transformers]:
            for line in branches:
                x_edge[2*i] = x[dps_uf.lookup_strings(line['from_bus'], ps.buses['name'])]
                y_edge[2*i] = y[dps_uf.lookup_strings(line['from_bus'], ps.buses['name'])]
                x_edge[2*i + 1] = x[dps_uf.lookup_strings(line['to_bus'], ps.buses['name'])]
                y_edge[2*i + 1] = y[dps_uf.lookup_strings(line['to_bus'], ps.buses['name'])]
                connect[2*i] = True
                i += 1

        #for load in ps.loads:


        dt = self.dt
        n_samples = 500
        dt = 20e-3

        self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255)
        # Phasor diagram
        self.graphWidget = pg.GraphicsLayoutWidget(show=True, title="Phasors")
        # self.setCentralWidget(self.graphWidget)

        #self.phasor_0 = np.array([0, 1, 0.9, 1, 0.9, 1]) + 1j * np.array([0, 0, -0.1, 0, 0.1, 0])
        plot_win = self.graphWidget.addPlot(title='Phasors')
        plot_win.setAspectLocked(True)

        plot_win.plot(x_edge, y_edge, connect=connect)
        plot_win.plot(x, y, pen=None, symbol='o', symbolBrush='b', symbolSize=15)
        plot_win.plot(x[ps.gen_bus_idx], y[ps.gen_bus_idx], pen=None, symbol='o', symbolBrush='r', symbolSize=15)

        #plot_win.plot(x[ps.gen_bus_idx], y[ps.gen_bus_idx], pen=None, symbol='o', symbolBrush='y', symbolSize=15)

        #self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255)
        #pen = pg.mkPen(color=self.colors(0), width=2)
        #self.graphWidget.plot(self.ts_keeper.time, np.zeros(n_samples), pen=pen)



        #scatter = pg.ScatterPlotItem(size=ps.n_bus, brush=pg.mkBrush(255, 255, 255, 120))

        #angle = self.rts.x[self.ps.gen_mdls['GEN'].state_idx['angle']]
        #magnitude = self.ps.e_q
        #phasors = magnitude*np.exp(1j*angle)

        #self.pl_ph = []
        #for i, phasor in enumerate(phasors[:, None]*self.phasor_0):
        #    pen = pg.mkPen(color=self.colors(i), width=2)
        #    self.pl_ph.append(plot_win_ph.plot(phasor.real, phasor.imag, pen=pen))
        #plot_win_ph.enableAutoRange('xy', False)



        self.graphWidget.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1000/update_freq)

    def update(self):
        pass
    #     # if not np.isclose(self.ts_keeper.time[-1], self.ps.time):
        # Phasors:
    #     angle = self.rts.x[self.ps.gen_mdls['GEN'].state_idx['angle']]
    #     magnitude = self.ps.e_q
    #     phasors = magnitude * np.exp(1j * angle)
    #     for i, (pl_ph, phasor) in enumerate(zip(self.pl_ph, phasors[:, None]*self.phasor_0)):
    #         pl_ph.setData(phasor.real, phasor.imag)


def main(rts):
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    # phasor_plot = gui.PhasorPlot(rts, update_freq=30)
    grid_plt = GridPlot(rts, update_freq=1)
    # ts_plot = gui.TimeSeriesPlot(rts, ['speed', 'angle'], update_freq=30)  # , 'speed', 'e_q_t', 'e_d_t', 'e_q_st', 'e_d_st'])
    # stats_plot = gui.SimulationStatsPlot(rts, update_freq=30)

    # Add Control Widgets
    # line_outage_ctrl = gui.LineOutageWidget(rts)
    # excitation_ctrl = gui.GenCtrlWidget(rts)

    console = PythonConsole()
    console.push_local_ns('rts', rts)
    console.push_local_ns('grid_plt', grid_plt)
    # console.push_local_ns('ts_plot', ts_plot)
    # console.push_local_ns('phasor_plot', phasor_plot)
    # console.push_local_ns('line_outage_ctrl', line_outage_ctrl)
    # console.push_local_ns('excitation_ctrl', excitation_ctrl)
    console.show()
    console.eval_in_thread()

    app.exec_()

    return app


if __name__ == '__main__':

    [importlib.reload(module) for module in [dps, dps_rts, gui]]

    import ps_models.ieee39 as model_data
    model = model_data.load()

    importlib.reload(dps)
    ps = dps.PowerSystemModel(model=model)
    ps.use_numba = False

    ps.power_flow()
    ps.init_dyn_sim()
    ps.build_y_bus_red(ps.buses['name'])
    ps.ode_fun(0, ps.x0)

    rts = dps_rts.RealTimeSimulator(ps, dt=10e-3, speed=0.5, solver=dps_uf.ModifiedEuler)
    rts.sol.n_it = 0
    rts.ode_fun(0, ps.x0)


    #plt.scatter(x, y)



    #rts.start()

    app = main(rts)
    #rts.stop()
