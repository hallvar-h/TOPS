import dynpssimpy.dynamic as dps
import dynpssimpy.real_time_sim as dps_rts
import importlib
import pyqtgraph as pg
import dynpssimpy.gui as dps_rts_gui
import time


if __name__ == '__main__':

    [importlib.reload(module) for module in [dps, dps_rts, dps_rts_gui]]
    import ps_models.ieee39 as model_data
    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.use_numba = True
    ps.power_flow()
    ps.init_dyn_sim()
    ps.ode_fun(0, ps.x0)
    ps.build_y_bus_red(ps.buses['name'])
    ps.x0[ps.angle_idx][0] += 1e-3
    rts = dps_rts.RealTimeSimulator(ps, dt=5e-3)
    rts.start()

    from threading import Thread
    # app = dps_rts.main(rts)
    time.sleep(2)
    rts.stop()