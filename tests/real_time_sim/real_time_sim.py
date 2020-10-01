import dynpssimpy.dynamic as dps
import dynpssimpy.real_time_sim as dps_rts
import importlib


if __name__ == '__main__':

    import ps_models.k2a as model_data
    model = model_data.load()

    [importlib.reload(mdl) for mdl in [dps, dps_rts]]
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
    rts = dps_rts.RealTimeSimulator(ps)
    rts.start()

    from threading import Thread
    app = dps_rts.main(rts)
    rts.stop()