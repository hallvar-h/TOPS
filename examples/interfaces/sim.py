import tops.dynamic as dps
from tops.simulator import Simulator


if __name__ == '__main__':

    import tops.ps_models.ieee39 as model_data
    # import ps_models.ieee39 as model_data
    # import ps_models.sm_ib as model_data

    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    ps.ode_fun(0, ps.x0)
    sim = Simulator(ps, dt=5e-3, t_end=2)
    sim.main_loop()

    print('Done')