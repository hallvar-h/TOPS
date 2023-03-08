from collections import defaultdict
try:
    import validation_functions as val_fun
except ImportError:
    import tests.test_validation.validation_functions as val_fun
import dynpssimpy.dynamic as dps
import importlib
from scipy.integrate import RK45
import pandas as pd
import sys
import pathlib
import numpy as np


if __name__ == '__main__':

    import dynpssimpy.ps_models.k2a_val as model_data
    model = model_data.load()

    [importlib.reload(mdl) for mdl in [model_data, dps]]

    model['avr'] = {
        'SEXS': [model['avr']['SEXS'][0], *model['avr']['SEXS'][2:]],
        'SCRX': [
            ['name',    'gen',  'T_b',  'T_a',      'K',    'T_e',  'C_switch', 'rc_rfd', 'E_min',    'E_max'],
            ['SCRX-1',  'G1',   13,     3.30005,    61,     0.05,   0,           0,         0,          4],
        ]
    }


    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(np.max(abs(ps.ode_fun(0, ps.x_0))))

    ps.gen['GEN'].P_m(ps.x_0, ps.v_0)
    x, v = x0, v0 = ps.x_0, ps.v_0
    # ps.avr['IEEET1'].output(ps.x_0, ps.v_0)
    from dynpssimpy.dyn_models.utils import auto_init
    output_0 = ps.gen['GEN']._input_values['E_f'][0:1]
    mdl = ps.avr['SCRX']

    mdl.output(x, v)
    mdl.v_t(x, v)
    mdl.v_error(x, v)
    mdl.time_constant_gain_K_Te.input(x, v)
    mdl.lead_lag_Tb_Ta.input(x, v)
    mdl.time_constant_gain_K_Te.output(x, v)
    # mdl.time_constant_gain_Ke_Te.output(x, v)
    # mdl.time_constant_gain_Ke_Te.output(x, v)
    
    # auto_init(mdl, x0, v0, output_0)

    t_end = 15
    max_step = 5e-3

    # PowerFactory result

    # os.chdir()
    file_path = pathlib.Path(__file__).parent
    pf_res = val_fun.load_pf_res(str(file_path) + '/k2a_scrx/powerfactory_res.csv')

    # Python result
    x0 = ps.x0
    sol = RK45(ps.ode_fun, 0, x0, t_end, max_step=5e-3)

    t = 0
    result_dict = defaultdict(list)
    sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]


    print('Running dynamic simulation')
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t / (t_end) * 100))

        # Simulate next step
        result = sol.step()
        t = sol.t
        x = sol.y
        v = ps.solve_algebraic(t, x)

        if t >= 1 and t <= 1.1:
            # print('Event!')
            ps.y_bus_red_mod[(sc_bus_idx,)*2] = 1e6
        else:
            ps.y_bus_red_mod[(sc_bus_idx,)*2] = 0

        # Store result variables
        result_dict['Global', 't'].append(sol.t)
        _ = [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, sol.y)]

    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    # Computing the error
    val_fun.generate_plots(ps, result, pf_res, max_step, choose_plots='basic')

    print(val_fun.compute_error(ps, result, pf_res, max_step))