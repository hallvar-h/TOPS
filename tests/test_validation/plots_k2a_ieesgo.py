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

    model['gov'] = {
        'TGOV1': [model['gov']['TGOV1'][0], *model['gov']['TGOV1'][2:]],
        'IEESGO': [
            ['name',        'gen',  'T_1',  'T_2',  'K_1',  'T_3',  'T_4',  'K_3',  'T_6',  'K_2',  'T_5',  'P_N',  'P_min',    'P_max' ],
            # ['IEESGO-1',    'G1',   0.01,	0,	    0,	    0.15,	0.3,    0.43,   0.4,	0.7,	8,	    1300,	0,	        1       ],
            ['IEESGO-1',    'G1',   0.3,    1.,     10.,    1.,     0.1,    0.4,    0.2,    0.4,    1.,     0.,     0.,         1.]   
        ]
    }

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(np.max(abs(ps.ode_fun(0, ps.x_0))))

    ps.gen['GEN'].P_m(ps.x_0, ps.v_0)

    t_end = 30
    max_step = 5e-3

    # PowerFactory result

    # os.chdir()
    __file__ = r'C:\Users\hallvarh\Coding\dynpssimpy\tests\test_validation\plots_k2a_ieesgo.py'
    file_path = pathlib.Path(__file__).parent
    pf_res = val_fun.load_pf_res(str(file_path) + '/k2a_ieesgo/powerfactory_res.csv')

    # Python result
    x0 = ps.x0
    sol = RK45(ps.ode_fun, 0, x0, t_end, max_step=5e-3)

    t = 0
    result_dict = defaultdict(list)
    load_bus_idx = ps.loads['Load'].bus_idx_red['terminal'][0]
    y_load_0 = ps.loads['Load'].y_load.real[0]


    print('Running dynamic simulation')
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t / (t_end) * 100))

        # Simulate next step
        result = sol.step()
        t = sol.t
        x = sol.y
        v = ps.solve_algebraic(t, x)

        if t > 1:
            ps.y_bus_red_mod[(load_bus_idx,) * 2] = y_load_0*0.1

        # Store result variables
        result_dict['Global', 't'].append(sol.t)
        _ = [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, sol.y)]

    index = pd.MultiIndex.from_tuples(result_dict)
    result = pd.DataFrame(result_dict, columns=index)

    # Computing the error
    val_fun.generate_plots(ps, result, pf_res, max_step, choose_plots='basic')

    print(val_fun.compute_error(ps, result, pf_res, max_step))