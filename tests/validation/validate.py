import unittest
from tests.validation.validation_functions import generate_plots, load_pf_res, compute_error
import pandas as pd
from collections import defaultdict
import dynpssimpy.dynamic as dps
from scipy.integrate import RK45


class MyTestCase(unittest.TestCase):
    def test_k2a(self):
        import ps_models.k2a as model_data
        model = model_data.load()

        ps = dps.PowerSystemModel(model=model)
        ps.avr['SEXS']['T_a'] = 2
        ps.avr['SEXS']['T_e'] = 0.1
        ps.gov['TGOV1']['T_1'] = 0.5
        ps.gov['TGOV1']['T_2'] = 1
        ps.gov['TGOV1']['T_3'] = 2
        ps.pf_max_it = 100
        ps.power_flow()
        ps.init_dyn_sim()

        t_end = 10
        max_step = 5e-3

        # PowerFactory result
        pf_res = load_pf_res('k2a/powerfactory_res.csv')

        # Python result
        x0 = ps.x0
        sol = RK45(ps.ode_fun, 0, x0, t_end, max_step=5e-3)

        t = 0
        result_dict = defaultdict(list)

        print('Running dynamic simulation')
        while t < t_end:
            # print(t)

            # Simulate next step
            result = sol.step()
            t = sol.t

            if t >= 1 and t <= 1.1:
                # print('Event!')
                ps.y_bus_red_mod[0, 0] = 1e6
            else:
                ps.y_bus_red_mod[0, 0] = 0

            # Store result variables
            result_dict['Global', 't'].append(sol.t)
            [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, sol.y)]

        index = pd.MultiIndex.from_tuples(result_dict)
        result = pd.DataFrame(result_dict, columns=index)

        error = compute_error(ps, result, pf_res, max_step)
        self.assertLessEqual(error, 0.02)

    def test_ieee39(self):
        import ps_models.ieee39 as model_data
        model = model_data.load()

        ps = dps.PowerSystemModel(model=model)
        ps.pf_max_it = 100
        ps.power_flow()
        ps.init_dyn_sim()
        print(max(abs(ps.ode_fun(0, ps.x0))))

        t_end = 10
        max_step = 5e-3

        # PowerFactory result
        pf_res = load_pf_res('ieee39/powerfactory_res.csv')

        x0 = ps.x0

        sol = RK45(ps.ode_fun, 0, x0, t_end, max_step=5e-3)

        t = 0
        result_dict = defaultdict(list)
        monitored_variables = ['e_q', 'v_g', 'v_g_dev', 'v_pss']

        print('Running dynamic simulation')
        while t < t_end:
            # print(t)

            # Simulate next step
            result = sol.step()
            t = sol.t

            if t >= 1 and t <= 1.05:
                # print('Event!')
                ps.y_bus_red_mod[0, 0] = 1e6
            else:
                ps.y_bus_red_mod[0, 0] = 0

            # Store result variables
            result_dict['Global', 't'].append(sol.t)
            for var in monitored_variables:
                [result_dict[(gen_name, var)].append(var_) for i, (var_, gen_name) in
                 enumerate(zip(getattr(ps, var), ps.generators['name']))]

            [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, sol.y)]

        index = pd.MultiIndex.from_tuples(result_dict)
        result = pd.DataFrame(result_dict, columns=index)

        error = compute_error(ps, result, pf_res, max_step)
        self.assertLessEqual(error, 0.02)


if __name__ == '__main__':
    unittest.main()
