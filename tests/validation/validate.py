import unittest
from tests.validation.validation_functions import load_pf_res, compute_error
import pandas as pd
from collections import defaultdict
import dynpssimpy.dynamic as dps
from scipy.integrate import RK45
import sys


class MyTestCase(unittest.TestCase):
    def test_k2a(self):
        # Test to compare simulation of K2A system with PowerFactory results.
        # Error should be bounded by specified value.
        import dynpssimpy.ps_models.k2a_val as model_data
        model = model_data.load()

        ps = dps.PowerSystemModel(model=model)
        # ps.power_flow()
        ps.init_dyn_sim()

        t_end = 10
        max_step = 5e-3

        # PowerFactory result
        pf_res = load_pf_res('k2a/powerfactory_res.csv')

        # Python result
        x0 = ps.x0
        sol = RK45(ps.ode_fun, 0, x0, t_end, max_step=max_step)

        t = 0
        result_dict = defaultdict(list)
        sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]

        print('Running dynamic simulation')
        while t < t_end:
            sys.stdout.write("\r%d%%" % (t / (t_end) * 100))

            # Simulate next step
            result = sol.step()
            t = sol.t

            if t >= 1 and t <= 1.1:
                # print('Event!')
                ps.y_bus_red_mod[(sc_bus_idx,)*2] = 1e6
            else:
                ps.y_bus_red_mod[(sc_bus_idx,)*2] = 0

            # Store result variables
            result_dict['Global', 't'].append(sol.t)
            [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, sol.y)]

        index = pd.MultiIndex.from_tuples(result_dict)
        result = pd.DataFrame(result_dict, columns=index)

        error = compute_error(ps, result, pf_res, max_step)
        self.assertLessEqual(error, 0.02)

    def test_ieee39(self):
        # Test to compare simulation of IEEE 39 bus system with PowerFactory results.
        # Error should be bounded by specified value.
        import dynpssimpy.ps_models.ieee39 as model_data
        model = model_data.load()

        ps = dps.PowerSystemModel(model=model)
        # ps.power_flow()
        ps.init_dyn_sim()

        t_end = 10
        max_step = 5e-3

        # PowerFactory result
        pf_res = load_pf_res('ieee39/powerfactory_res.csv')
        # pf_res = load_pf_res('tests/validation/ieee39/powerfactory_res.csv')

        x0 = ps.x0

        sol = RK45(ps.ode_fun, 0, x0, t_end, max_step=max_step)

        t = 0
        result_dict = defaultdict(list)
        # monitored_variables = ['e_q', 'v_g', 'v_g_dev', 'v_pss']
        sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]

        print('Running dynamic simulation')
        while t < t_end:
            sys.stdout.write("\r%d%%" % (t / (t_end) * 100))
            # print(t)

            # Simulate next step
            result = sol.step()
            t = sol.t

            if t >= 1 and t <= 1.05:
                # print('Event!')
                ps.y_bus_red_mod[(sc_bus_idx,)*2] = 1e6
            else:
                ps.y_bus_red_mod[(sc_bus_idx,)*2] = 0

            # Store result variables
            result_dict['Global', 't'].append(sol.t)
            # for var in monitored_variables:
            #     [result_dict[(gen_name, var)].append(var_) for i, (var_, gen_name) in
            #      enumerate(zip(getattr(ps, var), ps.generators['name']))]

            [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, sol.y)]

        index = pd.MultiIndex.from_tuples(result_dict)
        result = pd.DataFrame(result_dict, columns=index)

        error = compute_error(ps, result, pf_res, max_step)
        self.assertLessEqual(error, 0.02)

    def test_n44_init(self):
        # Test to verify that systems initialize properly.
        import dynpssimpy.ps_models.sm_ib as model_data_smib
        import dynpssimpy.ps_models.k2a as model_data_k2a
        import dynpssimpy.ps_models.ieee39 as model_data_ieee39
        import dynpssimpy.ps_models.n44 as model_data_n44

        for model_data in [model_data_smib, model_data_k2a, model_data_ieee39, model_data_n44]:
            model = model_data.load()
            ps = dps.PowerSystemModel(model=model)
            # ps.power_flow()
            ps.init_dyn_sim()
            diff_max = max(abs(ps.ode_fun(0, ps.x0)))
            self.assertLessEqual(diff_max, 1e-8)


if __name__ == '__main__':
    unittest.main()
