import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)
import numpy as np
import json

if __name__ == '__main__':
    #for iteration in np.arange(550,900,50):
        # Load model
        import Wind as model_data
        importlib.reload(model_data)
        model = model_data.load()
        model['loads'] = {'DynamicLoad': model['loads']}



    # 'vsc': {
    #         'VSC_PQ': [
    #             ['name', 'bus', 'S_n', 'p_ref', 'q_ref',  'k_p', 'k_q', 'T_p', 'T_q', 'k_pll','T_pll', 'T_i', 'i_max', 'K_SI, 'T_rocof'],
    #             ['VSC1', 'B1',    50,     1,       0,       1,      1,    0.1,   0.1,     5,      1,      0.01,    1.2, 10, 1],
    #         ],
    #     }

#     model['vsc'] =  {'VSC_PQ_SI' : [
#         ['name', 'bus', 'S_n', 'p_ref', 'q_ref',  'k_p', 'k_q', 'T_p', 'T_q', 'k_pll','T_pll', 'T_i', 'i_max', 'K_SI', 'T_rocof'],
#          ['VSC_SI', 'B1',    900,     0.7,       0.1,       1,      1,    0.1,   0.1,     5,      1,   1,   0.01, 1.2,   1, 1],
#          #['HVDC', 'B8',    900,     0.4,       0.1,       1,      1,    0.1,   0.1,     5,      1,    1,  0.01,  1.2,  0, 1],

# ]}
    
#         model['vsc'] =  {'VSC_PQ_SI': [
#                 ['name', 'bus', 'S_n', 'p_ref', 'q_ref',  'k_p', 'k_q', 'T_p', 'T_q', 'k_pll','T_pll', 'T_i', 'i_max', 'K_SI', 'T_rocof'],
#                 ['VSC_SI', 'B1',    900,     0.7,       0.1,       1,      1,    0.1,   0.1,     5,      1,      0.01,    1.2, 0, 0.5],
#                 ['VSC_HVDC', 'B8',    300,     1,       0.333,       1,      1,    0.1,   0.1,     5,      1,      0.01,    1.2, 0, 1],
# ]}

        model['vsc'] =  {'VSC_PQ': [
                ['name', 'bus', 'S_n', 'p_ref', 'q_ref',      'k_p', 'k_q', 'T_p', 'T_q', 'k_pll','T_pll', 'T_i', 'i_max'],
                ['VSC1', 'B1',    700,     1,       0.15,       0.1,      0.1,    1,   1,     5,      1,      1,    1.2],
                ['VSC1', 'B8',    300,     1,       0.33,       0.1,      0.1,    1,   1,     5,      1,      1,    1.2],
]}
    



    # model['vsc'] = {'VSC': [
    #     ['name',    'T_pll',    'T_i',  'bus',  'P_K_p',    'P_K_i',    'Q_K_p',    'Q_K_i',    'P_setp',   'Q_setp',   ],
    #     ['HVDC',    0.1,        1,      'B8',   0.1,        0.1,        0.1,        0.1,        300,          100],
    # ]}

        # Power system model
        ps = dps.PowerSystemModel(model=model)
        ps.init_dyn_sim()
        #print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

        t_end = 50
        x_0 = ps.x_0.copy()

        # Solver
        sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

        # Initialize simulation
        t = 0
        res = defaultdict(list)
        t_0 = time.time()

        sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]



    
    # Run simulation
    
        ffr = False
        t_ffr = 0
        t_start = 0
        P_fin = 0
        #ps.gen['GEN'].par['H'][3] = iteration
        while t < t_end:
            sys.stdout.write("\r%d%%" % (t/(t_end)*100))

            #if 10 <= t:
                #ps.loads['DynamicLoad'].set_input('g_setp', 1.6, 0)
                # ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][0], 'disconnect')

            
            
            

            # Simulate next step
            result = sol.step()
            x = sol.y
            v = sol.v
            t = sol.t

            # if((abs(v[0])<0.97)):
            #     q_con = (1-abs(v[0]))*1500
            #     ps.vsc['VSC'].set_input('Q_setp',100+q_con,1)

            
            # if (50+50*np.mean(ps.gen['GEN'].speed(x, v)) <=49.7) and ffr ==False:
            #     t_start = t+1.3
            #     t_ffr = t_start+30
            #     ffr = True
            
            # # if(t_start <= t <=t_ffr and ffr == True):
            # #     k = 180
            # #     f_dev = np.mean(ps.gen['GEN'].speed(x, v))
            # #     Pcontrol = 500-k*50*f_dev
            # #     ps.vsc['VSC'].set_input('P_setp', Pcontrol)
            # #     P_fin = Pcontrol


            # if(t_start <= t <=t_ffr and ffr == True):
            #     ps.vsc['VSC'].set_input('P_setp', 1000,0)




            dx = ps.ode_fun(0, ps.x_0)




            # Store result
            res['t'].append(t)
            res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
            res['v'].append(v.copy())
            res['gen_I'].append(ps.gen['GEN'].I(x, v).copy())
            res['gen_P'].append(ps.gen['GEN'].P_e(x, v).copy())
            res['load_P'].append(ps.loads['DynamicLoad'].P(x, v).copy())
            res['load_Q'].append(ps.loads['DynamicLoad'].Q(x, v).copy())
            # res['VSC_SI'].append(ps.vsc['VSC_PQ_SI'].p_e(x,v).copy()*ps.vsc['VSC_PQ_SI'].par['S_n'])
            res['VSC_SI'].append(ps.vsc['VSC_PQ'].p_e(x,v).copy())
            res['bus_names'].append(ps.buses['name'])

        print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))
        

        for key, value in res.items():
        # Iterate through the list of timesteps (assumed to be lists or arrays)
            for i, timestep in enumerate(value):  # Use enumerate to modify the list in-place
                if isinstance(timestep, np.ndarray):  # Check if it's a NumPy array
                    res[key][i] = timestep.tolist()  # Convert the NumPy array to a list
        for key, value in res.items():
            if(key != 't'):
            # Iterate through the list of timesteps (assumed to be lists or arrays)
                for i, timestep in enumerate(value):  # Use enumerate to modify the list in-place
                    for j, v in enumerate(res[key][i]):  # Iterate through each value in the timestep
                        if isinstance(v, complex):  # Check if it's a complex number
                            res[key][i][j] = str(v)  # Convert the complex number to a string
        with open('Results/SI/base2.json','w') as file:
            json.dump(res,file)
