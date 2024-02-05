# Synchronous machine connected to infinite bus

def load():
    return {
        'base_mva': 2200,
        'f': 60,
        'slack_bus': 'B2',

        'buses': [
            ['name',    'V_n'],
            ['B1',      24],
            ['B2',      24],
        ],

        'lines': [
            ['name',    'from_bus', 'to_bus',   'length',   'S_n',  'V_n',  'unit', 'R',    'X',      'B'],
            ['L1-2',    'B1',       'B2',       1,         2200,    24,     'p.u.',   0,   0.65,     0],
        ],

        'generators': {
            'GEN': [
                ['name',    'bus',  'S_n',  'V_n',  'P',    'V',    'H',    'D',    'X_d',  'X_q',  'X_d_t',    'X_q_t',    'X_d_st',   'X_q_st',   'T_d0_t',   'T_q0_t',   'T_d0_st',  'T_q0_st'],
                ['G1',      'B1',   2200,   24,     1998,   1,      3.5,    0,      1.81,   1.76,   0.3,        0.65,       0.23,        0.23,      8.0,        1,          0.03,       0.07],
                ['IB',      'B2',   2200*10, 24,    -1998,  0.995,  3.5e7,  0,      1.8,    1.8,    0.3,        0.65,       0.23,        0.23,      8,          1,          0.03,       0.07],
            ]
        },

        'gov': {
            'TGOV1': [
                ['name', 'gen', 'R', 'D_t', 'V_min', 'V_max', 'T_1', 'T_2', 'T_3'],
                ['GOV1', 'G1', 0.05, 0.02, 0, 1, 0.5, 2, 2],
            ]
        },

        'avr': {
            'SEXS': [
                ['name', 'gen', 'K', 'T_a', 'T_b', 'T_e', 'E_min', 'E_max'],
                ['AVR1', 'G1', 100, 4.0, 10.0, 0.1, -3, 3],
            ]
        }
    }
