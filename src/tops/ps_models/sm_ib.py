# Synchronous machine connected to infinite bus

def load():
    return {
        'base_mva': 900,
        'f': 50,
        'slack_bus': 'B2',

        'buses': [
            ['name',    'V_n'],
            ['B1',      20],
            ['B2',      20],
        ],

        'lines': [
            ['name',    'from_bus', 'to_bus',   'length',   'S_n',  'V_n',  'unit', 'R',    'X',      'B'],
            ['L1-2',    'B1',       'B2',       25,         900,    20,     'PF',   1e-4,   1e-3,     0],
        ],

        'generators': {
            'GEN': [
                ['name',    'bus',  'S_n',  'V_n',  'P',    'V',        'H',        'D',    'X_d',  'X_q',  'X_d_t',    'X_q_t',    'X_d_st',   'X_q_st',   'T_d0_t',   'T_q0_t',   'T_d0_st',  'T_q0_st'],
                ['G1',      'B1',   900,    20,     600,    1,          6.5,        0,      1.8,    1.7,    0.3,        0.3,        0.2,        0.2,        8.0,        0.6,        0.05,       0.05],
                ['IB',      'B2',   100000,  20,     -600,   1,          999999.,    0,      1.8,    1.8,    0.3,       0.3,        0.2,        0.2,        6.67,       6.67,       0.15,       0.15],
            ],
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
                ['AVR1', 'G1', 100, 2.0, 10.0, 0.1, -10, 10],
            ]
        }
    }
