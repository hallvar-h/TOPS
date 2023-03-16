# Synchronous machine connected to infinite bus

def load():
    return {
        'base_mva': 50,
        'f': 50,
        'slack_bus': 'B3',

        'buses': [
            ['name',    'V_n'],
            ['B1',      10],
            ['B2',      245],
            ['B3',      245],
        ],

        'lines': [
            ['name',    'from_bus', 'to_bus',   'length',   'S_n',  'V_n',  'unit', 'R',    'X',      'B'],
            ['L1-2',    'B2',       'B3',       250,         50,    245,     'ohm',   0,   0.4,     0],
        ],

        'transformers': [
            ['name',    'from_bus', 'to_bus',   'S_n',  'V_n_from', 'V_n_to',   'R',    'X'],
            ['T1',      'B1',       'B2',       50,    10,         245,        0,      0.1],
        ],

        'loads': [
            ['name', 'bus', 'P', 'Q', 'model'],
            ['L1', 'B2', 25, 0, 'Z'],
        ],

        'generators': {
                'GEN': [
                ['name',    'bus',  'S_n',  'V_n',  'P',        'V',      'H',  'D',   'X_d',  'X_q',  'X_d_t',    'X_q_t',    'X_d_st',   'X_q_st',   'T_d0_t',   'T_q0_t',   'T_d0_st',  'T_q0_st'],
                ['G1',       'B1',     50,     10,   40,  0.9331671,      3.1,    0,    1.05,   0.66,    0.328,       0.66,       0.254,      0.254,       2.49,        100,        0.06,       0.15], # x_q_st = 0.273
                ['IB',       'B3',     50,    245,    0,      0.898,  999999.,    0,   1e-10,  1e-10,    1e-10,      1e-10,       1e-10,      1e-10,        100,        100,         100,        100],
            ]
        },

        'gov': {
            'MYGOV': [
                ['name', 'gen', 'R', 'K','Kw'],
                ['GOV1', 'G1', 0.05, 100, 10],
                ]
        },

        'avr': {
            'SEXS': [
                ['name', 'gen', 'K', 'T_a', 'T_b', 'T_e', 'E_min', 'E_max'],
                ['AVR1', 'G1', 100, 2.0, 10.0, 0.5, -3, 3],
            ]
        },
    }