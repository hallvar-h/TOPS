# Synchronous machine connected to passive load

def load():
    return {
        'base_mva': 900,
        'f': 50,

        'buses': [
                ['name',    'V_n'],
                ['B1',      20],
                ['B2',      20],
        ],

        'lines': [
                ['name',    'from_bus',    'to_bus',    'length',   'S_n',  'V_n',  'unit',     'R',    'X',    'B'],
                ['L1-2',    'B1',          'B2',        25,         900,    20,     'PF',       1e-4,   1e-3,   1.75e-3*0],
        ],

        'loads': [
            ['name',    'bus',  'P',    'Q',    'model'],
            ['L1',      'B2',   600,    200,    'Z'],
        ],

        'generators': {
            'GEN': [
                ['name',    'bus',  'S_n',  'V_n',  'P',    'V',    'H',    'D',    'X_d',      'X_q',  'X_d_t',    'X_q_t',    'X_d_st',   'X_q_st',   'T_d0_t',   'T_q0_t',   'T_d0_st',  'T_q0_st'],
                ['G1',      'B1',   900,    20,     700,    1,      6.5,    0,      1.8,        1.7,    0.3,        0.3,        0.2,        0.2,        8.0,        0.6,        0.05,       0.05],
            ],
        },
    }