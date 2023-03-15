from dynpssimpy.dyn_models.blocks import *
from dynpssimpy.dyn_models.vsc import VSC

class WindGenerator(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def add_blocks(self):
        p = self.parza
        self.vsc = VSC(
            T_pll=p['T_pll'],
            T_i=p['T_i'],
            bus=p['from_bus'],
            P_K_p=p['P_K_p'],
            P_K_i=p['P_K_i'],
            Q_K_p=p['Q_K_p'],
            Q_K_i=p['Q_K_i'],
            P_setp=p['P_setp'],
            Q_setp=p['Q_setp']
        )

    def input_list(self):
        return ['P_setp', 'Q_setp']