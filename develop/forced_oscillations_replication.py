from dynpssimpy.dyn_models.utils import DAEModel, output
from dynpssimpy.dyn_models.blocks import *


class SecondOrderSystem1(DAEModel):
    def state_list(self):
        return ['x_1', 'x_2']

    @output
    def output(self, x, v):
        X = self.local_view(x)
        return X['x_2']

    def state_derivatives(self, dx, x, v):
        p = self.par
        dX = self.local_view(dx)
        X = self.local_view(x)
        u = self.input
        dX['x_1'][:] = -X['x_2']
        dX['x_2'][:] = 1/(p['a']*p['T_2']**2)*(X['x_1'] - p['T_1']*u(x, v))


class HYGOVBlockDiagram(DAEModel):
    def add_blocks(self):
        p = self.par
        self.integrator = Integrator(K=p['K_i'])
        self.servo = TimeConstant(T=p['T_y'])
        self.backlash = Backlash(...)
        self.second_order_block = SecondOrderSystem1(T_1=p['T_w'], T_2=p['T_e'])
        self.time_constant_T_a = TimeConstantVar(T=p['T_a'], e=p['e_g'])

        def sum_1(x, v): return self.input(x, v) - self.output(x, v)
        def sum_2(x, v): return sum_1(x, v) + p['b_p']*sum_3_pi
        def sum_3_pi(x, v): return p['K_p']*sum_2(x, v) + self.integrator.output(x, v)
        def h(x, v): return self.second_order_block.output

        self.integrator.input = sum_2
        self.output = self.time_constant_T_a.output
        self.servo.input = sum_3_pi
        self.backlash.input = self.servo.output
        y = self.backlash.output

        def sum_4(x, v): return p['e_qy']*y(x, v) + p['e_qx']*self.output(x, v) + p['e_qh']*h(x, v)
        def sum_5_P_m(x, v): return p['e_y']*y(x, v) + p['e_x']*self.output(x, v) + p['e_h']*h(x, v)

        def sum_6(x, v): return sum_5_P_m(x, v) - self.input_P_m0(x, v)
        self.time_constant_T_a.input = sum_6


if __name__ == '__main__':
    mdl = HYGOVBlockDiagram(
        K_p=[],
        K_i=[],
        b_p=[],
        T_y=[],
        T_a=[],
        db=[],
        e_qy=[],
        e_qx=[],
        e_qh=[],
        e_y=[],
        e_h=[],
        e_x=[],
        e_g=[],        
        )
    