"""InterfacerDirect: a base class for what is called by the simulator after every step."""
from types import SimpleNamespace

import tops.dynamic as dps
from tops.simulator import InterfacerDirect, Simulator


class Counter(InterfacerDirect):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def update(self, input):
        self.calls += 1


def test_an_interfacer_given_a_simulator_registers_itself_under_its_name():
    rts = SimpleNamespace(interface_functions={})
    interfacer = InterfacerDirect(rts)
    assert rts.interface_functions == {'InterfacerDirect': interfacer.interface_fun}


def test_the_name_can_be_chosen():
    rts = SimpleNamespace(interface_functions={})
    interfacer = InterfacerDirect(rts, name='controller')
    assert interfacer.interface_name == 'controller'
    assert rts.interface_functions == {'controller': interfacer.interface_fun}


def test_a_registered_interfacer_is_called_after_every_step_of_a_simulation():
    import tops.ps_models.k2a as model_data
    ps = dps.PowerSystemModel(model=model_data.load())
    ps.init_dyn_sim()
    sim = Simulator(ps, dt=5e-3, t_end=0.05)
    steps = []
    sim.interface_functions['steps'] = lambda sim: steps.append(sim.sol.t)
    interfacer = Counter(sim)
    sim.main_loop()
    assert interfacer.calls == len(steps) >= 10
