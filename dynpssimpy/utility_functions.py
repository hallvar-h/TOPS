import numpy as np


def jacobian_num(f, x, eps=1e-10, **params):
    # Numerical computation of Jacobian
    J = np.zeros([len(x), len(x)], dtype=np.float)

    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()

        x1[i] += eps
        x2[i] -= eps

        f1 = f(x1, **params)
        f2 = f(x2, **params)

        J[:, i] = (f1 - f2) / (2 * eps)

    return J


class DynamicModel:
    # Empty dummy-class for dynamic models (Gen, AVR, GOV, PSS etc.)
    def __init__(self):
        pass


class EventManager:
    def __init__(self, events, event_function):
        self.events = events
        self.event_flags = np.ones(len(self.events), dtype=bool)
        self.event_function = event_function

    def update(self, t_now):
        for i, (t_event, sub_events) in enumerate(self.events):
            if t_now >= t_event and self.event_flags[i]:
                self.event_flags[i] = False
                for element_type, name, action in sub_events:
                    self.event_function(element_type, name, action)
                    print(name + ' was ' + action + 'ed.')


