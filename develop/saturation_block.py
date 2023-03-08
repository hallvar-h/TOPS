from dynpssimpy.dyn_models.blocks import Saturation
from dynpssimpy.utility_functions import structured_array_from_list
import numpy as np
import matplotlib.pyplot as plt


par =structured_array_from_list(names=['name', 'E_1',      'S_e1',     'E_2',      'S_e2'], entries=[['sat', 6.5,        0.054,	    8,	        0.202]])
sat_block = Saturation(par)

input = []
output = []
for input_signal in np.arange(0, 10, 0.1):
    sat_block.input = lambda x, v: np.array([input_signal])
    input.append(sat_block.input(None, None))
    output.append(sat_block.output(None, None))

plt.plot(input, output)
plt.show()


sqrt = np.sqrt
U = np.arange(0.1, 2, 0.01)
E1 = 6.5
E2 = 8
SE1 = 0.054
SE2 = 0.202
K = SE1/SE2
A = sqrt(E1*E2)*(sqrt(E1) - sqrt(E2*K))/(sqrt(E2) - sqrt(E1*K))
B = SE2*(sqrt(E2) - sqrt(E1*K))**2/(E1 - E2)**2
SE = B*(U - A)**2/U

plt.plot(U, SE)
plt.show()