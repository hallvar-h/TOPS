import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import rcParams
rcParams['font.family'] = 'DejaVu Serif'
rcParams['font.serif'] = ['Computer Modern']

results = []
names = []
folder_path = Path('Results/Wind')
for name in sorted(folder_path.iterdir()):
    try:
        with open(name,'r') as file:
            res = json.load(file)
            results.append(res)
            formatted_string = name.stem.replace('_', '=')
            names.append(formatted_string)
    except:
        pass

for res in results:
    for key, value in res.items():
        if key != 't':
            # Iterate through the list of timesteps (assumed to be lists or arrays)
            for i, timestep in enumerate(value):  # Use enumerate to modify the list in-place
                for j, v in enumerate(res[key][i]):  # Iterate through each value in the timestep
                    if isinstance(v, str) and 'j' in v:  # Check if it's a string that represents a complex number
                        try:
                            res[key][i][j] = complex(v)  # Convert the string back to a complex number
                        except ValueError:
                            pass  # In case the string is not a valid complex number


plt.figure()
i = 0
for res in results:
    plt.plot(res['t'],50+50*np.mean((res['gen_speed']),axis = 1), label = names[i])
    i+=1
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('Frequency response for disconnection of a 400MW HVDC connection')
plt.grid()
plt.legend()
plt.show()