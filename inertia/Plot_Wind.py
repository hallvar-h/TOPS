import json
import numpy as np
import matplotlib.pyplot as plt
results = []
names = []
with open('Results/Wind/FFR0.json','r') as file:
    res = json.load(file)
    results.append(res)
    names.append('Basecase')
with open('Results/Wind/reg_V/reg_V_100.json','r') as file:
    res = json.load(file)
    results.append(res)
    names.append('Trying to regulate v k=100')
with open('Results/Wind/reg_V/reg_V_500.json','r') as file:
    res = json.load(file)
    results.append(res)
    names.append('Trying to regulate v k=500')
with open('Results/Wind/reg_V/reg_V_1500.json','r') as file:
    res = json.load(file)
    results.append(res)
    names.append('Trying to regulate v k=1500')

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


fig = plt.figure()
i = 0
for res in results:
    bus7 = [row[8] for row in res['v']]
    plt.plot(res['t'], np.abs(np.array(bus7)),label = names[i] + ' bus 7')
    i+=1
i = 0
for res in results:
    bus7 = [row[10] for row in res['v']]
    plt.plot(res['t'], np.abs(np.array(bus7)),label = names[i] + ' bus 9')
    i+=1
plt.xlabel('Time [s]')
plt.ylabel('Bus voltage')
plt.legend()
# plt.show()


# fig = plt.figure()
# plt.plot(res['t'], np.abs(res['VSC']))
# plt.xlabel('Time [s]')
# plt.ylabel('MW')


plt.figure()
i = 0
for res in results:
    plt.plot(res['t'],50+50*np.mean((res['gen_speed']),axis = 1), label = names[i])
    i+=1
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('Fast frequency response to a disconnection 400MW HVDC connection')
plt.grid()
plt.legend()
plt.show()