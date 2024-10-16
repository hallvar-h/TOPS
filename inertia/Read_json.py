import json
import numpy as np
import matplotlib.pyplot as plt
results = []
names = []
with open('Basecase.json','r') as file:
    res = json.load(file)
    results.append(res)
    names.append('Basecase')
with open('Wind.json','r') as file2:
    res2 = json.load(file2)
    results.append(res2)
    names.append('Wind')
# with open('FFR.json','r') as file3:
#     res3 = json.load(file3)
#     results.append(res3)
#     names.append('FFR')
with open('Ny_FFR.json','r') as file6:
    res6 = json.load(file6)
    results.append(res6)
    names.append('FFR')
with open('FFR_and_SC.json','r') as file4:
    res4 = json.load(file4)
    results.append(res4)
    names.append('FFR_and_SC')
with open('SC.json','r') as file5:
    res5 = json.load(file5)
    results.append(res5)
    names.append('SC')

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


# fig = plt.figure()
# i = 0
# for res in results:
#     bus1 = [row[8] for row in res['v']]
#     plt.plot(res['t'], np.abs(np.array(bus1)),label = names[i])
#     i+=1

# plt.xlabel('Time [s]')
# plt.ylabel('Bus voltage')
# plt.legend()
# # plt.show()


# fig = plt.figure()
# plt.plot(res['t'], np.abs(res['VSC']))
# plt.xlabel('Time [s]')
# plt.ylabel('MW')


plt.figure()
#plt.plot(res['t'], res['gen_speed'],label = ps.gen['GEN'].par['name'])
i = 0
for res in results:
    plt.plot(res['t'],50+50*np.mean((res['gen_speed']),axis = 1), label = names[i])
    i+=1
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.grid()
plt.legend()
plt.show()