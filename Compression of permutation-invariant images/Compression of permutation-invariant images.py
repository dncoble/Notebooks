import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# distance matrix -- 0s along diagonal, 1s in diagonals above and below, 2s above/below that, etc.
linear_dist = np.zeros((100,100))

for i in range(100):
    for j in range(100):
        linear_dist[i,j] = abs(i-j)

def row_energy(arr):
    a = arr @ arr.T
    b = a * linear_dist
    return np.sum(b)

def col_energy(arr):
    a = arr.T @ arr
    b = a * linear_dist
    return np.sum(b)

arr = np.random.rand(100, 100)
arrOriginal = np.copy(arr)

re = row_energy(arr)
ce = col_energy(arr)

print("starting row energy: %f"%re)
print("starting column energy: %f"%ce)

#%% minimizing row energy
i = 0; j = 1;
p = 0; # count of number of permutations
d = 0 # total number of iterations
i_max = i; j_max = j
while(i!=99 and j!=100):
    prop_arr = np.copy(arr)
    prop_arr[[i,j],:] = prop_arr[[j,i],:]
    prop_re = row_energy(prop_arr)
    if(prop_re < re):
        i = 0; j = 1; p += 1
        re = prop_re
        arr = prop_arr
    j += 1
    if(j==100):
        i += 1
        j = i+1
    # monitoring
    d += 1
    if(i > i_max):
        i_max = i
        j_max = 0
    if(i == i_max and j > j_max):
        j_max = j
    if(d % 10000 == 0):
        print("energy: %f  permutations: %d  iterations: %d  i_max: %d  j_max: %d"%(re, p, d, i_max, j_max))
#%% minimizing column energy
i = 0; j = 1;
p = 0; # count of number of permutations
d = 0 # total number of iterations
i_max = i; j_max = j
while(i!=99 and j!=100):
    prop_arr = np.copy(arr)
    prop_arr[:,[i,j]] = prop_arr[:,[j,i]]
    prop_ce = col_energy(prop_arr)
    if(prop_ce < ce):
        i = 0; j = 1; p += 1
        ce = prop_ce
        arr = prop_arr
    j += 1
    if(j==100):
        i += 1
        j = i+1
    # monitoring
    d += 1
    if(i > i_max):
        i_max = i
        if(j > j_max):
            j_max = j
    if(d % 10000 == 0):
        print("energy: %f  permutations: %d  iterations: %d  i_max: %d  j_max: %d"%(ce, p, d, i_max, j_max))
#%%
plt.figure()
plt.title("Original random image")
plt.imshow(arrOriginal, cmap=mpl.cm.binary, vmin=0, vmax=1, interpolation='nearest')
plt.tight_layout()
plt.show()

plt.figure()
plt.title("Permuted image")
plt.imshow(arr, cmap=mpl.cm.binary, vmin=0, vmax=1, interpolation='nearest')
plt.tight_layout()
plt.show()