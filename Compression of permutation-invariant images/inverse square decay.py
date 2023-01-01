import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

dim_n = 100
# connections decay as 1/2^k with k distance. negative as we want to maximize energy
decay = np.zeros((dim_n,dim_n))
for i in range(dim_n):
    for j in range(dim_n):
        if(i != j):
            decay[i,j] = -.5**(abs(i-j)-1)

def row_energy(arr):
    a = arr @ arr.T
    b = a * decay
    return np.sum(b)

def col_energy(arr):
    a = arr.T @ arr
    b = a * decay
    return np.sum(b)

arr = np.random.rand(dim_n, dim_n)
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
while(i!=dim_n-1 and j!=dim_n):
    prop_arr = np.copy(arr)
    prop_arr[[i,j],:] = prop_arr[[j,i],:]
    prop_re = row_energy(prop_arr)
    if(prop_re < re):
        i = 0; j = 1; p += 1
        re = prop_re
        arr = prop_arr
    j += 1
    if(j==dim_n):
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
while(i!=dim_n and j!=dim_n):
    prop_arr = np.copy(arr)
    prop_arr[:,[i,j]] = prop_arr[:,[j,i]]
    prop_ce = col_energy(prop_arr)
    if(prop_ce < ce):
        i = 0; j = 1; p += 1
        ce = prop_ce
        arr = prop_arr
    j += 1
    if(j==dim_n):
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