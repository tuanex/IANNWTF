import numpy as np

mu, sigma = 0, 1
arr = np.random.normal(mu,sigma, size=(5,5))

print(arr)

for i in range(5):
    for j in range(5):
        if arr[i,j] > 0.09:
            arr[i,j] = arr[i,j] ** 2
        else:
            arr[i,j] = 42

print("")
print(arr)