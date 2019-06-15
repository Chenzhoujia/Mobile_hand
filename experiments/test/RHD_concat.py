import numpy as np

base = np.random.randint(5,size=(10,3,3))
base_append = base[:3]
base_append = np.zeros_like(base_append)

base = np.concatenate((base, base_append), axis=0)
a = ['a1','a2']
a2 = ['a3']*3
a = a+a2
print(int(3/2))
print(base)
print(a)