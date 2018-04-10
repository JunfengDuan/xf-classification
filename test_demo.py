import numpy as np

for i in range(10):
    b = 88
    a = np.random.randint(10)
    print(a, end=' ')
    if a % 3 == 0:
        print('a-',a)

print(b)