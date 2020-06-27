# vectorised code vs for loop
import time
import numpy as np


if __name__ == "__main__":
    a = np.array([[1, 2, 3, 4, 5]])
    b = np.array([[1, 2, 3, 4, 5]])
    c_f = np.empty(a.shape)
    c_v = np.empty(a.shape)

    start_for = time.time()
    for i in range(len(a)):
        c_f += a[i] + b[i]
    end_for = time.time()
    print('time needed to run for loop: ', (end_for - start_for) * 1000, 'seconds')
    print('output of for loop: ', c_f)

    start_vec = time.time()
    c_v = a + b
    end_vec = time.time()
    print('time needed to run vectorized code: ', (end_vec - start_vec) * 1000, 'seconds')
    print('output of vectorized code: ', c_v)

    print('speed up factor: ', ((end_vec - start_vec) * 1000) / (end_for - start_for) * 1000)
