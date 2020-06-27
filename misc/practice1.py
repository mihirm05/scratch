# difference between * and dot()
# note: u and v are two column vector then (uT v) is inner and (u vT) is outer product

import numpy as np

A = np.random.randint(10, size=(2, 2))
B = np.random.randint(10, size=(2, 2))

print('A is: ', A)
print('B is: ', B)

# equivalent to multiplying single element (for this the shape of matrices should exactly be the same)
C = A*B
print('C is: ', C)

# equivalent to regular product of matrices  (axb) * (b*c) = (a*c)
D = np.dot(A, B)
print('D is: ', D)

