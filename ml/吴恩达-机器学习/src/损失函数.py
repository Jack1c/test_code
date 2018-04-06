from matplotlib import pyplot as plt

import numpy as np

X = [1, 2, 3]
y = [1, 2, 3]


# plt.plot(np.array(X), np.array(y))
# plt.show()


def j(theta):
    sum = 0
    for i, x in enumerate(X):
        sum += (theta * x - y[i]) ** 2

    return sum / (2 * len(X))


Jtheta_arr = []
theta_arr = []
line = []
jline = []
for i in range(200):
    Jtheta_arr.append(j(i / 100))
    theta_arr.append(i / 100)
    if (i % 20 == 0):
        line.append(i / 100)
        jline.append(j(i / 100))

plt.scatter(line, jline)

# theta图形
plt.plot(np.array(theta_arr), np.array(Jtheta_arr))
plt.show()

for t in jline:
    _x = []
    _y = []
    for i in range(2):
        _x.append(i)
        _y.append(i * t)

    plt.plot(_x, _y)

plt.show()
