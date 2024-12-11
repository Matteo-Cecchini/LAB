import pandas
import numpy
import matplotlib.pyplot as plt

a = pandas.read_csv("pot.csv")
b = a["VppCost"].values / .568
c = a["R2"].values / 14.3
d = 400e3 / (.707 * b)
e = 400e3 / (.707 * c)
for i,j in zip(b, c):
    print(i, j)

plt.plot(b, d)

plt.show()