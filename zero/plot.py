import random
import matplotlib.pyplot as plt

x = range(10)
y = [random.choice(range(10)) for i in range(10)]
print(x)
print(y)

plt.rcParams["figure.figsize"] = [12, 7.5]
plt.rcParams["figure.autolayout"] = True

plt.plot(x, y, color="red")
plt.title("grafico a caso")
plt.xlabel("numeri da 1 a 10")
plt.ylabel("sort tra 1 e 10")

plt.show()