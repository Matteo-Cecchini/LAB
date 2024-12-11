import numpy as np
import random
import matplotlib.pyplot as plt

#assi del grafico
y = random.choices(range(10), k=10)
x = np.sort(y)
#stampa degli assi
print("x = ", x)
print("y = ", y)
#definizione parametri grafico e visualizzazione
plt.rcParams["figure.figsize"] = [12, 7.5]
plt.rcParams["figure.autolayout"] = True

plt.plot(x, y, color="blue")
plt.title("Random graph")

plt.show()