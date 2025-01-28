import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/ceccoh/lab/tre_fotodiodo/12sec.csv")
df.columns = df.values[0]
df = df[1:]
df["Sequence"] = pd.to_numeric(df["Sequence"])
df["VOLT"] = pd.to_numeric(df["VOLT"])
mins = df.loc[df["VOLT"] < 1]

x, y = np.array(df["Sequence"]), np.array(df["VOLT"])

plt.plot(x, y)
plt.show()

xm = np.array(mins["Sequence"])
deltas = [xm[i] - xm[i - 1] for i in range(1, len(xm))]
print(deltas)
'''
for i in range(1, len(deltas) - 1):
    if deltas[i] == 1:
        deltas[i-1] += 0.5
        deltas[i+1] += 0.5'''

deltas = np.delete(deltas, [i == 1 for i in deltas])
deltas = 1.00E-2*deltas
mean, std = np.mean(deltas)*2, np.std(deltas, ddof=1)
g = 0.59*((2*np.pi)/mean)**2
dg = np.sqrt(((4*np.pi*0.001)/(mean**2))**2 + ((8*np.pi*0.59*std)/(mean**3))**2)
print(deltas)
print(mean, std)
print(g, dg)


#plt.plot(x, y)
#plt.plot(xm, ym)

#plt.show()