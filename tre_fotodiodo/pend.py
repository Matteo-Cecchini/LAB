import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ds1000z

csvs = ("12sec1.csv", "12sec2.csv", "12sec3.csv", "12sec4.csv", "12sec5.csv", "12sec6.csv")
data = [ds1000z.csv_formatter(pd.read_csv(i)) for i in csvs]
wire_len = pd.read_csv("lun.csv")
wire_len = np.array(wire_len["misura"])

plt.plot(data[0]["ch1_volt"].values)
plt.xlabel("tempo ($s^{-1}$)")
plt.ylabel("potenziale elettrico ($V$)")
plt.title("presa dati n°1")
plt.show()

array = [ds1000z.split_and_mean(i, "ch1_volt", "time") for i in data]
for i in array], axis=1)

print(np.mean(periods[0]))

acceleration = ds1000z.drop_g(periods, wire_len)
print(acceleration)