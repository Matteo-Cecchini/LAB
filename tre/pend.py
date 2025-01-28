import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ds1000z
import scipy.stats as ss

# analisi dati
csvs = ("12sec1.csv", "12sec2.csv", "12sec3.csv", "12sec4.csv", "12sec5.csv", "12sec6.csv")
datas = [ds1000z.csv_formatter(pd.read_csv(i)) for i in csvs]
data = [i[0] for i in datas]
increment = [i[1] for i in datas][0]
wire_len = pd.read_csv("lun.csv")
wire_len = np.array(wire_len["misura"])


fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 3, hspace = 0, wspace = 0)
axes = gs.subplots(sharex=True, sharey=True)
axes = axes.flatten()
print(axes)

colors = ['#1f77b4',  # Blu
          '#ff7f0e',  # Arancione
          '#2ca02c',  # Verde
          '#d62728',  # Rosso
          '#9467bd',  # Viola
          '#8c564b']  # Marrone
for i,j,k,l in zip(axes, data, colors, range(6)):
    i.plot(j["time"], j["ch1_volt"], c=k, label=f"dataset {l + 1}")
    i.legend(loc="upper right")
fig.suptitle("Grafici presa dati", size=15)
fig.supxlabel("Tempo (s)", size=15)
fig.supylabel("ddp (v)", size=15)
plt.show()

array = [ds1000z.split_and_mean(i, "ch1_volt", "time") for i in data]
periods = [ds1000z.time_deltas(i) for i in array]

periods = np.array([np.concatenate([i[0] for i in periods]), np.concatenate([i[1] for i in periods])]) * increment

acceleration = ds1000z.drop_g(periods, wire_len)
print()
for i in acceleration:
    print(i)