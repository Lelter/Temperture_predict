#%%
import numpy as np
import pandas as pd

d = pd.read_csv("初赛/train/train.csv")
print(d)
d.info()

#%%

d["time"] = pd.to_datetime(d["time"], origin='1970-01-01 08:00:00', unit="s")
d.set_index("time")
d.fillna(method="ffill", inplace=True)

#%%

d.info()
from matplotlib import pyplot as plt

plt.title("origin")
# x=list(train_set["time"])
# y=list(train_set["temperature"])
# plt.plot(train_set["time"],train_set["temperature"])

#%%

print(f"Total unique dates in the dataset: {len(set(d['time']))}")
print(f"Number of rows in the dataset: {d.shape[0]}")
#%%

features = ['temperature', '气压(室内)', '湿度(室内)']
#%%

# Aggregating to hourly level
d = d.groupby('time', as_index=False)[features].mean()

# Creating the data column
d['date'] = [x.date() for x in d['time']]
#%%

# Extracting the hour of day
d['hour'] = [x.hour for x in d['time']]

# Extracting the month of the year
d['month'] = [x.month for x in d['time']]
#%%
for i in range(d.shape[0]):
    if d.loc[i,"hour"]==0 or d.loc[i,"hour"]==12:
        d.loc[i,"time"]=d.loc[i,"time"]+pd.Timedelta(hours=12)


d[features].describe()
#%%

d.head(10)
#%%

d[['time', 'temperature']].tail(10)
#%%

plt.figure(figsize=(12, 8))
plt.plot('time', 'temperature', data=d)
plt.title('Hourly temperature graph')
plt.ylabel('Degrees in C')
plt.xlabel('Date')
plt.show()
#%%

plot_features = d[features]
plot_features.index = d.time

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(15, 10),
    facecolor="w",
    edgecolor="k"
)

for i, feature in enumerate(['气压(室内)', '湿度(室内)']):
    axes[i % 2].plot(plot_features[feature])
    axes[i % 2].set_title(f'{feature} Vilnius - hourly')

plt.tight_layout()

plt.show()
#%%

plt.figure(figsize=(8, 8))
plt.hist2d(d['气压(室内)'], d['temperature'], bins=(50, 50))
plt.colorbar()
ax = plt.gca()
plt.xlabel('Pressure, hPa')
plt.ylabel('Temperature, C')
ax.axis('tight')
plt.show()
#%%

plt.figure(figsize=(8, 8))
plt.hist2d(d['湿度(室内)'], d['temperature'], bins=(50, 50))
plt.colorbar()
ax = plt.gca()
plt.xlabel('moist, m/s')
plt.ylabel('Temperature, C')
ax.axis('tight')
plt.show()

#%%

d.boxplot('temperature', by='hour', figsize=(12, 8), grid=False)
#%%

# Creating the cyclical daily feature
d['day_cos'] = [np.cos(x * (2 * np.pi / 24)) for x in d['hour']]
d['day_sin'] = [np.sin(x * (2 * np.pi / 24)) for x in d['hour']]
#%%

dsin = d[['time', 'temperature', 'hour', 'day_sin', 'day_cos']].head(25).copy()
dsin['day_sin'] = [round(x, 3) for x in dsin['day_sin']]
dsin['day_cos'] = [round(x, 3) for x in dsin['day_cos']]

