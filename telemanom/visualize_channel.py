import numpy as np
import matplotlib.pyplot as plt

channel = 'T-1'
data = np.load(f'data/test/{channel}.npy')

telemetry_values = data[:, 0]

plt.figure(figsize=(14, 5))
plt.plot(telemetry_values, label='Telemetry reading')

anomaly_sequences = [[2399, 3898], [6550, 6585]]

for start, end in anomaly_sequences:
    plt.axvspan(start, end, color='red', alpha=0.3, label='Labeled anomaly')

plt.title(f'Channel {channel} (SMAP) - Telemetry over time')
plt.xlabel('Time step')
plt.ylabel('Telemetry value (scaled)')
plt.legend()
plt.show()