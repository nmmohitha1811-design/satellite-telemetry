import glob, os
import numpy as np
import matplotlib.pyplot as plt

channel = 'T-1'
L_S = 250  # the model needs 250 steps of history before its first prediction

# newest saved predictions for this channel
path = sorted(glob.glob(f'data/*/y_hat/{channel}.npy'), key=os.path.getmtime)[-1]
print('Using predictions from:', path)

y_hat = np.load(path)
test = np.load(f'data/test/{channel}.npy')[:, 0]

steps = np.arange(len(y_hat)) + L_S
actual = test[L_S:L_S + len(y_hat)]
error = np.abs(actual - y_hat)

true_anoms = [(2399, 3898), (6550, 6585)]
caught = [(2490, 2979)]
false_alarms = [(3960, 4029), (6620, 6829)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(steps, actual, label='Actual', linewidth=1)
ax1.plot(steps, y_hat, label='Model prediction', linewidth=1, alpha=0.8)
ax1.set_ylabel('Telemetry (scaled)')
ax1.set_title(f'{channel}: actual vs predicted')

ax2.plot(steps, error, color='black', linewidth=0.8)
ax2.set_ylabel('Absolute error')
ax2.set_xlabel('Test time step')
ax2.set_title('Prediction error')

def shade(ax, spans, color, label):
    for i, (s, e) in enumerate(spans):
        ax.axvspan(s, e, color=color, alpha=0.25, label=label if i == 0 else None)

for ax in (ax1, ax2):
    shade(ax, true_anoms, 'red', 'Labeled anomaly')
    shade(ax, caught, 'green', 'Model caught')
    shade(ax, false_alarms, 'orange', 'Model false alarm')

ax1.legend(loc='lower right')
plt.tight_layout()
plt.show()