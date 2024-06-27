import json 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import rc

names = {
    'normal': 'Non-Malicious Networks',
    'ours': 'SEER (Our Method)'
}
colors = {
    'normal': '#377eb8',
    'ours': '#4daf4a'
}

# Load data
with open("json/transmit.json", "r") as fp:
    m = json.load(fp) 
# boostedanalytical = infinity

# Plot metric 
plt.rcParams["figure.figsize"] = [14.5, 3.7]
plt.rcParams["figure.autolayout"] = True

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

fig, ax = plt.subplots()
for k in ['ours', 'normal']:
    v = m[k]
    if k == 'ours':
        y = 0.075
    else:
        y = 0
    ax.scatter(v, np.full(len(v), y), label=names[k], marker='x', color=colors[k], s=100)

ax.axvline(linewidth=2, x=1.0, ymin=0, ymax=0.5, linestyle='--', color='brown', zorder=0, alpha=0.8)

ax.spines[['right', 'top', 'left']].set_visible(False)
fig.legend(loc='upper left', bbox_to_anchor=(0.0001, 0.97), prop={'size': 19})
ax.get_yaxis().set_visible(False)
ax.set_ylim(-0.1, 0.61)
ax.set_xlim(0, 1.6)
ax.set_xticks(np.arange(0, 1.6, 0.5))
#ax.set_xticks([1, 5, 10])
ax.tick_params(axis='x', which='major', labelsize=22)
ax.set_xlabel('Transmission Signal-to-noise Ratio (Boosted Analytical = infinity)', fontsize=24)
plt.savefig('plots/transmit.pdf')
print('done')


