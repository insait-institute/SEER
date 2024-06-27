import json 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import rc

names = {
    'fishing': 'Example Disaggregation\n(Prior Work)',
    'normal': 'Non-Malicious Networks',
    'ours': 'SEER (Our Method)'
}
colors = {
    'fishing': '#e41a1c',
    'normal': '#377eb8',
    'ours': '#4daf4a'
}

# Load data
with open("json/dsnr.json", "r") as fp:
    m = json.load(fp) 
m['fishing'] = [0.72, 15, 52]

# Plot metric 
plt.rcParams["figure.figsize"] = [14.5, 3.7]
plt.rcParams["figure.autolayout"] = True

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

fig, ax = plt.subplots()
for k in ['fishing', 'ours', 'normal']:
    v = m[k]
    if k == 'ours':
        y = 0.075
    elif k == 'fishing':
        y = 0.15
    else:
        y = 0
    ax.scatter(v, np.full(len(v), y), label=names[k], marker='x', color=colors[k], s=100)

ax.axvline(linewidth=2, x=52, ymin=0.37, ymax=0.59, linestyle='--', color='red', zorder=0, alpha=0.8)
im52 = Image.open('./images/fishing_52.png')
newax = fig.add_axes([0.685, 0.63, 0.3, 0.3], anchor='NE', zorder=1)
newax.imshow(im52)
newax.axis('off')

ax.axvline(linewidth=2, x=15, ymin=0.37, ymax=0.59, linestyle='--', color='red', zorder=0, alpha=0.8)
im15 = plt.imread('./images/fishing_15.png')
newax = fig.add_axes([0.537, 0.63, 0.3, 0.3], anchor='NE', zorder=1)
newax.imshow(im15)
newax.axis('off')

ax.axvline(linewidth=2, x=0.72, ymin=0.37, ymax=0.59, linestyle='--', color='red', zorder=0, alpha=0.8)
im072 = plt.imread('./images/fishing_0.72.png')
newax = fig.add_axes([0.172, 0.63, 0.3, 0.3], anchor='NE', zorder=1)
newax.imshow(im072)
newax.axis('off')

ax.plot([0.65, 0.29], [0.075, 0.28], color='green', linewidth=2, linestyle='--', zorder=0, alpha=0.8)
ours = plt.imread('./images/img4_s2n=0.65.png')
newax = fig.add_axes([0.062, 0.63, 0.3, 0.3], anchor='NE', zorder=1)
newax.imshow(ours)
newax.axis('off')

icon_no = Image.open('./images/icon_no.png')
icon_yes = Image.open('./images/icon_yes.png')
icon_exclamation = Image.open('./images/icon_exclamation.png')
newax = fig.add_axes([0.676, 0.86, 0.1, 0.1], anchor='NE', zorder=1)
newax.imshow(icon_yes)
newax.axis('off')
newax = fig.add_axes([0.827, 0.86, 0.1, 0.1], anchor='NE', zorder=1)
newax.imshow(icon_yes)
newax.axis('off')
newax = fig.add_axes([0.310, 0.86, 0.1, 0.1], anchor='NE', zorder=1)
newax.imshow(icon_no)
newax.axis('off')
newax = fig.add_axes([0.200, 0.86, 0.1, 0.1], anchor='NE', zorder=1)
newax.imshow(icon_yes)
newax.axis('off')
newax = fig.add_axes([0.230, 0.86, 0.1, 0.1], anchor='NE', zorder=1)
newax.imshow(icon_exclamation)
newax.axis('off')

ax.spines[['right', 'top', 'left']].set_visible(False)
fig.legend(loc='upper left', bbox_to_anchor=(0.0001, 0.97), prop={'size': 19})
ax.set_xscale('log')
ax.get_yaxis().set_visible(False)
ax.set_ylim(-0.1, 0.61)
ax.set_xticks([1, 5, 10])
ax.tick_params(axis='x', which='major', labelsize=22)
ax.set_xlabel('Disaggregation Signal-to-noise Ratio (D-SNR; log scale)', fontsize=24)
plt.savefig('plots/dsnr.pdf')
print('done')


