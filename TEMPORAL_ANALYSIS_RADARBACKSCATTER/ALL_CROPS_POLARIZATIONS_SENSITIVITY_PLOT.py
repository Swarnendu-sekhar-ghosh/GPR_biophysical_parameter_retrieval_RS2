# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 17:18:25 2021

@author: mrslab13
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fig1 = plt.figure(figsize=(50,20))
ax1 = fig1.add_subplot()

data=pd.read_excel("please insert the xlsx file containing the Backscatter values for all dates for all crops respectively")

a=sns.violinplot(x=data['Date'],y=data['HH'],hue=data['CROP'],inner='box',scale='count',
                 gridsize=300,palette='husl',linewidth=6,width=0.7,saturation=20)

# plt.yticks(np.arange(3, -18, -4.0))
plt.ylim(-16, 0)

yticks = ax1.yaxis.get_major_ticks()

yticks[1].set_visible(False)

yticks[3].set_visible(False)
yticks[5].set_visible(False)
yticks[7].set_visible(False)

ax1.tick_params(which='major', length=40,width=10)

for axis in ['top','bottom','left','right']:
  ax1.spines[axis].set_linewidth(10)

plt.vlines(x=0.5, ymin=-18, ymax=1, colors='brown', ls=':', lw=6, label=None)
plt.vlines(x=1.5, ymin=-18, ymax=1, colors='brown', ls=':', lw=6, label=None)
plt.vlines(x=2.5, ymin=-18, ymax=1, colors='brown', ls=':', lw=6, label=None)


plt.xticks(fontsize=100)
plt.yticks(fontsize=100)
plt.xlabel("Date",size=100)
plt.ylabel("HH (dB)",size=100)
legend = ax1.legend()
legend.remove()
plt.show()

saveFileName = 'HH_ALL_CROPS.pdf'
file_savePath = 'put your folder path here where you want to save your image file' + saveFileName
print("Saved at: ",file_savePath)
fig1.savefig(file_savePath, bbox_inches = 'tight', dpi = 300, pad_inches = 0.08, 
            frameon = None)



fig2 = plt.figure(figsize=(50,20))
ax2 = fig2.add_subplot()

data=pd.read_excel("please insert the xlsx file containing the Backscatter values for all dates for all crops respectively")

b=sns.violinplot(x=data['Date'],y=data['HV'],hue=data['CROP'],inner='box',scale='count',
                 gridsize=300,palette='husl',linewidth=6,width=0.7,saturation=20)

plt.ylim(-30, -5)

# remove specfic ticks

# yticks = ax2.yaxis.get_major_ticks()
# yticks[1].set_visible(False)
# yticks[2].set_visible(False)
# yticks[4].set_visible(False)
# yticks[5].set_visible(False)
# yticks[7].set_visible(False)
# yticks[8].set_visible(False)

for axis in ['top','bottom','left','right']:
    ax2.spines[axis].set_linewidth(10)

ax2.tick_params(which='major', length=40,width=10)

plt.vlines(x=0.5, ymin=-30, ymax=-5, colors='brown', ls=':', lw=6, label=None)
plt.vlines(x=1.5, ymin=-30, ymax=-5, colors='brown', ls=':', lw=6, label=None)
plt.vlines(x=2.5, ymin=-30, ymax=-5, colors='brown', ls=':', lw=6, label=None)

plt.xticks(fontsize=90)
plt.yticks(fontsize=90)
plt.xlabel("Date",size=100)
plt.ylabel("HV (dB)",size=100)
legend1 = ax2.legend()
legend1.remove()
plt.show()

saveFileName = 'HV_ALL_CROPS.pdf'
file_savePath = 'F:/SMAPVEX_16_DATA/SMP16_DATA/EXTRACTED_DATA/SENSITIVITY_ANALYSIS/ALL_CROPS_PLOTS/' + saveFileName
print("Saved at: ",file_savePath)
fig2.savefig(file_savePath, bbox_inches = 'tight', dpi = 300, pad_inches = 0.08, 
            frameon = None)



fig3 = plt.figure(figsize=(50,20))
ax3 = fig3.add_subplot()

data=pd.read_excel("please insert the xlsx file containing the Backscatter values for all dates for all crops respectively")

c=sns.violinplot(x=data['Date'],y=data['VV'],hue=data['CROP'],inner='box',scale='count',
                 gridsize=300,palette='husl',linewidth=6,width=0.7,saturation=20)

plt.ylim(-18, 0)

# remove specfic ticks

# yticks = ax3.yaxis.get_major_ticks()
# yticks[1].set_visible(False)
# yticks[2].set_visible(False)
# yticks[4].set_visible(False)
# yticks[5].set_visible(False)
# yticks[7].set_visible(False)
# yticks[8].set_visible(False)


for axis in ['top','bottom','left','right']:
    ax3.spines[axis].set_linewidth(10)

ax3.tick_params(which='major', length=40,width=10)

plt.vlines(x=0.5, ymin=-18, ymax=0, colors='brown', ls=':', lw=6, label=None)
plt.vlines(x=1.5, ymin=-18, ymax=0, colors='brown', ls=':', lw=6, label=None)
plt.vlines(x=2.5, ymin=-18, ymax=0, colors='brown', ls=':', lw=6, label=None)

plt.xticks(fontsize=90)
plt.yticks(fontsize=90)
plt.xlabel("Date",size=100)
plt.ylabel("VV (dB)",size=100)
plt.legend(ncol=3,bbox_to_anchor=(0.5,-0.2),loc='upper center',fontsize=100,frameon=False)

plt.show()

saveFileName = 'VV_ALL_CROPS.pdf'
file_savePath = 'put your folder path here where you want to save your image file' + saveFileName
print("Saved at: ",file_savePath)
fig3.savefig(file_savePath, bbox_inches = 'tight', dpi = 300, pad_inches = 0.08, 
            frameon = None)



