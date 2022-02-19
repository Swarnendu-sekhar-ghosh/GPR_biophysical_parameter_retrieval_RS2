# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 09:17:41 2021

@author: mrslab13
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from numpy import arange

# objective function
def objective(x, a, b):
	return a * x + b


###################################################### SCATTER PLOT FOR WHEAT PAI #########################################################################

data1=pd.read_excel("insert the .xlsx file that you saved containing the predicted outputs from full-pol combination ")

x=data1["INSITU_PAI"]
y=data1["PREDICTED_PAI"]

fig1 = plt.figure(figsize=(12,12))
ax1 = fig1.add_subplot()
plt.xlim([0, 8])
plt.ylim([0, 8])
plt.plot([0, 8], [0, 8], 'k:',linewidth=7)
plt.scatter(x=data1["INSITU_PAI"],y=data1["PREDICTED_PAI"],c='g',s=800)
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.yticks(np.arange(0, 10, 2))
plt.xticks(np.arange(0, 10, 2))
plt.xlabel("Insitu PAI ($m^{2}m^{-2}$)",size=50)
plt.ylabel("Estimated PAI ($m^{2}m^{-2}$)",size=50)
plt.annotate(r'$\rho$ = 0.83', xy=(0.3, 7.3),size=50)
plt.annotate('RMSE = 1.01', xy=(0.3, 6.6),size=50)
plt.annotate('MAE =0.86', xy=(0.3, 6.0),size=50)
plt.annotate('HH+HV+VV',xy=(4.2,0.3),size=50)
popt, _ = curve_fit(objective, x, y)
a, b = popt
x_line = arange(min(x), max(x), 1)
y_line = objective(x_line, a, b)
plt.plot(x_line, y_line, '--', color='red',linewidth=7)
ax1.set_aspect('equal', adjustable='box')
plt.show()

saveFileName = 'HH_HV_VV_WHEAT_PAI.pdf'
file_savePath = 'F:/PHD/WORK_1/GPR_BIOPHYSICAL_PARAM_RETRIEVE_SMAPVEX16_DATA/SMP16_DATA/EXTRACTED_DATA/GPR/WHEAT/PAI/RECOVERED_FILES/PLOTS/' + saveFileName
print("Saved at: ",file_savePath)
fig1.savefig(file_savePath, bbox_inches = 'tight', dpi = 300, pad_inches = 0.08, 
            frameon = None)


data2=pd.read_excel("insert the .xlsx file that you saved containing the predicted outputs from HH+HV combination")
x=data2["INSITU_PAI"]
y=data2["PREDICTED_PAI"]

fig2 = plt.figure(figsize=(12,12))
ax2 = fig2.add_subplot()
plt.xlim([0, 8])
plt.ylim([0, 8])
plt.plot([0, 8], [0, 8], 'k:',linewidth=7)
plt.scatter(x=data2["INSITU_PAI"],y=data2["PREDICTED_PAI"],c='g',s=800)
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.yticks(np.arange(0, 10, 2))
plt.xticks(np.arange(0, 10, 2))
plt.xlabel("Insitu PAI ($m^{2}m^{-2}$)",size=50)
plt.ylabel("Estimated PAI ($m^{2}m^{-2}$)",size=50)
plt.annotate(r'$\rho$ = 0.75', xy=(0.3, 7.3),size=50) 
plt.annotate('RMSE = 1.19', xy=(0.3, 6.6),size=50)
plt.annotate('MAE = 0.97', xy=(0.3, 6.0),size=50)
plt.annotate('HH+HV',xy=(5.5,0.3),size=50)
popt, _ = curve_fit(objective, x, y)
a, b = popt
x_line = arange(min(x), max(x), 1)
y_line = objective(x_line, a, b)
plt.plot(x_line, y_line, '--', color='red',linewidth=7)
ax2.set_aspect('equal', adjustable='box')
plt.show()

saveFileName = 'HH_HV_WHEAT_PAI.pdf'
file_savePath = 'F:/PHD/WORK_1/GPR_BIOPHYSICAL_PARAM_RETRIEVE_SMAPVEX16_DATA/SMP16_DATA/EXTRACTED_DATA/GPR/WHEAT/PAI/RECOVERED_FILES/PLOTS/' + saveFileName
print("Saved at: ",file_savePath)
fig2.savefig(file_savePath, bbox_inches = 'tight', dpi = 300, pad_inches = 0.08, 
            frameon = None)



data3=pd.read_excel("insert the .xlsx file that you saved containing the predicted outputs from HV+VV combination")
x=data3["INSITU_PAI"]
y=data3["PREDICTED_PAI"]

fig3 = plt.figure(figsize=(12,12))
ax3 = fig3.add_subplot()
plt.xlim([0, 8])
plt.ylim([0, 8])
plt.plot([0, 8], [0, 8], 'k:',linewidth=7)
plt.scatter(x=data3["INSITU_PAI"],y=data3["PREDICTED_PAI"],c='g',s=800)
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.yticks(np.arange(0, 10, 2))
plt.xticks(np.arange(0, 10, 2))
plt.xlabel("Insitu PAI ($m^{2}m^{-2}$)",size=50)
plt.ylabel("Estimated PAI ($m^{2}m^{-2}$)",size=50)
plt.annotate(r'$\rho$ = 0.78 ', xy=(0.3, 7.3),size=50)  
plt.annotate('RMSE = 1.12', xy=(0.3, 6.6),size=50)
plt.annotate('MAE = 0.93', xy=(0.3, 6.0),size=50)
plt.annotate('HV+VV',xy=(5.5,0.3),size=50)
popt, _ = curve_fit(objective, x, y)
a, b = popt
x_line = arange(min(x), max(x), 1)
y_line = objective(x_line, a, b)
plt.plot(x_line, y_line, '--', color='red',linewidth=7)
ax3.set_aspect('equal', adjustable='box')
plt.show()

saveFileName = 'HV_VV_WHEAT_PAI.pdf'
file_savePath = 'F:/PHD/WORK_1/GPR_BIOPHYSICAL_PARAM_RETRIEVE_SMAPVEX16_DATA/SMP16_DATA/EXTRACTED_DATA/GPR/WHEAT/PAI/RECOVERED_FILES/PLOTS/' + saveFileName
print("Saved at: ",file_savePath)
fig3.savefig(file_savePath, bbox_inches = 'tight', dpi = 300, pad_inches = 0.08, 
            frameon = None)


data4=pd.read_excel("insert the .xlsx file that you saved containing the predicted outputs from HH+VV combination")
x=data4["INSITU_PAI"]
y=data4["PREDICTED_PAI"]

fig4 = plt.figure(figsize=(12,12))
ax4 = fig4.add_subplot()
plt.xlim([0, 8])
plt.ylim([0, 8])
plt.plot([0, 8], [0, 8], 'k:',linewidth=7)
plt.scatter(x=data4["INSITU_PAI"],y=data4["PREDICTED_PAI"],c='g',s=800)
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.yticks(np.arange(0, 10, 2))
plt.xticks(np.arange(0, 10, 2))
plt.xlabel("Insitu PAI ($m^{2}m^{-2}$)",size=50)
plt.ylabel("Estimated PAI ($m^{2}m^{-2}$)",size=50)
plt.annotate(r'$\rho$ = 0.64 ', xy=(0.3, 7.3),size=50) 
plt.annotate('RMSE = 1.35', xy=(0.3, 6.6),size=50)
plt.annotate('MAE = 1.02', xy=(0.3, 6.0),size=50)
plt.annotate('HH+VV',xy=(5.5,0.3),size=50)
popt, _ = curve_fit(objective, x, y)
a, b = popt
x_line = arange(min(x), max(x), 1)
y_line = objective(x_line, a, b)
plt.plot(x_line, y_line, '--', color='red',linewidth=7)
ax4.set_aspect('equal', adjustable='box')
plt.show()

saveFileName = 'HH_VV_WHEAT_PAI.pdf'
file_savePath = 'F:/PHD/WORK_1/GPR_BIOPHYSICAL_PARAM_RETRIEVE_SMAPVEX16_DATA/SMP16_DATA/EXTRACTED_DATA/GPR/WHEAT/PAI/RECOVERED_FILES/PLOTS/' + saveFileName
print("Saved at: ",file_savePath)
fig4.savefig(file_savePath, bbox_inches = 'tight', dpi = 300, pad_inches = 0.08, 
            frameon = None)



##################################################### SCATTER PLOT FOR WHEAT WETBIOMASS  ###################################################

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from numpy import arange

# objective function
def objective(x, a, b):
	return a * x + b


data5=pd.read_excel("insert the .xlsx file that you saved containing the predicted outputs from full-pol combination")

x=data5["INSITU_WETBIOMASS"]
y=data5["PREDICTED_WETBIOMASS"]

fig5 = plt.figure(figsize=(13,13))
ax5 = fig5.add_subplot()
plt.xlim([0, 6])
plt.ylim([0, 6])
plt.plot([0, 6], [0, 6], 'k:',linewidth=7)
plt.scatter(x=data5["INSITU_WETBIOMASS"],y=data5["PREDICTED_WETBIOMASS"],c='#8c564b',s=800,marker="P")
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.yticks(np.arange(0, 8, 2))
plt.xticks(np.arange(0, 8, 2))
plt.xlabel("Insitu WB ($Kgm^{-2}$)",size=50)
plt.ylabel("Estimated WB ($Kgm^{-2}$)",size=50)
plt.annotate(r'$\rho$ = 0.66 ', xy=(0.3, 5.5),size=50) 
plt.annotate('RMSE = 0.81', xy=(0.3, 4.9),size=50)
plt.annotate('MAE = 0.69', xy=(0.3, 4.3),size=50)
plt.annotate('HH+HV+VV',xy=(3.3,0.3),size=50)
popt, _ = curve_fit(objective, x, y)
a, b = popt
x_line = arange(min(x), max(x), 1)
y_line = objective(x_line, a, b)
plt.plot(x_line, y_line, '--', color='red',linewidth=7)
ax5.set_aspect('equal', adjustable='box')
plt.show()

saveFileName = 'HH_HV_VV_WHEAT_WB.pdf'
file_savePath = 'F:/PHD/WORK_1/GPR_BIOPHYSICAL_PARAM_RETRIEVE_SMAPVEX16_DATA/SMP16_DATA/EXTRACTED_DATA/GPR/WHEAT/BIOMASS/PLOTS/REVISED/WB/' + saveFileName
print("Saved at: ",file_savePath)
fig5.savefig(file_savePath, bbox_inches = 'tight', dpi = 300, pad_inches = 0.08, 
            frameon = None)


data6=pd.read_excel("insert the .xlsx file that you saved containing the predicted outputs from HH+HV combination")

x=data6["INSITU_WETBIOMASS"]
y=data6["PREDICTED_WETBIOMASS"]

fig6 = plt.figure(figsize=(13,13))
ax6 = fig6.add_subplot()
plt.xlim([0, 6])
plt.ylim([0, 6])
plt.plot([0, 6], [0, 6], 'k:',linewidth=7)
plt.scatter(x=data6["INSITU_WETBIOMASS"],y=data6["PREDICTED_WETBIOMASS"],c='#8c564b',s=800,marker="P")
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.yticks(np.arange(0, 8, 2))
plt.xticks(np.arange(0, 8, 2))
plt.xlabel("Insitu WB ($Kgm^{-2}$)",size=50)
plt.ylabel("Estimated WB ($Kgm^{-2}$)",size=50)
plt.annotate(r'$\rho$ = 0.65 ', xy=(0.3, 5.5),size=50)
plt.annotate('RMSE = 0.84', xy=(0.3, 4.9),size=50)
plt.annotate('MAE = 0.70', xy=(0.3, 4.3),size=50)
plt.annotate('HH+HV',xy=(4.3,0.3),size=50)
popt, _ = curve_fit(objective, x, y)
a, b = popt
x_line = arange(min(x), max(x), 1)
y_line = objective(x_line, a, b)
plt.plot(x_line, y_line, '--', color='red',linewidth=7)
ax6.set_aspect('equal', adjustable='box')
plt.show()

saveFileName = 'HH_HV_WHEAT_WB.pdf'
file_savePath = 'F:/PHD/WORK_1/GPR_BIOPHYSICAL_PARAM_RETRIEVE_SMAPVEX16_DATA/SMP16_DATA/EXTRACTED_DATA/GPR/WHEAT/BIOMASS/PLOTS/REVISED/WB/' + saveFileName
print("Saved at: ",file_savePath)
fig6.savefig(file_savePath, bbox_inches = 'tight', dpi = 300, pad_inches = 0.08, 
            frameon = None)


data7=pd.read_excel("insert the .xlsx file that you saved containing the predicted outputs from HV+VV combination")
x=data7["INSITU_WETBIOMASS"]
y=data7["PREDICTED_WETBIOMASS"]

fig7 = plt.figure(figsize=(13,13))
ax7 = fig7.add_subplot()
plt.xlim([0, 6])
plt.ylim([0, 6])
plt.plot([0, 6], [0, 6], 'k:',linewidth=7)
plt.scatter(x=data7["INSITU_WETBIOMASS"],y=data7["PREDICTED_WETBIOMASS"],c='#8c564b',s=800,marker="P")
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.yticks(np.arange(0, 8, 2))
plt.xticks(np.arange(0, 8, 2))
plt.xlabel("Insitu WB ($Kgm^{-2}$)",size=50)
plt.ylabel("Estimated WB ($Kgm^{-2}$)",size=50)
plt.annotate(r'$\rho$ = 0.64 ', xy=(0.3, 5.5),size=50)   
plt.annotate('RMSE = 0.83', xy=(0.3, 4.9),size=50)
plt.annotate('MAE = 0.63', xy=(0.3, 4.3),size=50)
plt.annotate('HV+VV',xy=(4.3,0.3),size=50)
popt, _ = curve_fit(objective, x, y)
a, b = popt
x_line = arange(min(x), max(x), 1)
y_line = objective(x_line, a, b)
plt.plot(x_line, y_line, '--', color='red',linewidth=7)
ax7.set_aspect('equal', adjustable='box')
plt.show()

saveFileName = 'HV_VV_WHEAT_WB.pdf'
file_savePath = 'F:/PHD/WORK_1/GPR_BIOPHYSICAL_PARAM_RETRIEVE_SMAPVEX16_DATA/SMP16_DATA/EXTRACTED_DATA/GPR/WHEAT/BIOMASS/PLOTS/REVISED/WB/' + saveFileName
print("Saved at: ",file_savePath)
fig7.savefig(file_savePath, bbox_inches = 'tight', dpi = 300, pad_inches = 0.08, 
            frameon = None)


data8=pd.read_excel("insert the .xlsx file that you saved containing the predicted outputs from HH+VV combination")
x=data8["INSITU_WETBIOMASS"]
y=data8["PREDICTED_WETBIOMASS"]

fig8 = plt.figure(figsize=(13,13))
ax8 = fig8.add_subplot()
plt.xlim([0, 6])
plt.ylim([0, 6])
plt.plot([0, 6], [0, 6], 'k:',linewidth=7)
plt.scatter(x=data8["INSITU_WETBIOMASS"],y=data8["PREDICTED_WETBIOMASS"],c='#8c564b',s=800,marker="P")
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.yticks(np.arange(0, 8, 2))
plt.xticks(np.arange(0, 8, 2))
plt.xlabel("Insitu WB ($Kgm^{-2}$)",size=50)
plt.ylabel("Estimated WB ($Kgm^{-2}$)",size=50)
plt.annotate(r'$\rho$ = 0.67 ', xy=(0.3, 5.5),size=50)  
plt.annotate('RMSE = 0.87', xy=(0.3, 4.9),size=50)
plt.annotate('MAE = 0.75', xy=(0.3, 4.3),size=50)
plt.annotate('HH+VV',xy=(4.3,0.3),size=50)
popt, _ = curve_fit(objective, x, y)
a, b = popt
x_line = arange(min(x), max(x), 1)
y_line = objective(x_line, a, b)
plt.plot(x_line, y_line, '--', color='red',linewidth=7)
ax8.set_aspect('equal', adjustable='box')
plt.show()

saveFileName = 'HH_VV_WHEAT_WB.pdf'
file_savePath = 'F:/PHD/WORK_1/GPR_BIOPHYSICAL_PARAM_RETRIEVE_SMAPVEX16_DATA/SMP16_DATA/EXTRACTED_DATA/GPR/WHEAT/BIOMASS/PLOTS/REVISED/WB/' + saveFileName
print("Saved at: ",file_savePath)
fig8.savefig(file_savePath, bbox_inches = 'tight', dpi = 300, pad_inches = 0.08, 
            frameon = None)


################################################## SCATTER PLOT FOR WHEAT VEGETATION WATER CONTENT##################################################


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from numpy import arange

# objective function
def objective(x, a, b):
	return a * x + b

data9=pd.read_excel("insert the .xlsx file that you saved containing the predicted outputs from full-pol combination")
x=data9["INSITU_VWC"]
y=data9["PREDICTED_VWC"]

fig9 = plt.figure(figsize=(13,13))
ax9 = fig9.add_subplot()
plt.xlim([0, 6])
plt.ylim([0, 6])
plt.plot([0, 6], [0, 6], 'k:',linewidth=7)
plt.scatter(x=data9["INSITU_VWC"],y=data9["PREDICTED_VWC"],c='#17becf',s=800,marker="X")
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.yticks(np.arange(0, 8, 2))
plt.xticks(np.arange(0, 8, 2))
plt.xlabel("Insitu VWC ($Kgm^{-2}$)",size=50)
plt.ylabel("Estimated VWC ($Kgm^{-2}$)",size=50)
plt.annotate(r'$\rho$ =  0.63', xy=(0.3, 5.5),size=50)  
plt.annotate('RMSE = 0.68', xy=(0.3,4.9),size=50)
plt.annotate('MAE = 0.56', xy=(0.3, 4.3),size=50)
plt.annotate('HH+HV+VV',xy=(3.3,0.3),size=50)
popt, _ = curve_fit(objective, x, y)
a, b = popt
x_line = arange(min(x), max(x), 1)
y_line = objective(x_line, a, b)
plt.plot(x_line, y_line, '--', color='red',linewidth=7)
ax9.set_aspect('equal', adjustable='box')
plt.show()

saveFileName = 'HH_HV_VV_WHEAT_VWC.pdf'
file_savePath = 'F:/PHD/WORK_1/GPR_BIOPHYSICAL_PARAM_RETRIEVE_SMAPVEX16_DATA/SMP16_DATA/EXTRACTED_DATA/GPR/WHEAT/BIOMASS/PLOTS/REVISED/VWC/' + saveFileName
print("Saved at: ",file_savePath)
fig9.savefig(file_savePath, bbox_inches = 'tight', dpi = 300, pad_inches = 0.08, 
            frameon = None)



data10=pd.read_excel("insert the .xlsx file that you saved containing the predicted outputs from HH+HV combination")
x=data10["INSITU_VWC"]
y=data10["PREDICTED_VWC"]

fig10 = plt.figure(figsize=(13,13))
ax10 = fig10.add_subplot()
plt.xlim([0, 6])
plt.ylim([0, 6])
plt.plot([0, 6], [0, 6], 'k:',linewidth=7)
plt.scatter(x=data10["INSITU_VWC"],y=data10["PREDICTED_VWC"],c='#17becf',s=800,marker="X")
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.yticks(np.arange(0, 8, 2))
plt.xticks(np.arange(0, 8, 2))
plt.xlabel("Insitu VWC ($Kgm^{-2}$)",size=50)
plt.ylabel("Estimated VWC ($Kgm^{-2}$)",size=50)
plt.annotate(r'$\rho$ = 0.57 ', xy=(0.3, 5.5),size=50)  
plt.annotate('RMSE = 0.71', xy=(0.3, 4.9),size=50)
plt.annotate('MAE = 0.58', xy=(0.3, 4.3),size=50)
plt.annotate('HH+HV',xy=(4.2,0.3),size=50)
popt, _ = curve_fit(objective, x, y)
a, b = popt
x_line = arange(min(x), max(x), 1)
y_line = objective(x_line, a, b)
plt.plot(x_line, y_line, '--', color='red',linewidth=7)
ax10.set_aspect('equal', adjustable='box')
plt.show()

saveFileName = 'HH_HV_WHEAT_VWC.pdf'
file_savePath = 'F:/PHD/WORK_1/GPR_BIOPHYSICAL_PARAM_RETRIEVE_SMAPVEX16_DATA/SMP16_DATA/EXTRACTED_DATA/GPR/WHEAT/BIOMASS/PLOTS/REVISED/VWC/' + saveFileName
print("Saved at: ",file_savePath)
fig10.savefig(file_savePath, bbox_inches = 'tight', dpi = 300, pad_inches = 0.08, 
            frameon = None)



data11=pd.read_excel("insert the .xlsx file that you saved containing the predicted outputs from HV+VV combination")
x=data11["INSITU_VWC"]
y=data11["PREDICTED_VWC"]

fig11 = plt.figure(figsize=(13,13))
ax11 = fig11.add_subplot()
plt.xlim([0, 6])
plt.ylim([0, 6])
plt.plot([0, 6], [0, 6], 'k:',linewidth=7)
plt.scatter(x=data11["INSITU_VWC"],y=data11["PREDICTED_VWC"],c='#17becf',s=800,marker="X")
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.yticks(np.arange(0, 8, 2))
plt.xticks(np.arange(0, 8, 2))
plt.xlabel("Insitu VWC ($Kgm^{-2}$)",size=50)
plt.ylabel("Estimated VWC ($Kgm^{-2}$)",size=50)
plt.annotate(r'$\rho$ = 0.60 ', xy=(0.3, 5.5),size=50)  
plt.annotate('RMSE = 0.69', xy=(0.3, 4.9),size=50)
plt.annotate('MAE = 0.55', xy=(0.3, 4.3),size=50)
plt.annotate('HV+VV',xy=(4.2,0.3),size=50)
popt, _ = curve_fit(objective, x, y)
a, b = popt
x_line = arange(min(x), max(x), 1)
y_line = objective(x_line, a, b)
plt.plot(x_line, y_line, '--', color='red',linewidth=7)
ax11.set_aspect('equal', adjustable='box')
plt.show()

saveFileName = 'HV_VV_WHEAT_VWC.pdf'
file_savePath = 'F:/PHD/WORK_1/GPR_BIOPHYSICAL_PARAM_RETRIEVE_SMAPVEX16_DATA/SMP16_DATA/EXTRACTED_DATA/GPR/WHEAT/BIOMASS/PLOTS/REVISED/VWC/' + saveFileName
print("Saved at: ",file_savePath)
fig11.savefig(file_savePath, bbox_inches = 'tight', dpi = 300, pad_inches = 0.08, 
            frameon = None)


data12=pd.read_excel("insert the .xlsx file that you saved containing the predicted outputs from HH+VV combination")

fig12 = plt.figure(figsize=(13,13))
ax12 = fig12.add_subplot()
plt.xlim([0, 6])
plt.ylim([0, 6])
plt.plot([0, 6], [0, 6], 'k:',linewidth=7)
plt.scatter(x=data12["INSITU_VWC"],y=data12["PREDICTED_VWC"],c='#17becf',s=800,marker="X")
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.yticks(np.arange(0, 8, 2))
plt.xticks(np.arange(0, 8, 2))
plt.xlabel("Insitu VWC ($Kgm^{-2}$)",size=50)
plt.ylabel("Estimated VWC ($Kgm^{-2}$)",size=50)
plt.annotate(r'$\rho$ = 0.63 ', xy=(0.3, 5.5),size=50)  
plt.annotate('RMSE = 0.72', xy=(0.3, 4.9),size=50)
plt.annotate('MAE = 0.62', xy=(0.3, 4.3),size=50)
plt.annotate('HH+VV',xy=(4.2,0.3),size=50)
popt, _ = curve_fit(objective, x, y)
a, b = popt
x_line = arange(min(x), max(x), 1)
y_line = objective(x_line, a, b)
plt.plot(x_line, y_line, '--', color='red',linewidth=7)
ax12.set_aspect('equal', adjustable='box')
plt.show()

saveFileName = 'HH_VV_WHEAT_VWC.pdf'
file_savePath = 'F:/PHD/WORK_1/GPR_BIOPHYSICAL_PARAM_RETRIEVE_SMAPVEX16_DATA/SMP16_DATA/EXTRACTED_DATA/GPR/WHEAT/BIOMASS/PLOTS/REVISED/VWC/' + saveFileName
print("Saved at: ",file_savePath)
fig12.savefig(file_savePath, bbox_inches = 'tight', dpi = 300, pad_inches = 0.08, 
            frameon = None)

















