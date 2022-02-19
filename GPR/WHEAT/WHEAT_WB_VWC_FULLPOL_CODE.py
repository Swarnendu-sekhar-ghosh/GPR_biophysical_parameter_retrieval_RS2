# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 10:53:06 2021

@author: Swarnendu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import GPy
from IPython.display import display
from scipy.special import boxcox, inv_boxcox
from sklearn import metrics

# ######################################### FULLPOL COMBINATION ######################################
 
# ################################## TRAIN DATASET ##########################################

data_TRAIN=pd.read_excel("insert your training dataset file here as .xlsx")

# reading the features 

HH_TRAIN=data_TRAIN["HH"]
HV_TRAIN=data_TRAIN["HV"]
VV_TRAIN=data_TRAIN["VV"]

#reading the target parameters

WB_TRAIN=data_TRAIN["WB"]
VWC_TRAIN=data_TRAIN["VWC"]


# # reading the features and target parameters in one dataframe to check skewness

X1_train=pd.concat([HH_TRAIN,HV_TRAIN,VV_TRAIN,WB_TRAIN,VWC_TRAIN],axis=1)

print('X_train Skew: %r' %X1_train.skew())

#SEPRATING FEATURES AND TARGET PARAMETERS BEFORE TRANSFORMATION

X_train=pd.concat([HH_TRAIN,HV_TRAIN,VV_TRAIN],axis=1)

Y_WB_train=X1_train["WB"]
Y_WB_train=np.array(Y_WB_train).reshape(-1,1);

Y_VWC_train=X1_train["VWC"]
Y_VWC_train=np.array(Y_VWC_train).reshape(-1,1);

# # BOXCOX TRANSFORMATION

# # transform training data & save lambda value

HH_train,fitted_lambda1 = stats.boxcox(X_train["HH"])
HV_train,fitted_lambda2 = stats.boxcox(X_train["HV"])
VV_train,fitted_lambda3 =stats.boxcox(X_train["VV"])
WB_train,fitted_lambda4 =stats.boxcox(Y_WB_train[:,0])
VWC_train,fitted_lambda5 =stats.boxcox(Y_VWC_train[:,0])


print(fitted_lambda1)
print(fitted_lambda2)
print(fitted_lambda3)
print(fitted_lambda4)
print(fitted_lambda5)

X_train_trans=np.column_stack((HH_train,HV_train,VV_train))

Y_WB_train_trans=WB_train

Y_WB_train_trans=np.array(Y_WB_train_trans).reshape(-1,1);

Y_VWC_train_trans=VWC_train
Y_VWC_train_trans=np.array(Y_VWC_train_trans).reshape(-1,1);


# #CHECKING FOR SKEWNESS OF TRANSFORMED DATASET

X_column_values=["HH","HV","VV"]

X_train_trans = pd.DataFrame(data = X_train_trans, columns = X_column_values)

print('X_train_trans Skew: %r' %X_train_trans.skew())

WB_column_values=["WB"]

Y_WB_train_trans=pd.DataFrame(data = Y_WB_train_trans, columns = WB_column_values)

print('Y_WB_train_trans Skew: %r' %Y_WB_train_trans.skew())

VWC_column_values=["VWC"]

Y_VWC_train_trans=pd.DataFrame(data = Y_VWC_train_trans, columns = VWC_column_values)

print('Y_VWC_train_trans Skew: %r' %Y_VWC_train_trans.skew())

# ###################################### GPY MODEL #############################################

# RBF kernel applied 


k1= GPy.kern.Linear(input_dim=3,ARD=False,name='linear')

k2 = GPy.kern.RBF(input_dim=3, lengthscale=1,ARD=False)


kernel=k1+k2

# Gaussian process regression model for wetbiomass

m1 = GPy.models.GPRegression(X_train_trans,Y_WB_train_trans,kernel)

display(m1)


# optimize of hyperparameters

m1.optimize(optimizer='scg',messages=True,max_iters=1000)

print(m1)

# Gaussian process regression model for vegetation water content

m2 = GPy.models.GPRegression(X_train_trans,Y_VWC_train_trans,kernel)

display(m2)

# optimize of hyperparameters

m2.optimize(optimizer='scg',messages=True,max_iters=1000)


print(m2)

########################### Predicting for the train set ########################################

X_train_trans=np.array(X_train_trans)

# when predicting for wetbiomass

Y_WB_train_pred=m1.predict(X_train_trans)

Y_WB_train_inv= inv_boxcox(Y_WB_train_trans,fitted_lambda4)

Y_WB_train_pred_inv= inv_boxcox(Y_WB_train_pred[0],fitted_lambda4)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

train_rmse1 = rmse(np.array(Y_WB_train_inv), np.array(Y_WB_train_pred_inv))

print('TRAIN WB RMSE: %f' %train_rmse1)

#train correlation coefficient

train_corr1,_=np.corrcoef(np.squeeze(np.array(Y_WB_train_inv)), np.squeeze(np.array(Y_WB_train_pred_inv)))

train_corr1,_
train_corr1=train_corr1[1]

print('TRAIN WB CORRELATION: %f' %train_corr1)

#train mean absolute error

train_mae1=metrics.mean_absolute_error(Y_WB_train_inv,Y_WB_train_pred_inv)

print('Train WB MAE: %f' %train_mae1)

# Train WB Plotting

fig1 = plt.figure()
ax1 = fig1.add_subplot()
plt.plot(Y_WB_train_inv,Y_WB_train_pred_inv, 'go',markersize=10,marker=".",color='g')
plt.xlim([0, 6])
plt.ylim([0, 6])
plt.xlabel("Insitu WetBiomass ($Kgm^{-2}$)")
plt.ylabel("Estimated WetBiomass ($Kgm^{-2}$)")
plt.plot([0, 6], [0, 6], 'k:')
plt.annotate('r = %.3f'%train_corr1, xy=(0.45, 5.7))           #round off upto 3decimals
plt.annotate('RMSE = %.3f'%train_rmse1, xy=(0.45, 5.2))
plt.annotate('MAE = %.3f'%train_mae1, xy=(0.45, 4.7))
plt.annotate('TRAIN',xy=(4,0.4))
plt.annotate('HH+HV+VV',xy=(4,1))
ax1.set_aspect('equal', adjustable='box')
plt.show()

# predicting for Vegetation Water Content (VWC)

Y_VWC_train_pred=m2.predict(X_train_trans)

Y_VWC_train_inv= inv_boxcox(Y_VWC_train_trans,fitted_lambda5)

Y_VWC_train_pred_inv= inv_boxcox(Y_VWC_train_pred[0],fitted_lambda5)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

train_rmse2 = rmse(np.array(Y_VWC_train_inv), np.array(Y_VWC_train_pred_inv))

print('TRAIN VWC RMSE: %f' %train_rmse2)

#train correlation coefficient

train_corr2,_=np.corrcoef(np.squeeze(np.array(Y_VWC_train_inv)), np.squeeze(np.array(Y_VWC_train_pred_inv)))

train_corr2,_
train_corr2=train_corr2[1]

print('TRAIN VWC CORRELATION: %f' %train_corr2)

#train mean absolute error

train_mae2=metrics.mean_absolute_error(Y_VWC_train_inv,Y_VWC_train_pred_inv)

print('Train VWC MAE: %f' %train_mae2)

# # Train VWC Plotting

fig2 = plt.figure()
ax2 = fig2.add_subplot()
plt.plot(Y_VWC_train_inv,Y_VWC_train_pred_inv, 'go',markersize=10,marker=".",color='g')
plt.xlim([0, 6])
plt.ylim([0, 6])
plt.xlabel("Insitu VWC ($Kgm^{-2}$)")
plt.ylabel("Estimated VWC ($Kgm^{-2}$)")
plt.plot([0, 6], [0, 6], 'k:')
plt.annotate('r = %.3f'%train_corr2, xy=(0.45, 5.7))           #round off upto 3decimals
plt.annotate('RMSE = %.3f'%train_rmse2, xy=(0.45, 5.2))
plt.annotate('MAE = %.3f'%train_mae2, xy=(0.45, 4.7))
plt.annotate('TRAIN',xy=(4,0.4))
plt.annotate('HH+HV+VV',xy=(4,1))
ax2.set_aspect('equal', adjustable='box')
plt.show()


################################################## TEST DATASET #################################################################

data_TEST=pd.read_excel("insert your test dataset file here as .xlsx")

# reading the features 

HH_TEST=data_TEST["HH"]
HV_TEST=data_TEST["HV"]
VV_TEST=data_TEST["VV"]

#reading the target parameters

WB_TEST=data_TEST["WB"]
VWC_TEST=data_TEST["VWC"]

# reading the features and target parameters in one dataframe to check skewness

X1_test=pd.concat([HH_TEST,HV_TEST,VV_TEST,WB_TEST,VWC_TEST],axis=1)

print('X_test Skew: %r' %X1_test.skew())

#SEPRATING FEATURES AND TARGET PARAMETERS BEFORE TRANSFORMATION

X_test=pd.concat([HH_TEST,HV_TEST,VV_TEST],axis=1)

Y_WB_test=X1_test["WB"]
Y_WB_test=np.array(Y_WB_test).reshape(-1,1);

Y_VWC_test=X1_test["VWC"]
Y_VWC_test=np.array(Y_VWC_test).reshape(-1,1);


# use lambda value to transform test data

HH_test= stats.boxcox(X_test["HH"], fitted_lambda1)
HV_test= stats.boxcox(X_test["HV"], fitted_lambda2)
VV_test= stats.boxcox(X_test["VV"], fitted_lambda3)
WB_test= stats.boxcox(Y_WB_test[:,0], fitted_lambda4)
VWC_test= stats.boxcox(Y_VWC_test[:,0], fitted_lambda5)

X_test_trans=np.column_stack((HH_test,HV_test,VV_test))

Y_WB_test_trans=WB_test
Y_WB_test_trans=np.array(Y_WB_test_trans).reshape(-1,1);

Y_VWC_test_trans=VWC_test
Y_VWC_test_trans=np.array(Y_VWC_test_trans).reshape(-1,1);

# Checking for skewness

X_column_values=["HH","HV","VV"]

X_test_trans = pd.DataFrame(data = X_test_trans, columns = X_column_values)

print('X_test_trans Skew: %r' %X_test_trans.skew())

WB_column_values=["WB"]

Y_WB_test_trans=pd.DataFrame(data = Y_WB_test_trans, columns = WB_column_values)

print('Y_WB_test_trans Skew: %r' %Y_WB_test_trans.skew())


WB_column_values=["VWC"]

Y_VWC_test_trans=pd.DataFrame(data = Y_VWC_test_trans, columns = VWC_column_values)

print('Y_VWC_test_trans Skew: %r' %Y_VWC_test_trans.skew())

########################### Predicting for the test set ########################################

X_test_trans=np.array(X_test_trans)

# when predicting for wetbiomass

Y_WB_test_pred=m1.predict(X_test_trans)

Y_WB_test_inv= inv_boxcox(Y_WB_test_trans,fitted_lambda4)

Y_WB_test_pred_inv= inv_boxcox(Y_WB_test_pred[0],fitted_lambda4)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

test_rmse1 = rmse(np.array(Y_WB_test_inv), np.array(Y_WB_test_pred_inv))

print('TEST WB RMSE: %f' %test_rmse1)

#train correlation coefficient

test_corr1,_=np.corrcoef(np.squeeze(np.array(Y_WB_test_inv)), np.squeeze(np.array(Y_WB_test_pred_inv)))

test_corr1,_
test_corr1=test_corr1[1]

print('TEST WB CORRELATION: %f' %test_corr1)

#train mean absolute error

test_mae1=metrics.mean_absolute_error(Y_WB_test_inv,Y_WB_test_pred_inv)

print('TEST WB MAE: %f' %test_mae1)

# # Test WB Plotting
# depict illustartion

fig3 = plt.figure()
ax3 = fig3.add_subplot()
plt.plot(Y_WB_test_inv,Y_WB_test_pred_inv, 'go',markersize=10,marker=".",color='g')
plt.xlim([0, 6])
plt.ylim([0, 6])
plt.xlabel("Insitu WetBiomass ($Kgm^{-2}$)")
plt.ylabel("Estimated WetBiomass ($Kgm^{-2}$)")
plt.plot([0, 6], [0, 6], 'k:')
plt.annotate('r = %.3f'%test_corr1, xy=(0.45, 5.7))           #round off upto 3decimals
plt.annotate('RMSE = %.3f'%test_rmse1, xy=(0.45, 5.2))
plt.annotate('MAE = %.3f'%test_mae1, xy=(0.45, 4.7))
plt.annotate('TEST',xy=(4,0.4))
plt.annotate('HH+HV+VV',xy=(4,1))
ax3.set_aspect('equal', adjustable='box')
plt.show()

# predicting for Vegetation Water Content (VWC)


Y_VWC_test_pred=m2.predict(X_test_trans)

Y_VWC_test_inv= inv_boxcox(Y_VWC_test_trans,fitted_lambda5)

Y_VWC_test_pred_inv= inv_boxcox(Y_VWC_test_pred[0],fitted_lambda5)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

test_rmse2 = rmse(np.array(Y_VWC_test_inv), np.array(Y_VWC_test_pred_inv))

print('TEST VWC RMSE: %f' %test_rmse2)

#train correlation coefficient

test_corr2,_=np.corrcoef(np.squeeze(np.array(Y_VWC_test_inv)), np.squeeze(np.array(Y_VWC_test_pred_inv)))

test_corr2,_
test_corr2=test_corr2[1]

print('TEST VWC CORRELATION: %f' %test_corr2)

#train mean absolute error

test_mae2=metrics.mean_absolute_error(Y_VWC_test_inv,Y_VWC_test_pred_inv)

print('TEST VWC MAE: %f' %test_mae2)

# # Test VWC Plotting


fig4 = plt.figure()
ax4 = fig4.add_subplot()
plt.plot(Y_VWC_test_inv,Y_VWC_test_pred_inv, 'go',markersize=10,marker=".",color='g')
plt.xlim([0, 6])
plt.ylim([0, 6])
plt.xlabel("Insitu VWC ($Kgm^{-2}$)")
plt.ylabel("Estimated VWC ($Kgm^{-2}$)")
plt.plot([0, 6], [0, 6], 'k:')
plt.annotate('r = %.3f'%test_corr2, xy=(0.45, 5.7))           #round off upto 3decimals
plt.annotate('RMSE = %.3f'%test_rmse2, xy=(0.45, 5.2))
plt.annotate('MAE = %.3f'%test_mae2, xy=(0.45, 4.7))
plt.annotate('TEST',xy=(4,0.4))
plt.annotate('HH+HV+VV',xy=(4,1))
ax4.set_aspect('equal', adjustable='box')
plt.show()


# INS1_column_values=["INSITU WETBIOMASS"]

# INS1=pd.DataFrame(data=np.array(Y_WB_train_inv),columns=INS1_column_values)

# PRED1_column_values=["PREDICTED WETBIOMASS"]

# PRED1=pd.DataFrame(data=Y_WB_train_pred_inv,columns=PRED1_column_values)

# INS1_PRED1=pd.concat([INS1,PRED1],axis=1)

# INS1_PRED1.to_excel('INS_HH_HV_VV_WB_PRED_TRAIN.xlsx', index=False)


# INS2_column_values=["INSITU WETBIOMASS"]

# INS2=pd.DataFrame(data=np.array(Y_WB_test_inv),columns=INS2_column_values)

# PRED2_column_values=["PREDICTED WETBIOMASS"]

# PRED2=pd.DataFrame(data=Y_WB_test_pred_inv,columns=PRED2_column_values)

# INS2_PRED2=pd.concat([INS2,PRED2],axis=1)

# INS2_PRED2.to_excel('INS_HH_HV_VV_WB_PRED_TEST.xlsx', index=False)


# INS3_column_values=["INSITU VWC"]

# INS3=pd.DataFrame(data=np.array(Y_VWC_train_inv),columns=INS3_column_values)

# PRED3_column_values=["PREDICTED VWC"]

# PRED3=pd.DataFrame(data=Y_VWC_train_pred_inv,columns=PRED3_column_values)

# INS3_PRED3=pd.concat([INS3,PRED3],axis=1)

# INS3_PRED3.to_excel('INS_HH_HV_VV_VWC_PRED_TRAIN.xlsx', index=False)


# INS4_column_values=["INSITU VWC"]

# INS4=pd.DataFrame(data=np.array(Y_VWC_test_inv),columns=INS4_column_values)

# PRED4_column_values=["PREDICTED VWC"]

# PRED4=pd.DataFrame(data=Y_VWC_test_pred_inv,columns=PRED4_column_values)

# INS4_PRED4=pd.concat([INS4,PRED4],axis=1)

# INS4_PRED4.to_excel('INS_HH_HV_VV_VWC_PRED_TEST.xlsx', index=False)


# ############################################ DUAL POL COMBINATION ###########################################
# ############################################ TRAIN DATASET ##################################################

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats
# import seaborn as sns
# import GPy
# from IPython.display import display
# from scipy.special import boxcox, inv_boxcox
# from sklearn import metrics

# data_TRAIN=pd.read_excel("F:\SMAPVEX_16_DATA\SMAPVEX16\Extracted_Crops_Data\WHEAT\BIOMASS\SMAPVEX16_WHEAT_BIOMASS_TRAIN_DATA.xlsx")

# # reading the features 

# #HH_TRAIN=data_TRAIN["HH"]
# HV_TRAIN=data_TRAIN["HV"]
# VV_TRAIN=data_TRAIN["VV"]

# #reading the target parameters

# WB_TRAIN=data_TRAIN["WB"]
# VWC_TRAIN=data_TRAIN["VWC"]


# # reading the features and target parameters in one dataframe to check skewness

# #X1_train=pd.concat([HH_TRAIN,HV_TRAIN,WB_TRAIN,VWC_TRAIN],axis=1)
# X1_train=pd.concat([HV_TRAIN,VV_TRAIN,WB_TRAIN,VWC_TRAIN],axis=1)
# #X1_train=pd.concat([HH_TRAIN,VV_TRAIN,WB_TRAIN,VWC_TRAIN],axis=1)


# print('X_train Skew: %r' %X1_train.skew())

# #SEPRATING FEATURES AND TARGET PARAMETERS BEFORE TRANSFORMATION

# #X_train=pd.concat([HH_TRAIN,HV_TRAIN],axis=1)
# X_train=pd.concat([HV_TRAIN,VV_TRAIN],axis=1)
# #X_train=pd.concat([HH_TRAIN,VV_TRAIN],axis=1)

# Y_WB_train=X1_train["WB"]
# Y_WB_train=np.array(Y_WB_train).reshape(-1,1);

# Y_VWC_train=X1_train["VWC"]
# Y_VWC_train=np.array(Y_VWC_train).reshape(-1,1);

# # BOXCOX TRANSFORMATION

# # transform training data & save lambda value

# #HH_train,fitted_lambda1 = stats.boxcox(X_train["HH"])
# HV_train,fitted_lambda2 = stats.boxcox(X_train["HV"])
# VV_train,fitted_lambda3 =stats.boxcox(X_train["VV"])
# WB_train,fitted_lambda4 =stats.boxcox(Y_WB_train[:,0])
# VWC_train,fitted_lambda5 =stats.boxcox(Y_VWC_train[:,0])


# #print(fitted_lambda1)
# print(fitted_lambda2)
# print(fitted_lambda3)
# print(fitted_lambda4)
# print(fitted_lambda5)


# #X_train_trans=np.column_stack((HH_train,HV_train))
# X_train_trans=np.column_stack((HV_train,VV_train))
# #X_train_trans=np.column_stack((HH_train,VV_train))

# Y_WB_train_trans=WB_train
# Y_WB_train_trans=np.array(Y_WB_train_trans).reshape(-1,1);

# Y_VWC_train_trans=VWC_train
# Y_VWC_train_trans=np.array(Y_VWC_train_trans).reshape(-1,1);


# #CHECKING FOR SKEWNESS OF TRANSFORMED DATASET

# #X_column_values=["HH","HV"]
# X_column_values=["HV","VV"]
# #X_column_values=["HH","VV"]


# X_train_trans = pd.DataFrame(data = X_train_trans, columns = X_column_values)

# print('X_train_trans Skew: %r' %X_train_trans.skew())

# WB_column_values=["WB"]

# Y_WB_train_trans=pd.DataFrame(data = Y_WB_train_trans, columns = WB_column_values)

# print('Y_WB_train_trans Skew: %r' %Y_WB_train_trans.skew())

# VWC_column_values=["VWC"]

# Y_VWC_train_trans=pd.DataFrame(data = Y_VWC_train_trans, columns = VWC_column_values)

# print('Y_VWC_train_trans Skew: %r' %Y_VWC_train_trans.skew())

# # ###################################### GPY MODEL #############################################

# # RBF kernel applied 

# #k0=GPy.kern.White(input_dim=3)

# k1= GPy.kern.Linear(input_dim=2,ARD=False,name='linear')

# k2 = GPy.kern.RBF(input_dim=2, lengthscale=1,ARD=False)

# k3= GPy.kern.Exponential(input_dim=2,variance=1.,lengthscale=1,ARD=False,name='Exponential')

# k4= GPy.kern.Matern32(input_dim=2,variance=1.,ARD=False,name='Mat32',lengthscale=1)

# k5= GPy.kern.Matern52(input_dim=2,variance=1.,ARD=False,name='Mat52',lengthscale=1)

# k6=GPy.kern.Poly(input_dim=2,variance=1,scale=1,bias=1,order=2,name='poly')

# k7=GPy.kern.RatQuad(input_dim=2,variance=1,lengthscale=1,power=2,ARD=False,name='RatQuad')

# kernel=k1+k2

# # Gaussian process regression model for wetbiomass

# m1 = GPy.models.GPRegression(X_train_trans,Y_WB_train_trans,kernel)

# display(m1)


# # optimize of hyperparameters

# m1.optimize(optimizer='scg',messages=True,max_iters=1000)

# print(m1)

# # Gaussian process regression model for vegetation water content

# m2 = GPy.models.GPRegression(X_train_trans,Y_VWC_train_trans,kernel)

# display(m2)

# # optimize of hyperparameters

# m2.optimize(optimizer='scg',messages=True,max_iters=1000)


# print(m2)

# ########################### Predicting for the train set ########################################

# X_train_trans=np.array(X_train_trans)

# # when predicting for wetbiomass

# Y_WB_train_pred=m1.predict(X_train_trans)

# Y_WB_train_inv= inv_boxcox(Y_WB_train_trans,fitted_lambda4)

# Y_WB_train_pred_inv= inv_boxcox(Y_WB_train_pred[0],fitted_lambda4)

# def rmse(predictions, targets):
#     return np.sqrt(((predictions - targets) ** 2).mean())

# train_rmse1 = rmse(np.array(Y_WB_train_inv), np.array(Y_WB_train_pred_inv))

# print('TRAIN WB RMSE: %f' %train_rmse1)

# #train correlation coefficient

# train_corr1,_=np.corrcoef(np.squeeze(np.array(Y_WB_train_inv)), np.squeeze(np.array(Y_WB_train_pred_inv)))

# train_corr1,_
# train_corr1=train_corr1[1]

# print('TRAIN WB CORRELATION: %f' %train_corr1)

# #train mean absolute error

# train_mae1=metrics.mean_absolute_error(Y_WB_train_inv,Y_WB_train_pred_inv)

# print('Train WB MAE: %f' %train_mae1)

# # Train WB Plotting

# fig1 = plt.figure()
# ax1 = fig1.add_subplot()
# plt.plot(Y_WB_train_inv,Y_WB_train_pred_inv, 'go',markersize=10,marker=".",color='g')
# plt.xlim([0, 7])
# plt.ylim([0, 7])
# plt.xlabel("Insitu WetBiomass ($Kgm^{-2}$)")
# plt.ylabel("Estimated WetBiomass ($Kgm^{-2}$)")
# plt.plot([0, 7], [0, 7], 'k:')
# plt.annotate('r = %.3f'%train_corr1, xy=(0.45, 6.7))           #round off upto 3decimals
# plt.annotate('RMSE = %.3f'%train_rmse1, xy=(0.45, 6.2))
# plt.annotate('MAE = %.3f'%train_mae1, xy=(0.45, 5.7))
# plt.annotate('TRAIN',xy=(5,0.4))
# plt.annotate('HV+VV',xy=(5,1))
# ax1.set_aspect('equal', adjustable='box')
# plt.show()

# # predicting for Vegetation Water Content (VWC)

# Y_VWC_train_pred=m2.predict(X_train_trans)

# Y_VWC_train_inv= inv_boxcox(Y_VWC_train_trans,fitted_lambda5)

# Y_VWC_train_pred_inv= inv_boxcox(Y_VWC_train_pred[0],fitted_lambda5)

# def rmse(predictions, targets):
#     return np.sqrt(((predictions - targets) ** 2).mean())

# train_rmse2 = rmse(np.array(Y_VWC_train_inv), np.array(Y_VWC_train_pred_inv))

# print('TRAIN VWC RMSE: %f' %train_rmse2)

# #train correlation coefficient

# train_corr2,_=np.corrcoef(np.squeeze(np.array(Y_VWC_train_inv)), np.squeeze(np.array(Y_VWC_train_pred_inv)))

# train_corr2,_
# train_corr2=train_corr2[1]

# print('TRAIN VWC CORRELATION: %f' %train_corr2)

# #train mean absolute error

# train_mae2=metrics.mean_absolute_error(Y_VWC_train_inv,Y_VWC_train_pred_inv)

# print('Train VWC MAE: %f' %train_mae2)

# # # Train VWC Plotting

# fig2 = plt.figure()
# ax2 = fig2.add_subplot()
# plt.plot(Y_VWC_train_inv,Y_VWC_train_pred_inv, 'go',markersize=10,marker=".",color='g')
# plt.xlim([0, 6])
# plt.ylim([0, 6])
# plt.xlabel("Insitu VWC ($Kgm^{-2}$)")
# plt.ylabel("Estimated VWC ($Kgm^{-2}$)")
# plt.plot([0, 6], [0, 6], 'k:')
# plt.annotate('r = %.3f'%train_corr2, xy=(0.45, 5.7))           #round off upto 3decimals
# plt.annotate('RMSE = %.3f'%train_rmse2, xy=(0.45, 5.2))
# plt.annotate('MAE = %.3f'%train_mae2, xy=(0.45, 4.7))
# plt.annotate('TRAIN',xy=(4,0.4))
# plt.annotate('HV+VV',xy=(4,1))
# ax2.set_aspect('equal', adjustable='box')
# plt.show()


# ###################################### TEST DATASET ###############################################

# data_TEST=pd.read_excel("F:\SMAPVEX_16_DATA\SMAPVEX16\Extracted_Crops_Data\WHEAT\BIOMASS\SMAPVEX16_WHEAT_BIOMASS_TEST_DATA.xlsx")

# # reading the features 

# #HH_TEST=data_TEST["HH"]
# HV_TEST=data_TEST["HV"]
# VV_TEST=data_TEST["VV"]

# #reading the target parameters

# WB_TEST=data_TEST["WB"]
# VWC_TEST=data_TEST["VWC"]

# # reading the features and target parameters in one dataframe to check skewness

# #X1_test=pd.concat([HH_TEST,HV_TEST,WB_TEST,VWC_TEST],axis=1)
# X1_test=pd.concat([HV_TEST,VV_TEST,WB_TEST,VWC_TEST],axis=1)
# #X1_test=pd.concat([HH_TEST,VV_TEST,WB_TEST,VWC_TEST],axis=1)

# print('X_test Skew: %r' %X1_test.skew())

# #SEPRATING FEATURES AND TARGET PARAMETERS BEFORE TRANSFORMATION

# #X_test=pd.concat([HH_TEST,HV_TEST],axis=1)
# X_test=pd.concat([HV_TEST,VV_TEST],axis=1)
# #X_test=pd.concat([HH_TEST,VV_TEST],axis=1)

# Y_WB_test=X1_test["WB"]
# Y_WB_test=np.array(Y_WB_test).reshape(-1,1);

# Y_VWC_test=X1_test["VWC"]
# Y_VWC_test=np.array(Y_VWC_test).reshape(-1,1);

# # use lambda value to transform test data

# #HH_test= stats.boxcox(X_test["HH"], fitted_lambda1)
# HV_test= stats.boxcox(X_test["HV"], fitted_lambda2)
# VV_test= stats.boxcox(X_test["VV"], fitted_lambda3)
# WB_test= stats.boxcox(Y_WB_test[:,0], fitted_lambda4)
# VWC_test= stats.boxcox(Y_VWC_test[:,0], fitted_lambda5)


# #X_test_trans=np.column_stack((HH_test,HV_test))
# X_test_trans=np.column_stack((HV_test,VV_test))
# #X_test_trans=np.column_stack((HH_test,VV_test))

# Y_WB_test_trans=WB_test
# Y_WB_test_trans=np.array(Y_WB_test_trans).reshape(-1,1);

# Y_VWC_test_trans=VWC_test
# Y_VWC_test_trans=np.array(Y_VWC_test_trans).reshape(-1,1);


# #X_column_values=["HH","HV"]
# X_column_values=["HV","VV"]
# #X_column_values=["HH","VV"]


# X_test_trans = pd.DataFrame(data = X_test_trans, columns = X_column_values)

# print('X_test_trans Skew: %r' %X_test_trans.skew())

# WB_column_values=["WB"]

# Y_WB_test_trans=pd.DataFrame(data = Y_WB_test_trans, columns = WB_column_values)

# print('Y_WB_test_trans Skew: %r' %Y_WB_test_trans.skew())

# VWC_column_values=["VWC"]

# Y_VWC_test_trans=pd.DataFrame(data = Y_VWC_test_trans, columns = VWC_column_values)

# print('Y_VWC_test_trans Skew: %r' %Y_VWC_test_trans.skew())

# ########################### Predicting for the test set ########################################

# X_test_trans=np.array(X_test_trans)

# # when predicting for wetbiomass

# Y_WB_test_pred=m1.predict(X_test_trans)

# Y_WB_test_inv= inv_boxcox(Y_WB_test_trans,fitted_lambda4)

# Y_WB_test_pred_inv= inv_boxcox(Y_WB_test_pred[0],fitted_lambda4)

# def rmse(predictions, targets):
#     return np.sqrt(((predictions - targets) ** 2).mean())

# test_rmse1 = rmse(np.array(Y_WB_test_inv), np.array(Y_WB_test_pred_inv))

# print('TEST WB RMSE: %f' %test_rmse1)

# #train correlation coefficient

# test_corr1,_=np.corrcoef(np.squeeze(np.array(Y_WB_test_inv)), np.squeeze(np.array(Y_WB_test_pred_inv)))

# test_corr1,_
# test_corr1=test_corr1[1]

# print('TEST WB CORRELATION: %f' %test_corr1)

# #train mean absolute error

# test_mae1=metrics.mean_absolute_error(Y_WB_test_inv,Y_WB_test_pred_inv)

# print('TEST WB MAE: %f' %test_mae1)

# # # Test WB Plotting
# # depict illustartion

# fig3 = plt.figure()
# ax3 = fig3.add_subplot()
# plt.plot(Y_WB_test_inv,Y_WB_test_pred_inv, 'go',markersize=10,marker=".",color='g')
# plt.xlim([0, 7])
# plt.ylim([0, 7])
# plt.xlabel("Insitu WetBiomass ($Kgm^{-2}$)")
# plt.ylabel("Estimated WetBiomass ($Kgm^{-2}$)")
# plt.plot([0, 7], [0, 7], 'k:')
# plt.annotate('r = %.3f'%test_corr1, xy=(0.45, 6.7))           #round off upto 3decimals
# plt.annotate('RMSE = %.3f'%test_rmse1, xy=(0.45, 6.2))
# plt.annotate('MAE = %.3f'%test_mae1, xy=(0.45, 5.7))
# plt.annotate('TEST',xy=(4,0.4))
# plt.annotate('HV+VV',xy=(4,1))
# ax3.set_aspect('equal', adjustable='box')
# plt.show()

# # predicting for Vegetation Water Content (VWC)


# Y_VWC_test_pred=m2.predict(X_test_trans)

# Y_VWC_test_inv= inv_boxcox(Y_VWC_test_trans,fitted_lambda5)

# Y_VWC_test_pred_inv= inv_boxcox(Y_VWC_test_pred[0],fitted_lambda5)

# def rmse(predictions, targets):
#     return np.sqrt(((predictions - targets) ** 2).mean())

# test_rmse2 = rmse(np.array(Y_VWC_test_inv), np.array(Y_VWC_test_pred_inv))

# print('TEST VWC RMSE: %f' %test_rmse2)

# #train correlation coefficient

# test_corr2,_=np.corrcoef(np.squeeze(np.array(Y_VWC_test_inv)), np.squeeze(np.array(Y_VWC_test_pred_inv)))

# test_corr2,_
# test_corr2=test_corr2[1]

# print('TEST VWC CORRELATION: %f' %test_corr2)

# #train mean absolute error

# test_mae2=metrics.mean_absolute_error(Y_VWC_test_inv,Y_VWC_test_pred_inv)

# print('TEST VWC MAE: %f' %test_mae2)

# # # Test VWC Plotting

# fig4 = plt.figure()
# ax4 = fig4.add_subplot()
# plt.plot(Y_VWC_test_inv,Y_VWC_test_pred_inv, 'go',markersize=10,marker=".",color='g')
# plt.xlim([0, 6])
# plt.ylim([0, 6])
# plt.xlabel("Insitu VWC ($Kgm^{-2}$)")
# plt.ylabel("Estimated VWC ($Kgm^{-2}$)")
# plt.plot([0, 6], [0, 6], 'k:')
# plt.annotate('r = %.3f'%test_corr2, xy=(0.45, 5.7))           #round off upto 3decimals
# plt.annotate('RMSE = %.3f'%test_rmse2, xy=(0.45, 5.2))
# plt.annotate('MAE = %.3f'%test_mae2, xy=(0.45, 4.7))
# plt.annotate('TEST',xy=(4,0.4))
# plt.annotate('HV+VV',xy=(4,1))
# ax4.set_aspect('equal', adjustable='box')
# plt.show()



