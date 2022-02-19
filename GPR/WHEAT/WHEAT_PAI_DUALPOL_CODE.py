# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 18:05:08 2021

@author: Swarnendu
"""

############################################ HH-HV COMBINATION ##############################



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import GPy
from IPython.display import display
from scipy.special import boxcox, inv_boxcox
from sklearn import metrics

#####################################################TRAINING DATASET ###########################################

data_TRAIN=pd.read_excel("insert your training dataset file here as .xlsx")

# reading the features 

HH_TRAIN=data_TRAIN["HH"]
HV_TRAIN=data_TRAIN["HV"]
#VV_TRAIN=data_TRAIN["VV"]

# reading the target parameters

PAI_TRAIN=data_TRAIN["PAI"]


# reading the features and target parameters in one dataframe to check skewness

X1_train=pd.concat([HH_TRAIN,HV_TRAIN,PAI_TRAIN],axis=1)
#X1_train=pd.concat([HV_TRAIN,VV_TRAIN,PAI_TRAIN],axis=1)
#X1_train=pd.concat([HH_TRAIN,VV_TRAIN,PAI_TRAIN],axis=1)


# SEPRATING FEATURES AND TARGET PARAMETERS BEFORE TRANSFORMATION

X_train=pd.concat([HH_TRAIN,HV_TRAIN],axis=1)

Y_train=X1_train["PAI"]
Y_train=np.array(Y_train).reshape(-1,1);

# BOXCOX TRANSFORMATION

# transform training data & save lambda value

HH_train,fitted_lambda1 = stats.boxcox(X_train["HH"])
HV_train,fitted_lambda2 = stats.boxcox(X_train["HV"])
PAI_train,fitted_lambda4 =stats.boxcox(Y_train[:,0])

print(fitted_lambda1)
print(fitted_lambda2)
print(fitted_lambda4)

X_train_trans=np.column_stack((HH_train,HV_train))


Y_train_trans=PAI_train
Y_train_trans=np.array(Y_train_trans).reshape(-1,1);


###################################### GPY MODEL #############################################

# RBF kernel applied 


k1= GPy.kern.Linear(input_dim=2,ARD=False,name='linear')

k2 = GPy.kern.RBF(input_dim=2, lengthscale=1,ARD=False)

kernel=k2+k1

# Gaussian process regression model

m = GPy.models.GPRegression(X_train_trans,Y_train_trans,kernel)

display(m)

# Optimize of hyperparameters and plot

m.optimize(optimizer='scg',messages=True,max_iters=1000)


print(m)

# ######################### Predicting for the train set ########################################

X_train_trans=np.array(X_train_trans)

Y_train_pred=m.predict(X_train_trans)

Y_train_inv= inv_boxcox(Y_train_trans,fitted_lambda4)

Y_train_pred_inv= inv_boxcox(Y_train_pred[0],fitted_lambda4)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

train_rmse = rmse(np.array(Y_train_inv[:,0]), np.array(Y_train_pred_inv[:,0]))

print('TRAIN RMSE: %f' %train_rmse)

#train correlation coefficient

train_corr,_=np.corrcoef(np.squeeze(np.array(Y_train_inv[:,0])), np.squeeze(np.array(Y_train_pred_inv[:,0])))

train_corr,_
train_corr=train_corr[1]

print('TRAIN CORRELATION: %f' %train_corr)

#train mean absolute error

train_mae=metrics.mean_absolute_error(Y_train_inv[:,0],Y_train_pred_inv[:,0])

print('Train MAE: %f' %train_mae)

# # Train LAI Plotting
# depict illustartion

fig1 = plt.figure()
ax1 = fig1.add_subplot()
plt.plot(Y_train_inv,Y_train_pred_inv, 'go',markersize=10,marker=".",color='g')
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.xlabel("Insitu PAI ($m^2m^{-2}$)")
plt.ylabel("Estimated PAI ($m^2m^{-2}$)")
plt.plot([0, 10], [0, 10], 'k:')
plt.annotate('r = %.3f'%train_corr, xy=(0.5, 9.2))           #round off upto 3decimals
plt.annotate('RMSE = %.3f'%train_rmse, xy=(0.5, 8.7))
plt.annotate('MAE = %.3f'%train_mae, xy=(0.5, 8.2))
plt.annotate('TRAIN',xy=(7,0.4))
plt.annotate('HH+HV',xy=(7,1))

ax1.set_aspect('equal', adjustable='box')
plt.show()


######################################### TEST DATASET ###############################################

data_TEST=pd.read_excel("insert your test dataset file here as .xlsx")

#reading the features 

HH_TEST=data_TEST["HH"]
HV_TEST=data_TEST["HV"]

# reading the target parameters

PAI_TEST=data_TEST["PAI"]

# SEPRATING FEATURES AND TARGET PARAMETERS BEFORE TRANSFORMATION

X_test=pd.concat([HH_TEST,HV_TEST],axis=1)

Y_test=PAI_TEST
Y_test=np.array(Y_test).reshape(-1,1);

# use lambda value to transform test data

HH_test= stats.boxcox(X_test["HH"], fitted_lambda1)
HV_test= stats.boxcox(X_test["HV"], fitted_lambda2)
PAI_test= stats.boxcox(Y_test[:,0], fitted_lambda4)

X_test_trans=np.column_stack((HH_test,HV_test))


Y_test_trans=PAI_test
Y_test_trans=np.array(Y_test_trans).reshape(-1,1);

########################## Predicting for the test set ###########################################

X_test_trans=np.array(X_test_trans)

Y_test_trans_pred=m.predict(X_test_trans)

Y_test_inv= inv_boxcox(Y_test_trans,fitted_lambda4)

Y_test_pred_inv= inv_boxcox(Y_test_trans_pred[0],fitted_lambda4)

# test RMSE

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

test_rmse = rmse(np.array(Y_test_inv[:,0]), np.array(Y_test_pred_inv[:,0]))


print('TEST RMSE: %f' %test_rmse)

#train correlation coefficient

test_corr,_=np.corrcoef(np.squeeze(np.array(Y_test_inv[:,0])), np.squeeze(np.array(Y_test_pred_inv[:,0])))

test_corr,_
test_corr=test_corr[1]

print('TEST CORRELATION: %f' %test_corr)


#test mean absolute error

test_mae=metrics.mean_absolute_error(Y_test_inv[:,0],Y_test_pred_inv[:,0])

print('Test MAE: %f' %test_mae)


# #  Test PAI Plotting

fig2 = plt.figure()
ax1 = fig2.add_subplot()
plt.plot(Y_test_inv,Y_test_pred_inv, 'go',markersize=10,marker=".",color='g')
plt.xlim([0,10])
plt.ylim([0, 10])
plt.xlabel("Insitu PAI ($m^2m^{-2}$)")
plt.ylabel("Estimated PAI ($m^2m^{-2}$)")
plt.plot([0, 10], [0, 10], 'k:')
plt.annotate('r = %.3f'%test_corr, xy=(0.5, 9.2))#round off upto 3decimals
plt.annotate('RMSE = %.3f'%test_rmse, xy=(0.5, 8.7))
plt.annotate('MAE = %.3f'%test_mae, xy=(0.5,8.2))
plt.annotate('TEST',xy=(7,0.4))
plt.annotate('HH+HV',xy=(7,1))
ax1.set_aspect('equal', adjustable='box')
plt.show()

# Save the predicted outputs 

# INS1_column_values=["INSITU PAI"]

# INS1=pd.DataFrame(data=Y_train_inv[:,0],columns=INS1_column_values)

# PRED1_column_values=["PREDICTED PAI"]

# PRED1=pd.DataFrame(data=Y_train_pred_inv[:,0],columns=PRED1_column_values)

# INS1_PRED1=pd.concat([INS1,PRED1],axis=1)

# INS1_PRED1.to_excel('INS_HH_HV_PRED_TRAIN.xlsx', index=False)


# INS2_column_values=["INSITU PAI"]

# INS2=pd.DataFrame(data=Y_test_inv[:,0],columns=INS2_column_values)

# PRED2_column_values=["PREDICTED PAI"]

# PRED2=pd.DataFrame(data=Y_test_pred_inv[:,0],columns=PRED2_column_values)

# INS2_PRED2=pd.concat([INS2,PRED2],axis=1)

# INS2_PRED2.to_excel('INS_HH_HV_PRED_TEST.xlsx', index=False)



############################################# HV-VV COMBINATION ###########################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import GPy
from IPython.display import display
from scipy.special import boxcox, inv_boxcox
from sklearn import metrics

#####################################################TRAINING DATASET ###########################################

data_TRAIN=pd.read_excel("insert your training dataset file here as .xlsx")

# reading the features 

HV_TRAIN=data_TRAIN["HV"]
VV_TRAIN=data_TRAIN["VV"]

# reading the target parameters

PAI_TRAIN=data_TRAIN["PAI"]


# reading the features and target parameters in one dataframe to check skewness

X1_train=pd.concat([HV_TRAIN,VV_TRAIN,PAI_TRAIN],axis=1)


# SEPRATING FEATURES AND TARGET PARAMETERS BEFORE TRANSFORMATION

X_train=pd.concat([HV_TRAIN,VV_TRAIN],axis=1)

Y_train=X1_train["PAI"]
Y_train=np.array(Y_train).reshape(-1,1);

# BOXCOX TRANSFORMATION

# transform training data & save lambda value

HV_train,fitted_lambda2 = stats.boxcox(X_train["HV"])
VV_train,fitted_lambda3 =stats.boxcox(X_train["VV"])
PAI_train,fitted_lambda4 =stats.boxcox(Y_train[:,0])

print(fitted_lambda2)
print(fitted_lambda3)
print(fitted_lambda4)

X_train_trans=np.column_stack((HV_train,VV_train))

Y_train_trans=PAI_train
Y_train_trans=np.array(Y_train_trans).reshape(-1,1);


####################################### GPY MODEL #############################################

# RBF kernel applied 


k1= GPy.kern.Linear(input_dim=2,ARD=False,name='linear')

k2 = GPy.kern.RBF(input_dim=2, lengthscale=1,ARD=False)


kernel=k2+k1

# Gaussian process regression model

m = GPy.models.GPRegression(X_train_trans,Y_train_trans,kernel)

display(m)

# optimize of hyperparameters and plot

m.optimize(optimizer='scg',messages=True,max_iters=1000)


print(m)

########################## Predicting for the train set ########################################

X_train_trans=np.array(X_train_trans)

Y_train_pred=m.predict(X_train_trans)

Y_train_inv= inv_boxcox(Y_train_trans,fitted_lambda4)

Y_train_pred_inv= inv_boxcox(Y_train_pred[0],fitted_lambda4)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

train_rmse = rmse(np.array(Y_train_inv[:,0]), np.array(Y_train_pred_inv[:,0]))

print('TRAIN RMSE: %f' %train_rmse)

#train correlation coefficient

train_corr,_=np.corrcoef(np.squeeze(np.array(Y_train_inv[:,0])), np.squeeze(np.array(Y_train_pred_inv[:,0])))

train_corr,_
train_corr=train_corr[1]

print('TRAIN CORRELATION: %f' %train_corr)

#train mean absolute error

train_mae=metrics.mean_absolute_error(Y_train_inv[:,0],Y_train_pred_inv[:,0])

print('Train MAE: %f' %train_mae)

# # Train LAI Plotting
# depict illustartion

fig1 = plt.figure()
ax1 = fig1.add_subplot()
plt.plot(Y_train_inv,Y_train_pred_inv, 'go',markersize=10,marker=".",color='g')
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.xlabel("Insitu PAI ($m^2m^{-2}$)")
plt.ylabel("Estimated PAI ($m^2m^{-2}$)")
plt.plot([0, 10], [0, 10], 'k:')
plt.annotate('r = %.3f'%train_corr, xy=(0.5, 9.2))           #round off upto 3decimals
plt.annotate('RMSE = %.3f'%train_rmse, xy=(0.5, 8.7))
plt.annotate('MAE = %.3f'%train_mae, xy=(0.5, 8.2))
plt.annotate('TRAIN',xy=(7,0.4))
plt.annotate('HV+VV',xy=(7,1))

ax1.set_aspect('equal', adjustable='box')
plt.show()


######################################### TEST DATASET ###############################################

data_TEST=pd.read_excel("insert your test dataset file here as .xlsx")

# reading the features 

HV_TEST=data_TEST["HV"]
VV_TEST=data_TEST["VV"]

# reading the target parameters

PAI_TEST=data_TEST["PAI"]

# SEPRATING FEATURES AND TARGET PARAMETERS BEFORE TRANSFORMATION

X_test=pd.concat([HV_TEST,VV_TEST],axis=1)

Y_test=PAI_TEST
Y_test=np.array(Y_test).reshape(-1,1);

# use lambda value to transform test data

HV_test= stats.boxcox(X_test["HV"], fitted_lambda2)
VV_test= stats.boxcox(X_test["VV"], fitted_lambda3)
PAI_test= stats.boxcox(Y_test[:,0], fitted_lambda4)

X_test_trans=np.column_stack((HV_test,VV_test))

Y_test_trans=PAI_test
Y_test_trans=np.array(Y_test_trans).reshape(-1,1);

# # ######################### Predicting for the test set ###########################################

X_test_trans=np.array(X_test_trans)

Y_test_trans_pred=m.predict(X_test_trans)

Y_test_inv= inv_boxcox(Y_test_trans,fitted_lambda4)

Y_test_pred_inv= inv_boxcox(Y_test_trans_pred[0],fitted_lambda4)

# test RMSE

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

test_rmse = rmse(np.array(Y_test_inv[:,0]), np.array(Y_test_pred_inv[:,0]))


print('TEST RMSE: %f' %test_rmse)

#train correlation coefficient

test_corr,_=np.corrcoef(np.squeeze(np.array(Y_test_inv[:,0])), np.squeeze(np.array(Y_test_pred_inv[:,0])))

test_corr,_
test_corr=test_corr[1]

print('TEST CORRELATION: %f' %test_corr)


#test mean absolute error

test_mae=metrics.mean_absolute_error(Y_test_inv[:,0],Y_test_pred_inv[:,0])

print('Test MAE: %f' %test_mae)


# #  Test PAI Plotting

fig2 = plt.figure()
ax1 = fig2.add_subplot()
plt.plot(Y_test_inv,Y_test_pred_inv, 'go',markersize=10,marker=".",color='g')
plt.xlim([0,10])
plt.ylim([0, 10])
plt.xlabel("Insitu PAI ($m^2m^{-2}$)")
plt.ylabel("Estimated PAI ($m^2m^{-2}$)")
plt.plot([0, 10], [0, 10], 'k:')
plt.annotate('r = %.3f'%test_corr, xy=(0.5, 9.2))#round off upto 3decimals
plt.annotate('RMSE = %.3f'%test_rmse, xy=(0.5, 8.7))
plt.annotate('MAE = %.3f'%test_mae, xy=(0.5,8.2))
plt.annotate('TEST',xy=(7,0.4))
plt.annotate('HV+VV',xy=(7,1))
ax1.set_aspect('equal', adjustable='box')
plt.show()

# Save the predicted outputs 


# INS1_column_values=["INSITU PAI"]

# INS1=pd.DataFrame(data=Y_train_inv[:,0],columns=INS1_column_values)

# PRED1_column_values=["PREDICTED PAI"]

# PRED1=pd.DataFrame(data=Y_train_pred_inv[:,0],columns=PRED1_column_values)

# INS1_PRED1=pd.concat([INS1,PRED1],axis=1)

# INS1_PRED1.to_excel('INS_HV_VV_PRED_TRAIN.xlsx', index=False)


# INS2_column_values=["INSITU PAI"]

# INS2=pd.DataFrame(data=Y_test_inv[:,0],columns=INS2_column_values)

# PRED2_column_values=["PREDICTED PAI"]

# PRED2=pd.DataFrame(data=Y_test_pred_inv[:,0],columns=PRED2_column_values)

# INS2_PRED2=pd.concat([INS2,PRED2],axis=1)

# INS2_PRED2.to_excel('INS_HV_VV_PRED_TEST.xlsx', index=False)


####################################### HH-VV COMBINATION ###################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import GPy
from IPython.display import display
from scipy.special import boxcox, inv_boxcox
from sklearn import metrics

################################# TRAINING DATASET #################################################

data_TRAIN=pd.read_excel("insert your training dataset file here as .xlsx")

# reading the features 

HH_TRAIN=data_TRAIN["HH"]
VV_TRAIN=data_TRAIN["VV"]

# reading the target parameters

PAI_TRAIN=data_TRAIN["PAI"]


# reading the features and target parameters in one dataframe to check skewness

X1_train=pd.concat([HH_TRAIN,VV_TRAIN,PAI_TRAIN],axis=1)


# SEPRATING FEATURES AND TARGET PARAMETERS BEFORE TRANSFORMATION

X_train=pd.concat([HH_TRAIN,VV_TRAIN],axis=1)

Y_train=X1_train["PAI"]
Y_train=np.array(Y_train).reshape(-1,1);

# BOXCOX TRANSFORMATION

# transform training data & save lambda value

HH_train,fitted_lambda1 = stats.boxcox(X_train["HH"])
VV_train,fitted_lambda3 =stats.boxcox(X_train["VV"])
PAI_train,fitted_lambda4 =stats.boxcox(Y_train[:,0])

print(fitted_lambda1)
print(fitted_lambda3)
print(fitted_lambda4)


X_train_trans=np.column_stack((HH_train,VV_train))

Y_train_trans=PAI_train
Y_train_trans=np.array(Y_train_trans).reshape(-1,1);


###################################### GPY MODEL #############################################

# RBF kernel applied 


k1= GPy.kern.Linear(input_dim=2,ARD=False,name='linear')

k2 = GPy.kern.RBF(input_dim=2, lengthscale=1,ARD=False)


kernel=k2+k1

# Gaussian process regression model

m = GPy.models.GPRegression(X_train_trans,Y_train_trans,kernel)

display(m)

# optimize of hyperparameters and plot

m.optimize(optimizer='scg',messages=True,max_iters=1000)


print(m)

######################### Predicting for the train set ########################################

X_train_trans=np.array(X_train_trans)

Y_train_pred=m.predict(X_train_trans)

Y_train_inv= inv_boxcox(Y_train_trans,fitted_lambda4)

Y_train_pred_inv= inv_boxcox(Y_train_pred[0],fitted_lambda4)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

train_rmse = rmse(np.array(Y_train_inv[:,0]), np.array(Y_train_pred_inv[:,0]))

print('TRAIN RMSE: %f' %train_rmse)

#train correlation coefficient

train_corr,_=np.corrcoef(np.squeeze(np.array(Y_train_inv[:,0])), np.squeeze(np.array(Y_train_pred_inv[:,0])))

train_corr,_
train_corr=train_corr[1]

print('TRAIN CORRELATION: %f' %train_corr)

#train mean absolute error

train_mae=metrics.mean_absolute_error(Y_train_inv[:,0],Y_train_pred_inv[:,0])

print('Train MAE: %f' %train_mae)

# # Train PAI Plotting
# depict illustartion

fig1 = plt.figure()
ax1 = fig1.add_subplot()
plt.plot(Y_train_inv,Y_train_pred_inv, 'go',markersize=10,marker=".",color='g')
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.xlabel("Insitu PAI ($m^2m^{-2}$)")
plt.ylabel("Estimated PAI ($m^2m^{-2}$)")
plt.plot([0, 10], [0, 10], 'k:')
plt.annotate('r = %.3f'%train_corr, xy=(0.5, 9.2))           #round off upto 3decimals
plt.annotate('RMSE = %.3f'%train_rmse, xy=(0.5, 8.7))
plt.annotate('MAE = %.3f'%train_mae, xy=(0.5, 8.2))
plt.annotate('TRAIN',xy=(7,0.4))
plt.annotate('HH+VV',xy=(7,1))

ax1.set_aspect('equal', adjustable='box')
plt.show()


######################################### TEST DATASET ###############################################

data_TEST=pd.read_excel("insert your test dataset file here as .xlsx")

# reading the features 

HH_TEST=data_TEST["HH"]
#HV_TEST=data_TEST["HV"]
VV_TEST=data_TEST["VV"]

# reading the target parameters

PAI_TEST=data_TEST["PAI"]

# SEPRATING FEATURES AND TARGET PARAMETERS BEFORE TRANSFORMATION

#X_test=pd.concat([HH_TEST,HV_TEST],axis=1)
#X_test=pd.concat([HV_TEST,VV_TEST],axis=1)
X_test=pd.concat([HH_TEST,VV_TEST],axis=1)

Y_test=PAI_TEST
Y_test=np.array(Y_test).reshape(-1,1);

# use lambda value to transform test data

HH_test= stats.boxcox(X_test["HH"], fitted_lambda1)
#HV_test= stats.boxcox(X_test["HV"], fitted_lambda2)
VV_test= stats.boxcox(X_test["VV"], fitted_lambda3)
PAI_test= stats.boxcox(Y_test[:,0], fitted_lambda4)

#X_test_trans=np.column_stack((HH_test,HV_test))
#X_test_trans=np.column_stack((HV_test,VV_test))
X_test_trans=np.column_stack((HH_test,VV_test))

Y_test_trans=PAI_test
Y_test_trans=np.array(Y_test_trans).reshape(-1,1);

# ######################### Predicting for the test set ###########################################

X_test_trans=np.array(X_test_trans)

Y_test_trans_pred=m.predict(X_test_trans)

Y_test_inv= inv_boxcox(Y_test_trans,fitted_lambda4)

Y_test_pred_inv= inv_boxcox(Y_test_trans_pred[0],fitted_lambda4)

# test RMSE

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

test_rmse = rmse(np.array(Y_test_inv[:,0]), np.array(Y_test_pred_inv[:,0]))


print('TEST RMSE: %f' %test_rmse)

#train correlation coefficient

test_corr,_=np.corrcoef(np.squeeze(np.array(Y_test_inv[:,0])), np.squeeze(np.array(Y_test_pred_inv[:,0])))

test_corr,_
test_corr=test_corr[1]

print('TEST CORRELATION: %f' %test_corr)


#test mean absolute error

test_mae=metrics.mean_absolute_error(Y_test_inv[:,0],Y_test_pred_inv[:,0])

print('Test MAE: %f' %test_mae)


#  Test PAI Plotting

fig2 = plt.figure()
ax1 = fig2.add_subplot()
plt.plot(Y_test_inv,Y_test_pred_inv, 'go',markersize=10,marker=".",color='g')
plt.xlim([0,10])
plt.ylim([0, 10])
plt.xlabel("Insitu PAI ($m^2m^{-2}$)")
plt.ylabel("Estimated PAI ($m^2m^{-2}$)")
plt.plot([0, 10], [0, 10], 'k:')
plt.annotate('r = %.3f'%test_corr, xy=(0.5, 9.2))#round off upto 3decimals
plt.annotate('RMSE = %.3f'%test_rmse, xy=(0.5, 8.7))
plt.annotate('MAE = %.3f'%test_mae, xy=(0.5,8.2))
plt.annotate('TEST',xy=(7,0.4))
plt.annotate('HH+VV',xy=(7,1))
ax1.set_aspect('equal', adjustable='box')
plt.show()

#   save the predicted results


# # INS1_column_values=["INSITU PAI"]

# # INS1=pd.DataFrame(data=Y_train_inv[:,0],columns=INS1_column_values)

# # PRED1_column_values=["PREDICTED PAI"]

# # PRED1=pd.DataFrame(data=Y_train_pred_inv[:,0],columns=PRED1_column_values)

# # INS1_PRED1=pd.concat([INS1,PRED1],axis=1)

# # INS1_PRED1.to_excel('INS_HH_VV_PRED_TRAIN.xlsx', index=False)


# # INS2_column_values=["INSITU PAI"]

# # INS2=pd.DataFrame(data=Y_test_inv[:,0],columns=INS2_column_values)

# # PRED2_column_values=["PREDICTED PAI"]

# # PRED2=pd.DataFrame(data=Y_test_pred_inv[:,0],columns=PRED2_column_values)

# # INS2_PRED2=pd.concat([INS2,PRED2],axis=1)

# # INS2_PRED2.to_excel('INS_HH_VV_PRED_TEST.xlsx', index=False)







