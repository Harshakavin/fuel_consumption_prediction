# @Harshakavin

from sklearn import linear_model
from sklearn import svm
import matplotlib.pyplot as pyplot
# import pandas as pd
# import pylab as pl
# import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from io import StringIO
import requests

''' NOTE
@ Real Data Url https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64
@ csv name (Original Fuel Consumption Ratings 2000-2014)
@ I modified some column names therefor I uploaded to my git repository
'''

#Get the Csv from my Git Url
url="https://raw.githubusercontent.com/Harshakavin/fuel_consumption_prediction/master/fuel_consumption.csv"
s=requests.get(url).text
df=pd.read_csv(StringIO(s),encoding='utf-8');
print(df)
# df=pd.read_csv('https://github.com/Harshakavin/fuel_consumption_prediction/blob/master/fuel_consumption.csv',encoding='utf-8')

#OR

# Get fuesl2 on this project file
# df = pd.read_csv("fuesl2.csv",encoding='utf-8')




# New dataframe contains selected columns from old dataframe
cdf = df[['ENGINESIZE', 'CYLINDERS', 'CO2EMISSIONS','TRANSMISSION','FUEL_TYPE','FUELCON_HWY','FUELCON_CITY','CON_MPG']]





# Columns names with TRANSMISSION and FUEL_TYPE DataFrame encodeding. Both Columns are contain strings names.
# rather than give a 0 or 1 , use one hot encoding in pandas library is better
cdf = pd.concat([pd.get_dummies(cdf, columns=['TRANSMISSION','FUEL_TYPE'])], axis=1)




# set median to nan field columns
cdf['ENGINESIZE']=cdf['ENGINESIZE'].fillna(cdf['ENGINESIZE'].median())
cdf['CYLINDERS']=cdf['CYLINDERS'].fillna(cdf['CYLINDERS'].median())




# set X and Y
X =cdf.drop(['CO2EMISSIONS'],axis=1)
Y=cdf['CO2EMISSIONS'].fillna(cdf['CO2EMISSIONS'].median())




# to check whether the data preprocessing working correct
print(X)
print(Y)



# Split data set to 20% to test and others data to train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


# train using LinearRegression
reg = linear_model.LinearRegression()



# train using Support Vector Regression
# reg=svm.SVR(kernel='rbf',gamma=0.001,C=1e3)


# Fit the data to the regression
reg.fit(X_train, Y_train)

#  #The Coefficients
#
# #print('Coefficients : ', svr_reg.coef_)




# Build a trained model using pickle
with open('fuel.pickle','wb') as f:
    pickle.dump(reg, f)



# Use the created pickle model for the prediction
pick_in=open('fuel.pickle','rb')
pickle_model =pickle.load(pick_in)



# To check view the ten CO2 EMISSIONS test data
print(X_test[0:10])
print(Y_test[0:10])


# Train Accuracy
print('Train Accuracy : %.2f' % reg.score(X_train, Y_train))



# Predict the test data
predict=pickle_model.predict(X_test)



# View the predicted CO2 EMISSIONS values
print(predict)
print(type(predict))



# get the trained accracy
accracy=pickle_model.score(X_test, Y_test)
print("Test Accuracy:",accracy)


# get predicted CO2 EMISSIONS for given data set
train_y_ = pickle_model.predict(X_test)



# compare CO2 EMISSIONS  with fuel consumption (Mile Per Gallon), limited to 100 rows

pyplot.scatter(X_test['CON_MPG'][0:500], Y_test[0:500], color='black', linestyle='-', label='dist (m)', linewidth=5)
pyplot.scatter(X_test['CON_MPG'][0:500] ,train_y_[0:500],color='yellow',linestyle='-', label='dist (m)', linewidth=5)
pyplot.ylabel("CO2 EMISSIONS")
pyplot.xlabel("Fuel consumption (Mile Per Gallon)")
pyplot.show()



# compare CO2 EMISSIONS  with ENGINE SIZE ,limited to 100 rows

pyplot.scatter(X_test['ENGINESIZE'][0:100], Y_test[0:100], color='black')
pyplot.scatter(X_test['ENGINESIZE'][0:100] ,train_y_[0:100],color='yellow', linestyle='dashed')
pyplot.ylabel("CO2 EMISSIONS")
pyplot.xlabel("ENGINE SIZE")
pyplot.show()




