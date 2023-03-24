#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opendatasets')


# In[2]:


import pandas as pd
import numpy as np
import opendatasets as od


# In[3]:


dataset = pd.read_csv("apartment-prices/rent_apartments.csv")


# In[4]:


dataset.head()


# In[5]:


dataset.shape


# In[6]:


dataset.info()


# In[7]:


#returns object containing counts of unique values
for column in dataset.columns:
    print(dataset[column].value_counts())
    print("*"*20)


# In[8]:


dataset.isna().sum()


# In[9]:


dataset.drop(columns=["Agency","link"],inplace=True)


# In[10]:


dataset.describe()


# In[11]:


dataset.info()


# In[12]:


dataset['Bathrooms'].value_counts()


# In[13]:


dataset["Bathrooms"]=dataset["Bathrooms"].fillna(dataset["Bathrooms"].median())


# In[14]:


dataset["Bedrooms"]=dataset["Bedrooms"].fillna(dataset["Bedrooms"].median())


# In[15]:


dataset["total_sqft"]=dataset["total_sqft"].fillna(dataset["total_sqft"].median())


# In[16]:


dataset.info()


# In[17]:


dataset["total_sqft"].unique()


# In[18]:


#drop ksh in price and the comma
dataset['Price'] = dataset['Price'].str.replace('KSh','',regex=True).str.replace(',','').astype(float)


# In[19]:


dataset.describe()


# In[20]:


dataset['location'].value_counts()


# In[21]:


dataset['locaton']=dataset['location'].apply(lambda x: x.strip())
location_count=dataset['location'].value_counts()


# In[22]:


location_count


# In[23]:


location_count_less_10=location_count[location_count<=10]
location_count_less_10


# In[24]:


dataset['location']=dataset['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)


# In[25]:


dataset['location'].value_counts()


# In[26]:


dataset.describe()


# In[27]:


dataset.shape


# In[28]:


dataset


# In[29]:


dataset.drop(columns=['locaton'],inplace=True)


# CLEANED DATA

# In[30]:


dataset.head()


# In[31]:


dataset.to_csv('Cleaned_data.csv')


# In[32]:


X=dataset.drop(columns=['Price'])
y=dataset['Price']


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[34]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 2)


# In[35]:


print(X_train.shape)
print(X_test.shape)


# APPLYING LINEAR REGRESSION

# In[36]:


column_trans=make_column_transformer((OneHotEncoder(sparse=False),["location"]),remainder='passthrough')


# In[37]:


scaler=StandardScaler()


# In[38]:


lr=LinearRegression(normalize=True)


# In[39]:


pipe=make_pipeline(column_trans,scaler,lr)


# In[40]:


pipe.fit(X_train,y_train)


# In[41]:


y_pred = pipe.predict(X_test)


# APPlYING LASSO

# In[42]:


lasso=Lasso()


# In[43]:


pipe=make_pipeline(column_trans,scaler, lasso)


# In[44]:


pipe.fit(X_train, y_train)


# In[45]:


y_pred_lasso=pipe.predict(X_test)
r2_score(y_test,y_pred_lasso)


# In[46]:


ridge=Ridge()


# In[47]:


pipe=make_pipeline(column_trans,scaler,ridge)


# In[48]:


pipe.fit(X_train, y_train)


# In[49]:


y_pred_ridge=pipe.predict(X_test)
r2_score(y_test,y_pred_ridge)


# In[50]:


import pickle


# In[51]:


print('No Regularization: ',r2_score(y_test,y_pred))
print('Lasso: ',r2_score(y_test,y_pred_lasso))
print('Ridge: ',r2_score(y_test,y_pred_ridge))


# In[52]:


pickle.dump(pipe, open('RegressorModel.pkl','wb'))


# In[56]:


pipe.predict(pd.DataFrame([["Parklands, Westlands","3000","3","3"]],columns=['location','total_sqft','Bedrooms','Bathrooms']))


# In[54]:


model =pickle.load(open('RegressorModel.pkl', 'rb'))

