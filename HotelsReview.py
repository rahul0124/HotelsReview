
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
df=pd.read_csv("f:/intern.csv")      # load the data into dataframe


# In[33]:


# check the shape of data
df.shape


# In[34]:


# convert categorical to numeric
import sklearn.preprocessing as pp
lb=pp.LabelBinarizer()
df.SwimmingPool=lb.fit_transform(df.SwimmingPool)
df.ExerciseRoom=lb.fit_transform(df.ExerciseRoom)
df.BasketballCourt=lb.fit_transform(df.BasketballCourt)
df.YogaClasses=lb.fit_transform(df.YogaClasses)
df.Club=lb.fit_transform(df.Club)
df.FreeWifi=lb.fit_transform(df.FreeWifi)


# In[35]:


# convert category variable into dummy variable
df1=pd.get_dummies(df.HotelStars,prefix="HotelStars")


# In[36]:


a=pd.concat([df,df1],axis=1)   # concatenation of dummy variables with the dataframe


# In[37]:


del a["HotelStars"]     # after concatenation delete dummy variable


# In[38]:


# to check out correlation b/w variables import a library i.e. statsmodels
import statsmodels.api as sm      
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sm.graphics.plot_corr(a.corr(),xnames=a.columns)


# In[39]:


# deletion of unnecessary varibles
del a["User country"]
del a["Period of stay"]
del a["Traveler type"]
del a["Hotel name"]
del a["User continent"]
del a["Review month"]
del a["Review weekday"]


# In[40]:


independent=a.columns
independent=independent.delete(3)
X=a[independent]
Y=a.Score


# In[41]:


# import outlier influence library
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
for i in range(len(independent)):
    myvif=[vif(X[independent].values,index) for index in range(len(independent))]
    print("VIF Values:",myvif)
    maxvif=max(myvif)
    myvif.index(maxvif)
    dindex=myvif.index(maxvif)
    print("Index",dindex,"Maxvif",maxvif,"Column",independent[dindex])
    if maxvif>10:
        independent=independent.delete(dindex)
print(independent)
X=a[independent]


# In[43]:


import statsmodels.api as sm
 # Eqn. -> Y=m1x1+m2x2+m3x3.............mnxn
# Y=Dependent,X=Independent,m=weights
model=sm.OLS(Y,X)

# train the model
# calculate weights using X and Y
model=model.fit()     # weights calculated here

# summary of model
model.summary()


# In[44]:


from pandas import DataFrame
df3=pd.DataFrame()
df3["Actual"]=Y
df3["Predicted"]=model.predict(X)
df3


# In[45]:


""" Deploy the model"""
# Take user input for house price predict
dict1={}
for column in X.columns:
    temp=int(input("Enter "+column+": "))
    dict1[column]=temp
dict1
    
# create a dataframe using dict1
user_input=pd.DataFrame(dict1,index=[0],columns=X.columns)
Score=model.predict(user_input)
print("Score of reviewer is: ",int(Score[0]))


# In[ ]:


""" Most relevant features are: """
-> NrReviews
-> NrHotelReviews
-> HelpfulVotes
-> BasketballCourt
-> YogaClasses
-> NrRooms
-> Memberyears
-> HotelStars_3
-> HotelStars_3,5
-> HotelStars_4
-> HotelStars_4

