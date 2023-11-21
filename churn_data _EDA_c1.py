#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns
import matplotlib.ticker as mtick  


# In[2]:


pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[3]:


df=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.columns


# In[7]:


df.shape


# In[8]:


df.dtypes


# In[9]:


df.describe()


# In[10]:


df['Churn'].value_counts().plot(kind='barh', figsize=(8, 6))
plt.xlabel("Count")
plt.ylabel("Target Variable")
plt.title("Count of TARGET Variable per category", y=1.02)


# In[11]:


100*df['Churn'].value_counts()/len(df['Churn'])


# In[12]:


df['Churn'].value_counts()


# In[13]:


df.info


# In[14]:


df.info(verbose = True)


# In[15]:


missing = pd.DataFrame((df.isnull().sum())*100/df.shape[0]).reset_index()
plt.figure(figsize=(16,5))
ax = sns.pointplot('index',0,data=missing)
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()


# In[16]:


#no missing values


# In[17]:


## Data cleaning


# In[18]:


dfc=df.copy()


# In[19]:


dfc


# In[20]:


dfc.TotalCharges = pd.to_numeric(dfc.TotalCharges, errors='coerce')
dfc.isnull().sum()


# In[21]:


dfc.loc[dfc ['TotalCharges'].isnull() == True]


# In[22]:


print(dfc['tenure'].max())


# In[23]:


labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

dfc['tenure_group'] = pd.cut(dfc.tenure, range(1, 80, 12), right=False, labels=labels)


# In[24]:


dfc['tenure_group'].value_counts()


# In[25]:


# removing columns not requierd for processing


# In[26]:


dfc.drop(columns= ['customerID','tenure'], axis=1, inplace=True)
dfc.head()


# In[27]:


##Data Exploration


# In[28]:


###Univariate Analysis


# In[29]:


for i, predictor in enumerate(dfc.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=dfc, x=predictor, hue='Churn')


# In[30]:


dfc['Churn'] = np.where(dfc.Churn == 'Yes',1,0)


# In[31]:


dfc.head()


# In[32]:


dfc_dummies = pd.get_dummies(dfc)
dfc_dummies.head()


# In[34]:


sns.lmplot(data=dfc_dummies, x='MonthlyCharges', y='TotalCharges', fit_reg=False)


# In[35]:


Mth = sns.kdeplot(dfc_dummies.MonthlyCharges[(dfc_dummies["Churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(dfc_dummies.MonthlyCharges[(dfc_dummies["Churn"] == 1) ],
                ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')


# In[36]:


Tot = sns.kdeplot(dfc_dummies.TotalCharges[(dfc_dummies["Churn"] == 0) ],
                color="Red", shade = True)
Tot = sns.kdeplot(dfc_dummies.TotalCharges[(dfc_dummies["Churn"] == 1) ],
                ax =Tot, color="Blue", shade= True)
Tot.legend(["No Churn","Churn"],loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Total Charges')
Tot.set_title('Total charges by churn')


# In[37]:


plt.figure(figsize=(20,8))
dfc_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


# In[38]:


##Bivariate Analysis


# In[39]:


new_df1_target0=dfc.loc[dfc["Churn"]==0]
new_df1_target1=dfc.loc[dfc["Churn"]==1]


# In[41]:


def uniplot(df,col,title,hue =None):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='bright') 
        
    plt.show()


# In[42]:


uniplot(new_df1_target1,col='Partner',title='Distribution of Gender for Churned Customers',hue='gender')


# In[43]:


uniplot(new_df1_target0,col='Partner',title='Distribution of Gender for Non Churned Customers',hue='gender')


# In[44]:


uniplot(new_df1_target1,col='PaymentMethod',title='Distribution of PaymentMethod for Churned Customers',hue='gender')


# In[45]:


uniplot(new_df1_target1,col='Contract',title='Distribution of Contract for Churned Customers',hue='gender')


# In[46]:


uniplot(new_df1_target1,col='TechSupport',title='Distribution of TechSupport for Churned Customers',hue='gender')


# In[47]:


uniplot(new_df1_target1,col='SeniorCitizen',title='Distribution of SeniorCitizen for Churned Customers',hue='gender')


# In[48]:


dfc_dummies.to_csv('df_churn.csv')


# In[ ]:




