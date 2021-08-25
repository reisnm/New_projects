#!/usr/bin/env python
# coding: utf-8

# ## Clustering script

# In[1]:


import pandas as pd
import numpy as np
import numpy as np; np.random.seed(0)
import scipy
import scipy.stats as stats
import datetime
import sklearn
from sklearn.preprocessing import StandardScaler


# In[2]:


eb2 = pd.read_csv('filepath')


# In[3]:


eb2.groupby (by ='customer_email').increment_id.count()
eb2['date'] = pd.to_datetime(eb2['date'])
eb2['LAST_TRANSACTION'] = pd.to_datetime(eb2['LAST_TRANSACTION'])


# In[4]:


#creating customer table
customer_table = pd.DataFrame()
customer_table['total_revenue'] = eb2.groupby(by='customer_email')['base_grand_total'].sum()
customer_table['average_transaction_value'] = eb2.groupby(by='customer_email')['base_grand_total'].mean()
customer_table['number_of_transactions'] = eb2.groupby(by='customer_email')['increment_id'].count()
customer_table['country'] = eb2.groupby(by='customer_email')['country'].agg(lambda x: stats.mode(x)[0][0])
customer_table['city'] = eb2.groupby(by='customer_email')['city'].agg(lambda x: stats.mode(x)[0][0])
customer_table['first_transaction'] = eb2.groupby(by='customer_email')['date'].min()
customer_table['last_transaction'] = eb2.groupby(by='customer_email')['date'].max()
customer_table['TYPE_USER'] = eb2.groupby(by='customer_email')['TYPE_USER'].agg(lambda x: stats.mode(x)[0][0])


# In[5]:


customer_table['first_transaction'].dt.month
customer_table ['first_transaction'].apply(lambda x : pd.datetime.now().date() - x.date())
customer_table['frequency'] = (customer_table ['first_transaction'].apply(lambda x : pd.datetime.now().date() - x.date())).dt.days/ customer_table['number_of_transactions']
customer_table['recency'] = customer_table ['last_transaction'].apply(lambda x : pd.datetime.now().date() - x.date())


# In[6]:


#defining RFM variables


# In[7]:


def ATV_score_function(df):
    
    if df['average_transaction_value'] <= 0.37:
        return 1
    elif df['average_transaction_value'] <=0.93:
        return 2
    elif df['average_transaction_value'] <=1.2:
        return 3
    elif df['average_transaction_value'] <=1.47:
        return 4
    else:
        return 5    
customer_table ['ATV_score'] = customer_table.apply(ATV_score_function, axis=1)


# In[8]:


def recency_score_function(df):
    
    if df['recency'].days >= 329:
        return 1
    elif df['recency'].days >=222:
        return 2
    elif df['recency'].days >=124:
        return 3
    elif df['recency'].days >=45:
        return 4
    else:
        return 5
customer_table ['recency_score'] = customer_table.apply(recency_score_function, axis=1)


# In[9]:


def number_of_transactions_score_function(df):
    
    if df['number_of_transactions'] >=12:
        return 5
    elif df['number_of_transactions'] >=6:
        return 4
    elif df['number_of_transactions'] >=4:
        return 3
    elif df['number_of_transactions'] >=2:
        return 2
    else:
        return 1
customer_table ['transaction_score'] = customer_table.apply(number_of_transactions_score_function, axis=1)


# In[10]:


def RFM_score_function(df):
    
    if df['ATV_score'] + df['recency_score'] + df['transaction_score'] >= 13: 
        return 5
    elif df['ATV_score'] + df['recency_score'] + df['transaction_score'] >= 10:
        return 4
    elif df['ATV_score'] + df['recency_score'] + df['transaction_score'] >= 7: 
        return 3
    elif df['ATV_score'] + df['recency_score'] + df['transaction_score'] >= 4: 
        return 2
    else:
        return 1
customer_table ['RFM'] = customer_table.apply(RFM_score_function, axis=1)


# In[11]:


customer_table ['customer_email'] = customer_table.index
customer_table = customer_table.reset_index(drop=True)
customer_table['cluster'] = customer_table['RFM']
customer_table['cluster'] = customer_table['cluster'].astype(str)
customer_table['cluster'] = customer_table['cluster'].replace({ '5': 'High value', '4': 'Promising', '3': 'Need attention', '2': 'Hibernating', '1':'Lost'})


# In[12]:


#saving customer table into CSV
current_date = datetime.datetime.now()
filename = 'customer_scores '+str(current_date.day)+'.'+str(current_date.month)+'.'+str(current_date.year)
customer_table.to_csv(filename + '.csv')

