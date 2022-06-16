###############################################################################################
#                                                                                             #
#   This program will take the file 'in-vehicle-coupon-recommendation.csv' and convert it     #
#   to dataframe where it will be cleaned and modified to be run through a few machine        # 
#   learning algorithms.  The newly created cleaned dataframe will be exported as csv file    #
#   named 'coupon.csv'                                                                        #
#                                                                                             #
###############################################################################################

#import of modules
import pandas as pd
import numpy as np

df = pd.read_csv(r"in-vehicle-coupon-recommendation.csv") #reads in CSV file and converts to df

drops = df.drop(['car'], axis=1) #drops column 'car' from the datafame

dropped = drops.shape[0] - drops.dropna().shape[0] #calculates the amount of dropped NaN from dataframe
drops.dropna(inplace=True) #drops all rows that have NaN element in a column
print("Dropped %d entries" % dropped)  #print statement that shows how many NaN have been dropped

#for col in drops.columns:
#    print(col)
#    print(drops[col].unique())
#    print('\n')for col in drops.columns:
#    print(col)
#    print(drops[col].value_counts())
#    print('\n')


#counts how many times an element occurs in a dataframe and assignse it to a variable
occup_dict = drops['occupation'].value_counts() 

#maps the volums to the column 'occulation' from the value_counts() above
drops['occupation'] = drops['occupation'].map(occup_dict) 

#all value mapping numerical assignments for
visit_freq = {'never':0, 'less1':1, '1~3':2, '4~8':6, 'gt8':9}
dest = {'No Urgent Place':3,  'Work':1, 'Home':2}
passenger = {'Alone':4,  'Friend(s)':3,  'Kid(s)':1,  'Partner':2}
weather = {'Sunny':3,  'Rainy':2,  'Snowy':1}
coup = {'Coffee House':5, 'Bar':2, 'Carry out & Take away':3 ,'Restaurant(<20)':4, 'Restaurant(20-50)':1}
exp = {'2h':1, '1d':24}
status = {'Married partner':5, 'Single':4, 'Unmarried partner':3, 'Divorced':2, 'Widowed':1}
edu = {'Some High School':1, 'High School Graduate':2, 'Some college - no degree':3, 'Associates degree':4,
       'Bachelors degree':5,'Graduate degree (Masters or Doctorate)':6}
income = {'Less than $12500':1, '$12500 - $24999':2, '$25000 - $37499':3, '$37500 - $49999':4,
          '$50000 - $62499':5, '$62500 - $74999':6, '$75000 - $87499':7, '$87500 - $99999':8, '$100000 or More':9}
gend = {'Male':1, 'Female':2}
tiktok = {'7AM':7, '10AM':10, '2PM':14, '6PM':16,  '10PM':22}
age = {'below21':16, '21':21,  '26':26,  '31':31,  '36':36,  '41':41,  '46':46, '50plus':55 }
#occup = {'Unemployed':25, 'Student':21, 'Computer & Mathematical':23, 'Sales & Related':22,
#         'Education&Training&Library':21, 'Management':20, 'Office & Administrative Support':19,
#         'Arts Design Entertainment Sports & Media':18, 'Business & Financial':17, 'Retired':16,
#         'Food Preparation & Serving Related':15, 'Healthcare Support':14, 'Healthcare Practitioners & Technical':13,
#         'Community & Social Services':12, 'Legal':11, 'Transportation & Material Moving':10, 'Architecture & Engineering':9,
#         'Protective Service':8, 'Life Physical Social Science':7, 'Construction & Extraction':6, 'Personal Care & Service':5,
#        'Installation Maintenance & Repair':4, 'Production Occupations':3, 'Building & Grounds Cleaning & Maintenance':2,
#        'Farming Fishing & Forestry':1}


# In[12]:


#replacing all "string" columns with mapped values
drops['destination'].replace(dest, inplace=True)
drops['passanger'].replace(passenger, inplace=True)
drops['weather'].replace(weather, inplace=True)
drops['Bar'].replace(visit_freq, inplace=True)
drops['CoffeeHouse'].replace(visit_freq, inplace=True) 
drops['CarryAway'].replace(visit_freq, inplace=True)
drops['RestaurantLessThan20'].replace(visit_freq, inplace=True)
drops['Restaurant20To50'].replace(visit_freq, inplace=True)
drops['coupon'].replace(coup, inplace=True)
drops['expiration'].replace(exp, inplace=True)
drops['maritalStatus'].replace(status, inplace=True)
drops['education'].replace(edu, inplace=True)
drops['income'].replace(income, inplace=True)
drops['gender'].replace(gend, inplace=True)
drops['time'].replace(tiktok, inplace=True)
#drops['occupation'].replace(occup, inplace=True)
drops['age'].replace(age, inplace=True)


corr = drops.corr().abs() #calculates the correlation and takes absolute value for dataframe
corr *= np.tri(*corr.values.shape, k=-1).T #takes correlation matrix and changes the bottom diagonal to 0's
corr_unstack = corr.unstack() #this changes the layout of the dataframe
corr_unstack.sort_values(inplace=True,ascending=False) #sorts values of the correlation df and places it highest to lowest order

print('Correlation (Top 10)')
print(corr_unstack.head(10))  #print top 10 correlation
print('\n')

print('Correlation with Class (Top 10)')
print(corr_unstack.get(key='Y').head(10)) #print top 10 correlation wth class (coupon usage)
print('\n')

cov = drops.cov().abs()  #takes absolute value of covariance of df
cov *= np.tri(*cov.values.shape, k=-1).T #repalces bottom diagonal with 0's on covariance matrix
cov_unstack = cov.unstack() #unstacks the covariance df
cov_unstack.sort_values(inplace=True,ascending=False) #sorts it

print('Covariance (Top 10)')
print(cov_unstack.head(10)) #prints top 10 
print('\n')

print('Covariance with Class (Top 10)')
print(cov_unstack.loc['Y'].head(10)) #prints top 10
print('\n')

drops.reset_index(drop=True, inplace=True) #resets index
 
drops.to_csv("coupon.csv") #converts dataframe to csv file with new name

