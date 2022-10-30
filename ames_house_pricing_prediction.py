#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This is Ames Housing Project. The goal is to predict residential house price in Ames, Iowa.
# We have 79 explanatory variables to use as our predictor
# Load all necessary modules
import pandas as pd # pandas module
import numpy as np # numpy module
import matplotlib.pyplot as plt # matplotlib module
import seaborn as sns # seaborn module


# In[2]:


# load the data 
ameHouse = pd.read_csv("train.csv")
ameHouse_test = pd.read_csv("test.csv")


# In[3]:


ameHouse.head()


# In[606]:


ameHouse_test.head()


# In[5]:


# check for na value
null_ames = [col for col in ameHouse.columns if ameHouse[col].isnull().sum() > 0]
ameHouse[null_ames].isnull().sum().sort_values(ascending=False)/len(ameHouse)


# In[6]:


# There are 4 features with missing value ratio more than 50%. 
# PoolQC, MiscFeature, Alley, and Fence
# We need to decide what need to be done upon all of these features.
# PoolQC --> Pool Quality
# MiscFeature ---> Contains feature no covered in other features
# Alley ---> Type of alley access
# Fence ---> Fence quality
# FireplaceQu ---> Fireplace Quality


# These 5 high rate missing ratio features, are very rare quality to has among common residential house, so no wonder if they have high numbers of missing value. Because of this reasoning, we can just drop this features

# In[7]:


# Drop features with missing value more than 50%
ameHouse = ameHouse.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1)
ameHouse.shape # the ncol now become 76


# In[8]:


# Don't forget to drop the same features from test dataset
ameHouse_test = ameHouse_test.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'], axis=1)
ameHouse_test.shape


# Why do we don't drop 'LotFrontage'?
# The reason is because the missing ratio is relatively tolerable, and based on the documentation, there is chance that this feature usable for the model.
# 
# 'LotFrontage' definition:
# *Linear feet of street connected to property*

# In[9]:


# Data summary for numerical features
ameHouse.describe()


# In[10]:


# Exploring data distribuiton
ame_05_quan = ameHouse.quantile(0.05)
ame_25_quan = ameHouse.quantile(0.25)
ame_50_quan = ameHouse.quantile(0.50) # same as median
ame_75_quan = ameHouse.quantile(0.75)
ame_95_quan = ameHouse.quantile(0.95)
quantile_ame = pd.DataFrame({'ame_05':ame_05_quan,
                            'ame_25':ame_25_quan,
                            'ame_50':ame_50_quan,
                            'ame_75':ame_75_quan,
                            'ame_95':ame_95_quan})
quantile_ame


# In[11]:


# Let's try to visualize SalePrice distribution (Because it is our target value, there are must be something we can learn)
fig, axSales = plt.subplots(1,2,figsize=(10,4))
sns.histplot(ax=axSales[0],x='SalePrice',data=ameHouse).set_title("Histogram of SalePrice")
sns.boxplot(ax=axSales[1],y = 'SalePrice', data=ameHouse, color='red').set_title("Boxplot of SalePrice")
plt.suptitle("Distribution of SalePrice", fontweight = 'bold')
plt.subplots_adjust(wspace=0.4)
plt.show()


# In[12]:


# The plot  become clear indication of outliers existence in SalePrice
# We can make it more form by doing normality test using qq plot
import statsmodels.api as sm

sm.qqplot(ameHouse['SalePrice'],line='45')
plt.show()


# 'SalePrice' data cleary do not follow 45 degree line, which is a strong indication that they don't follow normal distribution.

# In[13]:


# All numerical features
ames_numeric=ameHouse.columns[(ameHouse.dtypes=='int64') | (ameHouse.dtypes=='float64')]


# ### Understanding The Features
# To get better sense about ames housing dataset, i decide to analyze all feature individually.
# #### Id 
# There are no depper meaning of it. Use it as index or identity of each observations.

# #### MSSubClass
# *Indentifies the type of dwelling involved in the sale*; basically, it tells us about the residential house type
# The value is consist of discrete number, range from 20 up to 190.
# Here some definition of each number :
# - 20==>1-STORY 1946 & NEWER ALL STYLES
# - 30==>1-STORY 1945 & OLDER
# - 40==>1-STORY W/FINISHED ATTIC ALL AGES
# - 45==>1-1/2 STORY - UNFINISHED ALL AGES
# - 50==>1-1/2 STORY FINISHED ALL AGES
# - 60==>2-STORY 1946 & NEWER
# - 70==>2-STORY 1945 & OLDER
# etc...
# 
# This feature is something that i called 'half' numerical feature, because it doesn't really represent any measurement, instead *a set of quantified quality*.

# In[14]:


# Distribution of MSSubClass
fig, axMSClass = plt.subplots(1,2, figsize=(10,4))
sns.histplot(ax=axMSClass[0],x='MSSubClass',data=ameHouse, bins=16).set_title('MSSubClass in Hist')
sns.countplot(ax=axMSClass[1],x='MSSubClass', data=ameHouse).set_title('MSSubClass in Barplot')
plt.show()


# The two plots above show that the MSSubClass feature is more worthy of being treated as a **categorical feature**

# In[15]:


# let's drop MSSubClass from ames_numeric
#ames_numeric = ames_numeric.drop('MSSubClass')
ames_numeric


# #### LotFrontage
# 
# I've discussed this feature a bit before. Here I rewrite the definition based on the data documentation
# *Linear feet of street connected to property*.
# Based on [gimme-shelter](https://www.gimme-shelter.com/frontage-50043/), Frontage is "the width of the lot, measured at front part of the lot". When it comes to real estate, bigger frontage means more land, and more land means capacity to bigger house. So,logically speaking, this feature is necessary to predict house selling price. 

# In[16]:


# For future simplicity, i create a function called numDEA() to return all statistic summary of the feature
def numDEA(col,  df):
    stat_summary = pd.DataFrame({
        'Mean' : round(df[col].mean(),2),
        'Median' : round(df[col].median(),2),
        'Mode': df[col].mode()[0],
        'std' : round(df[col].std(),2),
        'Min': df[col].min(),
        'Max': df[col].max(),
        'Range': df[col].max() - df[col].min(),
        '5%': df[col].quantile(0.05),
        '25%': df[col].quantile(0.25),
        '50%': df[col].quantile(0.50),
        '75%': df[col].quantile(0.75),
        '90%': df[col].quantile(0.9),
        'IQR' : df[col].quantile(0.75) - df.quantile(0.25),
        'Count' : df[col].count(),
        'Unique': df[col].nunique(),
        'Missing Value' : df[col].isnull().sum()  
    }, index = [col])
    return stat_summary


# In[17]:


# LotFrontage's Statistical Summary

numDEA('LotFrontage',df= ameHouse)


# From central tendency, we can deduce that LotFrontage feature is **right-skewed data**, which indicate that most of the residential house in Ames, Iowa has frontage around 69 feet (21,03 meter), more or less. There are huge difference between 90% of house frontage and maximum frontage (313 - 96 = 217), which show us the existence of small group of "elite" house with wide frontage.

# In[18]:


# Distribution of LotFrontage
sns.histplot(x='LotFrontage',data=ameHouse).set_title('LotFrontage in Hist')
plt.show()


# In[19]:


sns.relplot(x='LotFrontage',y='SalePrice',data=ameHouse)
plt.title("SalePrice and LotFrontage")
plt.show()


# The scatterplot above gives us signs of a positive correlation between LotFrontage and SalePrice, albeit a bit of an anomaly. This anomaly give some indication of other factors influences SalePrice that i will investigate further.

# ### LotArea
# 
# From data documentation, 'LotArea' is Lot size in square feet.

# In[20]:


# Statistical Summary of LotArea
numDEA('LotArea',df= ameHouse)


# In[21]:


# LotArea Visualization
fig_LotArea, axLotArea = plt.subplots(1,2, figsize=(15,5))
sns.histplot(ax=axLotArea[0], x= 'LotArea',data=ameHouse).set_title("LotArea Distribution")
sns.scatterplot(ax=axLotArea[1],x='LotArea',y='SalePrice',data=ameHouse).set_title("SalePrice vs LotAreab")
plt.suptitle("LotArea Distribution and Relationship", fontweight="bold")
plt.show()


# Same as LotFrontage, LotArea is Right-Skewed. Majority of house in Ames, Iowa is small to medium house, while there are small fraction of house that has lotArea from 14.000 feet square (4.000 m2) up to 215.000 feet square (65.532 m2). 
# Because of the scale, scatterplot can not depict the relationship of LotArea and SalePrice really well. To fix it, i transform these two features with natural log. After transformation, the pattern start to show.

# In[22]:


viz_only_lotArea = pd.DataFrame({
    'nl_SalePrice':np.log(ameHouse['SalePrice']),
    'nl_LotArea':np.log(ameHouse['LotArea'])
})

sns.scatterplot(x='nl_LotArea',y='nl_SalePrice',data=viz_only_lotArea).set_title("Natural Log SalePrice vs Natural Log LotArea")
plt.show()


# #### OverallQual
# Based on data documentation,  'OverallQual' is Rates the overall material and finish of the house.
# The values are representation of a quality measure, such as following:
# 10 --- >  Very Excellent
# 9 --->  Excellent
# 8 --->  Very Good
# 7 --->	Good
# 6 --->	Above Average
# 5 --->	Average
# 4 --->	Below Average
# 3 --->	Fair
# 2 --->	Poor
# 1 --->	Very Poor
# 
# Because it has fixed set of values, we should treat 'OverallQual' as categorical variable, specifically, in *Ordinal* level measurement.

# In[23]:


# Drop OverallQual from ames_numeric
ames_numeric =ames_numeric.drop('OverallQual')
ames_numeric


# In[24]:


# OverallQual Exploration
# Frequency

countplot_overall = pd.DataFrame({
    'Count': ameHouse['OverallQual'].value_counts(),
    'Percent(%)':round((ameHouse['OverallQual'].value_counts()/len(ameHouse['OverallQual']))*100,2)
})
print("The mode of this feature is : {}".format(ameHouse['OverallQual'].mode()[0]))
print(countplot_overall)


# In[25]:


sns.countplot(x='OverallQual', data=ameHouse).set_title('OverallQual')
plt.show()


# Half of residential housing in Ames, Iowa (52,81%) has average to above average quality, with only 9,67% from total with below average to very poor quality housing. Based on this, i think it safe to deduce that Ames, Iowa is a good environment to live. 

# In[26]:


# SalePrice and OverallQual
sns.boxplot(x= 'OverallQual',y='SalePrice', data=ameHouse)
plt.title("OverallQual vs SalePrice")
plt.show()


# OverallQual able to divide SalePrice distribution rather clearly. It shows how each quality has distintive range of SalePrice, and it indicate that OverallQual is a good feature to predict SalePrice.

# #### OverallCond
# 
# From data documentation, OverallCond is Rates the overall condition of the house.
# More or less, it is similiar to OverallQual. 
# The set of values are :
# - 10 Very Excellent
# - 9	Excellent
# - 8	Very Good
# - 7	Good
# - 6	Above Average	
# - 5	Average
# - 4	Below Average	
# - 3	Fair
# - 2	Poor
# - 1	Very Poor

# In[27]:


# let's drop it from  ames_numeric
ames_numeric = ames_numeric.drop('OverallCond')
ames_numeric


# In[28]:


# OverallCond Data Exploration
allCond_count = pd.DataFrame({
    'Count': ameHouse['OverallCond'].value_counts(),
    'Percent(%)':round((ameHouse['OverallCond'].value_counts()/len(ameHouse['OverallCond'])*100),2)
})
print("The mode of this feature is : {}".format(ameHouse['OverallCond'].mode()[0]))
print(allCond_count)


# In[29]:


# OverallCond Visualization
figCond, axCond = plt.subplots(1,2, figsize=(15,5))
sns.countplot(ax=axCond[0],x='OverallCond',data=ameHouse).set_title("Frequency of OverallCond")
sns.boxplot(ax=axCond[1],x='OverallCond',y='SalePrice',data=ameHouse).set_title("OverallCond vs SalePrice")
plt.show()


# Different from OverallQual that able to divide SalePrice relatively well, OverallCond seems to have various relationship pattern to SalePrice. It shows by the boxplot, that house  with "average" condition has wide range of SalePrice. Moreover, there are a anomaly where "poor" condition house has higher price than house with better condition. This encourage an assumption that maybe these pattern caused by weak correlation between features, or there are feature with stronger effect involved.

# #### YearBuilt and YearRemodAdd
# 
# Based on data documentation, 
# YearBuilt is  *Original construction date*
# YearRemodAdd is *Remodel date (same as construction date if no remodeling or additions)*
# The values only specified in year, so nothing much we can do except to ensure there are not any null value, or miss entry

# In[30]:


print('''
Missing value : 
{}'''.format(ameHouse[['YearBuilt','YearRemodAdd']].isnull().sum()))


# #### MasVnrArea
# 
# Based on Data Documentation, MasVnrArea is Masonry veneer area in square feet
# Masonry veneer itself is single non-structural external layer of masonry, typically made of brick, stone or manufactured stone(source:[Wikipedia](https://en.wikipedia.org/wiki/Masonry_veneer))

# In[31]:


numDEA('MasVnrArea',df= ameHouse)


# In[32]:


# MasVnArea Visualization
figMasVn, axMasVn = plt.subplots(1,2, figsize=(15,5))
sns.histplot(ax=axMasVn[0], x= 'MasVnrArea',data=ameHouse).set_title("MasVnrArea Distribution")
sns.scatterplot(ax=axMasVn[1],x='MasVnrArea',y='SalePrice',data=ameHouse).set_title("SalePrice vs MasVnrArea")
plt.suptitle("MasVnrArea Distribution and Relationship", fontweight="bold")
plt.show()


# This data is a litle tricky. If we look upon it, almost all of house in Ames, Iowa has zero feet square of Masonry Venree. It is unclear whether they simply not use Masonry venree, or it is due to error in data entry

# #### BsmtFinSF1 and BsmtFinSF2
# 
# Based on data documentation, BsmtFinSF1 is Type 1 finished square feet. It is area of basement in certain type. BsmtFinSF2 is Type 2 finished square feet. These two features give us the same measurement, and the second only exist if a house has multiple type of basement. 

# In[33]:


print(numDEA('BsmtFinSF1',df=ameHouse))
print(numDEA('BsmtFinSF2',df=ameHouse))


# In[34]:


# BsmtFinSF1 Visualization
figBsmt1, axBsmt1 = plt.subplots(2,2, figsize=(15,10))
sns.histplot(ax=axBsmt1[0,0], x= 'BsmtFinSF1',data=ameHouse).set_title("BsmtFinSF1 Distribution")
sns.scatterplot(ax=axBsmt1[0,1],x='BsmtFinSF1',y='SalePrice',data=ameHouse).set_title("SalePrice vs BsmtFinSF1")
sns.histplot(ax=axBsmt1[1,0], x= 'BsmtFinSF2',data=ameHouse, color="red").set_title("BsmtFinSF2 Distribution")
sns.scatterplot(ax=axBsmt1[1,1],x='BsmtFinSF2',y='SalePrice',data=ameHouse,color="red").set_title("SalePrice vs BsmtFinSF2")
plt.suptitle("BsmtFinSF1 Distribution and Relationship", fontweight="bold")
plt.subplots_adjust(wspace=0.4)
plt.show()


# If we ignore observation with zero Basement Area, there is a distinct pattern which indicate positive relationship between Basement Area type 1.But, in BsmtFinSF2, there are no clear relationship with SalePrice. BsmtFinSF2 also has smaller value range with more than 90% of its value is 0. 

# #### BsmtUnfSF
# Unfinished square feet of basement area

# In[35]:


numDEA('BsmtUnfSF',ameHouse)


# In[36]:


figBsmtUnf, axBsmtUnf = plt.subplots(1,2, figsize=(15,5))
sns.histplot(ax=axBsmtUnf[0], x= 'BsmtUnfSF',data=ameHouse).set_title("BsmtUnfSF Distribution")
sns.scatterplot(ax=axBsmtUnf[1],x='BsmtUnfSF',y='SalePrice',data=ameHouse).set_title("SalePrice vs BsmtUnfSF")


# There are no clear pattern bertween BsmtUnfSF and SalePrice

# In[37]:


# There are quite a lot of variables here. So, let's just speed things up
# Within all ames_numeric,  we identify which one has smaller cardinality, and check it up to documentation, and decide whether 
# they are numeric or ordinal/categorical variable
# low cardinality, abritrary, under or equal 15
ames_numeric_low_car = [col for col in ames_numeric if ameHouse[col].nunique() <= 15]
ameHouse[ames_numeric_low_car].nunique()


# In[38]:


figMisc, axMisc = plt.subplots(2,3, figsize=(15,10))
sns.boxplot(ax=axMisc[0,0],x='MSSubClass',y="SalePrice",data=ameHouse)
sns.boxplot(ax=axMisc[0,1],x='BsmtFullBath',y="SalePrice",data=ameHouse)
sns.boxplot(ax=axMisc[0,2],x='BsmtHalfBath',y="SalePrice",data=ameHouse)
sns.boxplot(ax=axMisc[1,0],x='FullBath',y="SalePrice",data=ameHouse)
sns.boxplot(ax=axMisc[1,1],x='HalfBath',y="SalePrice",data=ameHouse)
sns.boxplot(ax=axMisc[1,2],x='Fireplaces',y="SalePrice",data=ameHouse)
plt.subplots_adjust(wspace=0.4)
plt.show()


# Based on the feature definition, all of this low cardinality feature (except for MSSubClass) are numerical discrete value. They represents a number of facility exists in a house. But, because of their value nature, the proper treatment is to group them together with categorical features.

# In[39]:


# Separate high cardinality numeric feature from low cardinality numeric feature
ames_numeric_high_car = [col for col in ames_numeric if col not in ames_numeric_low_car]
ames_numeric_high_car


# In[40]:


# call all statistic summary for ames numeric feature with high cardinality
for i in ames_numeric_high_car:
    print(numDEA(i,df=ameHouse),end="\n\n")


# From this summary, i find that, even in this high cardinality features, there are still few with high zero value percentage, like **'MasVnrArea', 'BsmtFinSF2','2ndFlrSF','WoodDeckSF',and 'ScreenPorch'**.

# In[41]:


# Let's check if if these features still worth to keep
figMisc2, axMisc2 = plt.subplots(2,3, figsize=(15,10))
sns.regplot(ax=axMisc2[0,0],x='MasVnrArea',y="SalePrice",data=ameHouse)
sns.regplot(ax=axMisc2[0,1],x='BsmtFinSF2',y="SalePrice",data=ameHouse)
sns.regplot(ax=axMisc2[0,2],x='2ndFlrSF',y="SalePrice",data=ameHouse)
sns.regplot(ax=axMisc2[1,0],x='WoodDeckSF',y="SalePrice",data=ameHouse)
sns.regplot(ax=axMisc2[1,1],x='ScreenPorch',y="SalePrice",data=ameHouse)
plt.subplots_adjust(wspace=0.4)
plt.show()


# Aside of 'MasVnrArea', all of these features doesn't seem worth to keep around. The rarity itself is a problem. Actually, looking back, all features with high zero percentage value is caused by rarity factor; it came back to the fact we found early about the existence of some elite house.
# It makes sense that not all house has pool, or wood deck,or 3 Season Porch, right? 

# In[42]:


# Keep all feature as numeric feature, except for MSSubClass
#ames_numeric = ames_numeric.drop('MSSubClass')
len(ames_numeric)


# In[43]:


# Analysis on Categorical Feature 
ames_cat = [col for col in ameHouse.columns if ameHouse[col].dtypes=='object']
print(len(ames_cat))
ameHouse[ames_cat].nunique()


# In[44]:


# Check missing value
ameHouse[ames_cat].isnull().sum()


# Here i have 38 features treated as categorical. In one glance, all missing value is caused by natural condition (no garage, no basement, etc).
# I build some visualization to make this rather presentable

# In[45]:


figCat, axCat = plt.subplots(3,3, figsize=(15,15))
sns.countplot(ax=axCat[0,0],x='MSZoning',data=ameHouse)
sns.countplot(ax=axCat[0,1],x='Street',data=ameHouse)
sns.countplot(ax=axCat[0,2],x='LotShape',data=ameHouse)
sns.countplot(ax=axCat[1,0],x='LandContour',data=ameHouse)
sns.countplot(ax=axCat[1,1],x='Utilities',data=ameHouse)
sns.countplot(ax=axCat[1,2],x='LotConfig',data=ameHouse)
sns.countplot(ax=axCat[2,0],x='LandSlope',data=ameHouse)
sns.countplot(ax=axCat[2,1],x='Neighborhood',data=ameHouse)
sns.countplot(ax=axCat[2,2],x='Condition1',data=ameHouse)
plt.subplots_adjust(wspace=0.4)
plt.show()


# In[46]:


figCat, axCat = plt.subplots(3,3, figsize=(15,15))
sns.boxplot(ax=axCat[0,0],x='MSZoning',y="SalePrice",data=ameHouse)
sns.boxplot(ax=axCat[0,1],x='Street',y="SalePrice",data=ameHouse)
sns.boxplot(ax=axCat[0,2],x='LotShape',y="SalePrice",data=ameHouse)
sns.boxplot(ax=axCat[1,0],x='LandContour',y="SalePrice",data=ameHouse)
sns.boxplot(ax=axCat[1,1],x='Utilities',y="SalePrice",data=ameHouse)
sns.boxplot(ax=axCat[1,2],x='LotConfig',y="SalePrice",data=ameHouse)
sns.boxplot(ax=axCat[2,0],x='LandSlope',y="SalePrice",data=ameHouse)
sns.boxplot(ax=axCat[2,1],x='Neighborhood',y="SalePrice",data=ameHouse)
sns.boxplot(ax=axCat[2,2],x='Condition1',y="SalePrice",data=ameHouse)
plt.subplots_adjust(wspace=0.4)
plt.show()


# Judging from the distribution and count value per category, almost all of nine categoricals we visualized has unbalanced proportion, dominated by certain group.

# In[47]:


### Preprocessing Data
# As we start to pre processing data, it's time to split the training dataset
# Import train_test_split module
from sklearn.model_selection import train_test_split


# In[48]:


# separate target variable from the rest
# in this project, the target is SalePrice
X = ameHouse.copy(deep=True)
y = X.pop("SalePrice")


# In[49]:


X.head()


# In[50]:


y.head()


# In[51]:


train_X, val_X, train_y, val_y = train_test_split(X,y,train_size=0.8,test_size=0.2, random_state=0)


# In[52]:


print(train_X.shape)
print(val_X.shape)
print(train_y.shape)
print(val_y.shape)


# #### Handling  Missing Value
# 
# As we know from previous DEA result, there are some features contain missing value. Some of them has been dropped because of their high percentage of missing value. What left is to deal with the remaining problem.
# 
# It's easy to get rid of entire rows of data with missing values, but I don't want to risk losing too much important information.
# 
# To do that, i will try to do some reasoning on each feature with missing values, to find out the cause of the missing values. If the cause is natural, then I will remove the row or replace it with a zero value. On the other hand, if the cause of the missing value is an error, then I will do imputation.

# In[53]:


# Recall all the missing
miss_col = [col for col in train_X.columns if train_X[col].isnull().sum()>0]
print(len(miss_col))
train_X[miss_col].isnull().sum()


# In[54]:


# There are 14 features contains missing value.
# Let's examine each of it


# In[55]:


# LotFrontage, as stated before, is the linear feet of street connected to the property
# First, let's think it. Is it possible for a house, a residential house, to not has any street or alley in its front?
# While it still possible in a rural area, it seems not be the case in this dataset.
# To Prove it,there are few features we can use.

# I extract all index with null value of LotFrontage 
miss_lotFrontage = train_X[train_X['LotFrontage'].isnull()].index.tolist()

# Street == Type of road access to property
# This feature consist of two categories, ie 'Gravel' and 'Paved'
sns.countplot(x=train_X.loc[miss_lotFrontage,'Street'])
plt.title("Street Access of house with null value of LotFrontage", fontweight = "bold")
plt.show()
print("Missing value :{}".format(train_X['Street'].isnull().sum())) 
# no missing value, means all houses in our train dataset have access to the road`
# Most of house with null LotFrontage has Pave road connected to it, so it is make no sense. 


# In[56]:


# Now i know that missing value in LotFrontage is caused by error factor. The next step is to do some imputation to fill these
# empty value
# For this case, i want to use median value of LotFrontage based on its MSZoning and MSSubClass. This abritrary decision.
median_lotFrontage_by_ZoningClass = train_X.groupby(['MSZoning','MSSubClass'])['LotFrontage'].median()
train_X.groupby(['MSZoning','MSSubClass'])['LotFrontage'].median()

# Start imputation


# In[57]:


# Imputation
# Python may give warning like this: 
# SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame.
# Disable this by write this code instead
# pd.options.mode.chained_assignment = None  # default='warn'
#.loc[row_index,col_indexer] = value instead
pd.options.mode.chained_assignment = None
miss_lotFrontage = train_X[train_X['LotFrontage'].isnull()].index.tolist()
for j in miss_lotFrontage:
    for i in range(len(median_lotFrontage_by_ZoningClass)):
        if(train_X.loc[j,'MSZoning'] == median_lotFrontage_by_ZoningClass.index[i][0] and train_X.loc[j,'MSSubClass']==median_lotFrontage_by_ZoningClass.index[i][1]):
            train_X.loc[j,'LotFrontage'] = median_lotFrontage_by_ZoningClass[i]


# In[58]:


# Imputation Success
train_X.loc[miss_lotFrontage,'LotFrontage']


# In[59]:


train_X['LotFrontage'].isnull().sum() # no more missing value


# In[60]:


# Don't forget to do the same thing to validation and test set
print(val_X['LotFrontage'].isnull().sum()) # 47 missing values in validation dataset
print(ameHouse_test['LotFrontage'].isnull().sum()) # 227 missing values in test dataset


# In[61]:


Valid_median_lotFrontage_by_ZoningClass = val_X.groupby(['MSZoning','MSSubClass'])['LotFrontage'].median()
Test_median_lotFrontage_by_ZoningClass = ameHouse_test.groupby(['MSZoning','MSSubClass'])['LotFrontage'].median()


# In[62]:


Valid_median_lotFrontage_by_ZoningClass
# There is a null value of median in validation dataset. I replace it with the median from the same grouping in test dataset
Valid_median_lotFrontage_by_ZoningClass[5] = 60
Valid_median_lotFrontage_by_ZoningClass


# In[63]:


# Apparently, test data set has different columns set with missing value compared to valid and training data set. 
# I put it aside for now, and will deal with it in separate section.
Test_median_lotFrontage_by_ZoningClass
# ameHouse_test.isnull().sum() #uncomment this line to show which columns has missing value in test dataset 


# In[64]:


# get all row of valid dataset with missing lotFrontage value
Valid_miss_lotFrontage = val_X[val_X['LotFrontage'].isnull()].index.tolist()
# Imputed  the value with median
for j in Valid_miss_lotFrontage:
    for i in range(len(Valid_median_lotFrontage_by_ZoningClass)):
        if(val_X.loc[j,'MSZoning'] == Valid_median_lotFrontage_by_ZoningClass.index[i][0] and val_X.loc[j,'MSSubClass']==Valid_median_lotFrontage_by_ZoningClass.index[i][1]):
            val_X.loc[j,'LotFrontage'] = Valid_median_lotFrontage_by_ZoningClass[i]


# In[65]:


# Check the imputed value
val_X['LotFrontage'].isnull().sum() # No More Missing Value


# In[66]:


# MasVnrType and MasVnrArea Imputation
# The reason i group these two together is because they are represents the same thing, that is Masonry Venree.
# Based on data documentation, a house without Masonry Venree valued as None.
# Logically, there will be no Masonry area if no Masonry Venree is build
# and that's why MasVnrType and MasVnrArea has the same number of missing value
# To impute this features, i will replace None value with string "None" for MasVnrType and 0 value for MasVnrArea
miss_MasVnr = train_X[train_X['MasVnrType'].isnull()].index.tolist()
# Imputing MasVnrType
train_X.loc[miss_MasVnr, 'MasVnrType'] = "None"
# Imputing MasVnrArea
train_X.loc[miss_MasVnr, 'MasVnrArea'] = 0.0


# In[67]:


# check
print(train_X['MasVnrType'].isnull().sum()) # no missing value
print(train_X['MasVnrArea'].isnull().sum()) # no missing value


# In[68]:


# Don't forget to do the same thing to validation and test dataset
# Check the missing value 
print("Missing value of MasVnrType in val_X: {}".format(val_X['MasVnrType'].isnull().sum()))
print("Missing value of MasVnrArea in val_X: {}".format(val_X['MasVnrArea'].isnull().sum()))
print("Missing value of MasVnrType in ameHouse_test: {}".format(ameHouse_test['MasVnrType'].isnull().sum()))
print("Missing value of MasVnrArea in ameHouse_test: {}".format(ameHouse_test['MasVnrArea'].isnull().sum()))


# In[69]:


# get the index of missing value in validation dataset
Val_miss_MasVnr = val_X[val_X['MasVnrType'].isnull()].index.tolist()
# Start imputing
val_X.loc[Val_miss_MasVnr,'MasVnrType'] = "None"
val_X.loc[Val_miss_MasVnr,'MasVnrArea'] = 0.0


# In[70]:


# Check missing
print("Missing value of MasVnrType in val_X: {}".format(val_X['MasVnrType'].isnull().sum()))
print("Missing value of MasVnrArea in val_X: {}".format(val_X['MasVnrArea'].isnull().sum()))


# In[71]:


# Imputing test dataset MassVnr
Test_miss_MasVnr = ameHouse_test[ameHouse_test['MasVnrType'].isnull()].index.tolist()
ameHouse_test.loc[Test_miss_MasVnr, 'MasVnrType'] = "None"
ameHouse_test.loc[Test_miss_MasVnr, 'MasVnrArea'] = 0.0
print("Missing value of MasVnrType in ameHouse_test: {}".format(ameHouse_test['MasVnrType'].isnull().sum()))
print("Missing value of MasVnrArea in ameHouse_test: {}".format(ameHouse_test['MasVnrArea'].isnull().sum()))


# In[72]:


#  Imputing BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
# Same as MasVnr... , These 4 features represents basement. House without basement  will valued as NA
# Because all of them are categorical features, i will replace it with string 'NoBasement'
Train_miss_Bsmt=train_X[train_X['BsmtQual'].isnull()][['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']].index.tolist()
train_X.loc[Train_miss_Bsmt,['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']]='NoBasement'


# In[73]:


print("Missing value of BsmtQual:{}".format(train_X['BsmtQual'].isnull().sum()))
print("Missing value of BsmtCond:{}".format(train_X['BsmtCond'].isnull().sum()))
print("Missing value of BsmtExposure:{}".format(train_X['BsmtExposure'].isnull().sum()))
print("Missing value of BsmtFinType1:{}".format(train_X['BsmtFinType1'].isnull().sum()))
print("Missing value of BsmtFinType2:{}".format(train_X['BsmtFinType2'].isnull().sum()))


# In[74]:


train_X[train_X['BsmtFinType2'].isnull()][['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF2']]


# In[75]:


# i use this proportion of finished basement type 2 for imputing the only remaining missing value of basement type 2
train_X.loc[train_X['BsmtFinType2']!="Unf",'BsmtFinType2'].value_counts()


# In[76]:


train_X.loc[332, 'BsmtFinType2'] = "LwQ"
print("Missing value of BsmtFinType2:{}".format(train_X['BsmtFinType2'].isnull().sum()))


# In[77]:


print(val_X[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].isnull().sum(),end="\n\n")
print(ameHouse_test[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].isnull().sum())


# In[78]:


val_miss_bsmt = val_X[val_X['BsmtExposure'].isnull()].index.tolist()
val_X.loc[val_miss_bsmt,['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']]


# In[79]:


sns.countplot(x='BsmtQual',hue='BsmtExposure',data=val_X)
plt.show()


# In[80]:


# Based on the chart above, it seems that basement with Good Quality(90-99 inch) has higher propotion for basement with
# Good Exposure. It may come from the fact that higher basement able to get optimum sunlight and air circulation exposure.
# I use this as my imputation basis for row number 984

val_X.loc[948,'BsmtExposure'] = 'Gd'


# In[81]:


# Let's update the missing value index
val_miss_bsmt = val_X[val_X['BsmtExposure'].isnull()].index.tolist()
val_miss_bsmt
# Imputed the remaining rows
val_X.loc[val_miss_bsmt,['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']]='NoBasement'


# In[82]:


# Check the value
val_X.loc[val_miss_bsmt,['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']]
val_X[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].isnull().sum() # Clear


# In[83]:


# Basement in Test data set 
Test_miss_bsmt = ameHouse_test.loc[ameHouse_test['BsmtFinType1'].isnull()].index.tolist()
ameHouse_test.loc[Test_miss_bsmt, ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']]


# In[84]:


# Impute this rows with the same method as Train and Valid dataset
ameHouse_test.loc[Test_miss_bsmt,['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']] = 'NoBasement'
ameHouse_test[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].isnull().sum()


# In[85]:


Test2_miss_bsmt = ameHouse_test[ameHouse_test['BsmtExposure'].isnull()].index.tolist()


# In[86]:


Test_inquiry_bsmt = ameHouse_test.loc[(ameHouse_test['BsmtQual']=='Gd') & (ameHouse_test['BsmtCond']=='TA') & (ameHouse_test['BsmtFinType1']=='Unf') 
                 & (ameHouse_test['BsmtFinType2']=='Unf')].index.tolist()
ameHouse_test.loc[Test_inquiry_bsmt,'BsmtExposure'].value_counts()
# We mining all rows from test dataset that has the same condition 
# Start from this, i will impute the missing BsmtExposure with 'No'


# In[87]:


ameHouse_test.loc[Test2_miss_bsmt,'BsmtExposure'] = 'No'
ameHouse_test[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].isnull().sum()


# In[88]:


# imputing the BsmtQual in test dataset
Test3_miss_bsmt = ameHouse_test.loc[ameHouse_test['BsmtQual'].isnull()].index.tolist()
ameHouse_test.loc[Test3_miss_bsmt,['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']]


# In[89]:


Test4_miss_bsmt = ameHouse_test.loc[(ameHouse_test['BsmtCond']=='Fa') | (ameHouse_test['BsmtCond']=='TA') & (ameHouse_test['BsmtExposure']=="No") &
                                   (ameHouse_test['BsmtFinType1']=="Unf") & (ameHouse_test['BsmtFinType2']=='Unf')].index.tolist()
ameHouse_test.loc[Test4_miss_bsmt,'BsmtQual'].value_counts()


# In[90]:


# Impute BsmtQual in test dataset with 'TA'

ameHouse_test.loc[Test4_miss_bsmt,'BsmtQual'] = 'TA'
ameHouse_test[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].isnull().sum()


# In[91]:


# Imputing BsmtCond in Test dataset
Test5_miss_bsmt = ameHouse_test.loc[ameHouse_test['BsmtCond'].isnull()].index.tolist()
ameHouse_test.loc[Test5_miss_bsmt,['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']]


# In[92]:


# I use median for this SimpleImputer strategy that is most_frequent
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# define the imputer
simpleImp= SimpleImputer(missing_values = np.nan, strategy='most_frequent')
# Define the ColumnTransformer
column_trans = ColumnTransformer([('imputed_BsmtCond',simpleImp,[30])])
imputed_ameHouse_test = column_trans.fit_transform(ameHouse_test)

#new data
imputed_ameHouse_test


# In[93]:


ameHouse_test['imputed_BsmtCond'] = imputed_ameHouse_test
ameHouse_test.columns
ameHouse_test.drop('BsmtCond',axis=1,inplace=True)
ameHouse_test.shape


# In[94]:


ameHouse_test['imputed_BsmtCond']


# In[95]:


ameHouse_test.isnull().sum()


# In[96]:


# Electrical Missing Value
miss_elec = train_X.loc[train_X['Electrical'].isnull()].index
miss_elec


# In[97]:


figElec,elc = plt.subplots(1,2, figsize=(15,5))
sns.countplot(ax = elc[0],x='Electrical',hue="MSZoning",data=train_X).set_title('Electrical System by MSZoning')
elc[0].legend(loc="upper right",title='MSZoning')
sns.countplot(ax=elc[1],x='Electrical',hue='MSSubClass',data=train_X).set_title('Electrical System by MSSubClass')
elc[1].legend(loc="upper right",title='MSSubClass')
plt.show()


# In[98]:


# From these two visualization, we acquire insight about which electrical system used by house from across certain category most
train_X.loc[miss_elec,['MSZoning','MSSubClass']]
# Impute the missing value with SBrkr
train_X.loc[miss_elec,'Electrical'] = 'SBrkr'
print("Missing value of Electrical: {} ".format(train_X['Electrical'].isnull().sum()))


# In[99]:


# Let's do the same for validation and test dataset
print("Missing value of Electrical in validation dataset : {}".format(val_X['Electrical'].isnull().sum()))
print("Missing value of Electrical in test dataset : {}".format(ameHouse_test['Electrical'].isnull().sum()))


# In[100]:


#At first glance, the Garage feature has the same imputation properties as basement and Masonry. But look in more detail, 
#and we realize that there is one feature called GarageYrBlt that cannot be imputed arbitrarily. Logically, when there is 
#no garage, there is no development. Imputing the GarageYrBlt variable with various techniques, as far as I know, 
# will only make this feature lose its essence. Therefore, I decided to ditch the row where these features miss.
 
miss_garage = train_X.loc[train_X['GarageType'].isnull()].index.tolist()
train_X.loc[miss_garage,['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageCars']]


# In[101]:


train_X.drop(index=miss_garage, inplace=True)
train_X[['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageCars']].isnull().sum()


# In[102]:


# Check all features
train_X[miss_col].isnull().sum()
# Check the data dimension
train_X.shape # From 1168 to 1110, i lost 58 rows, but fortunately, still retain 95% data


# In[103]:


# Garage Features in validation and test dataset
print(val_X[['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']].isnull().sum(),end="\n\n")
print(ameHouse_test[['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']].isnull().sum())


# In[104]:


Valid_miss_garage = val_X.loc[val_X['GarageType'].isnull()].index.tolist()
val_X.loc[Valid_miss_garage,['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']]


# In[105]:


# drop all Garage Features with missing value
val_X.drop(index=Valid_miss_garage, inplace=True)
val_X[['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageCars']].isnull().sum()


# In[106]:


val_X.shape # I able to keep 92,12% of the data after drop  imputation


# In[107]:


Test_miss_garage = ameHouse_test.loc[ameHouse_test['GarageFinish'].isnull()].index.tolist()
print(ameHouse_test.loc[Test_miss_garage,['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']])


# In[108]:


ameHouse_test.loc[(ameHouse_test['GarageType'].notnull()) & (ameHouse_test['GarageFinish'].isnull())]
ameHouse_test.loc[[666,1116],['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']]


# In[109]:


# There is a litle confusion from test dataset, but i consider that losing 78 rows or trying to imputate two rows is clear 
# decision line. I decide to drop all 78 rows, because it's simpler and the cost are relatively low.
#ameHouse_test.drop(index=Test_miss_garage, inplace=True)
ameHouse_test[['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageCars']].isnull().sum()


# In[110]:


ameHouse_test.shape # i still have 94,65 of the original dataset


# In[111]:


# Check for missing values in train and validation data
columns_miss_train = len([col for col in train_X.columns if train_X[col].isnull().sum()>0])
columns_miss_val = len([col for col in val_X.columns if val_X[col].isnull().sum()>0])
columns_miss_test = len([col for col in ameHouse_test.columns if ameHouse_test[col].isnull().sum()>0])

print(columns_miss_train)
print(columns_miss_val)
print(columns_miss_test)


# In[112]:


# Check for miss value columns in test dataset
miss_test_col = [col for col in ameHouse_test.columns if ameHouse_test[col].isnull().sum()>0]
ameHouse_test[miss_test_col].isnull().sum()


# In[113]:


# For efficiency, i decide to use KNN Imputation technique for numerical features and simpleimputer "most_frequent" for string
from sklearn.impute import KNNImputer
KnnImp = KNNImputer(n_neighbors = 6)


# In[114]:


ames_test_cat =ameHouse_test.select_dtypes("object").columns.tolist()
ames_test_num =ameHouse_test.select_dtypes(["int64","float64"]).columns.tolist()


# In[115]:


# ColumnTransformer for two Columns
catTestImputer = SimpleImputer(strategy = 'most_frequent')
test_transformer = ColumnTransformer([('imputed_cat',catTestImputer,ames_test_cat),
                                     ('imputed_num',KnnImp,ames_test_num)])
test_imputed_ames = test_transformer.fit_transform(ameHouse_test)


# In[116]:


new_test_col = ameHouse_test.columns
new_test_col


# In[117]:


imputed_ames_test_new = pd.DataFrame(test_imputed_ames, columns=new_test_col)


# In[118]:


new_col =ames_test_cat + ames_test_num
imputed_ames_test_new = pd.DataFrame(test_imputed_ames, columns=new_col).reset_index()
imputed_ames_test_new.drop('Id',axis=1,inplace=True)
imputed_ames_test_new


# In[119]:


# Check for missing value
print("Missing value in test dataset: {}".format(len([col for col in imputed_ames_test_new.columns if imputed_ames_test_new[col].isnull().sum()>0])))
print("Missing value in train dataset: {}".format(len([col for col in train_X.columns if train_X[col].isnull().sum()>0])))
print("Missing value in validation dataset: {}".format(len([col for col in val_X.columns if val_X[col].isnull().sum()])))


# ### Next step
# 
# Now that we are done with missing value in all dataset, we proceed to the next steps. That is, another cleaning. Here i explain what i would do for the next few steps.
# - Dealing with categorical features
# - Dealing with outliers and possibility of scaling
# - Parsing 
# - Check for inconsistenty
# 
# After that, i would continue with **FEATURE ENGINEERING**, **MODEL BUILDING, MODEL ENSEMBLING**, and finally, draw one or some conclusions from the data. 

# #### Dealing with Categorical and Ordinal Features
# 
# Typically, there are two method to work with categorical/ordinal features (3 actually, but i ditch it because it just about drop non-numerical features), i.e:
# - Ordinal encoding
# - One-Hot encoding
# The difference between the two lies in the output of the data encoding. Ordinal encoding will convert category data into ordinal data, while one-hot encoding will convert category data into nominal data, where each category will have its own column.

# In[120]:


print("Object-type data column in train_X :{}".format(len(train_X.select_dtypes('object').columns))) # 38
print(train_X.select_dtypes('object').nunique()) 
ames_cat_col = train_X.select_dtypes('object').columns.tolist()


# In[121]:


# Another thing we need to know especially for the Ames, Iowa housing data, is that there are numerical features that we can 
# treat as categorical data, or specifically as ordinal data type.
# To choose the right features that we would treat as ordinal, first, we filter the one with low cardinality, 
# and second, we read the data documentation to make sure that they are indeed have orndinal nature.
low_num_card = []
count_num = train_X.select_dtypes(['int64','float64']).nunique()
for i in count_num.index:
    if count_num[i] <= 15:
        low_num_card.append(i)
low_num_card


# There are 15 numerical features with low cardinality. Then, after double check with data documentation, here is the result:
# - MSSubClass --- > Nominal
# - OverallQual --- > Ordinal
# - OverallCond --- > Ordinal
# - BsmtFullBath --- > Ordinal
# - BsmtHalfBath --- > Ordinal
# - FullBath --- > Ordinal
# - HalfBath --- > Ordinal
# - BedroomAbvGr --- > Ordinal
# - KitchenAbvGr --- > Ordinal
# - TotRmsAbvGrd --- > Ordinal
# - FirePlaces --- > Ordinal
# - GarageCars --- > Ordinal
# - PoolArea --- > Numeric
# - MoSold --- > Numeric (DateTime)
# - YrSold --- > Numeric (DateTime)
# 
# So, 12 out of 15 low cardinality numeric features can be treated as Ordinal/nominal features. Hence, i will take them out from numeric features group later.

# In[122]:


ames_cat_col = ames_cat_col + low_num_card[:11]
ames_cat_col


# In[123]:


# From categorical / ordinal group of features, i need to decide which one should be encoded with Ordinal or One-Hot technique
# based on it's nature
# Usually i need to read the data documentation to understand what each features represents
# But, there are too many features.  So, let's take the most efficient strategy
# Basically, almost all features with string value would be considered to be encoded with One-Hot encoding,
# Exception for some, features like 'ExterQual','ExterCond', 'BsmtCond', 'BsmtQual','BsmtExposure','BsmtFinType1',
# 'BsmtFinType2','HeatingQC','KitchenQual','GarageQual', and 'GarageCond' would be encoded with One-Hot encoding
# The reason why i encode some features above with Ordinal method is because they inherently represents quality level,
# and presummably, those leveling would better represented as ordinal data.

# Ordinal Encoding
ame_cat_to_ordinal = ['ExterQual','ExterCond', 'BsmtCond', 'BsmtQual','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC',
                      'KitchenQual','GarageQual','GarageCond']+ames_cat_col[38:]
# create a copy for this column in different dataframe
label_ordinal_train_X = train_X[ame_cat_to_ordinal].copy(deep=True)
label_ordinal_val_X = val_X[ame_cat_to_ordinal].copy(deep=True)


# In[346]:


# for test dataset
imputed_ames_test_new.rename(columns={'index':'Id','imputed_BsmtCond':'BsmtCond'},inplace=True)


# In[351]:


label_ordinal_test = imputed_ames_test_new[ame_cat_to_ordinal].copy(deep=True)
label_ordinal_test.nunique()


# In[124]:


label_ordinal_val_X.nunique()


# In[125]:


# import OrdinalEncoder module
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=15)
label_ordinal_train_X[:] = ordinal_encoder.fit_transform(label_ordinal_train_X)
label_ordinal_val_X[:] = ordinal_encoder.transform(label_ordinal_val_X)


# In[352]:


# For test dataset
label_ordinal_test[:] = ordinal_encoder.transform(label_ordinal_test)


# In[353]:


label_ordinal_train_X
label_ordinal_val_X
label_ordinal_test


# In[127]:


ame_cat_to_onehot = [col for col in ames_cat_col if col not in ame_cat_to_ordinal]
label_onehot_train_X = train_X[ame_cat_to_onehot].copy(deep=True)
label_onehot_val_X = val_X[ame_cat_to_onehot].copy(deep=True)


# In[354]:


# For test dataset
label_onehot_test = imputed_ames_test_new[ame_cat_to_onehot].copy(deep=True)
label_onehot_test


# In[128]:


label_onehot_train_X


# In[129]:


# One Hot Encoder
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(label_onehot_train_X[:]))
OH_cols_val = pd.DataFrame(OH_encoder.transform(label_onehot_val_X[:]))


# In[355]:


# For test dataset
OH_cols_test = pd.DataFrame(OH_encoder.transform(label_onehot_test[:]))
OH_cols_test.index = label_onehot_test.index
OH_cols_test


# In[130]:


OH_cols_train.index = label_onehot_train_X.index
OH_cols_val.index = label_onehot_val_X.index


# In[131]:


OH_encoder.inverse_transform(OH_cols_train)
OH_col_names = OH_encoder.get_feature_names().tolist()


# In[132]:


# Change the columns name
OH_cols_train.columns = OH_col_names
OH_cols_val.columns = OH_col_names


# In[356]:


OH_cols_test.columns = OH_col_names
OH_cols_test


# In[133]:


# join two data into one dataframe filled with transformed categorical features
ames_transform_all_cat_train = pd.concat([label_ordinal_train_X,OH_cols_train],axis=1)
ames_transform_all_cat_val = pd.concat([label_ordinal_val_X,OH_cols_val],axis=1)


# In[357]:


# For test dataset
ames_transform_all_cat_test = pd.concat([label_ordinal_test,OH_cols_test],axis=1)
ames_transform_all_cat_test


# In[134]:


ames_transform_all_cat_train.head()


# In[135]:


ames_transform_all_cat_val.head()


# In[136]:


# don't forget to drop all categorical/ordinal features from the original dataset
num_train_X = train_X.drop(ames_cat_col, axis = 1)
num_val_X = val_X.drop(ames_cat_col, axis = 1)


# In[358]:


# for test dataset
num_test = imputed_ames_test_new.drop(ames_cat_col,axis=1)
num_test


# In[137]:


num_train_X.head()


# In[138]:


# Now merge the only-number dataframe with transformed categorical dataframe
all_num_cat_trans_train_X = pd.concat([num_train_X,ames_transform_all_cat_train],axis=1)
all_num_cat_trans_val_X = pd.concat([num_val_X,ames_transform_all_cat_val],axis=1)


# In[360]:


# For test dataset
all_num_cat_trans_test = pd.concat([num_test,ames_transform_all_cat_test],axis=1)
all_num_cat_trans_test.shape


# In[139]:


print(all_num_cat_trans_train_X.shape)
print(all_num_cat_trans_val_X.shape)


# #### Standarization, and Normalization
# We are done with categorical features. The features increase drastically, and perhaps would impact the model performance significantly, for better or worse.
# one thing for sure is all data features are now of numeric type. They all have different range and distribution; My next job is to examine and make sure this difference will not reduce model's performance. 
# 
# In this part, i will do normality test using Shapiro-Wilk against only the continuous or 'original number features' so i called. Then, features that do not follow Gaussian distribution will go through a normalization procedure using Robust Scaler.
# 
# *Side note*:
# Shapiro-Wilk Test's hypothesis are:
# H0 : The population is distributed normally
# H1 : The population is not distributed normally

# In[140]:


# first of all,  i will make 'Id' as dataframe index
all_num_cat_trans_train_X = all_num_cat_trans_train_X.set_index('Id')
all_num_cat_trans_val_X = all_num_cat_trans_val_X.set_index('Id')


# In[367]:


# For test dataset
all_num_cat_trans_test = all_num_cat_trans_test.set_index('Id')


# In[141]:


# Next, i will exclude dummy and ordinal features from strandarization process
num_features = [col for col in all_num_cat_trans_train_X.columns if col in num_train_X]
len(num_features)


# In[142]:


# import shapiro module
from scipy.stats import shapiro

# Perform the test
def ShapiroTest(df, sig):
    p_val = []
    decs = []
    cols = []
    for i in df.columns:
        shap,res = shapiro(df[i])
        cols.append(i)
        p_val.append(res)
        if res > sig:
            decs.append("Normal Distribution")
        else:
            decs.append("Not Normal Distributrion")
    res_df = pd.DataFrame({'Feature': cols, 'P-Value':p_val, 'Conclusion': decs})
    return res_df


# In[143]:


Shapiro_test_train = ShapiroTest(all_num_cat_trans_train_X[num_features],0.05)


# In[144]:


Shapiro_test_train


# In[145]:


Shapiro_test_train['Conclusion'].value_counts() # All features do not follow Gaussian Distribution


# In[146]:


Shapiro_test_val = ShapiroTest(all_num_cat_trans_val_X[num_features],0.05)
Shapiro_test_val 


# In[369]:


# Now, we know that all numerical features, either from train or validation dataset are not comes from normal distribution
# So i will perform normalization for all of them. But before that, 'MoSold','YrSold', and 'GarageYrBlt' should be separate
# because they are datetime features, and standarization in any form will make no sense.
year_feature = ['YearBuilt','YearRemodAdd','GarageYrBlt','MoSold','YrSold']
num_features = [col for col in num_features if col not in year_feature]
len(num_features)


# In[148]:


# Distribution Visual
figDis,axDis = plt.subplots(5,4,figsize=(15,15))
sns.histplot(ax=axDis[0,0],x='LotFrontage',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[0,1],x='LotArea',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[0,2],x='MasVnrArea',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[0,3],x='BsmtFinSF1',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[1,0],x='BsmtFinSF2',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[1,1],x='BsmtUnfSF',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[1,2],x='TotalBsmtSF',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[1,3],x='1stFlrSF',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[2,0],x='2ndFlrSF',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[2,1],x='GrLivArea',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[2,2],x='GarageCars',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[2,3],x='GarageArea',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[3,0],x='WoodDeckSF',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[3,1],x='OpenPorchSF',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[3,2],x='EnclosedPorch',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[3,3],x='3SsnPorch',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[4,0],x='ScreenPorch',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[4,1],x='PoolArea',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[4,2],x='MiscVal',data=all_num_cat_trans_train_X)
sns.histplot(ax=axDis[4,3],x='LowQualFinSF',data=all_num_cat_trans_train_X)
plt.subplots_adjust(wspace=0.4)
plt.show()


# In[377]:


# Scaling predictor and target features
from sklearn.preprocessing import RobustScaler
num_feature = num_features[:20]
num_feature
robustScaler = RobustScaler()
scaled_num_robust_train_X = pd.DataFrame(robustScaler.fit_transform(all_num_cat_trans_train_X[num_feature]),
                                       columns=num_feature)
scaled_num_robust_val_X = pd.DataFrame(robustScaler.transform(all_num_cat_trans_val_X[num_feature]),
                                      columns=num_feature)


# In[378]:


# For Test dataset
num_feature = num_features[:20]
scaled_num_robust_test = pd.DataFrame(robustScaler.transform(all_num_cat_trans_test[num_feature]),
                                     columns=num_feature)
scaled_num_robust_test


# In[379]:


all_num_cat_trans_train_X.shape
cat_features = [col for col in all_num_cat_trans_train_X.columns if col not in num_features]
len(cat_features) # 202 cat features
len(scaled_num_robust_train_X.columns) # 20 features


# In[380]:


scaled_num_robust_train_X.index = all_num_cat_trans_train_X.index
scaled_num_robust_val_X.index = all_num_cat_trans_val_X.index


# In[381]:


# drop all number features from original dataset in train and val

all_cat_trans_train_X = all_num_cat_trans_train_X[cat_features]
all_cat_trans_val_X = all_num_cat_trans_val_X[cat_features]
len(all_cat_trans_train_X.columns)


# In[382]:


# For test dataset
all_cat_trans_test = all_num_cat_trans_test[cat_features]
len(all_cat_trans_test.columns)


# In[153]:


scaled_all_num_cat_robust_trainX = pd.concat([scaled_num_robust_train_X,all_cat_trans_train_X],axis=1)
scaled_all_num_cat_robust_valX = pd.concat([scaled_num_robust_val_X,all_cat_trans_val_X],axis=1)
scaled_all_num_cat_robust_trainX


# In[384]:


# Concate for test dataset
#scaled_all_num_cat_robust_test = pd.concat([scaled_num_robust_test,all_cat_trans_test],axis=1)
scaled_all_num_cat_robust_test


# In[154]:


scaled_all_num_cat_robust_trainX


# In[155]:


scaled_all_num_cat_robust_valX


# In[156]:


figDis,axDis = plt.subplots(5,4,figsize=(15,15))
sns.histplot(ax=axDis[0,0],x='LotFrontage',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[0,1],x='LotArea',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[0,2],x='MasVnrArea',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[0,3],x='BsmtFinSF1',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[1,0],x='BsmtFinSF2',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[1,1],x='BsmtUnfSF',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[1,2],x='TotalBsmtSF',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[1,3],x='1stFlrSF',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[2,0],x='2ndFlrSF',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[2,1],x='GrLivArea',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[2,2],x='GarageCars',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[2,3],x='GarageArea',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[3,0],x='WoodDeckSF',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[3,1],x='OpenPorchSF',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[3,2],x='EnclosedPorch',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[3,3],x='3SsnPorch',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[4,0],x='ScreenPorch',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[4,1],x='PoolArea',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[4,2],x='MiscVal',data=scaled_all_num_cat_robust_trainX)
sns.histplot(ax=axDis[4,3],x='LowQualFinSF',data=scaled_all_num_cat_robust_trainX)
plt.subplots_adjust(wspace=0.4)
plt.show()


# In[157]:


# Experiment to normalize data using 
from sklearn.preprocessing import PowerTransformer
powerTrans = PowerTransformer()
normalize_num_train_X = pd.DataFrame(powerTrans.fit_transform(all_num_cat_trans_train_X[num_features]),columns=num_features)
normalize_num_train_X.shape


# In[158]:


figNorm,axNorm = plt.subplots(5,4,figsize=(15,15))
sns.histplot(ax=axNorm[0,0],x='LotFrontage',data=normalize_num_train_X)
sns.histplot(ax=axNorm[0,1],x='LotArea',data=normalize_num_train_X)
sns.histplot(ax=axNorm[0,2],x='MasVnrArea',data=normalize_num_train_X)
sns.histplot(ax=axNorm[0,3],x='BsmtFinSF1',data=normalize_num_train_X)
sns.histplot(ax=axNorm[1,0],x='BsmtFinSF2',data=normalize_num_train_X)
sns.histplot(ax=axNorm[1,1],x='BsmtUnfSF',data=normalize_num_train_X)
sns.histplot(ax=axNorm[1,2],x='TotalBsmtSF',data=normalize_num_train_X)
sns.histplot(ax=axNorm[1,3],x='1stFlrSF',data=normalize_num_train_X)
sns.histplot(ax=axNorm[2,0],x='2ndFlrSF',data=normalize_num_train_X)
sns.histplot(ax=axNorm[2,1],x='GrLivArea',data=normalize_num_train_X)
sns.histplot(ax=axNorm[2,2],x='GarageCars',data=normalize_num_train_X)
sns.histplot(ax=axNorm[2,3],x='GarageArea',data=normalize_num_train_X)
sns.histplot(ax=axNorm[3,0],x='WoodDeckSF',data=normalize_num_train_X)
sns.histplot(ax=axNorm[3,1],x='OpenPorchSF',data=normalize_num_train_X)
sns.histplot(ax=axNorm[3,2],x='EnclosedPorch',data=normalize_num_train_X)
sns.histplot(ax=axNorm[3,3],x='3SsnPorch',data=normalize_num_train_X)
sns.histplot(ax=axNorm[4,0],x='ScreenPorch',data=normalize_num_train_X)
sns.histplot(ax=axNorm[4,1],x='PoolArea',data=normalize_num_train_X)
sns.histplot(ax=axNorm[4,2],x='MiscVal',data=normalize_num_train_X)
sns.histplot(ax=axDis[4,3],x='LowQualFinSF',data=normalize_num_train_X)
plt.subplots_adjust(wspace=0.4)
plt.show()


# In[159]:


# We can see that almost all numeric features distribution move closer to Gaussian distribution.

ShapiroTest(normalize_num_train_X,0.05)
# even so, all of them still can't pass normality test.
# So, here i decide to proceed my analysis based only on scaled data.  


# In[564]:


# Transform y-feature (salePrice)
new_train_y = train_y.values.reshape(-1,1)
new_val_y = val_y.values.reshape(-1,1)
robustScaler_forPrice = RobustScaler()
scaled_train_y = pd.DataFrame(robustScaler_forPrice.fit_transform(new_train_y),columns=['SalePrice'])
scaled_val_y = pd.DataFrame(robustScaler_forPrice.transform(new_val_y),columns=['SalePrice'])
scaled_train_y.index = train_y.index+1
scaled_val_y.index = val_y.index+1


# In[565]:


scaled_train_y.head()


# In[162]:


scaled_val_y.head()


# In[163]:


scaled_all_num_cat_robust_trainX.index


# In[576]:


# Adjust train and validation Y index
scaled_train_y = scaled_train_y.loc[scaled_all_num_cat_robust_trainX.index]
scaled_train_y


# In[165]:


scaled_all_num_cat_robust_valX.index


# In[166]:


# Adjust train and validation Y index
scaled_val_y = scaled_val_y.loc[scaled_all_num_cat_robust_valX.index]
scaled_val_y


# In[167]:


# Outliers
# I wouldn't do any outlier detection at this point
# The reason are : i already scaling it, and any outlier detection after this stage seems to generate a condition where
# all rows considered outliers. To prove this, run the following code to compare the result 
# of outlier detection using scaled and original dataset


## NOTICE : It is not a general rule, but maybe it would be more meaningfull to detect and treat outlier before all 
## preprocessing steps. Each case and each method of transformation has different impact.
## uncomment one of these part 

###############################################
##### Outliers detection in scaled data set
#Q1 = scaled_all_num_cat_robust_trainX.quantile(0.25)
#Q3 = scaled_all_num_cat_robust_trainX.quantile(0.75)

# calculate the IQR
#QR = Q3 - Q1

# filter the dataset with the IQR
#IQR_outliers = scaled_all_num_cat_robust_trainX[((scaled_all_num_cat_robust_trainX < (Q1 - 1.5 * IQR)) |(scaled_all_num_cat_robust_trainX > (Q3 + 1.5 * IQR))).any(axis=1)]
#IQR_outliers


###############################################

##### Outliers detection in original data set
#Q1 = train_X.quantile(0.25)
#Q3 = train_X.quantile(0.75)

# calculate the IQR
#QR = Q3 - Q1

# filter the dataset with the IQR
#IQR_outliers = train_X[((train_X < (Q1 - 1.5 * IQR)) |(train_X > (Q3 + 1.5 * IQR))).any(axis=1)]
#IQR_outliers


# ### Feature Selection and Engineering
# 
# Ames Housing dataset has 222 features to use for model building. While it is a bliss to have lot of data, it is also a curse. More data means higher cost of maintaining database and collecting data. It is also burden our computing machine, and make time spend to train our model longer
# 
# Not all features need to be included in model, and apparently, some important predictor can only be obtained by processing and manipulating existing features to generate a new one.
# 
# For this section, i will conduct three feature selection method, i.e: 
# - Mutual Information
# - Manual significance test
# - Foward Sequential Feature Selection (FSFS) 
# 
# For a detailed explanation of each method, please read more credible sources. Personally, I recommend articles from [analyticsvidhya.com](https://www.analyticsvidhya.com) and [kdnuggets.com](https://www.kdnuggets.com)
# 
# **Before i perform all selection process**, i will do some feature engineering instinctively. If all process generate result as expected, then i would proceed feature selection with all method i mentioned before.
# 
# At the end of this section, i will have 3 different subsets of data ready to trained with model

# In[168]:


# Feature Engineering

# Year Feature Vs SalePrice
figYr,axYr = plt.subplots(2,2, figsize=(10,8))
sns.lineplot(ax=axYr[0,0],x=scaled_all_num_cat_robust_trainX['YearBuilt'], y = scaled_train_y['SalePrice'])
sns.lineplot(ax=axYr[0,1],x=scaled_all_num_cat_robust_trainX['YearRemodAdd'],y=scaled_train_y['SalePrice'])
sns.lineplot(ax=axYr[1,0],x=scaled_all_num_cat_robust_trainX['MoSold'], y = scaled_train_y['SalePrice'])
sns.lineplot(ax=axYr[1,1],x=scaled_all_num_cat_robust_trainX['YrSold'],y=scaled_train_y['SalePrice'])
plt.show()


# SalePrice has increasing trend against YearBuilt and YearRemodAdd. There are other factors involved like price level that should be considered, but because of the limitation of data, i decide to use this price as my analysis basis. Future research may be conducted by adding inflation level feature so the real price effect can be examined.
# 
# A support basis for this decision comes from Selim, H (2009) that analyzing determinant of house prices in Turkey. Selim,H use similiar features as Ames, Iowa dataset to break down how housing price in Turkey determined by applied and compared predicted result from Hedonic Regression and Artificial Neural Network. Selim,H(2009) also use nominal price selling as target features, and transform it using natural logarithm.
# 
# As opposite, SalePrice seems to have decreasing trend against YrSold. This may have connection to the fact that the sales record only available from 2006 to 2010, a time period around Global Financial Crisis 2009 due to subprime mortgage. This short period of time(4 years of record) is rather unbalance compared to YearBuilt or YearRemodAdd, hence, i probably won't use it as house price determinant because of the earlier reasoning and significant event that occurs at that period.
# 
# Rather than use year feature as it is, i will use age of the house. House age has been use as determinant feature and proved to be statistically significant by some papers, such as [Selim, H(2009)](https://www.sciencedirect.com/science/article/abs/pii/S0957417408000596), and [Xu, Y., Zhang, Q., Zheng, S. et al(2018)](https://www.sciencedirect.com/science/article/abs/pii/S0264837714001884)
# 
# I will generate two features, that is age based on building year and age based on remodelling year, to see the difference and its impact.

# In[405]:


# engineering AgeBuilt
scaled_all_num_cat_robust_trainX['AgeBuilt'] = scaled_all_num_cat_robust_trainX['YrSold'] - scaled_all_num_cat_robust_trainX['YearBuilt']
# engineering AgeRemodAdd
scaled_all_num_cat_robust_trainX['AgeRemodAdd'] = scaled_all_num_cat_robust_trainX['YrSold'] - scaled_all_num_cat_robust_trainX['YearRemodAdd']


# In[406]:


figAge, axAge = plt.subplots(1,2,figsize=(10,5))
sns.lineplot(ax=axAge[0],x=scaled_all_num_cat_robust_trainX['AgeBuilt'],y=scaled_train_y['SalePrice'])
sns.lineplot(ax=axAge[1],x=scaled_all_num_cat_robust_trainX['AgeRemodAdd'],y=scaled_train_y['SalePrice'])
plt.show()


# In[407]:


# Generate the same features in validation data
# engineering AgeBuilt
scaled_all_num_cat_robust_valX['AgeBuilt'] = scaled_all_num_cat_robust_valX['YrSold'] - scaled_all_num_cat_robust_valX['YearBuilt']
# engineering AgeRemodAdd
scaled_all_num_cat_robust_valX['AgeRemodAdd'] = scaled_all_num_cat_robust_valX['YrSold'] - scaled_all_num_cat_robust_valX['YearRemodAdd']


# In[408]:


# for test dataset
scaled_all_num_cat_robust_test['AgeBuilt'] = scaled_all_num_cat_robust_test["YrSold"] - scaled_all_num_cat_robust_test['YearBuilt']
scaled_all_num_cat_robust_test['AgeRemodAdd'] = scaled_all_num_cat_robust_test['YrSold'] - scaled_all_num_cat_robust_test['YearBuilt']


# In[409]:


figAge, axAge = plt.subplots(1,2,figsize=(10,5))
sns.lineplot(ax=axAge[0],x=scaled_all_num_cat_robust_valX['AgeBuilt'],y=scaled_val_y['SalePrice'])
sns.lineplot(ax=axAge[1],x=scaled_all_num_cat_robust_valX['AgeRemodAdd'],y=scaled_val_y['SalePrice'])
plt.show()


# In[410]:


# Scaling Age Features
robustScaler_forAge = RobustScaler()
scaled_age_train_X = pd.DataFrame(robustScaler_forAge.fit_transform(scaled_all_num_cat_robust_trainX[['AgeBuilt','AgeRemodAdd']]),
                                      columns=['AgeBuilt','AgeRemodAdd'])
scaled_age_val_X = pd.DataFrame(robustScaler_forAge.transform(scaled_all_num_cat_robust_valX[['AgeBuilt','AgeRemodAdd']]),
                                      columns=['AgeBuilt','AgeRemodAdd'])


# In[411]:


# Scaling age features for test dataset
scaled_age_test = pd.DataFrame(robustScaler_forAge.transform(scaled_all_num_cat_robust_test[['AgeBuilt','AgeRemodAdd']]),
                              columns=['AgeBuilt','AgeRemodAdd'])


# In[412]:


# Index
scaled_age_train_X.index = scaled_all_num_cat_robust_trainX.index
scaled_age_val_X.index = scaled_all_num_cat_robust_valX.index
scaled_age_train_X


# In[413]:


scaled_age_test.index = scaled_all_num_cat_robust_test.index
scaled_age_test


# In[414]:


# drop unscaling AgeBuilt and AgeRemodAdd

scaled_all_num_cat_robust_trainX = scaled_all_num_cat_robust_trainX.drop(['AgeBuilt','AgeRemodAdd'],axis=1) 
scaled_all_num_cat_robust_valX = scaled_all_num_cat_robust_valX.drop(['AgeBuilt','AgeRemodAdd'],axis=1) 
scaled_all_num_cat_robust_trainX


# In[415]:


# For test dataset
scaled_all_num_cat_robust_test = scaled_all_num_cat_robust_test.drop(['AgeBuilt','AgeRemodAdd'],axis=1) 


# In[416]:


# concat the scaled features
scaled_all_num_cat_robust_trainX = pd.concat([scaled_all_num_cat_robust_trainX,scaled_age_train_X],axis=1)
scaled_all_num_cat_robust_valX = pd.concat([scaled_all_num_cat_robust_valX,scaled_age_val_X],axis=1)
scaled_all_num_cat_robust_trainX


# In[417]:


# For test dataset
scaled_all_num_cat_robust_test = pd.concat([scaled_all_num_cat_robust_test,scaled_age_test],axis=1)
scaled_all_num_cat_robust_test.shape


# In[439]:


# Feature Engineering Using KMeans

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=6) # 6 is abritrary, we can do experiment with different cluster
# For trainX
#scaled_all_num_cat_robust_trainX['Cluster'] = kmeans.fit_predict(scaled_all_num_cat_robust_trainX)
#scaled_all_num_cat_robust_trainX['Cluster'] = scaled_all_num_cat_robust_trainX['Cluster'].astype('category')
scaled_all_num_cat_robust_trainX['Cluster'].value_counts()
# For valX
scaled_all_num_cat_robust_valX['Cluster'] = kmeans.fit_predict(scaled_all_num_cat_robust_valX)
# for test
scaled_all_num_cat_robust_test['Cluster'] = kmeans.fit_predict(scaled_all_num_cat_robust_test)


# In[441]:


# Feature Selection
cat_features.append('Cluster')
# Mutual Information

# Mutual Information treat discrete feature differently with continous feature. First, we need to sort them
disc_features = scaled_all_num_cat_robust_trainX.columns.isin(cat_features).tolist()
# in our case, because of standarization, all value of has been become float64. That's why i sort them by using 
# cat_features and num_features i had created before
disc_features


# In[442]:


# Feature Selection

# Mutual Information

# Mutual Information treat discrete feature differently with continous feature. First, we need to sort them
from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
ames_mi_score = make_mi_scores(scaled_all_num_cat_robust_trainX,scaled_train_y['SalePrice'],discrete_features=disc_features)


# In[443]:


# Top 10 related features
ames_mi_score.head(10)


# In[444]:


# Bottom 10 related features
ames_mi_score.tail(10)


# In[445]:


# To make comparison easier, here the plot bar of all MI score
plt.figure(figsize=(20,15))
sns.barplot(y=ames_mi_score.index[:50],x=ames_mi_score[:50])
plt.show()


# In[464]:


# filter out feature with Mutual Information lower than threshold; abritrary = 0.3

highest_mi_score =ames_mi_score[ames_mi_score > 0.3].index.tolist()
mi_all_num_cat_robust_trainX = scaled_all_num_cat_robust_trainX[highest_mi_score]
mi_all_num_cat_robust_valX = scaled_all_num_cat_robust_valX[highest_mi_score]

mi_all_num_cat_robust_trainX


# In[250]:


# One thing i first notice is : No Location features, ie Neighborhood, include in Top 10 related features.
# I try to visualize it to see if really there is no strong corellation between Neighborhood and SalePrice

sns.boxplot(x='Neighborhood',y='SalePrice',data=ameHouse)
plt.show()


# In[257]:


# Based on some research about real estate, Neighborhood proved to played significant role here. But my mutual_information 
# regression said the opposite.
# Here i decide to do a comparison, a model built using this top features based on mutual information without neighborhood 
# and the same features with additional features from all neighborhood componenet.

# but before i do that, i will continue to perform another feature selection procedure.


# In[447]:


# Manually statistical test
# Statistics Significance test is slassified based on sample value 
plt.figure(figsize=(15,15))
sns.heatmap(scaled_all_num_cat_robust_trainX.corr())
plt.show()


# In[448]:


# Spearman's rank Correlation with response feature
from scipy.stats import spearmanr
rhols = []
pls = []
for col in scaled_all_num_cat_robust_trainX.columns:
    rho, p = spearmanr(scaled_all_num_cat_robust_trainX[col], scaled_train_y['SalePrice'])
    rhols.append(rho)
    pls.append(p)


# In[449]:


# Convert it to dataframe
sign = []
for i in pls:
    if i < 0.05:
        sign.append('Significant')
    else:
        sign.append('Not significant')

spear_corr = pd.DataFrame({
    'Feature' : scaled_all_num_cat_robust_trainX.columns,
    'Corr' : rhols,
    'p-value': pls,
    'Significance': sign
})


# In[450]:


# show the number of significant feature
len(spear_corr.loc[spear_corr['Significance']=='Significant'])


# In[451]:


#145 out of 224 features proved to be statistically significant correlate to SalePrice
#first, let's drop all insignificant features
sig_features = spear_corr.loc[spear_corr['Significance']=='Significant']['Feature'].tolist()
sig_all_num_cat_robust_trainX = scaled_all_num_cat_robust_trainX[sig_features]
sig_all_num_cat_robust_valX = scaled_all_num_cat_robust_valX[sig_features]


# In[473]:


# I will use these 145 features as model building component. 
# but, now i am curious, what if, instead use all of them, i just use features with correlation above certain threshold
# For this experiment, i will use absolute mean of correlation rank as threshold
shapiro(spear_corr.loc[spear_corr['Significance']=='Significant']['Corr'])[1]
threshold_corr = spear_corr.loc[spear_corr['Significance']=='Significant']['Corr'].abs().median()

# filter out the features with correlation below median correlation value
sig_high_features = spear_corr[(spear_corr['Corr'] > threshold_corr) & (spear_corr['Significance']=='Significant')]['Feature'].tolist()
sig_high_num_cat_robust_trainX = scaled_all_num_cat_robust_trainX[sig_high_features]
sig_high_num_cat_robust_valX = scaled_all_num_cat_robust_valX[sig_high_features]


# In[475]:


sig_high_num_cat_robust_trainX.shape
sig_high_num_cat_robust_valX.shape


# In[335]:


# Now, i am already have 6 datasets to train and compared: train-val data with highest MI score, train-val data with
# significant spearman correlation, and train-val data highest and significant spearman correlation


# ### Modelling
# 
# In this section, i'll start to build prediction model using these following algorithms:
# - Random Forest
# - XGBoost
# Here i will train and validate the model quality using cross-validation method, then choose the highest value as the best model and predictor

# In[528]:


# Function to score model
from sklearn.model_selection import cross_val_score

def score_model(X,y,model):
    score= -1 * cross_val_score(model,X,y,cv=5,scoring="neg_mean_absolute_error")
    score_mean = score.mean()
    return score_mean


# In[527]:


# For accuracy scoring purpose, i will concat train and val dataset
# This because cross-validation will shuffle all observation into n-parameters randomly, train-predict the data, 
# and give average score of accurancy from each prediction subsets

# import Random Forest Regressor Module
from sklearn.ensemble import RandomForestRegressor
# import XGBRegressor Module
from xgboost import XGBRegressor

# merge train and validation dataset
forest_model = RandomForestRegressor(n_estimators = 200, random_state=0)
xgboost_model = XGBRegressor(n_estimators =500, learning_rate=0.01, random_state=0)


# In[487]:


#baseline dataset
concat_scaled_all_num_cat_robust = pd.concat([scaled_all_num_cat_robust_trainX,scaled_all_num_cat_robust_valX],axis=0)
concat_scaled_all_num_cat_robust.shape


# In[477]:


# Highest MI dataset
concat_mi_all_num_cat_robust = pd.concat([mi_all_num_cat_robust_trainX,mi_all_num_cat_robust_valX],axis=0)
concat_mi_all_num_cat_robust.shape


# In[478]:


# All Significant Dataset
concat_sig_all_num_cat_robust = pd.concat([sig_all_num_cat_robust_trainX,sig_all_num_cat_robust_valX],axis=0)
concat_sig_all_num_cat_robust.shape


# In[482]:


# High Correlated Significant Dataset
concat_sig_high_num_cat_robust = pd.concat([sig_high_num_cat_robust_trainX,sig_high_num_cat_robust_valX],axis=0)
concat_sig_high_num_cat_robust.shape


# In[490]:


concat_scaled_y = pd.concat([scaled_train_y,scaled_val_y],axis=0)
concat_scaled_y.shape


# In[529]:


print("Mean Absolute Error of baseline dataset in Random Forest Model: {} ".format(score_model(concat_scaled_all_num_cat_robust,
                                                                                      concat_scaled_y['SalePrice'],forest_model)))


# In[530]:


print("Mean Absolute Error of baseline dataset in XGBoost Model: {} ".format(score_model(concat_scaled_all_num_cat_robust,
                                                                                      concat_scaled_y['SalePrice'],xgboost_model)))


# In[531]:


print("Mean Absolute Error of MI_highest dataset in Random Forest Model: {} ".format(score_model(concat_mi_all_num_cat_robust,
                                                                                      concat_scaled_y['SalePrice'],forest_model)))


# In[532]:


print("Mean Absolute Error of MI_highest dataset in XGBoost Model: {} ".format(score_model(concat_mi_all_num_cat_robust,
                                                                                      concat_scaled_y['SalePrice'],xgboost_model)))


# In[533]:


print("Mean Absolute Error of all Significant dataset in Random Forest Model: {} ".format(score_model(concat_sig_all_num_cat_robust,
                                                                                      concat_scaled_y['SalePrice'],forest_model)))


# In[534]:


print("Mean Absolute Error of all Significant dataset in XGBoost Model: {} ".format(score_model(concat_sig_all_num_cat_robust,
                                                                                      concat_scaled_y['SalePrice'],xgboost_model)))


# In[535]:


print("Mean Absolute Error of High Significant dataset in Random Forest Model: {} ".format(score_model(concat_sig_high_num_cat_robust,
                                                                                      concat_scaled_y['SalePrice'],forest_model)))


# In[536]:


print("Mean Absolute Error of High Significant dataset in XGBoost Model: {} ".format(score_model(concat_sig_high_num_cat_robust,
                                                                                      concat_scaled_y['SalePrice'],xgboost_model)))


# In[542]:


# Create test dataset with all significant features only
sig_scaled_all_num_cat_robust_test = scaled_all_num_cat_robust_test[sig_features]


# In[555]:


sig_scaled_all_num_cat_robust_test[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']] = sig_scaled_all_num_cat_robust_test[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].astype(np.int64)
sig_scaled_all_num_cat_robust_test[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].dtypes


# In[581]:


# After assessment, turns out the best model for this case is using all significant dataset, trained by XGBoostRegression model.
# So, i will use this to predict test data
xgboost_model.fit(sig_all_num_cat_robust_trainX,scaled_train_y)
salePrice_predict = xgboost_model.predict(sig_scaled_all_num_cat_robust_test).reshape(-1,1)


# In[595]:


salePrice_predict
inversed_salePrice_predict = robustScaler_forPrice.inverse_transform(salePrice_predict)
inversed_salePrice_predict = inversed_salePrice_predict.tolist()


# In[599]:


flat_salePrice_predict = sum(inversed_salePrice_predict,[])
flat_salePrice_predict


# In[608]:


sub_salePrice_prediction = pd.DataFrame({'Id':ameHouse_test['Id'], 'SalePrice':flat_salePrice_predict})
sub_salePrice_prediction
sub_salePrice_prediction.to_csv("submission_ames_iowa_house_price_preiction.csv", index=False)

