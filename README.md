# Credit Risk Prediction<br/>
Author: Waldy Setiono (waldysetiono@gmail.com)

**Background**: A finance company has been providing loans to people in the past several decades. Sometimes the borrowers can provide repayments without fail, sometimes some other borrowers fail to meet the legal obligations of the loans. Now the company wants to minimize the default rate by approving loan applications more wisely. This project aims to a make predictive model that can help the company assess applicants' decency in getting the loans by learning from past experiences.

**Data**: The data used in this project is from [IBM Github page]((https://github.com/IBM)).

##**Data Wrangling and Exploratory Data Analysis**

**Importing packages**


```python
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
```

**Loading data**


```python
# Load the data
data = pd.read_csv("https://raw.githubusercontent.com/waldysetio/credit-risk/main/data/german_credit_data.csv")
```


```python
# See what the data set looks like
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>CustomerID</th>
      <th>CheckingStatus</th>
      <th>LoanDuration</th>
      <th>CreditHistory</th>
      <th>LoanPurpose</th>
      <th>LoanAmount</th>
      <th>ExistingSavings</th>
      <th>EmploymentDuration</th>
      <th>InstallmentPercent</th>
      <th>Sex</th>
      <th>OthersOnLoan</th>
      <th>CurrentResidenceDuration</th>
      <th>OwnsProperty</th>
      <th>Age</th>
      <th>InstallmentPlans</th>
      <th>Housing</th>
      <th>ExistingCreditsCount</th>
      <th>Job</th>
      <th>Dependents</th>
      <th>Telephone</th>
      <th>ForeignWorker</th>
      <th>Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>713a336c-a255-4e2d-9d57-90b3e99e2f06</td>
      <td>0_to_200</td>
      <td>31</td>
      <td>credits_paid_to_date</td>
      <td>other</td>
      <td>1889</td>
      <td>100_to_500</td>
      <td>less_1</td>
      <td>3</td>
      <td>female</td>
      <td>none</td>
      <td>3</td>
      <td>savings_insurance</td>
      <td>32</td>
      <td>none</td>
      <td>own</td>
      <td>1</td>
      <td>skilled</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
      <td>No Risk</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>140b363f-a3fe-4828-a33f-7284dfdb3969</td>
      <td>less_0</td>
      <td>18</td>
      <td>credits_paid_to_date</td>
      <td>car_new</td>
      <td>462</td>
      <td>less_100</td>
      <td>1_to_4</td>
      <td>2</td>
      <td>female</td>
      <td>none</td>
      <td>2</td>
      <td>savings_insurance</td>
      <td>37</td>
      <td>stores</td>
      <td>own</td>
      <td>2</td>
      <td>skilled</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
      <td>No Risk</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>43b7b51d-5eda-4860-b461-ebef3d3436f4</td>
      <td>less_0</td>
      <td>15</td>
      <td>prior_payments_delayed</td>
      <td>furniture</td>
      <td>250</td>
      <td>less_100</td>
      <td>1_to_4</td>
      <td>2</td>
      <td>male</td>
      <td>none</td>
      <td>3</td>
      <td>real_estate</td>
      <td>28</td>
      <td>none</td>
      <td>own</td>
      <td>2</td>
      <td>skilled</td>
      <td>1</td>
      <td>yes</td>
      <td>no</td>
      <td>No Risk</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>f40eaf08-e6d1-4765-ab20-c5f7faca1635</td>
      <td>0_to_200</td>
      <td>28</td>
      <td>credits_paid_to_date</td>
      <td>retraining</td>
      <td>3693</td>
      <td>less_100</td>
      <td>greater_7</td>
      <td>3</td>
      <td>male</td>
      <td>none</td>
      <td>2</td>
      <td>savings_insurance</td>
      <td>32</td>
      <td>none</td>
      <td>own</td>
      <td>1</td>
      <td>skilled</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
      <td>No Risk</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1728910a-d3ff-4799-ac50-203a3a58a3fb</td>
      <td>no_checking</td>
      <td>28</td>
      <td>prior_payments_delayed</td>
      <td>education</td>
      <td>6235</td>
      <td>500_to_1000</td>
      <td>greater_7</td>
      <td>3</td>
      <td>male</td>
      <td>none</td>
      <td>3</td>
      <td>unknown</td>
      <td>57</td>
      <td>none</td>
      <td>own</td>
      <td>2</td>
      <td>skilled</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
      <td>Risk</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>4995</td>
      <td>e77fa77b-78da-4607-a2fa-ede36c1e968f</td>
      <td>greater_200</td>
      <td>27</td>
      <td>credits_paid_to_date</td>
      <td>furniture</td>
      <td>4650</td>
      <td>less_100</td>
      <td>1_to_4</td>
      <td>3</td>
      <td>male</td>
      <td>none</td>
      <td>4</td>
      <td>savings_insurance</td>
      <td>40</td>
      <td>none</td>
      <td>own</td>
      <td>1</td>
      <td>skilled</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
      <td>No Risk</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>4996</td>
      <td>6e71db4b-375c-42e1-b4a8-3292c007967a</td>
      <td>0_to_200</td>
      <td>11</td>
      <td>prior_payments_delayed</td>
      <td>furniture</td>
      <td>250</td>
      <td>greater_1000</td>
      <td>4_to_7</td>
      <td>3</td>
      <td>male</td>
      <td>none</td>
      <td>3</td>
      <td>car_other</td>
      <td>32</td>
      <td>bank</td>
      <td>own</td>
      <td>1</td>
      <td>unemployed</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
      <td>No Risk</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>4997</td>
      <td>db501d22-e0c9-4f38-bf05-7f2c1df35395</td>
      <td>no_checking</td>
      <td>32</td>
      <td>outstanding_credit</td>
      <td>appliances</td>
      <td>6536</td>
      <td>unknown</td>
      <td>greater_7</td>
      <td>5</td>
      <td>male</td>
      <td>co-applicant</td>
      <td>5</td>
      <td>unknown</td>
      <td>54</td>
      <td>stores</td>
      <td>own</td>
      <td>2</td>
      <td>unskilled</td>
      <td>2</td>
      <td>yes</td>
      <td>yes</td>
      <td>Risk</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>4998</td>
      <td>802055d6-6aa7-4532-bf1c-0b8b114b483d</td>
      <td>0_to_200</td>
      <td>38</td>
      <td>outstanding_credit</td>
      <td>other</td>
      <td>1597</td>
      <td>500_to_1000</td>
      <td>greater_7</td>
      <td>3</td>
      <td>female</td>
      <td>co-applicant</td>
      <td>3</td>
      <td>savings_insurance</td>
      <td>27</td>
      <td>stores</td>
      <td>own</td>
      <td>1</td>
      <td>skilled</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
      <td>Risk</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>4999</td>
      <td>53094239-82f4-4b14-b2e9-7a0355a10839</td>
      <td>less_0</td>
      <td>12</td>
      <td>all_credits_paid_back</td>
      <td>car_new</td>
      <td>4152</td>
      <td>less_100</td>
      <td>1_to_4</td>
      <td>2</td>
      <td>male</td>
      <td>none</td>
      <td>3</td>
      <td>savings_insurance</td>
      <td>29</td>
      <td>none</td>
      <td>own</td>
      <td>2</td>
      <td>skilled</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
      <td>No Risk</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 23 columns</p>
</div>



Let's check if there are missing values.

**Missing values**


```python
# Identify missing values
missing_data = data.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")
```

    Unnamed: 0
    False    5000
    Name: Unnamed: 0, dtype: int64
    
    CustomerID
    False    5000
    Name: CustomerID, dtype: int64
    
    CheckingStatus
    False    5000
    Name: CheckingStatus, dtype: int64
    
    LoanDuration
    False    5000
    Name: LoanDuration, dtype: int64
    
    CreditHistory
    False    5000
    Name: CreditHistory, dtype: int64
    
    LoanPurpose
    False    5000
    Name: LoanPurpose, dtype: int64
    
    LoanAmount
    False    5000
    Name: LoanAmount, dtype: int64
    
    ExistingSavings
    False    5000
    Name: ExistingSavings, dtype: int64
    
    EmploymentDuration
    False    5000
    Name: EmploymentDuration, dtype: int64
    
    InstallmentPercent
    False    5000
    Name: InstallmentPercent, dtype: int64
    
    Sex
    False    5000
    Name: Sex, dtype: int64
    
    OthersOnLoan
    False    5000
    Name: OthersOnLoan, dtype: int64
    
    CurrentResidenceDuration
    False    5000
    Name: CurrentResidenceDuration, dtype: int64
    
    OwnsProperty
    False    5000
    Name: OwnsProperty, dtype: int64
    
    Age
    False    5000
    Name: Age, dtype: int64
    
    InstallmentPlans
    False    5000
    Name: InstallmentPlans, dtype: int64
    
    Housing
    False    5000
    Name: Housing, dtype: int64
    
    ExistingCreditsCount
    False    5000
    Name: ExistingCreditsCount, dtype: int64
    
    Job
    False    5000
    Name: Job, dtype: int64
    
    Dependents
    False    5000
    Name: Dependents, dtype: int64
    
    Telephone
    False    5000
    Name: Telephone, dtype: int64
    
    ForeignWorker
    False    5000
    Name: ForeignWorker, dtype: int64
    
    Risk
    False    5000
    Name: Risk, dtype: int64
    
    

It looks there is no missing value in the data set. Great! Now let's see the data type of each column and how they are correlated with one another. 

**Descriptive Statistics**


```python
# Print the data type
print(data.dtypes)
```

    Unnamed: 0                   int64
    CustomerID                  object
    CheckingStatus              object
    LoanDuration                 int64
    CreditHistory               object
    LoanPurpose                 object
    LoanAmount                   int64
    ExistingSavings             object
    EmploymentDuration          object
    InstallmentPercent           int64
    Sex                         object
    OthersOnLoan                object
    CurrentResidenceDuration     int64
    OwnsProperty                object
    Age                          int64
    InstallmentPlans            object
    Housing                     object
    ExistingCreditsCount         int64
    Job                         object
    Dependents                   int64
    Telephone                   object
    ForeignWorker               object
    Risk                        object
    dtype: object
    


```python
# Find out the correlation among columns
data.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>LoanDuration</th>
      <th>LoanAmount</th>
      <th>InstallmentPercent</th>
      <th>CurrentResidenceDuration</th>
      <th>Age</th>
      <th>ExistingCreditsCount</th>
      <th>Dependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>1.000000</td>
      <td>0.000293</td>
      <td>-0.006283</td>
      <td>0.004290</td>
      <td>0.023128</td>
      <td>0.007087</td>
      <td>0.003582</td>
      <td>-0.001247</td>
    </tr>
    <tr>
      <th>LoanDuration</th>
      <td>0.000293</td>
      <td>1.000000</td>
      <td>0.670614</td>
      <td>0.687898</td>
      <td>0.557946</td>
      <td>0.546914</td>
      <td>0.489787</td>
      <td>0.293867</td>
    </tr>
    <tr>
      <th>LoanAmount</th>
      <td>-0.006283</td>
      <td>0.670614</td>
      <td>1.000000</td>
      <td>0.722800</td>
      <td>0.614925</td>
      <td>0.641352</td>
      <td>0.562614</td>
      <td>0.338788</td>
    </tr>
    <tr>
      <th>InstallmentPercent</th>
      <td>0.004290</td>
      <td>0.687898</td>
      <td>0.722800</td>
      <td>1.000000</td>
      <td>0.657493</td>
      <td>0.617009</td>
      <td>0.516673</td>
      <td>0.324706</td>
    </tr>
    <tr>
      <th>CurrentResidenceDuration</th>
      <td>0.023128</td>
      <td>0.557946</td>
      <td>0.614925</td>
      <td>0.657493</td>
      <td>1.000000</td>
      <td>0.542147</td>
      <td>0.420342</td>
      <td>0.283789</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.007087</td>
      <td>0.546914</td>
      <td>0.641352</td>
      <td>0.617009</td>
      <td>0.542147</td>
      <td>1.000000</td>
      <td>0.536857</td>
      <td>0.316068</td>
    </tr>
    <tr>
      <th>ExistingCreditsCount</th>
      <td>0.003582</td>
      <td>0.489787</td>
      <td>0.562614</td>
      <td>0.516673</td>
      <td>0.420342</td>
      <td>0.536857</td>
      <td>1.000000</td>
      <td>0.335467</td>
    </tr>
    <tr>
      <th>Dependents</th>
      <td>-0.001247</td>
      <td>0.293867</td>
      <td>0.338788</td>
      <td>0.324706</td>
      <td>0.283789</td>
      <td>0.316068</td>
      <td>0.335467</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print basic statistics of the data
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>LoanDuration</th>
      <th>LoanAmount</th>
      <th>InstallmentPercent</th>
      <th>CurrentResidenceDuration</th>
      <th>Age</th>
      <th>ExistingCreditsCount</th>
      <th>Dependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2499.500000</td>
      <td>21.393000</td>
      <td>3480.145000</td>
      <td>2.982400</td>
      <td>2.854200</td>
      <td>35.932400</td>
      <td>1.465800</td>
      <td>1.164600</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1443.520003</td>
      <td>11.162843</td>
      <td>2488.232783</td>
      <td>1.127096</td>
      <td>1.115702</td>
      <td>10.648536</td>
      <td>0.565415</td>
      <td>0.370856</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>250.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>19.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1249.750000</td>
      <td>13.000000</td>
      <td>1326.750000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>28.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2499.500000</td>
      <td>21.000000</td>
      <td>3238.500000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>36.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3749.250000</td>
      <td>29.000000</td>
      <td>5355.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>44.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4999.000000</td>
      <td>64.000000</td>
      <td>11676.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>74.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the statistics including columns with object data type
data.describe(include=['object'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>CheckingStatus</th>
      <th>CreditHistory</th>
      <th>LoanPurpose</th>
      <th>ExistingSavings</th>
      <th>EmploymentDuration</th>
      <th>Sex</th>
      <th>OthersOnLoan</th>
      <th>OwnsProperty</th>
      <th>InstallmentPlans</th>
      <th>Housing</th>
      <th>Job</th>
      <th>Telephone</th>
      <th>ForeignWorker</th>
      <th>Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5000</td>
      <td>5000</td>
      <td>5000</td>
      <td>5000</td>
      <td>5000</td>
      <td>5000</td>
      <td>5000</td>
      <td>5000</td>
      <td>5000</td>
      <td>5000</td>
      <td>5000</td>
      <td>5000</td>
      <td>5000</td>
      <td>5000</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>5000</td>
      <td>4</td>
      <td>5</td>
      <td>11</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>a12cc897-401f-4ad4-869b-1a48acca80bf</td>
      <td>no_checking</td>
      <td>prior_payments_delayed</td>
      <td>car_new</td>
      <td>less_100</td>
      <td>1_to_4</td>
      <td>male</td>
      <td>none</td>
      <td>savings_insurance</td>
      <td>none</td>
      <td>own</td>
      <td>skilled</td>
      <td>none</td>
      <td>yes</td>
      <td>No Risk</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>1993</td>
      <td>1686</td>
      <td>945</td>
      <td>1856</td>
      <td>1470</td>
      <td>3104</td>
      <td>4173</td>
      <td>1660</td>
      <td>3517</td>
      <td>3195</td>
      <td>3400</td>
      <td>2941</td>
      <td>4877</td>
      <td>3330</td>
    </tr>
  </tbody>
</table>
</div>



###**Data Visualization**

Let's see how many customers who have chance of default on the loan and how many customers whose loans will be paid off based on some of the features.


```python
# Functions of risk status plot settings
def plot_stacked_bars(dataframe, title_, size_=(17, 10), rot_=0, legend_="upper right"):
    ax = dataframe.plot(kind="bar",
                        stacked=True,
                        figsize=size_,
                        rot=rot_,
                        title=title_)
    annotate_stacked_bars(ax, textsize=14)
    plt.legend(["No Risk", "Risk"], loc=legend_)
    plt.ylabel("Customer base")
    plt.show()

def annotate_stacked_bars(ax, pad=0.99, colour="white", textsize=13):
    for p in ax.patches:
        value = str(round(p.get_height(),1))
        if value == '0.0':
          continue
        ax.annotate(value,
                    ((p.get_x()+ p.get_width()/2)*pad-0.05, (p.get_y()+p.get_height()/2)*pad),
                    color=colour,
                    size=textsize,
                   )
```

**Credit history**


```python
# Plot risk related to credit history
credit_history = data.groupby([data["CreditHistory"], data["Risk"]])["CustomerID"].count().unstack(level=1).fillna(0)
plot_stacked_bars(credit_history, "Credit History", legend_="upper left")
```


![png](output_22_0.png)


**Loan duration**


```python
# Plot risk related to loan duration
loan_duration = data.groupby([data["LoanDuration"], data["Risk"]])["CustomerID"].count().unstack(level=1).fillna(0)
plot_stacked_bars(loan_duration, "Loan Duration (month)", legend_="upper left")
```


![png](output_24_0.png)


**Job**


```python
# Plot risk related to job
job = data.groupby([data["Job"], data["Risk"]])["CustomerID"].count().unstack(level=1).fillna(0)
plot_stacked_bars(job, "Job", legend_="upper left")
```


![png](output_26_0.png)


**Employment duration**


```python
# Plot risk related to loan amount
employment_duration = data.groupby([data["EmploymentDuration"], data["Risk"]])["CustomerID"].count().unstack(level=1).fillna(0)
plot_stacked_bars(employment_duration, "Employment Duration (year)", legend_="upper left")
```


![png](output_28_0.png)


**Checking status**


```python
# Plot risk related to checking status
checking_status = data.groupby([data["CheckingStatus"], data["Risk"]])["CustomerID"].count().unstack(level=1).fillna(0)
plot_stacked_bars(checking_status, "Checking Status", legend_="upper left")
```


![png](output_30_0.png)


**Loan purpose**


```python
# Plot risk related to loan purpose
loan_purpose = data.groupby([data["LoanPurpose"], data["Risk"]])["CustomerID"].count().unstack(level=1).fillna(0)
plot_stacked_bars(loan_purpose, "Loan Purpose", legend_="upper left")
```


![png](output_32_0.png)


**Existing savings**


```python
# Plot risk related to existing savings
existing_savings = data.groupby([data["ExistingSavings"], data["Risk"]])["CustomerID"].count().unstack(level=1).fillna(0)
plot_stacked_bars(existing_savings, "Existing Savings", legend_="upper left")
```


![png](output_34_0.png)


**Current residence duration**


```python
# Plot risk related to current residence duration
current_residence_duration = data.groupby([data["CurrentResidenceDuration"], data["Risk"]])["CustomerID"].count().unstack(level=1).fillna(0)
plot_stacked_bars(current_residence_duration, "Current Residence Duration", legend_="upper left")
```


![png](output_36_0.png)


**Domestic or foreign worker**


```python
# Plot risk related worker status
foreign_worker = data.groupby([data["ForeignWorker"], data["Risk"]])["CustomerID"].count().unstack(level=1).fillna(0)
plot_stacked_bars(foreign_worker, "Foreign Worker", legend_="upper left")
```


![png](output_38_0.png)


**Housing**


```python
# Plot risk related to housing
housing = data.groupby([data["Housing"], data["Risk"]])["CustomerID"].count().unstack(level=1).fillna(0)
plot_stacked_bars(housing, "Housing", legend_="upper left")
```


![png](output_40_0.png)


**Property ownership**


```python
# Plot risk related to property ownership
owns_property = data.groupby([data["OwnsProperty"], data["Risk"]])["CustomerID"].count().unstack(level=1).fillna(0)
plot_stacked_bars(owns_property, "Owns Property", legend_="upper left")
```


![png](output_42_0.png)


**Age**


```python
# Plot risk related to age
age = data.groupby([data["Age"], data["Risk"]])["CustomerID"].count().unstack(level=1).fillna(0)
plot_stacked_bars(age, "Age", legend_="upper left")
```


![png](output_44_0.png)


###**Data Cleaning**


```python
# Remove column "Unnamed: 0"
data = data.drop(columns=["Unnamed: 0"])
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>CheckingStatus</th>
      <th>LoanDuration</th>
      <th>CreditHistory</th>
      <th>LoanPurpose</th>
      <th>LoanAmount</th>
      <th>ExistingSavings</th>
      <th>EmploymentDuration</th>
      <th>InstallmentPercent</th>
      <th>Sex</th>
      <th>OthersOnLoan</th>
      <th>CurrentResidenceDuration</th>
      <th>OwnsProperty</th>
      <th>Age</th>
      <th>InstallmentPlans</th>
      <th>Housing</th>
      <th>ExistingCreditsCount</th>
      <th>Job</th>
      <th>Dependents</th>
      <th>Telephone</th>
      <th>ForeignWorker</th>
      <th>Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>713a336c-a255-4e2d-9d57-90b3e99e2f06</td>
      <td>0_to_200</td>
      <td>31</td>
      <td>credits_paid_to_date</td>
      <td>other</td>
      <td>1889</td>
      <td>100_to_500</td>
      <td>less_1</td>
      <td>3</td>
      <td>female</td>
      <td>none</td>
      <td>3</td>
      <td>savings_insurance</td>
      <td>32</td>
      <td>none</td>
      <td>own</td>
      <td>1</td>
      <td>skilled</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
      <td>No Risk</td>
    </tr>
    <tr>
      <th>1</th>
      <td>140b363f-a3fe-4828-a33f-7284dfdb3969</td>
      <td>less_0</td>
      <td>18</td>
      <td>credits_paid_to_date</td>
      <td>car_new</td>
      <td>462</td>
      <td>less_100</td>
      <td>1_to_4</td>
      <td>2</td>
      <td>female</td>
      <td>none</td>
      <td>2</td>
      <td>savings_insurance</td>
      <td>37</td>
      <td>stores</td>
      <td>own</td>
      <td>2</td>
      <td>skilled</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
      <td>No Risk</td>
    </tr>
    <tr>
      <th>2</th>
      <td>43b7b51d-5eda-4860-b461-ebef3d3436f4</td>
      <td>less_0</td>
      <td>15</td>
      <td>prior_payments_delayed</td>
      <td>furniture</td>
      <td>250</td>
      <td>less_100</td>
      <td>1_to_4</td>
      <td>2</td>
      <td>male</td>
      <td>none</td>
      <td>3</td>
      <td>real_estate</td>
      <td>28</td>
      <td>none</td>
      <td>own</td>
      <td>2</td>
      <td>skilled</td>
      <td>1</td>
      <td>yes</td>
      <td>no</td>
      <td>No Risk</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f40eaf08-e6d1-4765-ab20-c5f7faca1635</td>
      <td>0_to_200</td>
      <td>28</td>
      <td>credits_paid_to_date</td>
      <td>retraining</td>
      <td>3693</td>
      <td>less_100</td>
      <td>greater_7</td>
      <td>3</td>
      <td>male</td>
      <td>none</td>
      <td>2</td>
      <td>savings_insurance</td>
      <td>32</td>
      <td>none</td>
      <td>own</td>
      <td>1</td>
      <td>skilled</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
      <td>No Risk</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1728910a-d3ff-4799-ac50-203a3a58a3fb</td>
      <td>no_checking</td>
      <td>28</td>
      <td>prior_payments_delayed</td>
      <td>education</td>
      <td>6235</td>
      <td>500_to_1000</td>
      <td>greater_7</td>
      <td>3</td>
      <td>male</td>
      <td>none</td>
      <td>3</td>
      <td>unknown</td>
      <td>57</td>
      <td>none</td>
      <td>own</td>
      <td>2</td>
      <td>skilled</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
      <td>Risk</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>e77fa77b-78da-4607-a2fa-ede36c1e968f</td>
      <td>greater_200</td>
      <td>27</td>
      <td>credits_paid_to_date</td>
      <td>furniture</td>
      <td>4650</td>
      <td>less_100</td>
      <td>1_to_4</td>
      <td>3</td>
      <td>male</td>
      <td>none</td>
      <td>4</td>
      <td>savings_insurance</td>
      <td>40</td>
      <td>none</td>
      <td>own</td>
      <td>1</td>
      <td>skilled</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
      <td>No Risk</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>6e71db4b-375c-42e1-b4a8-3292c007967a</td>
      <td>0_to_200</td>
      <td>11</td>
      <td>prior_payments_delayed</td>
      <td>furniture</td>
      <td>250</td>
      <td>greater_1000</td>
      <td>4_to_7</td>
      <td>3</td>
      <td>male</td>
      <td>none</td>
      <td>3</td>
      <td>car_other</td>
      <td>32</td>
      <td>bank</td>
      <td>own</td>
      <td>1</td>
      <td>unemployed</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
      <td>No Risk</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>db501d22-e0c9-4f38-bf05-7f2c1df35395</td>
      <td>no_checking</td>
      <td>32</td>
      <td>outstanding_credit</td>
      <td>appliances</td>
      <td>6536</td>
      <td>unknown</td>
      <td>greater_7</td>
      <td>5</td>
      <td>male</td>
      <td>co-applicant</td>
      <td>5</td>
      <td>unknown</td>
      <td>54</td>
      <td>stores</td>
      <td>own</td>
      <td>2</td>
      <td>unskilled</td>
      <td>2</td>
      <td>yes</td>
      <td>yes</td>
      <td>Risk</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>802055d6-6aa7-4532-bf1c-0b8b114b483d</td>
      <td>0_to_200</td>
      <td>38</td>
      <td>outstanding_credit</td>
      <td>other</td>
      <td>1597</td>
      <td>500_to_1000</td>
      <td>greater_7</td>
      <td>3</td>
      <td>female</td>
      <td>co-applicant</td>
      <td>3</td>
      <td>savings_insurance</td>
      <td>27</td>
      <td>stores</td>
      <td>own</td>
      <td>1</td>
      <td>skilled</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
      <td>Risk</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>53094239-82f4-4b14-b2e9-7a0355a10839</td>
      <td>less_0</td>
      <td>12</td>
      <td>all_credits_paid_back</td>
      <td>car_new</td>
      <td>4152</td>
      <td>less_100</td>
      <td>1_to_4</td>
      <td>2</td>
      <td>male</td>
      <td>none</td>
      <td>3</td>
      <td>savings_insurance</td>
      <td>29</td>
      <td>none</td>
      <td>own</td>
      <td>2</td>
      <td>skilled</td>
      <td>1</td>
      <td>none</td>
      <td>yes</td>
      <td>No Risk</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 22 columns</p>
</div>




```python
# Checking duplicates
data[data.duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>CheckingStatus</th>
      <th>LoanDuration</th>
      <th>CreditHistory</th>
      <th>LoanPurpose</th>
      <th>LoanAmount</th>
      <th>ExistingSavings</th>
      <th>EmploymentDuration</th>
      <th>InstallmentPercent</th>
      <th>Sex</th>
      <th>OthersOnLoan</th>
      <th>CurrentResidenceDuration</th>
      <th>OwnsProperty</th>
      <th>Age</th>
      <th>InstallmentPlans</th>
      <th>Housing</th>
      <th>ExistingCreditsCount</th>
      <th>Job</th>
      <th>Dependents</th>
      <th>Telephone</th>
      <th>ForeignWorker</th>
      <th>Risk</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



##**Feature Engineering and Selection**

**One-hot encoding**


```python
# Copying data from data to train just in case
train = data
```


```python
# Checking unique values of each column
print(train.apply(lambda col: col.unique()))
```

    CustomerID                  [713a336c-a255-4e2d-9d57-90b3e99e2f06, 140b363...
    CheckingStatus                   [0_to_200, less_0, no_checking, greater_200]
    LoanDuration                [31, 18, 15, 28, 32, 9, 16, 11, 35, 5, 27, 29,...
    CreditHistory               [credits_paid_to_date, prior_payments_delayed,...
    LoanPurpose                 [other, car_new, furniture, retraining, educat...
    LoanAmount                  [1889, 462, 250, 3693, 6235, 9604, 1032, 3109,...
    ExistingSavings             [100_to_500, less_100, 500_to_1000, unknown, g...
    EmploymentDuration            [less_1, 1_to_4, greater_7, 4_to_7, unemployed]
    InstallmentPercent                                         [3, 2, 6, 5, 4, 1]
    Sex                                                            [female, male]
    OthersOnLoan                                  [none, co-applicant, guarantor]
    CurrentResidenceDuration                                   [3, 2, 5, 4, 1, 6]
    OwnsProperty                [savings_insurance, real_estate, unknown, car_...
    Age                         [32, 37, 28, 57, 41, 36, 22, 49, 19, 34, 40, 4...
    InstallmentPlans                                         [none, stores, bank]
    Housing                                                     [own, free, rent]
    ExistingCreditsCount                                             [1, 2, 3, 4]
    Job                         [skilled, management_self-employed, unskilled,...
    Dependents                                                             [1, 2]
    Telephone                                                         [none, yes]
    ForeignWorker                                                       [yes, no]
    Risk                                                          [No Risk, Risk]
    dtype: object
    


```python
# Replacing binary columns values with 0 and 1 
train["Sex"]=train["Sex"].replace(["female", "male"],[0,1])
train["Telephone"]=train["Telephone"].replace(["none", "yes"],[0,1])
train["ForeignWorker"]=train["ForeignWorker"].replace(["no", "yes"],[0,1])
train["Risk"]=train["Risk"].replace(["No Risk", "Risk"],[0,1])
```

###**Make categorical data and dummy variables**


```python
def create_dummy(train_column, categories_prefix):
    # Transform to categorical data type
    train[train_column] = train[train_column].astype("category")

    # Count value of each category
    print(pd.DataFrame({"Samples in category": train[train_column].value_counts()}))

    #Create dummy variables
    categories_column = pd.get_dummies(train[train_column], prefix = categories_prefix)
    return categories_column
```


```python
# Dummy variables for credit history
categories_CreditHistory = create_dummy("CreditHistory", "CH")
categories_CreditHistory
```

                            Samples in category
    prior_payments_delayed                 1686
    credits_paid_to_date                   1490
    outstanding_credit                      938
    all_credits_paid_back                   769
    no_credits                              117
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CH_all_credits_paid_back</th>
      <th>CH_credits_paid_to_date</th>
      <th>CH_no_credits</th>
      <th>CH_outstanding_credit</th>
      <th>CH_prior_payments_delayed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 5 columns</p>
</div>




```python
# Dummy variables for loan purpose
categories_LoanPurpose = create_dummy("LoanPurpose", "LP")
categories_LoanPurpose
```

                Samples in category
    car_new                     945
    furniture                   853
    car_used                    808
    radio_tv                    755
    appliances                  561
    repairs                     283
    vacation                    205
    education                   167
    retraining                  164
    business                    146
    other                       113
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LP_appliances</th>
      <th>LP_business</th>
      <th>LP_car_new</th>
      <th>LP_car_used</th>
      <th>LP_education</th>
      <th>LP_furniture</th>
      <th>LP_other</th>
      <th>LP_radio_tv</th>
      <th>LP_repairs</th>
      <th>LP_retraining</th>
      <th>LP_vacation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 11 columns</p>
</div>




```python
# Dummy variables for CheckingStatus
categories_CheckingStatus = create_dummy("CheckingStatus", "CS")
categories_CheckingStatus
```

                 Samples in category
    no_checking                 1993
    less_0                      1398
    0_to_200                    1304
    greater_200                  305
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CS_0_to_200</th>
      <th>CS_greater_200</th>
      <th>CS_less_0</th>
      <th>CS_no_checking</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 4 columns</p>
</div>




```python
# Dummy variables for Existing Savings
categories_ExistingSavings = create_dummy("ExistingSavings", "ES")
categories_ExistingSavings
```

                  Samples in category
    less_100                     1856
    100_to_500                   1133
    500_to_1000                  1078
    greater_1000                  558
    unknown                       375
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ES_100_to_500</th>
      <th>ES_500_to_1000</th>
      <th>ES_greater_1000</th>
      <th>ES_less_100</th>
      <th>ES_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 5 columns</p>
</div>




```python
# Dummy variables for Employment Duration
categories_EmploymentDuration = create_dummy("EmploymentDuration", "ED")
categories_EmploymentDuration
```

                Samples in category
    1_to_4                     1470
    4_to_7                     1400
    greater_7                   930
    less_1                      904
    unemployed                  296
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ED_1_to_4</th>
      <th>ED_4_to_7</th>
      <th>ED_greater_7</th>
      <th>ED_less_1</th>
      <th>ED_unemployed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 5 columns</p>
</div>




```python
# Dummy variables for Others OnLoan
categories_OthersOnLoan = create_dummy("OthersOnLoan", "OL")
categories_OthersOnLoan
```

                  Samples in category
    none                         4173
    co-applicant                  717
    guarantor                     110
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OL_co-applicant</th>
      <th>OL_guarantor</th>
      <th>OL_none</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 3 columns</p>
</div>




```python
# Dummy variables for Owns Property
categories_OwnsProperty = create_dummy("OwnsProperty", "OP")
categories_OwnsProperty
```

                       Samples in category
    savings_insurance                 1660
    car_other                         1540
    real_estate                       1087
    unknown                            713
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OP_car_other</th>
      <th>OP_real_estate</th>
      <th>OP_savings_insurance</th>
      <th>OP_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 4 columns</p>
</div>




```python
# Dummy variables for Installment Plans
categories_InstallmentPlans = create_dummy("InstallmentPlans", "IP")
categories_InstallmentPlans
```

            Samples in category
    none                   3517
    stores                 1017
    bank                    466
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IP_bank</th>
      <th>IP_none</th>
      <th>IP_stores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 3 columns</p>
</div>




```python
# Dummy variables for Housing
categories_Housing = create_dummy("Housing", "Housing")
categories_Housing
```

          Samples in category
    own                  3195
    rent                 1066
    free                  739
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Housing_free</th>
      <th>Housing_own</th>
      <th>Housing_rent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 3 columns</p>
</div>




```python
# Dummy variables for Existing Credits Count
categories_ExistingCreditsCount = create_dummy("ExistingCreditsCount", "ECC")
categories_ExistingCreditsCount
```

       Samples in category
    1                 2847
    2                 1978
    3                  174
    4                    1
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ECC_1</th>
      <th>ECC_2</th>
      <th>ECC_3</th>
      <th>ECC_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 4 columns</p>
</div>




```python
# Dummy variables for Job
categories_Job = create_dummy("Job", "Job")
categories_Job
```

                              Samples in category
    skilled                                  3400
    unskilled                                 673
    management_self-employed                  641
    unemployed                                286
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Job_management_self-employed</th>
      <th>Job_skilled</th>
      <th>Job_unemployed</th>
      <th>Job_unskilled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 4 columns</p>
</div>



**Merge dummy variables with the main dataframe**


```python
# Merge train and categorical variables using common index
train = pd.merge(train, categories_CreditHistory, left_index=True, right_index=True)
train = pd.merge(train, categories_LoanPurpose, left_index=True, right_index=True)
train = pd.merge(train, categories_CheckingStatus, left_index=True, right_index=True)
train = pd.merge(train, categories_ExistingSavings, left_index=True, right_index=True)
train = pd.merge(train, categories_EmploymentDuration, left_index=True, right_index=True)
train = pd.merge(train, categories_OthersOnLoan, left_index=True, right_index=True)
train = pd.merge(train, categories_OwnsProperty, left_index=True, right_index=True)
train = pd.merge(train, categories_InstallmentPlans, left_index=True, right_index=True)
train = pd.merge(train, categories_Housing, left_index=True, right_index=True)
train = pd.merge(train, categories_ExistingCreditsCount, left_index=True, right_index=True)
train = pd.merge(train, categories_Job, left_index=True, right_index=True)
```


```python
# Drop the variables that has been represented by dummy variables 
train.drop(columns=["CreditHistory", "LoanPurpose", "CheckingStatus", 
                    "ExistingSavings", "EmploymentDuration", "OthersOnLoan", "OwnsProperty",
                    "InstallmentPlans", "Housing", "ExistingCreditsCount", "Job"],inplace=True)
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>LoanDuration</th>
      <th>LoanAmount</th>
      <th>InstallmentPercent</th>
      <th>Sex</th>
      <th>CurrentResidenceDuration</th>
      <th>Age</th>
      <th>Dependents</th>
      <th>Telephone</th>
      <th>ForeignWorker</th>
      <th>Risk</th>
      <th>CH_all_credits_paid_back</th>
      <th>CH_credits_paid_to_date</th>
      <th>CH_no_credits</th>
      <th>CH_outstanding_credit</th>
      <th>CH_prior_payments_delayed</th>
      <th>LP_appliances</th>
      <th>LP_business</th>
      <th>LP_car_new</th>
      <th>LP_car_used</th>
      <th>LP_education</th>
      <th>LP_furniture</th>
      <th>LP_other</th>
      <th>LP_radio_tv</th>
      <th>LP_repairs</th>
      <th>LP_retraining</th>
      <th>LP_vacation</th>
      <th>CS_0_to_200</th>
      <th>CS_greater_200</th>
      <th>CS_less_0</th>
      <th>CS_no_checking</th>
      <th>ES_100_to_500</th>
      <th>ES_500_to_1000</th>
      <th>ES_greater_1000</th>
      <th>ES_less_100</th>
      <th>ES_unknown</th>
      <th>ED_1_to_4</th>
      <th>ED_4_to_7</th>
      <th>ED_greater_7</th>
      <th>ED_less_1</th>
      <th>ED_unemployed</th>
      <th>OL_co-applicant</th>
      <th>OL_guarantor</th>
      <th>OL_none</th>
      <th>OP_car_other</th>
      <th>OP_real_estate</th>
      <th>OP_savings_insurance</th>
      <th>OP_unknown</th>
      <th>IP_bank</th>
      <th>IP_none</th>
      <th>IP_stores</th>
      <th>Housing_free</th>
      <th>Housing_own</th>
      <th>Housing_rent</th>
      <th>ECC_1</th>
      <th>ECC_2</th>
      <th>ECC_3</th>
      <th>ECC_4</th>
      <th>Job_management_self-employed</th>
      <th>Job_skilled</th>
      <th>Job_unemployed</th>
      <th>Job_unskilled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>713a336c-a255-4e2d-9d57-90b3e99e2f06</td>
      <td>31</td>
      <td>1889</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>32</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>140b363f-a3fe-4828-a33f-7284dfdb3969</td>
      <td>18</td>
      <td>462</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>43b7b51d-5eda-4860-b461-ebef3d3436f4</td>
      <td>15</td>
      <td>250</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f40eaf08-e6d1-4765-ab20-c5f7faca1635</td>
      <td>28</td>
      <td>3693</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>32</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1728910a-d3ff-4799-ac50-203a3a58a3fb</td>
      <td>28</td>
      <td>6235</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>57</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>e77fa77b-78da-4607-a2fa-ede36c1e968f</td>
      <td>27</td>
      <td>4650</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>6e71db4b-375c-42e1-b4a8-3292c007967a</td>
      <td>11</td>
      <td>250</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>32</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>db501d22-e0c9-4f38-bf05-7f2c1df35395</td>
      <td>32</td>
      <td>6536</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>54</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>802055d6-6aa7-4532-bf1c-0b8b114b483d</td>
      <td>38</td>
      <td>1597</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>53094239-82f4-4b14-b2e9-7a0355a10839</td>
      <td>12</td>
      <td>4152</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>29</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 62 columns</p>
</div>



###**Visualize features distribution**


```python
fig, axs = plt.subplots(nrows=5, figsize=(9,25))

# Plot histograms
sns.histplot((train["LoanDuration"]), ax=axs[0])
sns.histplot((train["LoanAmount"]), ax=axs[1])
train["InstallmentPercent"].astype(int)
sns.histplot((train["InstallmentPercent"]), discrete=True, ax=axs[2])
train["CurrentResidenceDuration"].astype(int)
sns.histplot((train["CurrentResidenceDuration"]), discrete=True, ax=axs[3])
sns.histplot((train["Age"]), ax=axs[4])
plt.show()
```


![png](output_70_0.png)


Let's see the distribution through boxplot to spot outliers.


```python
# Plot LoanDuration
sns.boxplot(x=train["LoanDuration"], data=train["LoanDuration"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f2b9d682a10>




![png](output_72_1.png)



```python
# Plot LoanAmount
sns.boxplot(data=train["LoanAmount"], x=train["LoanAmount"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f2b9d519590>




![png](output_73_1.png)



```python
# Plot InstallmentPercent 
sns.boxplot(data=train["InstallmentPercent"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f2b9d64f350>




![png](output_74_1.png)



```python
# Plot CurrentResidenceDuration
sns.boxplot(data=train["CurrentResidenceDuration"], x=train["CurrentResidenceDuration"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f2b9d7caa50>




![png](output_75_1.png)



```python
# Plot Age
sns.boxplot(data=train["Age"], x=train["Age"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f2b9d571a90>




![png](output_76_1.png)


**Calculate the correlation of the variables**


```python
correlation = train.corr()
```


```python
# Plot correlation
plt.figure(figsize=(23,20))
sns.heatmap(correlation, xticklabels=correlation.columns.values,
yticklabels=correlation.columns.values, annot = True, annot_kws={'size':10})
# Axis ticks size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
```


![png](output_79_0.png)


**Address multicollinearity**

There seems to be multicollinearity among predictors as we can see in the heat map above. Let's remove the features that can relatively be represented by other predictor. For example, ECC_1 and ECC_2 are highly correlated hence we're going to remove ECC_1 and rename ECC_2 as ECC_1_or_2. We will do the same thing to OL_None with OL_Coapplicant.


```python
# Rename ECC_2 and OL_co-applicant
train.rename(columns={'ECC_2': 'ECC_1_or_2', 'OL_co-applicant': 'OL_co-applicant_or_none'}, inplace=True)
```


```python
# Drop column ECC_1 and OL_none
train.drop(['ECC_1', 'OL_none'], axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>LoanDuration</th>
      <th>LoanAmount</th>
      <th>InstallmentPercent</th>
      <th>Sex</th>
      <th>CurrentResidenceDuration</th>
      <th>Age</th>
      <th>Dependents</th>
      <th>Telephone</th>
      <th>ForeignWorker</th>
      <th>Risk</th>
      <th>CH_all_credits_paid_back</th>
      <th>CH_credits_paid_to_date</th>
      <th>CH_no_credits</th>
      <th>CH_outstanding_credit</th>
      <th>CH_prior_payments_delayed</th>
      <th>LP_appliances</th>
      <th>LP_business</th>
      <th>LP_car_new</th>
      <th>LP_car_used</th>
      <th>LP_education</th>
      <th>LP_furniture</th>
      <th>LP_other</th>
      <th>LP_radio_tv</th>
      <th>LP_repairs</th>
      <th>LP_retraining</th>
      <th>LP_vacation</th>
      <th>CS_0_to_200</th>
      <th>CS_greater_200</th>
      <th>CS_less_0</th>
      <th>CS_no_checking</th>
      <th>ES_100_to_500</th>
      <th>ES_500_to_1000</th>
      <th>ES_greater_1000</th>
      <th>ES_less_100</th>
      <th>ES_unknown</th>
      <th>ED_1_to_4</th>
      <th>ED_4_to_7</th>
      <th>ED_greater_7</th>
      <th>ED_less_1</th>
      <th>ED_unemployed</th>
      <th>OL_co-applicant_or_none</th>
      <th>OL_guarantor</th>
      <th>OP_car_other</th>
      <th>OP_real_estate</th>
      <th>OP_savings_insurance</th>
      <th>OP_unknown</th>
      <th>IP_bank</th>
      <th>IP_none</th>
      <th>IP_stores</th>
      <th>Housing_free</th>
      <th>Housing_own</th>
      <th>Housing_rent</th>
      <th>ECC_1_or_2</th>
      <th>ECC_3</th>
      <th>ECC_4</th>
      <th>Job_management_self-employed</th>
      <th>Job_skilled</th>
      <th>Job_unemployed</th>
      <th>Job_unskilled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>713a336c-a255-4e2d-9d57-90b3e99e2f06</td>
      <td>31</td>
      <td>1889</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>32</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>140b363f-a3fe-4828-a33f-7284dfdb3969</td>
      <td>18</td>
      <td>462</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>43b7b51d-5eda-4860-b461-ebef3d3436f4</td>
      <td>15</td>
      <td>250</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f40eaf08-e6d1-4765-ab20-c5f7faca1635</td>
      <td>28</td>
      <td>3693</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>32</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1728910a-d3ff-4799-ac50-203a3a58a3fb</td>
      <td>28</td>
      <td>6235</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>57</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>e77fa77b-78da-4607-a2fa-ede36c1e968f</td>
      <td>27</td>
      <td>4650</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>6e71db4b-375c-42e1-b4a8-3292c007967a</td>
      <td>11</td>
      <td>250</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>32</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>db501d22-e0c9-4f38-bf05-7f2c1df35395</td>
      <td>32</td>
      <td>6536</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>54</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>802055d6-6aa7-4532-bf1c-0b8b114b483d</td>
      <td>38</td>
      <td>1597</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>53094239-82f4-4b14-b2e9-7a0355a10839</td>
      <td>12</td>
      <td>4152</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>29</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 60 columns</p>
</div>



###**Remove outliers**

Let's replace outliers with the mean values using Z score.


```python
def replace_outliers_z_score(dataframe, column, Z=3):

    from scipy.stats import zscore
    df = dataframe.copy(deep=True)
    df.dropna(inplace=True, subset=[column])

    # Calculate mean without outliers
    df["zscore"] = zscore(df[column])
    mean_ = df[(df["zscore"] > -Z) & (df["zscore"] < Z)][column].mean()

    # Replace with mean values
    dataframe[column] = dataframe[column].fillna(mean_)
    dataframe["zscore"] = zscore(dataframe[column])
    no_outliers = dataframe[(dataframe["zscore"] < -Z) | (dataframe["zscore"] > Z)].shape[0]
    dataframe.loc[(dataframe["zscore"] < -Z) | (dataframe["zscore"] > Z),column] = mean_
    
    # Print message
    print("Replaced:", no_outliers, " outliers in ", column)
    return dataframe.drop(columns="zscore")
```


```python
# Replace outliers with mean values
train = replace_outliers_z_score(train,"LoanDuration")
train = replace_outliers_z_score(train,"LoanAmount")
train = replace_outliers_z_score(train,"InstallmentPercent")
train = replace_outliers_z_score(train,"CurrentResidenceDuration")
```

    Replaced: 11  outliers in  LoanDuration
    Replaced: 4  outliers in  LoanAmount
    Replaced: 0  outliers in  InstallmentPercent
    Replaced: 0  outliers in  CurrentResidenceDuration
    


```python
# Reset index of the dataframe
train.reset_index(drop=True, inplace=True)
```

**Plot the distribution after removing outliers**


```python
# Plot LoanDuration
sns.boxplot(x=train["LoanDuration"], data=train["LoanDuration"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f2b9b47fb90>




![png](output_90_1.png)


As we can see, the outliers of more than 60 in loan duration has been replaced and there are still ones slightly above 50 but it is due to the Z score threshold of 3.


```python
# Plot LoanAmount
sns.boxplot(data=train["LoanAmount"], x=train["LoanAmount"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f2b9b45bd90>




![png](output_92_1.png)


##**Modeling and Evaluation**

**Check the dataframe**


```python
# See the columns of the dataframe
pd.DataFrame({"Dataframe columns": train.columns})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dataframe columns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CustomerID</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LoanDuration</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LoanAmount</td>
    </tr>
    <tr>
      <th>3</th>
      <td>InstallmentPercent</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sex</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>57</th>
      <td>ECC_4</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Job_management_self-employed</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Job_skilled</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Job_unemployed</td>
    </tr>
    <tr>
      <th>61</th>
      <td>Job_unskilled</td>
    </tr>
  </tbody>
</table>
<p>62 rows × 1 columns</p>
</div>



**Split the data**


```python
# Split the data into "risk" as response and the rest as features
y = train["Risk"]
X = train.drop(labels = ["CustomerID","Risk"], axis = 1)
```


```python
# Split the data into training and validation data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)
```

###**Modeling**


```python
# Import algorithms and metrics evaluation packages
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
```


```python
# Function for evaluating model
def evaluate(model_, X_test_, y_test_): 

    # Get the model predictions
    prediction_test_ = model_.predict(X_test_)

    # Print the evaluation metrics as pandas dataframe 
    results = pd.DataFrame({"Accuracy" : [accuracy_score(y_test_, prediction_test_)], 
                            "Precision" : [precision_score(y_test_, prediction_test_)], 
                            "Recall" : [recall_score(y_test_, prediction_test_)],
                            "F_score" : f1_score(y_test_, prediction_test_, average='binary')})
    return results
```

**Logistic Regression**


```python
# Fit features train to target train using Logistic Regression
LogResmodel = LogisticRegression().fit(X_train, y_train)
LogResmetrics = evaluate(LogResmodel, X_test, y_test)
LogResmetrics
```

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.8152</td>
      <td>0.755952</td>
      <td>0.630273</td>
      <td>0.687415</td>
    </tr>
  </tbody>
</table>
</div>



**Random Forest**


```python
# Fit features train to target train using Random Forest
RFmodel = RandomForestClassifier().fit(X_train, y_train)
RFmetrics = evaluate(RanFormodel, X_test, y_test)
RFmetrics
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.8112</td>
      <td>0.779264</td>
      <td>0.578164</td>
      <td>0.663818</td>
    </tr>
  </tbody>
</table>
</div>



**K-NN**


```python
# Fit features train to target train using KNN
KNNmodel = KNeighborsClassifier(n_neighbors=8).fit(X_train, y_train)
KNNmetrics = evaluate(KNNmodel, X_test, y_test)
KNNmetrics
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.7376</td>
      <td>0.636364</td>
      <td>0.434243</td>
      <td>0.516224</td>
    </tr>
  </tbody>
</table>
</div>



**Gaussian Naive Bayes**


```python
# Fit features train to target train using Gaussian Naive Bayes
GaussianNBmodel = GaussianNB().fit(X_train, y_train)
GaussianNBmetrics = evaluate(GaussianNBmodel, X_test, y_test)
GaussianNBmetrics
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.744</td>
      <td>0.581854</td>
      <td>0.73201</td>
      <td>0.648352</td>
    </tr>
  </tbody>
</table>
</div>



**Gradient Boosting Classifier**


```python
# Fit features train to target train using Gradient Boosting Classifier
GBmodel = GradientBoostingClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
GBmetrics = evaluate(GBCmodel, X_test, y_test)
GBmetrics
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.8096</td>
      <td>0.752294</td>
      <td>0.610422</td>
      <td>0.673973</td>
    </tr>
  </tbody>
</table>
</div>



**Extreme Gradient Boosting**


```python
# Fit features train to target train using XGBoost
XGBmodel = xgb.XGBClassifier(learning_rate=0.1,max_depth=6,n_estimators=500,n_jobs=-1).fit(X_train, y_train)
XGBmetrics = evaluate(XGBmodel, X_test, y_test)
XGBmetrics
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.776</td>
      <td>0.680352</td>
      <td>0.575682</td>
      <td>0.623656</td>
    </tr>
  </tbody>
</table>
</div>



**Support Vector Classification**


```python
SVCmodel = svm.SVC().fit(X_train, y_train)
SVCmetrics = evaluate(SVMmodel, X_test, y_test)
SVCmetrics
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.7352</td>
      <td>0.640625</td>
      <td>0.406948</td>
      <td>0.497724</td>
    </tr>
  </tbody>
</table>
</div>



**Decision Tree**


```python
DTmodel = tree.DecisionTreeClassifier().fit(X_train, y_train)
DTmetrics = evaluate(DTmodel, X_test, y_test)
DTmetrics
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.7232</td>
      <td>0.565517</td>
      <td>0.610422</td>
      <td>0.587112</td>
    </tr>
  </tbody>
</table>
</div>



**Multi-layer Perceptron**


```python
MLPmodel = MLPClassifier().fit(X_train, y_train)
MLPmetrics = evaluate(MLPmodel, X_test, y_test)
MLPmetrics
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.736</td>
      <td>0.566972</td>
      <td>0.766749</td>
      <td>0.651899</td>
    </tr>
  </tbody>
</table>
</div>



###**Models Comparison**


```python
# Function to make a dataframe of sorted metrics
def metricscomparison(modelmetrics_):

    # Make a dataframe containing models and their respective metrics
    models = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", 
                  "KNN", "Gaussian Naive Bayes", 
                  "Gradient Boosting", "XGBoost", 
                  "Support Vector Classification", "Decision Tree", 
                  "Multi-layer Perceptron"],
        modelmetrics_: [
            LogResmetrics[modelmetrics_].values, 
            RFmetrics[modelmetrics_].values,  
            KNNmetrics[modelmetrics_].values, 
            GaussianNBmetrics[modelmetrics_].values, 
            GBmetrics[modelmetrics_].values, 
            XGBmetrics[modelmetrics_].values,
            SVCmetrics[modelmetrics_].values,
            DTmetrics[modelmetrics_].values,
            MLPmetrics[modelmetrics_].values
        ]})

    # Change metrics data type to float
    models[modelmetrics_] = models[modelmetrics_].astype(float)

    # Short models based on metrics
    models = models.sort_values(by=modelmetrics_, ascending=False)
    return models
```


```python
# Function to plot sorted metrics
def metricsplot(metrics_column, metrics_dataframe):
    # metricscolumn e.g. dataframe.accuracy
    # metrics_dataframe e.g. dataframe

    # Set barplot attributes
    plt.figure(figsize=(15,5))
    ax = sns.barplot(x=metrics_column, y='Model', data=metrics_dataframe, palette='Blues_r')
    y = metrics_column

    # Adjust the barplot size and number rounding
    for i, v in enumerate(y):
        ax.text(v+0.003, i+.1, str(round(v,2)), fontweight='bold')

    # Plot sorted metrics of the models
    plt.title('Models Evaluation')
    plt.show()
```

**Accuracy**


```python
# Call function of models sorting
accuracy_comparison = metricscomparison('Accuracy')

# Call function of ploting sorted models
metricsplot(accuracy_comparison.Accuracy, accuracy_comparison)
```


![png](output_124_0.png)


**Precision**


```python
# Call function of models sorting
precision_comparison = metricscomparison('Precision')

# Call function of ploting sorted models
metricsplot(precision_comparison.Precision, precision_comparison)
```


![png](output_126_0.png)


**Recall**


```python
# Call function of models sorting
recall_comparison = metricscomparison('Recall')

# Call function of ploting sorted models
metricsplot(recall_comparison.Recall, recall_comparison)
```


![png](output_128_0.png)


**F-score**


```python
# Call function of models sorting
fscore_comparison = metricscomparison('F_score')

# Call function of ploting sorted models
metricsplot(fscore_comparison.F_score, fscore_comparison)
```


![png](output_130_0.png)


###**Choose a Model**

As we can see, there is no single algorithm that tops the other in all metrics but Logistic Regression is always included in the top three in every category. We can choose a model based on what metrics is more important for us. For example, if our focus is to gather positivity correctly (like covid cases) then the model with high sensitivity/recall will suit. If our focus is to know the absence of something then we will prefer model with higher specificity. There are many more possibilities of consideration that can be used related to metrics in choosing the best algorithm for a given problem. In this occasion **Logistic Regression will be the model of this project**.

Let's see the detail of how Logistic Regression make predictions in this project.

**Confussion Matrix**


```python
# Import confussion matrix library
from sklearn.metrics import plot_confusion_matrix

# Plot confussion matrix
class_names = ['0', '1']
disp = plot_confusion_matrix(LogResmodel, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 values_format = '.0f')
plt.grid(False)  
plt.show(disp)
```


![png](output_135_0.png)


**ROC and AUC**


```python
def calculate_roc_auc(model_, X_test_, y_test_):
    """
    Evaluate the roc-auc score
    """
    # Get the model predictions
    # We are using the prediction for the class 1 -> risk
    prediction_test_ = model_.predict_proba(X_test_)[:,1] 
    
    # Compute roc-auc
    fpr, tpr, thresholds = metrics.roc_curve(y_test_, prediction_test_)

    # Print the evaluation metrics as pandas dataframe
    score = pd.DataFrame({"ROC-AUC" : [metrics.auc(fpr, tpr)]}) 
   
    return fpr, tpr, score
```


```python
def plot_roc_auc(fpr,tpr): 
    """
    Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates.
    """
    # Initialize plot
    f, ax = plt.subplots(figsize=(14,8)) # Plot ROC
    
    # Plot ROC
    roc_auc = metrics.auc(fpr, tpr) 
    ax.plot(fpr, tpr, lw=2, alpha=0.3, label="AUC = %0.2f" % (roc_auc)) 

    # Plot the random line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color='r', label="Random", alpha=.8)
  
    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05]) 
    ax.set_ylim([-0.05, 1.05]) 
    ax.set_xlabel("False Positive Rate (FPR)") 
    ax.set_ylabel("True Positive Rate (TPR)") 
    ax.set_title("ROC-AUC") 
    ax.legend(loc="lower right")
    plt.show()
```


```python
# Calculate AUC score
fpr, tpr, auc_score = calculate_roc_auc(LogResmodel, X_test, y_test)
auc_score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ROC-AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.857992</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot ROC AUC
plot_roc_auc(fpr, tpr)
plt.show()
```


![png](output_140_0.png)

