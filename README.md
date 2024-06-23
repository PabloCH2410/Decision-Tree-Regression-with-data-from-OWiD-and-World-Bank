# Decision-Tree-Regression-with-data-from-OWiD-and-World-Bank
With this project we will use Python to extract data from Our World in Data and the World Bank through APIs, combine it and generate different insights through exploratory analysis and the implementation of a decision tree for regression.

Specifically, we will use the owid libraries for the Our World in Data (owid-catalog), and wbgapi for the World Bank data. Both libraries provide APIs that allow us to import the data catalogue from these sites into our project and use them as dataframes.

## Contents

- [Libraries](#libraries)
- [Connection](#connection)
- [Combining datasets](#combining-datasets)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Decision Tree Regression](#decision-tree-regression)
- [Comparison between models and conclusions](#comparison-between-models-and-conclusions)
- [References](#references)

## Libraries

### Data manipulation
```python
import pandas as pd
import numpy as np
from collections import Counter
```

### API connection
```python
from owid import catalog #Our World in Data API
import wbgapi as wb #World Bank data API
```

### Modeling
```python
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_text
```

### Graphs
```python
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.ticker as ticker
import seaborn as sns
```

### Graphs configuration
```python
plt.rcParams['image.cmap'] = "bwr" #change default colorbar
style.use('ggplot') or plt.style.use('ggplot') #graphs style: ggplot
```

## Connection

### Our World in Data

First, we are going to do the connection with Our World in Data
```python
owid_catalog = catalog.find() #To see all the catalog of OWiD

 #This is the way to go through 50 by 50 lines (or more) of the catalogue until you get the dataset you want.
print(owid_catalog.iloc[1000:1050])

                      table  ...    formats
1000    _3_9_3_sh_sta_poisn  ...  [feather]
1001     _3_a_1_sh_prv_smok  ...  [feather]
1002     _3_a_1_sh_prv_smok  ...  [feather]
1003     _3_b_1_sh_acs_dtp3  ...  [feather]
1004     _3_b_1_sh_acs_dtp3  ...  [feather]
1005      _3_b_1_sh_acs_hpv  ...  [feather]
...
[50 rows x 9 columns]

owid_catalog.columns

Index(['table', 'dataset', 'version', 'namespace', 'channel', 'is_public',
       'dimensions', 'path', 'formats'],
      dtype='object')
```
We find the datasets we want and then we load them. In this case we want to get data on renewable energy and fossil fuel production.
```python
catalog.find('renewable_electricity_capacity')
catalog.find('fossil_fuel_production')

#the .iloc[0] because there are two datasets containing the word renewable_electricity_capacity
renewable_electricity_capacity = catalog.find('renewable_electricity_capacity').iloc[0].load().reset_index()

#the .iloc[0] because there are two datasets containing the word fossil_fuel_production
fossil_fuel_production = catalog.find('fossil_fuel_production').iloc[0].load().reset_index()
```

```python
catalog.find('renewable_electricity_capacity')
catalog.find('fossil_fuel_production')

#the .iloc[0] because there are two datasets containing the word renewable_electricity_capacity
renewable_electricity_capacity = catalog.find('renewable_electricity_capacity').iloc[0].load().reset_index()

renewable_electricity_capacity.head()

       country  year  ...  total_renewable_energy  wind_energy
   Afghanistan  2000  ...              191.503006          NaN
   Afghanistan  2001  ...              191.503006          NaN
   Afghanistan  2002  ...              191.505997          NaN
   Afghanistan  2003  ...              191.619003          NaN
   Afghanistan  2004  ...              191.718002          NaN
[5 rows x 21 columns]

renewable_electricity_capacity.columns

Index(['country', 'year', 'bagasse', 'bioenergy', 'biogas',
       'concentrated_solar_power', 'geothermal_energy', 'hydropower',
       'liquid_biofuels', 'marine_energy', 'offshore_wind_energy',
       'onshore_wind_energy', 'other_solid_biofuels', 'pure_pumped_storage',
       'renewable_hydropower_including_mixed_plants',
       'renewable_municipal_waste', 'solar_energy', 'solar_photovoltaic',
       'solid_biofuels_and_renewable_waste', 'total_renewable_energy',
       'wind_energy'],
      dtype='object')

#the .iloc[0] because there are two datasets containing the word fossil_fuel_production
fossil_fuel_production = catalog.find('fossil_fuel_production').iloc[0].load().reset_index()

fossil_fuel_production.head()

       country  ...  gas_production_per_capita__kwh
   Afghanistan  ...                             0.0
   Afghanistan  ...                             0.0
   Afghanistan  ...                             0.0
   Afghanistan  ...                             0.0
   Afghanistan  ...                             0.0
[5 rows x 14 columns]

fossil_fuel_production.columns

Index(['country', 'year', 'coal_production__twh', 'gas_production__twh',
       'oil_production__twh', 'annual_change_in_coal_production__pct',
       'annual_change_in_coal_production__twh',
       'annual_change_in_oil_production__pct',
       'annual_change_in_oil_production__twh',
       'annual_change_in_gas_production__pct',
       'annual_change_in_gas_production__twh',
       'coal_production_per_capita__kwh', 'oil_production_per_capita__kwh',
       'gas_production_per_capita__kwh'],
      dtype='object')
```
With these two dataframes we have data on renewable energy capacity per country and fossil fuel production capacity per country. The first dataframe gives information on numerous renewable energy sources, while in the second dataset the most relevant columns are coal, gas and oil production.

### World Bank

Second, we do the connection with the World Bank
```python
#We search for all the datasets related to climate change
wb.search('climate change')

========
Series: AG.LND.ARBL.HA

Developmentrelevance: ...are intrinsically linked to global challenges of food insecurity and poverty, climate change adaptation and mitigation, as well as degradation and depletion of natural...
========

#World Bank Series: AG.LND.ARBL.HA, Arable land (hectares)

wb_AL = wb.data.DataFrame('AG.LND.ARBL.HA')

wb_AL.head()

         YR1960     YR1961     YR1962  ...     YR2021  YR2022  YR2023
economy                                ...                           
ABW         NaN     2000.0     2000.0  ...     2000.0     NaN     NaN
AFE         NaN        NaN        NaN  ...        NaN     NaN     NaN
AFG         NaN  7650000.0  7700000.0  ...  7829000.0     NaN     NaN
AFW         NaN        NaN        NaN  ...        NaN     NaN     NaN
AGO         NaN  2670000.0  2700000.0  ...  5373000.0     NaN     NaN
[5 rows x 64 columns]

#economy is the index of this dataframe
wb_AL.columns

Index(['YR1960', 'YR1961', 'YR1962', 'YR1963', 'YR1964', 'YR1965', 'YR1966',
       'YR1967', 'YR1968', 'YR1969', 'YR1970', 'YR1971', 'YR1972', 'YR1973',
       'YR1974', 'YR1975', 'YR1976', 'YR1977', 'YR1978', 'YR1979', 'YR1980',
       'YR1981', 'YR1982', 'YR1983', 'YR1984', 'YR1985', 'YR1986', 'YR1987',
       'YR1988', 'YR1989', 'YR1990', 'YR1991', 'YR1992', 'YR1993', 'YR1994',
       'YR1995', 'YR1996', 'YR1997', 'YR1998', 'YR1999', 'YR2000', 'YR2001',
       'YR2002', 'YR2003', 'YR2004', 'YR2005', 'YR2006', 'YR2007', 'YR2008',
       'YR2009', 'YR2010', 'YR2011', 'YR2012', 'YR2013', 'YR2014', 'YR2015',
       'YR2016', 'YR2017', 'YR2018', 'YR2019', 'YR2020', 'YR2021', 'YR2022',
       'YR2023'],
      dtype='object')
```

We need to do a pivot loger to get a Year column
```python
wb_AL_columns = list(wb_AL.columns)

wb_AL = wb.data.DataFrame('AG.LND.ARBL.HA').reset_index() #we do a reset_index() because the index is 'economy' by default

wb_AL = pd.melt(wb_AL, id_vars='economy', value_vars=wb_AL_columns)
```
Know, we use this total population dataset to know the name of every country code, so we use it in order to extract every country code from World Bank
```python
total_pop = wb.data.DataFrame('SP.POP.TOTL', time=2015, labels=True).reset_index()
total_pop = total_pop.drop('SP.POP.TOTL',axis=1)
```
Finally, we do a inner join between these two datasets
```python
arable_land = pd.merge(wb_AL,total_pop,on='economy') #deafault: inner join

arable_land['variable'] = arable_land['variable'].str[2:7] #to not take YR (wich appears in every column)
arable_land.rename(columns = {'variable':'year'}, inplace = True)
arable_land.rename(columns = {'value':'hectares'}, inplace = True)
```

## Combining datasets

Now we have all the data we want for our analysis. All these datasets share country and year variable. Thus, we can create a dataset combining these three datasets and take just into account the most recent year common to all by country. First, lets see how many countries there are in each dataset
```python
np.unique(arable_land['Country'].values)
Counter(np.unique(arable_land['Country'].values)) #how many unique values do we have in column Country from arable_land
len(np.unique(arable_land['Country'].values)) #We have 266 countries in our dataset

arable_land_countries = [country for country, df in 
                         arable_land.groupby('Country')]

###

np.unique(fossil_fuel_production['country'].values)
Counter(np.unique(fossil_fuel_production['country'].values))
len(np.unique(fossil_fuel_production['country'].values)) #We have 257 countries in our dataset

fossil_fuel_production_countries = [country for country, df in 
                         fossil_fuel_production.groupby('country')]

###

np.unique(renewable_electricity_capacity['country'].values)
Counter(np.unique(renewable_electricity_capacity['country'].values))
len(np.unique(renewable_electricity_capacity['country'].values)) #We have 244 countries in our dataset

renewable_electricity_capacity_countries = [country for country, df in 
                         renewable_electricity_capacity.groupby('country')]
```
This is a way to know how many unique values we have in the Country column of each dataset we have imported from the API. That is, how many countries we have per dataset. In addition, we create lists with these unique values per country (e.g. fossil_fuel_production_countries).

However, this is repetitive code, so we can group it all together in one function and apply it to get the same information.
```python
def unique_countries_info(df, column_name):
    unique_countries = np.unique(df[column_name].values)
    count_unique = len(unique_countries)
    return unique_countries, Counter(unique_countries), count_unique, list(df.groupby(column_name).groups.keys())

# Arable Land
arable_land_unique, arable_land_counter, arable_land_count, arable_land_countries = unique_countries_info(arable_land, 'Country')
print(f'We have {arable_land_count} countries in our arable_land dataset')

We have 266 countries in our arable_land dataset

# Fossil Fuel Production
fossil_fuel_unique, fossil_fuel_counter, fossil_fuel_count, fossil_fuel_countries = unique_countries_info(fossil_fuel_production, 'country')
print(f'We have {fossil_fuel_count} countries in our fossil_fuel_production dataset')

We have 257 countries in our fossil_fuel_production dataset

# Renewable Electricity Capacity
renewable_electricity_unique, renewable_electricity_counter, renewable_electricity_count, renewable_electricity_countries = unique_countries_info(renewable_electricity_capacity, 'country')
print(f'We have {renewable_electricity_count} countries in our renewable_electricity_capacity dataset')

We have 244 countries in our renewable_electricity_capacity dataset
```
Now, these datasets also have a time variable, so we need to know what years they share between them in order to combine them.

We want to take into account the most recent year: we will take 2016 as the three dataset share this year.
```python
np.unique(arable_land['year'].values) #1960 - 2021
np.unique(fossil_fuel_production['year'].values) #1900 - 2022
np.unique(renewable_electricity_capacity['year'].values) #2000 - 2021

arable_land_2016 = arable_land.loc[arable_land['year'] == "2016"]
arable_land_2016 = arable_land_2016.drop('year',axis=1)

fossil_fuel_production_2016 = fossil_fuel_production.loc[fossil_fuel_production['year'] == 2016]
fossil_fuel_production_2016 = fossil_fuel_production_2016.drop('year',axis=1)

renewable_electricity_capacity_2016 = renewable_electricity_capacity.loc[renewable_electricity_capacity['year'] == 2016]
renewable_electricity_capacity_2016 = renewable_electricity_capacity_2016.drop('year',axis=1)
```
Now, we are going to do an inner join to join these datasets in one
```python
arable_land_2016.rename(columns = {'Country':'country'}, inplace = True)

dataset_2016 = arable_land_2016.merge(fossil_fuel_production_2016,on='country').merge(renewable_electricity_capacity_2016,on='country')
dataset_2016 = dataset_2016.drop('economy',axis=1)

dataset_2016 = dataset_2016[~dataset_2016['country'].isin(['World'])] #we "delete" World values
```
We can also create another dataset for the evolution of these variables from 2000 to 2016
```python
arL = arable_land.loc[arable_land['year'].isin(["2000","2001","2002","2003","2004","2005","2006",
                                          "2007","2008","2009","2010","2011","2012","2013",
                                          "2014","2015","2016"])]

fP = fossil_fuel_production.loc[fossil_fuel_production['year'].isin([2000,2001,2002,2003,2004,2005,2006,
                                                                     2007,2008,2009,2010,2011,2012,2013,
                                                                     2014,2015,2016])]

rec = renewable_electricity_capacity.loc[renewable_electricity_capacity['year'].isin([2000,2001,2002,2003,2004,2005,2006,
                                                                     2007,2008,2009,2010,2011,2012,2013,
                                                                     2014,2015,2016])]

arL.rename(columns = {'Country':'country'}, inplace = True)

arL['year'] = arL['year'].astype('uint64')

dataset = arL.merge(fP, how='inner', left_on=['country', 'year'], right_on=['country', 'year'])

dataset = dataset.merge(rec, how='inner', left_on=['country', 'year'], right_on=['country', 'year'])

dataset = dataset.drop('economy',axis=1)

dataset = dataset[~dataset['country'].isin(['World'])]
```

## Exploratory Data Analysis

We have created two dataset which are quite similar: one with all the variables in 2016 (dataset_2016), and other with the evolution of the variables from 2000 to 2016 (dataset)
```python
dataset_2016.head()

    hectares               country  ...  total_renewable_energy  wind_energy
0     2000.0                 Aruba  ...               38.099998         30.0
1  7729000.0           Afghanistan  ...              349.313995          0.1
2  5347000.0                Angola  ...             1764.816040          NaN
3   620300.0               Albania  ...             1914.000000          NaN
4    44500.0  United Arab Emirates  ...              141.516006          0.0

dataset_2016.tail()

       hectares       country  ...  total_renewable_energy  wind_energy
171     25760.0         Samoa  ...               16.653999         0.55
172         NaN        Kosovo  ...               81.220001         1.00
173  12000000.0  South Africa  ...             4651.853027      1473.00
174   3700000.0        Zambia  ...             2431.250977          NaN
175   4000000.0      Zimbabwe  ...              880.021973          NaN

dataset_2016.shape #we have 175 rows (countries) and 33 variables

dataset_2016.columns

Index(['hectares', 'country', 'coal_production__twh', 'gas_production__twh',
       'oil_production__twh', 'annual_change_in_coal_production__pct',
       'annual_change_in_coal_production__twh',
       'annual_change_in_oil_production__pct',
       'annual_change_in_oil_production__twh',
       'annual_change_in_gas_production__pct',
       'annual_change_in_gas_production__twh',
       'coal_production_per_capita__kwh', 'oil_production_per_capita__kwh',
       'gas_production_per_capita__kwh', 'bagasse', 'bioenergy', 'biogas',
       'concentrated_solar_power', 'geothermal_energy', 'hydropower',
       'liquid_biofuels', 'marine_energy', 'offshore_wind_energy',
       'onshore_wind_energy', 'other_solid_biofuels', 'pure_pumped_storage',
       'renewable_hydropower_including_mixed_plants',
       'renewable_municipal_waste', 'solar_energy', 'solar_photovoltaic',
       'solid_biofuels_and_renewable_waste', 'total_renewable_energy',
       'wind_energy'],
      dtype='object')

#We can use describe() method in order to know the main statistics of all columns (or a bunch of them if qe want)

dataset_2016.describe()

dataset_2016[['oil_production__twh', 'onshore_wind_energy']].describe()

       oil_production__twh  onshore_wind_energy
count           175.000000           118.000000
mean            285.059784          4602.003418
std            1118.035522         18436.279297
min               0.000000             0.000000
25%               0.000000             1.933500
50%               0.000000            72.452499
75%              35.096937           835.275024
max           10351.818359        147036.890625
```
Now, we could do some plots in order to explore our dataset. For instance, which is the average energy production by year? In the graph below we see increasing trends in the production of both coal, oil and gas but from 2014 onwards coal production starts to decline and is overtaken by oil. We also observe that gas production is far behind the other two raw materials but with an upward slope during the 16 years analysed.
```python
dataset.groupby(['year'])['coal_production__twh',
                          'gas_production__twh',
                          'oil_production__twh'].mean().plot(title='Average Fossil Fuel Production by Year')

```
![Average Fossil Fuel Production by Year](https://github.com/PabloCH2410/Decision-Tree-Regression-with-data-from-OWiD-and-World-Bank/assets/172182349/cdfa006e-3567-4ec6-b377-7a43b14243b1)

Which are the 5 countries with the highest average coal in terawatt-hours (TWh) in 2016? A their porduction of oil and gas?
```python
dataset_2016.groupby(['country'])['coal_production__twh',
                          'gas_production__twh',
                          'oil_production__twh'].mean().nlargest(5,
                                                    ['coal_production__twh'])

              coal_production__twh  gas_production__twh  oil_production__twh
country                                                                      
China                  19671.232422          1379.416138          2322.336426
North America           4534.779297          9682.548828         10351.818359
United States           4083.574463          7273.561035          6312.184082
Australia               3563.055664           940.796997           178.546112
India                   3301.831055           265.823151           472.008911
```
And for China, which produced the most coal, what were the top 5 years that produced the most coal on average?
```python
dataset[dataset['country'] == 'China'].groupby('year')[['coal_production__twh', 
                                                        'gas_production__twh', 
                                                        'oil_production__twh']].mean().nlargest(5, 'coal_production__twh')

      coal_production__twh  gas_production__twh  oil_production__twh
year                                                                
2013          22034.136719          1218.107056          2441.826660
2012          21789.302734          1114.780762          2412.969238
2014          21680.761719          1311.808472          2458.919189
2011          21535.068359          1061.656006          2359.442139
2015          21273.957031          1356.688721          2495.332764
```

So what we can see is that although on average the 175 countries in our dataset (those producing coal) started to reduce coal production from 2014 onwards, China, which in 2016 was the largest coal producer, produces in the previous 5 years its largest amount of coal when the ‘global’ production curve (global in our dataset) starts to flatten before decreasing.


And what about the solar and wind capacity production between 2000 and 2016? In this case we see that, of the 175 countries we have in our dataset over 16 years, solar energy production is below onshore wind energy production, although both are experiencing an upward trend with similar speed (slope).
```python
dataset.groupby(['year'])['solar_energy',
                          'onshore_wind_energy',
                          'total_renewable_energy'].mean().plot(title='Average Solar and Onshore Wind Production by Year')
```
![Average Solar and Onshore Wind Production by Year](https://github.com/PabloCH2410/Decision-Tree-Regression-with-data-from-OWiD-and-World-Bank/assets/172182349/10739d93-7d07-49c3-bf90-aeee281da499)

Which are the 5 countries with the highest average  onshore wind energy production in 2016? And solar energy?                                                         
```python
dataset_2016.groupby(['country'])['onshore_wind_energy'].mean().nlargest(5)

country
China            147036.890625
North America     99019.726562
United States     81473.062500
Germany           45303.000000
India             28700.439453
Name: onshore_wind_energy, dtype: float32


dataset_2016.groupby(['country'])['solar_energy'].mean().nlargest(5)

country
China            77818.796875
Japan            42040.000000
Germany          40679.000000
North America    39969.125000
United States    35433.675781
Name: solar_energy, dtype: float32
```
We can see that there are many variables with NaN values, andwe can see that we have NaN in both datasets and for the same variables. Only coal, gas, oil and total renewable energy have no NaN, so the NaN values come from the renewable electricity capacity
```python
dataset_2016.isna().sum().sort_values() #number of null values in each column of the DataFrame

annual_change_in_oil_production__twh             0
country                                          0
coal_production__twh                             0
gas_production__twh                              0
oil_production__twh                              0
gas_production_per_capita__kwh                   0
annual_change_in_coal_production__twh            0
oil_production_per_capita__kwh                   0
total_renewable_energy                           0
coal_production_per_capita__kwh                  0
annual_change_in_gas_production__twh             0
hectares                                         4
solar_energy                                     5
solar_photovoltaic                               5
renewable_hydropower_including_mixed_plants     33
hydropower                                      33
bioenergy                                       56
onshore_wind_energy                             57
wind_energy                                     57
solid_biofuels_and_renewable_waste              72
biogas                                          87
annual_change_in_oil_production__pct            90
annual_change_in_gas_production__pct            95
other_solid_biofuels                           103
annual_change_in_coal_production__pct          118
bagasse                                        122
pure_pumped_storage                            137
renewable_municipal_waste                      138
geothermal_energy                              151
liquid_biofuels                                157
marine_energy                                  158
concentrated_solar_power                       160
offshore_wind_energy                           160
dtype: int64


dataset.isna().sum().sort_values()

year                                              0
gas_production_per_capita__kwh                    0
oil_production_per_capita__kwh                    0
coal_production_per_capita__kwh                   0
oil_production__twh                               0
total_renewable_energy                            0
coal_production__twh                              0
country                                           0
gas_production__twh                               0
annual_change_in_coal_production__twh             3
annual_change_in_oil_production__twh              3
annual_change_in_gas_production__twh              3
hectares                                         60
hydropower                                      319
renewable_hydropower_including_mixed_plants     319
solar_energy                                    661
solar_photovoltaic                              665
bioenergy                                      1005
solid_biofuels_and_renewable_waste             1179
onshore_wind_energy                            1338
wind_energy                                    1338
annual_change_in_oil_production__pct           1367
annual_change_in_gas_production__pct           1439
annual_change_in_coal_production__pct          1743
other_solid_biofuels                           1758
biogas                                         1773
bagasse                                        1904
pure_pumped_storage                            2106
renewable_municipal_waste                      2254
geothermal_energy                              2333
liquid_biofuels                                2532
marine_energy                                  2563
offshore_wind_energy                           2569
concentrated_solar_power                       2590
dtype: int64
```
Taking this into account, we can assume those NaN values as 0 (we will imagine that if a country has NaN it means that they do not have renewable capacity in this source). Regarding preprocessing  we can delete rows/columns containing NaN or use other variables to complete these NaN (impute). Deleting rows/columns could not be the best way as we will lose lot of info. Imputing values has the problem that there are variables which have a lot of weight in the model, which could lead to misleading imputation

We have SimpleImputer (constant value or statistic), IterativeImputer (put a value on the column taking into account the other columns) and KNNImputer. So, taking all of this into account we have to make a decission, and it seems that the best solution (and the easiest one) is to replace NaN with 0. [Reference](https://machinelearningknowledge.ai/how-to-use-sklearn-simple-imputer-simpleimputer-for-filling-missing-values-in-dataset)
```python
constant_imputer = SimpleImputer(strategy='constant',fill_value=0)
result_constant_imputer = constant_imputer.fit_transform(dataset_2016)
dataset_2016 = pd.DataFrame(result_constant_imputer, columns=list(dataset_2016.columns))

constant_imp= SimpleImputer(strategy='constant',fill_value=0)
result_constant_imp = constant_imp.fit_transform(dataset)
dataset = pd.DataFrame(result_constant_imp, columns=list(dataset.columns)
```
Now we do not have NaN values
```python
dataset_2016.isna().sum().sort_values()

hectares                                       0
solid_biofuels_and_renewable_waste             0
solar_photovoltaic                             0
solar_energy                                   0
renewable_municipal_waste                      0
renewable_hydropower_including_mixed_plants    0
pure_pumped_storage                            0
other_solid_biofuels                           0
onshore_wind_energy                            0
offshore_wind_energy                           0
marine_energy                                  0
liquid_biofuels                                0
hydropower                                     0
geothermal_energy                              0
concentrated_solar_power                       0
total_renewable_energy                         0
biogas                                         0
bagasse                                        0
gas_production_per_capita__kwh                 0
oil_production_per_capita__kwh                 0
coal_production_per_capita__kwh                0
annual_change_in_gas_production__twh           0
annual_change_in_gas_production__pct           0
annual_change_in_oil_production__twh           0
annual_change_in_oil_production__pct           0
annual_change_in_coal_production__twh          0
annual_change_in_coal_production__pct          0
oil_production__twh                            0
gas_production__twh                            0
coal_production__twh                           0
country                                        0
bioenergy                                      0
wind_energy                                    0
dtype: int64
```
Now, for the EDA we are going to select a cluster of countries
```python
columns_to_drop = ['annual_change_in_coal_production__pct',
                   'annual_change_in_coal_production__twh',
                   'annual_change_in_oil_production__pct',
                   'annual_change_in_oil_production__twh',
                   'annual_change_in_gas_production__pct',
                   'annual_change_in_gas_production__twh']

dataset_2016 = dataset_2016.drop(columns=columns_to_drop)
dataset = dataset.drop(columns=columns_to_drop)


dataset_2016_cluster = dataset_2016[dataset_2016['country'].isin(['Argentina','Australia','Austria','Belgium','Bangladesh','Spain','Japan','South Africa'])]

dataset_cluster = dataset[dataset['country'].isin(['Argentina','Australia','Austria','Belgium','Bangladesh','Spain','Japan','South Africa'])]
```
Now, we can perform EDA for all the countries (in order to know the distribution for 2016 of the variables for all countries)
```python
dataset_2016_cluster.set_index("country", inplace=True) # Shape (8,26)

# Calculate the number of rows and columns required
num_columns = len(dataset_2016_cluster.columns)
nrows = int(np.ceil(num_columns / 5))
ncols = min(num_columns, 5)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 10))
axes = axes.flatten()

for i, colum in enumerate(dataset_2016_cluster.columns):
    sns.histplot(
        data=dataset_2016_cluster,
        x=colum,
        stat="count",
        kde=True,  # Kernel Density Estimation
        line_kws={'linewidth': 2},
        bins=20,  # Number of intervals
        alpha=0.3,  # Opacity of the bars
        ax=axes[i]
    )
    axes[i].set_title(colum, fontsize=12)
    axes[i].tick_params(labelsize=6)
    axes[i].set_xlabel("")

# Remove unused axes
if num_columns < len(axes):
    for j in range(num_columns, len(axes)):
        fig.delaxes(axes[j])

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Distribution of numerical variables of our cluster')
plt.show()
```
The result is a figure with multiple subplots, each showing the distribution  of a specific numerical variable from the dataset_2016_cluster. This allows us to compare distributionsand see how different variables are distributed within the dataset. We can also identify patterns.

From this graph we can see how the distribution of practically all variables is negatively skewed, except for bioenergy production, which is more similar to a normal distribution. Thus, we can see how most of the production of these countries that we have clustered is concentrated in the lower limits of these countries, both fossil fuels and renewables.
![Distribution of numerical variables of our cluster](https://github.com/PabloCH2410/Decision-Tree-Regression-with-data-from-OWiD-and-World-Bank/assets/172182349/161d01a6-682d-48bb-87a7-20eac93669af)

Lets see onshore wind energy variable (analyse one variable for all countries). Of the cluster of countries we have chosen, all show a growing trend in onshore wind energy production, but Spain stands out by far for its higher speed (slope) and production figure.
```python
ax = sns.lineplot(dataset_cluster['year'], 
             dataset_cluster['onshore_wind_energy'], 
             hue = dataset_cluster['country'])
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

print(dataset_2016_cluster.groupby(['country'])['onshore_wind_energy'].mean().nlargest(5))

country
Spain        22985.000000
Australia     4324.000000
Japan         3187.000000
Austria       2729.996094
Belgium       1621.599976
Name: onshore_wind_energy, dtype: float64
```
![Onshore wind energy by country](https://github.com/PabloCH2410/Decision-Tree-Regression-with-data-from-OWiD-and-World-Bank/assets/172182349/75be2b0f-016b-4f2b-9ec0-60784cbbc4fe)

We can also analyze the evolution of the variable for a country (analyze all variables of one country)
```python
dataset_cluster_Australia = dataset_cluster.loc[dataset_cluster['country'] == 'Australia']
dataset_cluster_Australia = dataset_cluster_Australia.drop('country',axis=1) #shape(17,27)

fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(18, 10)) 
axes = axes.flat

for i, colum in enumerate(dataset_cluster_Australia.columns):
    sns.lineplot(
        dataset_cluster_Australia['year'],
        dataset_cluster_Australia[colum],
        ax = axes[i]
    )
    axes[i].set_title(colum, fontsize = 11)
    axes[i].yaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].xaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].tick_params(labelsize = 9)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    
# Remove unused axes
for i in [0, 27, 28, 29, 30, 31, 32, 33, 34, 35]:  # Indexes of empty plots
    if i < len(axes):
        fig.delaxes(axes[i])
        
fig.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle('Distribution of variables in Australia by year')
```
![Distribution of variables in Australia by year](https://github.com/PabloCH2410/Decision-Tree-Regression-with-data-from-OWiD-and-World-Bank/assets/172182349/ca71bbb4-2051-4f7e-bd29-83ec7c6d1125)

This graph helps us to quickly see how Australia's production variables evolve over the years of our dataset. For example, we can see that in these 16 years Australia has not produced hydropower or offshore wind energy, or that biogas experienced a large increase until around 2009, when its production began to decline. 

Coming back to our objective, we would like to focus on onshore_wind_energy variable. Lets see the behaviour of this variable compared to the rest of variables of our cluster
```python
dataset_2016_cluster = dataset_2016_cluster.astype('float64')

columns = dataset_2016_cluster.drop('onshore_wind_energy',axis=1) #all except onshore_wind_energy

columns = columns.astype('float64')

fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(18,10))
axes = axes.flat

for i, colum in enumerate(columns):
    sns.regplot(
        x           = dataset_2016_cluster[colum],
        y           = dataset_2016_cluster['onshore_wind_energy'],
        color       = "gray",
        marker      = '.',
        scatter_kws = {"alpha":0.4},
        line_kws    = {"color":"r","alpha":0.7},
        ax          = axes[i]
    )
    axes[i].set_title(f" vs {colum}", fontsize = 11)
    axes[i].yaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].xaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].tick_params(labelsize = 9)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

# Remove unused axes
for i in range(25, 36):  # Indexes of empty plots
    if i < len(axes):
        fig.delaxes(axes[i])
    
fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Correlation with onshore wind energy', fontsize = 10, fontweight = "bold");
```
![Correlation with onshore wind energy](https://github.com/PabloCH2410/Decision-Tree-Regression-with-data-from-OWiD-and-World-Bank/assets/172182349/8cca7e88-1750-48dc-adb6-e72d1b7d3a9c)

With this graph we can see the relationship of each variable in our dataset with onshore wind energy. Thus, we can see that there is a negative correlation between onshore wind energy production and oil and gas production (in the case of coal we see that there is hardly any influence), and a strong positive correlation with biogas production. Another curious insight we can draw from the relationship graph and the correlation matrix is with the hectares of arable land. We see that there is a negative correlation between the production of onshore wind energy and these hectares, which seems logical: to produce more onshore wind energy we need more space for wind farms for example, which reduces the possible land space that can be used for cultivation or other agricultural activity.

We can do all this visually, see the direction of the relationship between two variables (positive or negative) and the strength of this relationship (slope of the curve drawn on the graph), but to be sure of what we see a very good way is a correlation matrix.

With this we see that our previous insights on the correlations of gas, oil, coal and biogas production are correct.
```python
mask = np.triu(np.ones_like(dataset_2016_cluster.corr()))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
sns.heatmap(
    dataset_2016_cluster.corr(method = 'pearson'),
    annot = True,
    mask = mask,
    ax = ax
    )
ax.tick_params(labelsize = 12)
```
![Correlation matrix](https://github.com/PabloCH2410/Decision-Tree-Regression-with-data-from-OWiD-and-World-Bank/assets/172182349/ae325520-3773-4b39-ac84-c8dc38f33b48)

## Decision Tree Regression

We could repeat the same analysis for the number of countries and variables we want, but as we have more than 20 variables, it would a great idea to look for a way to summarise these info. So, we are going to do a decission tree, in this case a regression tree in order to predict the value of onshore_wind_energy. With this, we will be able to see the importance of some predictors over others. From here on we have followed the modelling steps outlined in the great article [Decision Trees with Python](https://cienciadedatos.net/documentos/py07_arboles_decision_python): Regression and Classification from Ciencia de Datos.net

In this code we have first divided our dataset into two parts: one to train the model (train) and one to see that it actually works (test). In our case, we separate our dataset between onshore wind energy (the response variable we want to predict) and the rest of the variables. For this we use the famous train_test_split method of SKlearn.

Then we use the DecisionTreeRegressor method and we "integrate" using .fit the split we have made of our data. With this we get the number of layers (depth) and termination nodes of our decision tree: it proposes a tree of 14 layers with 85 terminal nodes. This is best seen with a graph of the tree. In this case note that we have not done any tuning to the DecisionTreeRegressor class.
```python
dataset_2016.set_index("country", inplace = True)

dataset_2016 = dataset_2016.astype('float64')

X_train, X_test, y_train, y_test = train_test_split(
                                        dataset_2016.drop(columns = "onshore_wind_energy"),
                                        dataset_2016['onshore_wind_energy'],
                                        random_state = 123)

model = DecisionTreeRegressor(random_state = 123).fit(X_train, y_train)

print(f"Tree depth: {model.get_depth()}")
print(f"Number of terminal nodes: {model.get_n_leaves()}")

Tree depth: 14
Number of terminal nodes: 85

fig, ax = plt.subplots(figsize=(40,35))

plot_tree(
            decision_tree = model,
            feature_names = dataset_2016.drop(columns = 'onshore_wind_energy').columns,
            class_names   = 'onshore_wind_energy',
            filled        = True,
            impurity      = False, #When set to True, show the impurity at each node
            fontsize      = 10,
            precision     = 2, #Number of digits of precision for floating point
            ax            = ax
         )
```
![First decision tree](https://github.com/PabloCH2410/Decision-Tree-Regression-with-data-from-OWiD-and-World-Bank/assets/172182349/f137e64d-840d-4daa-99ff-c9e504026b89)

As we can see with this huge tree it is almost impossible to figure it something. Thus, we have to tune it a little bit.
```python
#max_depth- so we can control the depth of the tree in order to figure it out

model1 = DecisionTreeRegressor(max_depth = 5,
                               random_state = 123).fit(X_train, y_train)

#We have a depth of 5 and 13 terminal nodes
print(f"Tree depth: {model1.get_depth()}")
print(f"Number of terminal nodes: {model1.get_n_leaves()}")

fig, ax = plt.subplots(figsize=(22,9))

plot_tree(
            decision_tree = model1,
            feature_names = dataset_2016.drop(columns = 'onshore_wind_energy').columns,
            class_names   = 'onshore_wind_energy',
            filled        = True,
            impurity      = False, #When set to True, show the impurity at each node
            fontsize      = 10,
            precision     = 2, #Number of digits of precision for floating point
            ax            = ax
         )
```
```python
text_model1 = export_text(
                    decision_tree = model1,
                    feature_names = list(dataset_2016.drop(columns = "onshore_wind_energy").columns)
               )
print(text_model1)

Tree depth: 5
Number of terminal nodes: 13
|--- solar_energy <= 23673.85
|   |--- solar_energy <= 6853.03
|   |   |--- wind_energy <= 2183.32
|   |   |   |--- wind_energy <= 504.50
|   |   |   |   |--- wind_energy <= 127.32
|   |   |   |   |   |--- value: [10.88]
|   |   |   |   |--- wind_energy >  127.32
|   |   |   |   |   |--- value: [243.55]
|   |   |   |--- wind_energy >  504.50
|   |   |   |   |--- wind_energy <= 1125.23
|   |   |   |   |   |--- value: [760.91]
|   |   |   |   |--- wind_energy >  1125.23
|   |   |   |   |   |--- value: [1405.82]
|   |   |--- wind_energy >  2183.32
|   |   |   |--- gas_production_per_capita__kwh <= 2332.22
|   |   |   |   |--- hectares <= 1773905.00
|   |   |   |   |   |--- value: [5124.10]
|   |   |   |   |--- hectares >  1773905.00
|   |   |   |   |   |--- value: [5989.50]
|   |   |   |--- gas_production_per_capita__kwh >  2332.22
|   |   |   |   |--- oil_production_per_capita__kwh <= 4806.34
|   |   |   |   |   |--- value: [3033.86]
|   |   |   |   |--- oil_production_per_capita__kwh >  4806.34
|   |   |   |   |   |--- value: [4116.36]
|   |--- solar_energy >  6853.03
|   |   |--- renewable_municipal_waste <= 315.57
|   |   |   |--- value: [22985.00]
|   |   |--- renewable_municipal_waste >  315.57
|   |   |   |--- liquid_biofuels <= 1.83
|   |   |   |   |--- value: [10832.53]
|   |   |   |--- liquid_biofuels >  1.83
|   |   |   |   |--- value: [11566.56]
|--- solar_energy >  23673.85
|   |--- renewable_municipal_waste <= 1186.07
|   |   |--- value: [81473.06]
|   |--- renewable_municipal_waste >  1186.07
|   |   |--- value: [99019.73]
```
![Second decision tree](https://github.com/PabloCH2410/Decision-Tree-Regression-with-data-from-OWiD-and-World-Bank/assets/172182349/a723ebde-9533-4c47-ac78-501367f96bf6)

To give an interpretation of a decision tree we take as an example the rightmost branch: the onshore wind energy production is 81473.06 when the solar energy production is greater than 23673.85 and the renewable municipal waste is less than or equal to 1186.07, and if the renewable municipal waste is greater than 1186.07 then the onshore wind energy production is 99019.73

### Importance of tree variables

In addition, we can know the importance of the predictors of this model that we have created with a maximum of 5 layers of depth. For this we will use the .feature_importances_ method. In our case we see that solar energy, renewable municipal waste and wind energy are the most relevant variables in our max_depth 5 decision tree.
```python
# Predictor importance in the model reflects the total (normalized) reduction in the splitting criterion. 
# Predictors not selected in any split have zero significance.

importance = pd.DataFrame(
                            {'predictor': dataset_2016.drop(columns = 'onshore_wind_energy').columns,
                             'importance': model1.feature_importances_}
                            )
importance.sort_values('importance', ascending=False)

                                      predictor  importance
20                                 solar_energy    0.975478
19                    renewable_municipal_waste    0.014658
24                                  wind_energy    0.009177
6                gas_production_per_capita__kwh    0.000537
5                oil_production_per_capita__kwh    0.000105
0                                      hectares    0.000030
13                              liquid_biofuels    0.000016
4               coal_production_per_capita__kwh    0.000000
16                         other_solid_biofuels    0.000000
23                       total_renewable_energy    0.000000
22           solid_biofuels_and_renewable_waste    0.000000
21                           solar_photovoltaic    0.000000
2                           gas_production__twh    0.000000
3                           oil_production__twh    0.000000
18  renewable_hydropower_including_mixed_plants    0.000000
17                          pure_pumped_storage    0.000000
15                         offshore_wind_energy    0.000000
7                                       bagasse    0.000000
14                                marine_energy    0.000000
1                          coal_production__twh    0.000000
11                            geothermal_energy    0.000000
10                     concentrated_solar_power    0.000000
9                                        biogas    0.000000
8                                     bioenergy    0.000000
12                                   hydropower    0.000000
```
### Tree pruning

Pruning aims to find the simplest tree (minimal size) that delivers optimal predictions. Specifying the ccp_alpha parameter is essential for pruning, as it controls the degree of complexity penalty. A higher ccp_alpha value leads to more aggressive pruning and a smaller tree size. Cross-validation is utilized to determine the optimal ccp_alpha value.

The GridSearchCV method allows us to pass a grid of self-defined parameters (in this case an array of 60 values between 0 and 80) to find the best hyperparameters of an estimator (in this case our estimator is a DecisionTreeRegressor with a configuration). GridSearchCV will test the parameter grid we have passed it to see which are the best DecisionTreeRegressor estimators based on what we have indicated. It is very important to highlight the parameter cv in which we indicate the number of folds in which we are going to divide the dataset.

That is, in this case we have defined an array of 60 values between 0 and 80 for ccp_alpha, and with the GridSearchCV method we will search in all these ccp_alpha values which is the most optimal for the DecisionTreeRegressor estimator that we have defined. For this, we will use the cross-validation technique by randomly dividing the dataset in 10 approximately equal parts, and in each iteration 9 folds will be used to train the estimator (model) with the ccp_alpha values of the grid (hyperparameters to be tested) and the remaining fold will be used to validate the model. This process is repeated so that each of the 10 folds is used exactly once as a validation set (the model is trained and validated 10 times). The results of the 10 iterations are averaged to obtain a more accurate and robust estimate of the model performance. For further information about hyperparameter optimization with Grid Search check Michael Fuchs blog about [Decision Trees](https://michael-fuchs-python.netlify.app/2019/11/30/introduction-to-decision-trees/).
```python
#max_depht: depht of the tree
#min_samples_split: the minimum number of samples required to split an internal node (default 2). The higher the value, the less flexible the model is.
#min_samples_leaf: The minimum number of samples required to be at a leaf node (default 1)
#max_leaf_nodes: max terminal node (default none). The lower the value, the less flexible the model is.
#radom_state: seed to make the results reproducible. Controls the randomness of the estimator

#np.linspace will return an array of 60 values between 0 and 80
param_grid = {'ccp_alpha':np.linspace(0, 80, 60)}

grid = GridSearchCV(
        # The biggest possible tree to then be prune
        estimator = DecisionTreeRegressor(
                            max_depth         = None,
                            min_samples_split = 2,
                            min_samples_leaf  = 1,
                            random_state      = 123
                       ),
        param_grid = param_grid,
        cv         = 10,
        refit      = True,
        return_train_score = True
      )

grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_) #ccp_alpha is 4.067796610169491

final_model = grid.best_estimator_ 

#Depth of 8 and 32 terminal nodes
print(f"Tree depth: {final_model.get_depth()}")
print(f"Number of terminal nodes: {final_model.get_n_leaves()}")
```
```python
fig, ax = plt.subplots(figsize=(30,16))

plot_tree(
            decision_tree = final_model,
            feature_names = dataset_2016.drop(columns = 'onshore_wind_energy').columns,
            class_names   = 'onshore_wind_energy',
            filled        = True,
            impurity      = False, #When set to True, show the impurity at each node
            fontsize      = 10,
            precision     = 2, #Number of digits of precision for floating point
            ax            = ax
         )
```
![Third decision tree](https://github.com/PabloCH2410/Decision-Tree-Regression-with-data-from-OWiD-and-World-Bank/assets/172182349/5b5f2bda-c847-47f2-b5a8-1bd5cc2227ec)

```python
importance_fm = pd.DataFrame(
                            {'predictor': dataset_2016.drop(columns = 'onshore_wind_energy').columns,
                             'importance':final_model.feature_importances_}
                            )
importance_fm.sort_values('importance', ascending=False)

                                      predictor    importance
20                                 solar_energy  9.754344e-01
24                                  wind_energy  9.203058e-03
14                                marine_energy  9.152119e-03
10                     concentrated_solar_power  5.505084e-03
6                gas_production_per_capita__kwh  5.373881e-04
5                oil_production_per_capita__kwh  1.044998e-04
22           solid_biofuels_and_renewable_waste  2.968261e-05
13                              liquid_biofuels  1.601628e-05
0                                      hectares  1.014420e-05
9                                        biogas  7.186374e-06
3                           oil_production__twh  1.758228e-07
17                          pure_pumped_storage  1.242617e-07
19                    renewable_municipal_waste  1.070126e-07
2                           gas_production__twh  4.856037e-08
8                                     bioenergy  0.000000e+00
11                            geothermal_energy  0.000000e+00
1                          coal_production__twh  0.000000e+00
7                                       bagasse  0.000000e+00
4               coal_production_per_capita__kwh  0.000000e+00
15                         offshore_wind_energy  0.000000e+00
16                         other_solid_biofuels  0.000000e+00
18  renewable_hydropower_including_mixed_plants  0.000000e+00
21                           solar_photovoltaic  0.000000e+00
23                       total_renewable_energy  0.000000e+00
12                                   hydropower  0.000000e+00
```
Let's try to squeeze even more out of GridSearchCV. Now we will not only pass a grid for ccp_alpha, but also for max_depth, min_samples_split and min_samples_leaf. 
```python
param_grid = {
    'ccp_alpha': np.linspace(0, 80, 60),
    'max_depth': [None] + list(range(1, 5)),
    'min_samples_split': range(2, 5),
    'min_samples_leaf': range(1, 3)
}

grid = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=123),
    param_grid=param_grid,
    cv=5,
    refit=True,
    return_train_score=True
)

grid.fit(X_train, y_train)

#Best parameters: {'ccp_alpha': 0.0, 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}
print("Best parameters:", grid.best_params_)

final_model1 = grid.best_estimator_ 

#Depth of 3 and 6 terminal nodes
print(f"Tree depth: {final_model1.get_depth()}")
print(f"Number of terminal nodes: {final_model1.get_n_leaves()}")

fig, ax = plt.subplots(figsize=(30,16))

plot_tree(
            decision_tree = final_model1,
            feature_names = dataset_2016.drop(columns = 'onshore_wind_energy').columns,
            class_names   = 'onshore_wind_energy',
            filled        = True,
            impurity      = False, #When set to True, show the impurity at each node
            fontsize      = 10,
            precision     = 2, #Number of digits of precision for floating point
            ax            = ax
         )

importance_fm = pd.DataFrame(
                            {'predictor': dataset_2016.drop(columns = 'onshore_wind_energy').columns,
                             'importance':final_model1.feature_importances_}
                            )
importance_fm.sort_values('importance', ascending=False)

                                      predictor  importance
20                                 solar_energy    0.976680
17                          pure_pumped_storage    0.009164
24                                  wind_energy    0.008644
8                                     bioenergy    0.005512
13                              liquid_biofuels    0.000000
23                       total_renewable_energy    0.000000
22           solid_biofuels_and_renewable_waste    0.000000
21                           solar_photovoltaic    0.000000
19                    renewable_municipal_waste    0.000000
18  renewable_hydropower_including_mixed_plants    0.000000
16                         other_solid_biofuels    0.000000
15                         offshore_wind_energy    0.000000
14                                marine_energy    0.000000
0                                      hectares    0.000000
1                          coal_production__twh    0.000000
11                            geothermal_energy    0.000000
10                     concentrated_solar_power    0.000000
9                                        biogas    0.000000
7                                       bagasse    0.000000
6                gas_production_per_capita__kwh    0.000000
5                oil_production_per_capita__kwh    0.000000
4               coal_production_per_capita__kwh    0.000000
3                           oil_production__twh    0.000000
2                           gas_production__twh    0.000000
12                                   hydropower    0.000000
```
![Fourth decision tree](https://github.com/PabloCH2410/Decision-Tree-Regression-with-data-from-OWiD-and-World-Bank/assets/172182349/dc2aefdc-7207-4d36-90a6-a79ad3053648)

## Comparison between models and conclusions
```python
models = [model, model1, final_model, final_model1]
model_names = ['model', 'model1', 'final_model', 'final_model1']

for mdl, mdl_name in zip(models, model_names):
    predictions = mdl.predict(X=X_test)
    rmse = mean_squared_error(y_true=y_test, y_pred=predictions, squared=False)
    print(f"Error (RMSE) of test of {mdl_name}: {rmse} (this values show the deviation from the real value of what we want to precit)")

Error (RMSE) of test of model: 16629.042940444888 (this values show the deviation from the real value of what we want to precit)
Error (RMSE) of test of model1: 14992.59296232735 (this values show the deviation from the real value of what we want to precit)
Error (RMSE) of test of final_model: 16629.043169691166 (this values show the deviation from the real value of what we want to precit)
Error (RMSE) of test of final_model1: 17330.175656042833 (this values show the deviation from the real value of what we want to precit)
```
With this we can see how the deviation (error) of the actual result of the onshore wind energy production of the initial model (where we do not indicate anything from the regression tree) and the final model where we use cross-validation to obtain the best hyperparameters of the model hardly differ (model: 16629.042 VS final_model: 16629.043), while the model in which we specify a tree depth of 5 layers reduces the deviation from the actual prediction quite a bit (model1: 14992.59). Similarly, despite the computational weight of the second GridSearchCV (final_model1) we use, the error is even larger than the other models. However, we have seen a way to apply this amazing SKlearn function in more depth.

Thus, we see that by limiting the depth of the tree we limit the error of its prediction. However, with a better selection of hyperparameters in GridSearchCV we will probably achieve a much greater error reduction.

With this project we have seen a real use case of Python for data analysis and data modelling to apply machine learning algorithms. First, we have made the connection to the APIs of Our World in Data and the World Bank. After searching and extracting the data we needed, we have used Pandas together with other libraries to transform the data and get the datasets with what we want to analyse and model. Then we performed an exploratory data analysis (EDA) to get interesting insights from our data. Finally, we have used a regression tree to predict the onshore wind energy production as a function of the other variables. This last point is of great interest as this algorithm allows us to interpret our data in an intuitive and user-friendly way, but without losing the robustness of one of the most widely used ML algorithms.

Therefore, the implementation and application of complex algorithms and procedures becomes much easier using Python and its various data analysis and modelling libraries.

## References

[Ciencia de Datos.net: Árboles de decisión con Python: regresión y clasificación](https://cienciadedatos.net/documentos/py07_arboles_decision_python)

[Introduction to Decision Trees by Michael Fuchs](https://michael-fuchs-python.netlify.app/2019/11/30/introduction-to-decision-trees/)

[An Introduction to Statistical Learning](https://www.statlearning.com/)

[Ciencia de Datos.net: Machine learning con Python y Scikit-learn](https://cienciadedatos.net/documentos/py06_machine_learning_python_scikitlearn)

[How To Use Sklearn Simple Imputer (SimpleImputer) for Filling Missing Values in Dataset](https://machinelearningknowledge.ai/how-to-use-sklearn-simple-imputer-simpleimputer-for-filling-missing-values-in-dataset/#google_vignette)

[owid-catalog 0.3.11](https://pypi.org/project/owid-catalog/)

[wbgapi 1.0.12](https://pypi.org/project/wbgapi/)

[DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

[Decision Tree Regression example](https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html#sphx-glr-auto-examples-tree-plot-tree-regression-py)

[GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

[Plotting a diagonal correlation matrix](https://seaborn.pydata.org/examples/many_pairwise_correlations.html)
