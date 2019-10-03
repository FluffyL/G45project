import pandas as pd
import datetime
import numpy as np
from sklearn.cluster import KMeans

taxi = pd.read_csv('taxi_train.csv', sep=',')
taxi.head(10)

import geohash as gh
gh.decode('dr5rxth4yu4c')

# Task 1 - Data Preprocessing and Statistics
#
#
# Task 1.1 Read Taxi Data in using the API Pandas.read_csv so that the column 'pickup_datetime' is read as datetime64.
#                 Hint: use the parameter parse_dates
#                  ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
#                 To make sure your code works you might want to read only the first 1000 rows and expand it later
#taxi = pd.read_csv('taxi_train.csv', parse_dates = ['pickup_datetime'], nrows = 1000)
taxi = pd.read_csv('taxi_train.csv', parse_dates = ['pickup_datetime'])

# Task 1.2 Convert the field pickup_geohashed and dropoff_geohashed into x-y coordinate using the API Geohash.decode
#                  The package Geohash can be found from pip. You might encounter the problem 
#                   'python3.5.2 can't find the module '  ref: https://github.com/vinsci/geohash/issues/4
#                    This can be fixed very easily. Or
#                   You might directly use the fixed version of Geohash in our project package.
#                   The precision of each coordinate is with 6 decimal places
pickup_x = []
pickup_y = []
dropoff_x = []
dropoff_y = []
distance = []
for index, row in taxi.iterrows():
	row['pickup_geohash'] = list(round(i,6) for i in gh.decode(row['pickup_geohash']))
	pickup_x.append(row['pickup_geohash'][0])
	pickup_y.append(row['pickup_geohash'][1])
	row['dropoff_geohash'] = list(round(i,6) for i in gh.decode(row['dropoff_geohash']))
	dropoff_x.append(row['dropoff_geohash'][0])
	dropoff_y.append(row['dropoff_geohash'][1])
	dx = row['dropoff_geohash'][0]-row['pickup_geohash'][0]
	dy = row['dropoff_geohash'][1]-row['pickup_geohash'][1]
	distance.append(np.sqrt(np.square(dx)+np.square(dy)))

# Task 1.2.1 Create unpack the decoded pickup_geohashed and dropoff_geohashed into the pair of columns 
#                  pickup_x pickup_y  and dropoff_x, dropoff_y respectively. 
#                  Namely: if pickup is [40.712278, -73.84161]; pickup_x should contain 40.712278 and pickup_y should contain -73.84161
taxi['pickup_x'] = pickup_x
taxi['pickup_y'] = pickup_y
taxi['dropoff_x'] = dropoff_x
taxi['dropoff_y'] = dropoff_y

# Task 1.3 Create the column 'distance' based on the Euclidean distance that the ride has traveled.
taxi['distance'] = distance

# Task 1.4.1  Check the memory you have spent by the API .info()
print(taxi.info())

# Task 1.4.2  Fetch the first 10 lines of your data to preview it.
print(taxi.head(20))

# Task 1.5 Remove rows with invalid geohashed. Count the number of rows removed.
taxi = taxi.dropna(axis=0,how='any')

# Task 1.6  Display the count, mean, standard derviation of the int type variable and 
# display the earliest and latest pickup_time.
int_name = taxi.select_dtypes(include = int)
count = []
mean = []
std = []
for i in int_name:
	count.append(taxi[i].value_counts().shape[0])
	mean.append(taxi[i].mean())
	std.append(taxi[i].std())
print('count: ',count)
print('mean: ',mean)
print('std: ', std)

# Task 1.7 Find the number of order between (8am to 9am)   and the order between (1am to 2am)
#                 Note: Instead of using only the first 1000 rows, expand your selection of rows to collect enough data.
#                Hint: try the API between_time of DataFrame. 
#           ref:  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.between_time.html#pandas.DataFrame.between_time
taxi = taxi.set_index(pd.DatetimeIndex(taxi['pickup_datetime']))
start = datetime.time(8,0,0)
end = datetime.time(9,0,0)
between8_9 = taxi.between_time(start, end).shape[0]
start = datetime.time(13,0,0)
end = datetime.time(14,0,0)
between1_2 = taxi.between_time(start, end).shape[0]
print('order between 8:00-9:00: ', between8_9)
print('order between 13:00-14:00: ', between1_2)


# Task 2 - Data Clustering 
#
#
# Task 2.1 Create a DataFrame that contains two columns. The first column (the index) is a time series 0:00, 0:15, 0:30, 0:45, 
#                  1:00,... 23:00, 23:15, 23:30, 23:45
#                  The second column is an integer that counts the number of ride between in the interval. For example, 0:00 should contains all order happens on or after 0:00 to 0:15.
#  This task is less straight forward, at least in our solution. So let's break down a little bit.



#  Task 2.1.1 Create a list of string containing the series '0:00', '0:15', '0:30', ... '23:45' 
#                     Hint: A double loop with if-else can do the job.
interval = datetime.timedelta(minutes = 15)
start = datetime.time(0,0,0)
end = datetime.time(23,59,59)
step = datetime.time(00,15,00)
time_step = pd.date_range('00:00', '23:59', freq = '15min').time

# Task 2.1.2 Count the number of orders. You might use between_time again.
cluster = pd.DataFrame(data=[],index=[],columns=[])
cluster['interval'] = time_step
number = []
for i in range(len(time_step)-1):
	number.append(taxi.between_time(time_step[i], time_step[i+1]).shape[0])
number.append(taxi.between_time(time_step[i], datetime.time(23,59,59)).shape[0])
cluster['number'] = number
print(cluster)

# Task 2.2.1  Use K-mean algorithm to find 30 cluster centers of the coordinates obtained from  Task 1.2.1
#                  You may implement your own K-mean algorithm or simply adopt the API sklearn.cluster.KMeans
#                   ref: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
                
kmeans = KMeans(n_clusters=30, random_state=0).fit(cluster)
#print(kmeans)
    
# Task 2.2.2 Describe how many % of order has started from a cluster centers and ends at the same cluster centers.  


# Task 3 - Simple Data Visualization 
#
#
# Task 3.1 Using data obtained from Task 1.6. Plot a curveof the volume of order in different times of a day.
#                    Hint: try the API DataFrame.plot


# Task 3.2 Using data obtained from Task 2.2.1. Plot a histogram of the volume of order in different cluster centers


# Task 3.3 Scatter plot the 100 random location of the pickups in blue, 100 random location of dropoffs in red, plot also the cluster centers in black

# Task 4 - Frequent Pattern Mining 
#
#
# Task 4.1 Apply FPGrowth algorithm, either using existing API or write your own, to identify which set of users are likely to go-together. 
#                 Definition of go-together: they starts at the same cluster centers and their start time is in the same 15-minutes timeslot.
 #                This task may not be straight forward as you may need to build the list of transaction first.


df = pd.read_csv('training.csv', sep=',', nrows=10)
df.head(10)

monday8am = pd.read_csv('5_1_testing.csv', sep=',', nrows=10, index_col=0)
monday8am.head(10)

best_tips = pd.read_csv('5_2_testing.csv', sep=',', nrows=10)
best_tips.head(10)



