# Flight Delay Analysis

This repository contains a set of flight data and an analysis script in Jupyter Notebook to study the factors that affect flight delays. The main objective is to predict whether a flight will be delayed by more than 3 hours and to analyse the most common types of delays in such circumstances.

## Content

1. **`data.csv`**: CSV file with flight data.
2. **`flight_delay_analysis.ipynb`**: Jupyter Notebook script that performs data analysis, pre-processing, and modelling for flight delay prediction.

## Dataset

The `data.csv` file contains the following information about the flights:
- **DayOfWeek**: Day of the week (1 = Monday, 7 = Sunday)
- **Date**: Scheduled date
- **DepTime**: Actual departure time (local, hhmm)
- **ArrTime**: Actual arrival time (local, hhmm)
- **CRSArrTime**: Scheduled arrival time (local, hhmm)
- **UniqueCarrier**: Carrier code
- **Airline**: Airline
- **FlightNum**: Flight number
- **TailNum**: Aircraft tail number
- **ActualElapsedTime**: Actual time in the air (in minutes) with TaxiIn/Out
- **CRSElapsedTime**: Estimated flight time (in minutes)
- **AirTime**: Flight time (in minutes)
- **ArrDelay**: Difference in minutes between scheduled and actual arrival time
- **Origin**: IATA code of the airport of origin
- **Org_Airport**: Name of the airport of origin
- **Dest**: IATA code of the destination airport
- **Dest_Airport**: Name of the destination airport
- **Distance**: Distance between airports (miles)
- **TaxiIn**: Time of arrival and arrival at the door of the destination airport, in minutes
- **TaxiOut**: Time elapsed between departure from the origin airport and take-off, in minutes
- **Cancelled**: Was the flight cancelled? 1 = yes, 0 = no
- **CancellationCode**: Reason for cancellation
- **Diverted**: Was the flight diverted? 1 = yes, 0 = no
- **CarrierDelay**: Delay on the part of the carrier (in minutes)
- **WeatherDelay**: Delay due to weather (in minutes)
- **NASDelay**: Delay by the national aviation system (in minutes)
- **SecurityDelay**: Delay due to security (in minutes)
- **LateAircraftDelay**: Delay for this reason (in minutes)

## Requisites

To run Jupyter Notebook, you will need to have the following packages installed in your Python environment:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install these packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Execute

1. Clone this repository:
```bash
git clone https://github.com/nevalego/flights-delay.git
```
2. Navegate to the repo directory:
```bash
cd flights-delay
```
3. Open Jupyter Notebook:
```bash
jupyter notebook flight_delay_analysis.ipynb
```

## Description of the Analysis

### 1. Characterisation of the Dataset

1. Display the first rows of the dataset: An initial sample of the data is inspected to get an overview.
2. General information and statistical description: Detailed information about the data types and a statistical description of the numerical variables is provided.
    3. Number of classes of the target variable: The number of classes for the target variable (delay > 3 hours) and the type of values it takes are determined.
4. Total number of instances: The total size of the dataset is displayed.
5. Missing values: Missing values in the dataset are identified and quantified.

### 2. Exploratory Data Analysis (EDA)

1. Distribution of arrival delays: The distribution of arrival delays is displayed using histograms and density graphs.
2. Arrival delays by delay type: Arrival delays are analysed based on whether the delay exceeds 3 hours or not.
    3. Relationship between delays and distance: The relationship between flight distance and arrival delay is explored using scatter plots.
4. Actual time vs estimated flight time: Actual flight time is compared with estimated time to assess the accuracy of estimates.

### 3. Data Preprocessing

1. Treatment of missing values: Missing values are replaced with the mode of each column.
2. Treatment of duplicate values: Duplicate rows are removed to ensure the quality of the dataset.
3. Coding of categorical variables: Categorical variables are coded into numerical values for use in modelling.


### 4. Modelling

 1. Dataset division: The dataset is divided into training and test sets (80% training, 20% test).
2. Linear regression model training: A linear regression model is trained using the training set.
3. Model evaluation: The model's performance is evaluated using metrics such as mean square error (MSE) and coefficient of determination (R^2). The model's predictions are compared with the actual values in a graph.
