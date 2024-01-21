# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 23:17:38 2024

@author: dileep
"""

# Updated code with new indicators and years

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns

# Function to read data from a CSV file and preprocess it
def read_and_preprocess_data(filename):
    # Read the CSV file and skip the first 4 rows
    df = pd.read_csv(filename, skiprows=4)

    # Drop unnecessary columns
    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    df = df.drop(cols_to_drop, axis=1)

    # Rename columns for clarity
    df = df.rename(columns={'Country Name': 'Country'})

    # Reshape the data using the melt function
    df = df.melt(id_vars=['Country', 'Indicator Name'], var_name='Year', value_name='Value')

    # Convert 'Year' and 'Value' columns to numeric
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # Create pivot tables for years and countries
    df_years = df.pivot_table(index=['Country', 'Indicator Name'], columns='Year', values='Value')
    df_countries = df.pivot_table(index=['Year', 'Indicator Name'], columns='Country', values='Value')

    # Drop columns with all NaN values
    df_years = df_years.dropna(how='all', axis=1)
    df_countries = df_countries.dropna(how='all', axis=1)

    return df_years, df_countries

# Function to subset data based on countries, indicators, and years
def subset_data_for_indicators(df_years, countries, indicators, start_year, end_year):
    df = df_years.loc[(countries, indicators), start_year:end_year]
    df = df.transpose()
    return df

# Function to plot a violin plot of correlation coefficients
def plot_corr_violin(df):
    df_corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")
    sns.violinplot(x=df_corr.values.flatten(), inner="quartile", palette="viridis")
    plt.title('Correlation Violin Plot for Selected Countries', fontsize=14)
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.show()

# Function to normalize a dataframe using StandardScaler
def normalize_dataframe(df):
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized

# Function to perform K-Means clustering on a dataframe
def perform_kmeans(df, num_clusters):
    # Handling missing values by imputing with mean
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Applying K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(df_imputed)
    return cluster_labels, kmeans.cluster_centers_

# Function to print a summary of data points in each cluster
def print_cluster_summary(cluster_labels):
    cluster_counts = np.bincount(cluster_labels)
    for i, count in enumerate(cluster_counts):
        print(f"Cluster {i+1}: {count} data points")

# Function to filter data based on countries, indicators, and years
def filter_data_for_indicators(filename, countries, indicators, start_year, end_year):
    df = pd.read_csv(filename, skiprows=4)
    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    df = df.drop(cols_to_drop, axis=1)
    df = df.rename(columns={'Country Name': 'Country'})
    df = df[df['Country'].isin(countries) & df['Indicator Name'].isin(indicators)]
    df = df.melt(id_vars=['Country', 'Indicator Name'], var_name='Year', value_name='Value')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.pivot_table(index=['Country', 'Indicator Name'], columns='Year', values='Value')
    df = df.loc[:, start_year:end_year]
    return df

# Function for an exponential growth model
def exponential_growth(x, a, b):
    return a * np.exp(b * x)

# Function to calculate confidence interval for curve fitting
def confidence_interval(xdata, ydata, popt, pcov, alpha=0.05):
    n = len(ydata)
    m = len(popt)
    df = max(0, n - m)
    tval = -1 * stats.t.ppf(alpha / 2, df)
    residuals = ydata - exponential_growth(xdata, *popt)
    stdev = np.sqrt(np.sum(residuals**2) / df)
    ci = tval * stdev * np.sqrt(1 + np.diag(pcov))
    return ci

# Function to plot future values based on an exponential growth model
def plot_future_for_indicator(df, countries, indicator, start_year, end_year):
    data = filter_data_for_indicators(df, countries, [indicator], start_year, end_year)
    growth_rate = np.zeros(data.shape)
    for i in range(data.shape[0]):
        popt, pcov = curve_fit(exponential_growth, np.arange(data.shape[1]), data.iloc[i])
        ci = confidence_interval(np.arange(data.shape[1]), data.iloc[i], popt, pcov)
        growth_rate[i] = popt[1]

    fig, ax = plt.subplots()
    for i in range(data.shape[0]):
        ax.plot(np.arange(data.shape[1]), data.iloc[i],
                label=data.index.get_level_values('Country')[i])
    ax.set_xlabel('Year')
    ax.set_ylabel(f'{indicator} Value')
    ax.set_title(indicator)
    ax.legend(loc='best')
    plt.show()

# Function to plot clustered data points with K-Means centers
def plot_clustered_data(df, cluster_labels, cluster_centers):
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1],
                         c=cluster_labels, cmap='rainbow')
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
               s=200, marker='h', c='black')
    ax.set_xlabel(df.columns[0], fontsize=12)
    ax.set_ylabel(df.columns[1], fontsize=12)
    ax.set_title("K-Means Clustering Results", fontsize=14)
    ax.grid(True)
    plt.colorbar(scatter)
    plt.show()

if __name__ == '__main__':
    # Read data from the specified CSV file
    df_years, df_countries = read_and_preprocess_data(r"worlddata.csv")

    # Define countries, indicators, and subset the data
    indicators = [
        'Arable land (% of land area)',
        'Population growth (annual %)'
    ]
    countries = ['India', 'United States', 'United Kingdom']
    start_year, end_year = 1980, 2018
    df = subset_data_for_indicators(df_years, countries, indicators, start_year, end_year)

    # Normalize the dataframe using StandardScaler
    df_normalized = normalize_dataframe(df)

    # Perform K-Means clustering
    num_clusters = 3
    cluster_labels, cluster_centers = perform_kmeans(df_normalized, num_clusters)

    # Display clustering results and plot clustered data
    print("Clustering Results points cluster_centers")
    print(cluster_centers)
    plot_clustered_data(df_normalized, cluster_labels, cluster_centers)

    # Plot future values based on exponential growth model
    plot_future_for_indicator(r"worlddata.csv", countries, 'Population growth (annual %)', start_year, end_year)

    # Plot a violin plot of correlation coefficients
    plot_corr_violin(df)

    # Display a summary of data points in each cluster
    print_cluster_summary(cluster_labels)
