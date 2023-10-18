#F0D5F1

import pandas as pd
import random
import matplotlib.pyplot as plt

from pathlib import Path



def change_price_to_float(input_df):
    new_df = input_df.copy()
    new_df['item_price'] = new_df['item_price'].str.replace('$', '').astype(float)

    return new_df



def number_of_observations(input_df):
    new_df = input_df.copy()
    return len(new_df)

def items_and_prices(input_df):
    new_df = input_df.copy()
    return new_df[['item_name', 'item_price']]

def sorted_by_price(input_df):
    new_df = input_df.copy()
    return new_df.sort_values(by='item_price', ascending=False)

def avg_price(input_df):
    new_df = input_df.copy()
    return new_df['item_price'].mean()


def unique_items_over_ten_dollars(input_df):
    new_df = input_df.copy()
    filtered_df = new_df[new_df['item_price'] > 10]

    unique_items = filtered_df.drop_duplicates(subset=['item_name', 'choice_description', 'item_price'])

    return unique_items [['item_name', 'choice_description', 'item_price']]


def items_starting_with_s(input_df):
     new_df = input_df['item_name'][input_df['item_name'].str.startswith('S')]
     return new_df['item_name']

def first_three_columns(input_df):
    new_df = input_df.copy()
    return new_df.iloc[:, :3]


def every_column_except_last_two(input_df):
    new_df = input_df.copy()
    return new_df.iloc[:, :-2]

def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    filtered_df = input_df[input_df[column_to_filter].isin(rows_to_keep)]
    return filtered_df[columns_to_keep]

def generate_quartile(input_df):
    def assign_quartile(item_price):
        if item_price >= 30:
            return 'premium'
        elif 20 <= item_price < 30:
            return 'high-cost'
        elif 10 <= item_price < 20:
            return 'medium-cost'
        else:
            return 'low-cost'

    input_df['Quartile'] = input_df['item_price'].apply(assign_quartile)

    return input_df

def average_price_in_quartiles(input_df):
    new_df = input_df.copy()
    return new_df.groupby('Quartile')['item_price'].mean()

def minmaxmean_price_in_quartile(input_df):
    new_df = input_df.copy()
    return new_df.groupby('Quartile')['item_price'].agg(['min', 'max', 'mean'])

import random
from typing import List
def gen_uniform_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory, seed=42):
    random.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):
        trajectory = []
        cumulative_sum = 0.0

        for _ in range(length_of_trajectory):
            random_value = random.uniform(0, 1)
            cumulative_sum += random_value
            avg_value = cumulative_sum / (len(trajectory) + 1)
            trajectory.append(avg_value)

        trajectories.append(trajectory)

    return trajectories

def gen_logistic_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory, seed=42) -> List[List[float]]:
    random.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):
        trajectory = []
        cumulative_sum = 0.0

        for _ in range(length_of_trajectory):
            random_value = random.random()
            cumulative_sum += random_value
            avg_value = cumulative_sum / (len(trajectory) + 1)
            trajectory.append(avg_value)

        trajectories.append(trajectory)

    return trajectories

def gen_laplace_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory, seed=42) -> List[List[float]]:
    random.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):
        trajectory = []
        cumulative_sum = 0.0

        for _ in range(length_of_trajectory):
            random_value = laplace_distribution.gen.rand(1, 3.3)
            cumulative_sum += random_value
            trajectory.append(cumulative_sum)

        trajectories.append(trajectory)

    return trajectories

def gen_cauchy_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory, seed=42) -> List[List[float]]:
    random.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):
        trajectory = []
        cumulative_sum = 0.0

        for _ in range(length_of_trajectory):
            random_value = distribution(2, 4)
            cumulative_sum += random_value
            trajectory.append(cumulative_sum)

        trajectories.append(trajectory)

    return trajectories

def gen_chi2_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory, seed=42) -> List[List[float]]:
    random.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):
        trajectory = []
        cumulative_sum = 0.0

        for _ in range(length_of_trajectory):
            random_value = random.uniform(3)
            cumulative_sum += random_value
            avg_value = cumulative_sum / (len(trajectory) + 1)
            trajectory.append(avg_value)

        trajectories.append(trajectory)

    return trajectories