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
    new_df = input_df[(input_df['item_name'].str.startswith('S'))]
    return new_df['item_name'].drop_duplicates()

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
    new_df = input_df.copy()
    def quartile(item_price):
        if 30 <= item_price:
            return 'premium'
        elif 20 <= item_price <= 29.99:
            return 'high-cost'
        elif 10 <= item_price <= 19.99:
            return 'medium-cost'
        else:
            return 'low-cost'

    new_df['Quartile'] = new_df['item_price'].apply(quartile)
    return new_df

def average_price_in_quartiles(input_df):
    new_df = input_df.copy()
    return new_df.groupby('Quartile')['item_price'].mean()

def minmaxmean_price_in_quartile(input_df):
    new_df = input_df.copy()
    return new_df.groupby('Quartile')['item_price'].agg(['min', 'max', 'mean'])

import random
from typing import List

from src.utils import distributions as dist
from src.weekly import weekly_test_1 as w1
from src.weekly import weekly_test_2 as w2


def gen_uniform_mean_trajectories(distribution: dist.UniformDistribution, number_of_trajectories, length_of_trajectory):
    res = []
    distribution.rand.seed(42)
    for i in range(0, number_of_trajectories):
        rand_num = []
        for j in range(0, length_of_trajectory):
            rand_num.append(distribution.gen_rand())
        res.append(w1.cumavg_list(rand_num))
    return res

def gen_logistic_mean_trajectories(distribution: dist.LogisticDistribution, number_of_trajectories,
                                   length_of_trajectory):
    res = []
    distribution.rand.seed(42)
    for i in range(0, number_of_trajectories):
        rand_num = []
        for j in range(0, length_of_trajectory):
            rand_num.append(distribution.gen_rand())
        res.append(w1.cumavg_list(rand_num))
    return res



def gen_laplace_mean_trajectories(distribution: w2.LaplaceDistribution, number_of_trajectories, length_of_trajectory):
    res = []
    distribution.rand.seed(42)
    for i in range(0, number_of_trajectories):
        rand_num = []
        for j in range(0, length_of_trajectory):
            rand_num.append(distribution.gen_rand())
        res.append(w1.cumavg_list(rand_num))
    return res


def gen_cauchy_mean_trajectories(distribution: dist.CauchyDistribution, number_of_trajectories, length_of_trajectory):
    res = []
    distribution.rand.seed(42)
    for i in range(0, number_of_trajectories):
        rand_num = []
        for j in range(0, length_of_trajectory):
            rand_num.append(distribution.gen_rand())
        res.append(w1.cumavg_list(rand_num))
    return res


def gen_chi2_mean_trajectories(distribution: dist.ChiSquaredDistribution, number_of_trajectories, length_of_trajectory):
    res = []
    distribution.rand.seed(42)
    for i in range(0, number_of_trajectories):
        rand_num = []
        for j in range(0, length_of_trajectory):
            rand_num.append(distribution.gen_rand())
        res.append(w1.cumavg_list(rand_num))
    return res
