#F0D5F1

import pandas
import pandas as pd
from typing import List, Dict
import matplotlib
import matplotlib.pyplot as plt

euro12 = pd.read_csv('../data/Euro_2012_stats_TEAM.csv')

def number_of_participants(input_df):
    res = len(input_df['Team'])
    return res

def goals(input_df):
    return input_df[['Team', 'Goals']]

def sorted_by_goal(input_df):
    df = goals(input_df)
    return df.sort_values(by = 'Goals', ascending = False)

def avg_goal(input_df):
    return input_df['Goals'].mean()


def countries_over_five(input_df):
    selected_df = input_df[input_df['Goals'] >= 6]
    return selected_df

def countries_starting_with_g(input_df):
    selected_df = input_df[input_df['Team'].str.startswith('G')]
    return selected_df['Team']

def first_seven_columns(input_df):
    return input_df.iloc[:, :7]

def every_column_except_last_three(input_df):
    return input_df.iloc[:, :-3]

def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    selected_columns = input_df[columns_to_keep]
    filtered_rows = input_df[input_df[column_to_filter].isin(rows_to_keep)]
    result_df = filtered_rows[selected_columns]
    return result_df


def generate_quartile(input_df):
    input_df['Quartile'] = pd.cut(input_df['Goals'], bins=[-1, 2, 4, 5, 12], labels=[4, 3, 2, 1])
    return input_df


def average_yellow_in_quartiles(input_df):
    input_df['Quartile'] = pd.cut(input_df['Goals'], bins=[-1, 2, 4, 5, 12], labels=[4, 3, 2, 1])
    quartile_passes = input_df.groupby('Quartile')['Passes'].mean().reset_index()
    return quartile_passes


def minmax_block_in_quartile(input_df):
    input_df['Quartile'] = pd.cut(input_df['Goals'], bins=[-1, 2, 4, 5, 12], labels=[4, 3, 2, 1])
    quartile_minmax = input_df.groupby('Quartile')['Blocks'].agg(['min', 'max']).reset_index()
    return quartile_minmax

def scatter_goals_shots(input_df):
    plt.figure(figsize=(8, 6))
    plt.scatter(input_df['Goals'], input_df['Shots on target'])
    plt.title('Goals and Shot on target')
    plt.xlabel('Goals')
    plt.ylabel('Shots on target')
    plt.grid(True)

    return plt

def scatter_goals_shots_by_quartile(input_df):
    input_df['Quartile'] = pd.cut(input_df['Goals'], bins=[-1, 2, 4, 5, 12], labels=[4, 3, 2, 1])
    colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange'}
    plt.figure(figsize=(8, 6))
    for quartile, color in colors.items():
        subset = input_df[input_df['Quartile'] == quartile]
        plt.scatter(subset['Goals'], subset['Shots on target'], label=f'Quartile {quartile}', color=color)

    plt.title('Goals and Shot on target')
    plt.xlabel('Goals')
    plt.ylabel('Shots on target')
    plt.grid(True)
    plt.legend(title='Quartiles')

    return plt

from typing import List
from random import seed, random
import matplotlib.pyplot as plt

def gen_pareto_mean_trajectories(pareto_distribution, number_of_trajectories, length_of_trajectory):
    seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):
        samples = [pareto_distribution.rvs() for _ in range(length_of_trajectory)]
        cumulative_means = [sum(samples[:i + 1]) / (i + 1) for i in range(length_of_trajectory)]
        trajectories.append(cumulative_means)

    return trajectories