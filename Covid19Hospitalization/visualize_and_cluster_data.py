import numpy as np
import pandas as pd
import os
from numpy import genfromtxt
from matplotlib import pyplot as plt
import seaborn as sns;

from src.make_df import normalize_and_scale_df, merge_dfs

sns.set_theme();

states_data = np.array(
    ['US-AK', 'US-DC', 'US-DE', 'US-HI', 'US-ID', 'US-ME', 'US-MT', 'US-ND', 'US-NE', 'US-NH', 'US-NM', 'US-RI',
     'US-SD', 'US-VT', 'US-WV', 'US-WY'], dtype=object)
dates_data = np.array(
    ['2020-03-09', '2020-03-16', '2020-03-23', '2020-03-30', '2020-04-06', '2020-04-13', '2020-04-20', '2020-04-27',
     '2020-05-04', '2020-05-11',
     '2020-05-18', '2020-05-25', '2020-06-01', '2020-06-08', '2020-06-15', '2020-06-22', '2020-06-29', '2020-07-06',
     '2020-07-13', '2020-07-20',
     '2020-07-27', '2020-08-03', '2020-08-10', '2020-08-17', '2020-08-24', '2020-08-31', '2020-09-07', '2020-09-14',
     '2020-09-21'], dtype=object)

PROJ_DIR = os.path.dirname(os.getcwd())


def visualize_search_trends_evolution(df):
    # focusing on Aphonia, Dysautonomia, Shallow breathing, Ventricular fibrillation (threshhold of 0.67)
    # column 2, 5, 11, 14
    # hard coded since we manually selected the symptoms we want to track
    # everything shifted by 1 column from adding dates column
    symptoms_wanted = df.values[:, [2, 5, 11, 14]].astype(np.float)

    aphonia_data = symptoms_wanted[:, 0].reshape(len(dates_data), len(states_data))
    dysautonomia_data = symptoms_wanted[:, 1].reshape(len(dates_data), len(states_data))
    shallow_breathing_data = symptoms_wanted[:, 2].reshape(len(dates_data), len(states_data))
    ventricular_fibrillation_data = symptoms_wanted[:, 3].reshape(len(dates_data), len(states_data))

    # Plotting graphs
    plt.figure(figsize=(30, 30))

    plt.subplot(2, 2, 1)
    aphonia_graph = sns.heatmap(aphonia_data, yticklabels=dates_data, cmap="PiYG", vmin=-3, vmax=4,
                                cbar_kws={'label': 'Normalized Popularity'})
    aphonia_graph.set_title('Evolution of search trends for Aphonia', fontdict={'fontsize': 14})
    aphonia_graph.set_xlabel('States', fontdict={'fontsize': 14})
    aphonia_graph.set_ylabel('Dates', fontdict={'fontsize': 14})
    aphonia_graph.set_xticklabels(states_data)

    plt.subplot(2, 2, 2)
    dysautonomia_graph = sns.heatmap(dysautonomia_data, yticklabels=dates_data, cmap="PiYG", vmin=-3, vmax=4,
                                     cbar_kws={'label': 'Normalized Popularity'})
    dysautonomia_graph.set_title('Evolution of search trends for Dysautonomia', fontdict={'fontsize': 14})
    dysautonomia_graph.set_xlabel('States', fontdict={'fontsize': 14})
    dysautonomia_graph.set_ylabel('Dates', fontdict={'fontsize': 14})
    dysautonomia_graph.set_xticklabels(states_data)

    plt.subplot(2, 2, 3)
    shallow_breathing_graph = sns.heatmap(shallow_breathing_data, yticklabels=dates_data, cmap="PiYG", vmin=-3, vmax=4,
                                          cbar_kws={'label': 'Normalized Popularity'})
    shallow_breathing_graph.set_title('Evolution of search trends for Shallow Breathing', fontdict={'fontsize': 14})
    shallow_breathing_graph.set_xlabel('States', fontdict={'fontsize': 14})
    shallow_breathing_graph.set_ylabel('Dates', fontdict={'fontsize': 14})
    shallow_breathing_graph.set_xticklabels(states_data)

    plt.subplot(2, 2, 4)
    ventricular_fibrillation_graph = sns.heatmap(ventricular_fibrillation_data, yticklabels=dates_data, cmap="PiYG",
                                                 vmin=-3, vmax=4, cbar_kws={'label': 'Normalized Popularity'})
    ventricular_fibrillation_graph.set_title('Evolution of search trends for Ventricular Fibrillation',
                                             fontdict={'fontsize': 14})
    ventricular_fibrillation_graph.set_xlabel('States', fontdict={'fontsize': 14})
    ventricular_fibrillation_graph.set_ylabel('Dates', fontdict={'fontsize': 14})
    ventricular_fibrillation_graph.set_xticklabels(states_data)

    plt.show()

def visualize_hospital_data_evolution(df):
    hospital_data = df.values[:, [16,17]].astype(np.float)
    hospital_cum = hospital_data[:, 0].reshape(len(dates_data), len(states_data))
    hospital_new = hospital_data[:, 1].reshape(len(dates_data), len(states_data))

    # Plotting graphs
    plt.figure(figsize=(30, 15))
    plt.subplot(1, 2, 1)
    hospital_graph_cum = sns.heatmap(hospital_cum, yticklabels=dates_data, cmap="PiYG", vmin=-3, vmax=4,
                                cbar_kws={'label': 'Normalized Popularity'})
    hospital_graph_cum.set_title('Evolution of cumulative hospitalized', fontdict={'fontsize': 14})
    hospital_graph_cum.set_xlabel('States', fontdict={'fontsize': 14})
    hospital_graph_cum.set_ylabel('Dates', fontdict={'fontsize': 14})
    hospital_graph_cum.set_xticklabels(states_data)

    plt.subplot(1, 2, 2)

    hospital_graph_new = sns.heatmap(hospital_new, yticklabels=dates_data, cmap="PiYG", vmin=-3, vmax=4,
                                 cbar_kws={'label': 'Normalized Popularity'})
    hospital_graph_new.set_title('Evolution of new hospitalized', fontdict={'fontsize': 14})
    hospital_graph_new.set_xlabel('States', fontdict={'fontsize': 14})
    hospital_graph_new.set_ylabel('Dates', fontdict={'fontsize': 14})
    hospital_graph_new.set_xticklabels(states_data)

    plt.savefig(os.path.join(PROJ_DIR, 'figures', 'hospital_data_normalized_by_date'), dpi=300)

    plt.show()

if __name__ == '__main__':
    merge = merge_dfs()
    scaled_date, _ = normalize_and_scale_df(merge, 'date')
    scaled_rgn, _ = normalize_and_scale_df(merge, 'region')
    visualize_hospital_data_evolution(scaled_date)
    # visualize_hospital_data_evolution(scaled_rgn)

    visualize_search_trends_evolution(scaled_date)
    # visualize_search_trends_evolution(scaled_rgn)
