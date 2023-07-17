# module imports
import os
import time
# import matplotlib.pyplot as plt
import pandas
import threading

# files import
from Preprocessing import *
from Calculation import *
from Analyze import *


def read_csv(pathOfCSV: str) -> pandas.core.frame.DataFrame:
    """
    reads csv file and returns its data
    :param pathOfCSV: directory path where population.csv is stored
    :return: dataframe of the csv. index=["fov", "label", "cluster_name"]
    """

    col_names = ["fov", "label", "cluster_name"]
    data = pandas.read_csv(pathOfCSV, usecols=col_names)
    return data


def handle(name: str, labels: pandas.core.frame.DataFrame) -> None:
    """
    procedure called for every fov. most important function
    :param labels: csv file as dataframe
    :param name: name of fov
    :return: [0]: entire raw data- distances. [1]: analyzed (basic analyze) data
    """

    # debug
    print(f'{name} started!')

    # remove noise of fov
    mask = remove_noise(name)

    # get cells mass center points to find distance from segmentation
    segmentsPath = str(
        "C:\\Users\\roeyb\\PycharmProjects\\Alpha\\data\\data\\" + name + "\\TIFs\\segmentation_labels_merged.tif")
    centers = read_cell_segments(segmentsPath)

    # calculate distances
    # create distance transform from edges
    distance_map = distance_transform(mask)
    # create array of cell index * distance
    final_distance_list = indexByDistance(distance_map, centers)

    # identify using csv

    raw_data = assign(labels, name, final_distance_list)

    # update global dicts
    # assign- raw data. basic analysis
    global entire_data
    global analyzed_data
    entire_data[name], analyzed_data[name] = raw_data, basic_analyze(raw_data)

    # debug
    print(f'{name} ended!')

    return


def main() -> None:
    """
    the main function. calls everything and assembles result dictionaries
    :return: None
    """

    global entire_data
    global analyzed_data
    global final_data

    # debug
    # start counting time
    t0 = time.time()

    # directory path
    path = "C:\\Users\\roeyb\\PycharmProjects\\Alpha\\data\\data"
    dir_list = os.listdir(path)

    # ignore everything
    np.seterr(all='ignore')

    pathOfCSV = "C:\\Users\\roeyb\\PycharmProjects\\Alpha\\data\\population_updated.csv"
    labels = read_csv(pathOfCSV)

    # main loop
    threads = []
    for fov in dir_list:
        threads.append(threading.Thread(target=handle, args=(fov, labels, )))

    # start all the threads in the list
    for t in threads:
        t.start()

    # join all the threads in the list
    for t in threads:
        t.join()

    # create density plots for each cell class
    #final_data = create_final_data(analyzed_data)

    # debug
    # stop counting time
    t1 = time.time()
    print(f"Run time: {t1 - t0} s")

    # box plot means
    box_plot(analyzed_data)

    return


if __name__ == "__main__":
    # code starts here
    entire_data = {}
    analyzed_data = {}
    final_data = {}
    main()
    exit(0)
