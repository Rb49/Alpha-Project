# module imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import os


def assign(data: pd.core.frame.DataFrame, name: str, final_distance_list: np.ndarray) -> dict:
	"""
    creates a list of the distances of every cell from a cell class for every type
    :param data: dataframe of the population csv
    :param name: name of fov
    :param final_distance_list: dict. key = cell type. value = list of distances
    :return:
    """

	global _cell_classes

	# create dict
	distancesClass = {cell: [] for cell in _cell_classes}

	fov_data = data[data['fov'] == name]

	for label, cluster_name in zip(fov_data['label'], fov_data['cluster_name']):
		distancesClass[cluster_name].append(final_distance_list[label][0])

	return distancesClass


def basic_analyze(entire_data: dict) -> dict:
	"""
    analyzes the raw data and normalizes it
    :param entire_data: dictionary of the entire raw data of the fov
    :return: dict of analyzed data (basic)
    stats = {"Sorted normalized data": [], "Length of data": 0, "Mean": 0, "Median": 0, "Standard deviation": 0}
    if distance list is empty, value is None
    """

	global _cell_classes

	# just some basic statistic data
	analyzeClass = {cell: {"Sorted normalized data": [], "Length of data": 0, "Mean": 0,
						   "Median": 0, "Standard deviation": 0} for cell in _cell_classes}

	for cell_class in _cell_classes:

		arr = entire_data[cell_class]
		if len(arr) > 0:

			# rescaling (min-max normalization)- doesn't have effect
			# minX, maxX = min(arr), max(arr)
			# min_val, max_val = minX, maxX
			# arr = sorted(list(map(lambda x: (x - minX) / (maxX - minX) * (max_val - min_val) + min_val, arr)))
			arr = sorted(arr)

			analyzeClass[cell_class]["Sorted normalized data"] = arr
			# length
			analyzeClass[cell_class]["Length of data"] = len(arr)
			# mean
			analyzeClass[cell_class]["Mean"] = np.mean(arr)
			# median
			analyzeClass[cell_class]["Median"] = arr[len(arr) // 2]
			# standard deviation
			analyzeClass[cell_class]["Standard deviation"] = np.std(arr)

		else:
			analyzeClass[cell_class] = None

	return analyzeClass


def create_final_data(entire_data: dict) -> dict:
	"""
    combines all the analyzed data of all fov
    creates density plots for every cell type and displays them
    :param entire_data:
    :return: dict. key = cell type. value = plt.plot() type
    """

	global _cell_classes

	final_data = {cell: {fov: None for fov in entire_data.keys()} for cell in _cell_classes}

	for cell in final_data.keys():
		for fov in final_data[cell].keys():
			counts = entire_data[fov][cell]

			# create density plots
			plot = plt.figure()

			plt.hist(counts, figure=plot, alpha=1, color="blue")

			# name title and axis and legend
			plt.xlabel("Distance (pixels)", figure=plot)
			plt.ylabel("Count", figure=plot)
			plt.title(fov, figure=plot)

			# save plot to dict
			final_data[cell][fov] = plot

			# show plot
			# plt.show()
			# close plt to not flood kernel
			plt.close()

	return final_data


def box_plot(analyzed_data: dict) -> None:
	"""
    create box plots for means
    :param analyzed_data:
    :return:
    """

	cell_classes = ["Goblet", "Paneth","CD4 T", "CD8 T",
					 "Neutrophils", "Macrophages", "Plasma"]

	data1 = {cell: [] for cell in cell_classes}
	data2 = {cell: [] for cell in cell_classes}

	for fov in analyzed_data:
		if fov.count('Control') != 0:
			for cell_class in cell_classes:
				if analyzed_data[fov][cell_class]:
					data1[cell_class].append(analyzed_data[fov][cell_class]['Mean'])
		else:
			for cell_class in cell_classes:
				if analyzed_data[fov][cell_class]:
					data2[cell_class].append(analyzed_data[fov][cell_class]['Mean'])

	values1 = [data1[key] for key in data1.keys()]
	values2 = [data2[key] for key in data2.keys()]
	plt.boxplot(values1, patch_artist=True, boxprops=dict(facecolor='blue'), positions=np.arange(len(values1)) + 0.2,
				widths=0.3)
	plt.boxplot(values2, patch_artist=True, boxprops=dict(facecolor='orange'), positions=np.arange(len(values2)) - 0.2,
				widths=0.3)
	plt.xticks(range(0, len(data1.keys())), data1.keys(), rotation=45, fontsize=16)
	plt.yticks(fontsize=16)
	plt.grid(axis='y')

	# Create a legend
	red_patch = mpatches.Patch(color='orange', label='GvHD')
	blue_patch = mpatches.Patch(color='blue', label='Control')
	plt.legend(handles=[blue_patch, red_patch], loc="upper right", fontsize=20)

	plt.ylabel('Distance (pixels)', fontsize=20)
	plt.xlabel('Cell type', fontsize=20)
	plt.title('Medians of cell types by distance', fontsize=22)

	# Show the plot
	plt.show()
	plt.close()


def subplots18(entire_data: dict) -> None:
	"""

    :param entire_data:
    :return:
    """

	for cell in _cell_classes:
		names = ['2021-12-01_Slide133_GVHD_Cohort_Slide1_run3_FOV1_GVHD_1_FOV4',
				 '2021-12-01_Slide133_GVHD_Cohort_Slide1_run5_FOV1_GVHD_5_FOV1',
				 '2021-12-02T_Slide_134_GVHD_Cohort_Slide2_run1_FOV5_GVHD_9_FOV2_SF',
				 '2021-12-10_Slide142_GVHD_Cohort_Slide10_run2_FOV3_GVHD_53_FOV_3',
				 '2021-12-10_Slide142_GVHD_Cohort_Slide10_run2_FOV6_GVHD_54_FOV_2',
				 '2021-12-13_Slide_141_GVHD_Cohort_Slide_9_run1_FOV6_GVHD_48_FOV_4',
				 '2021-12-15_Slide136_GVHD_Cohort_Slide4_run2_FOV8_GVHD_22_FOV_2',
				 '2021-12-17_Slide137_GVHD_Cohort_Slide5_run3_FOV3_GVHD_27_FOV_5',
				 '2021-12-18_Slide_138_GVHD_Cohort_Slide6_run2_FOV1_GVHD_32_FOV_1',
				 '2021-12-18_Slide_138_GVHD_Cohort_Slide6_run2_FOV2_GVHD_32_FOV_2',
				 '2021-12-18_Slide_138_GVHD_Cohort_Slide6_run2_FOV9_GVHD_34_FOV_2',
				 '2021-12-19_Slide_139_GVHD_Cohort_Slide7_run1_FOV2_GVHD_35_FOV_2',
				 'Control_10_FOV_2',
				 'Control_5_FOV_2',
				 'Control_6_FOV_2',
				 'Control_7_FOV_1',
				 'Control_8_FOV_1',
				 'Control_8_FOV_2']
		names = names[::-1]

		fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(16, 9))
		# plt.tight_layout() # causes problems
		data = []

		for fov in names:
			data.append(entire_data[fov][cell])

		for i, ax in enumerate(axes.flat):
			if i < 18:
				# set name
				if len(names[i].split('GVHD')) == 3:
					subName = names[i].split('GVHD')[2]
					subName = subName[1:]
				else:
					subName = names[i].split('Control')[1]
					subName = subName[1:]

				# create plot hist
				ax.grid(True)

				s = sum(map(lambda x: len(entire_data[names[i]][x]), entire_data[names[i]]))
				x = len(entire_data[names[i]][cell])
				subName = f'{subName}\n{round(x / s * 100, 2)}% of cells'

				ax.hist(data[i], alpha=1, color="blue", density=True)
				# name title and axis and legend
				ax.set_xlabel("Distance (pixels)", fontsize=10)
				ax.set_ylabel("Probability density", fontsize=10)
				ax.set_title(subName, fontsize=12)

		plt.subplots_adjust(wspace=0.6, hspace=0.6)
		fig.suptitle(cell, fontsize=16)
		# plt.show()
		# plt.close()
		plt.savefig(os.path.join('C:\\Users\\roeyb\\PycharmProjects\\Alpha\\hists', f'{cell}.png'))


# global to module only
# cell types that appear in the population csv


_cell_classes = ["Goblet", "Enterocytes", "Fibroblasts", "Plasma", "Neurons",
				 "Macrophages", "Unidentified", "Muscles", "CD4 T", "T_DN",
				 "Mast cells", "Endothel", "CD8 T", "Endocrine", "Other_Immune",
				 "Neutrophils", "Paneth", "Tregs", "B cells"]





'''
def box_plot(analyzed_data: dict) -> None:
    for fov in analyzed_data:
        data1 = {cell: [] for cell in _cell_classes}
        data2 = {cell: [] for cell in _cell_classes}
        if fov.count('Control') != 0:
            for cell_class in analyzed_data[fov]:
                if analyzed_data[fov][cell_class]:
                    data1[cell_class].extend(analyzed_data[fov][cell_class]['Sorted normalized data'])
        else:
            for cell_class in analyzed_data[fov]:
                if analyzed_data[fov][cell_class]:
                    data2[cell_class].extend(analyzed_data[fov][cell_class]['Sorted normalized data'])

        values1 = [data1[key] for key in data1.keys()]
        values2 = [data2[key] for key in data2.keys()]
        plt.boxplot(values1, patch_artist=True, boxprops=dict(facecolor='blue'), positions=np.arange(len(values1))+0.2,widths=0.3)
        plt.boxplot(values2, patch_artist=True, boxprops=dict(facecolor='orange'), positions=np.arange(len(values2))-0.2, widths=0.3)
        plt.xticks(range(0, len(data1.keys())), data1.keys(), rotation=45)
        plt.grid(axis='y')

        # Create a legend
        red_patch = mpatches.Patch(color='orange', label='GVHD')
        blue_patch = mpatches.Patch(color='blue', label='Control')
        plt.legend(handles=[blue_patch, red_patch], loc="upper right")

        plt.ylabel('Distance (pixels)')
        plt.xlabel('Cell type')
        plt.title(f'Means of cell types by distance {fov}')

        # Show the plot
        plt.show()
        plt.close()
'''
