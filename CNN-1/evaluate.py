import csv
import os
import numpy as np

from six.moves import cPickle as pickle

from log import log
import util


def write_results_to_csv(results, header=None):
	
	
	file_name = "results_" + util.get_current_time_as_string() + ".csv"
	

	with open(file_name, "w", newline='' ) as f:
		writer = csv.writer(f)
		if header is not None:
			writer.writerow(header)

		for row in results:
			writer.writerow(row)
			#x, y, y_pred = row
			#y_pred = round(y_pred.flatten().flatten().tolist()[0])
			#writer.writerow([x, y, y_pred])


	log("Eval results written to: " + file_name)

	


