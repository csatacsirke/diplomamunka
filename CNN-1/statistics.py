import csv
import os
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


from log import log
import util

def get_default_file_name(postfix=""):
	return "results_" + util.get_current_time_as_string() + postfix + ".csv"

def write_results_to_csv(results, header=None, postfix="", file_name=None):
	if file_name is None:
		file_name = get_default_file_name(postfix=postfix)
	
	

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


def create_roc_curve_plot(roc_curve, roc_auc):
	fpr, tpr, thresholds = roc_curve

	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
			 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.show()	


def normalize(array):
	array = np.subtract(array, np.min(array))
	array = np.divide(array, np.max(array))
	return array

def one_minus(array):
	array = np.subtract(1.0, array)
	return array

def mean_squared_error(a, b):
	return np.sum(np.square(a-b))


class IntervalTransform:

	def __init__(self, array):
		self.min = np.min(array)
		self.max = np.max(array)

	def transform(self, value):
		value = np.subtract(value, self.min)
		value = np.divide(value, self.max - self.min)
		value = one_minus(value)
		return value 

	def inverse_transform(self, value):
		value = one_minus(value)
		value = value * (self.max - self.min) + self.min
		return value


def eval(results):
	# record = (MSE, y, file_name)

	r = [list(a) for a in zip(*results)]
	#r = list(zip(*results))
	#print(r)

	MSE_list = r[0]
	y_list = r[1]

	
	y_list = np.array(y_list).astype(np.float)
	MSE_list = np.array(MSE_list).astype(np.float)

	tr = IntervalTransform(MSE_list)

	MSE_list = tr.transform(MSE_list)
	#MSE_list = one_minus(normalize(MSE_list))

	print(MSE_list)
	#MSE_list = np.divide(MSE_list, np.max(MSE_list))
	#MSE_list = MSE_list / max(MSE_list)

	# TODO
	#ex_has_prob_hatar = 0.5
	ex_has_prob_hatar = 0.8

	visualize_data = True
	if visualize_data:
		roc_auc_score = metrics.roc_auc_score(y_list, MSE_list)
		roc_curve = metrics.roc_curve(y_list, MSE_list)
		create_roc_curve_plot(roc_curve, roc_auc_score)
	
	ex_has_hatar = tr.inverse_transform(ex_has_prob_hatar)

	log("hatar: ", ex_has_hatar)

	predictions = []
	for record in results:
		# (MSE, y, file_name)
		MSE, y, file_name = record
		MSE = float(MSE)
		pred = 1 if MSE < ex_has_hatar else 0


		#log(MSE, "->", tr.transform(MSE))
		log(MSE, "->", pred, "/", y)

		predictions.append((MSE, y, pred, file_name))
		#predictions = {
		#	"MSE": MSE,
		#	"y": y,
		#	"y_pred": pred,
		#	"file" : file_name
		#}
	header = ("MSE", "y", "pred", "file_name")
	#write_results_to_csv(predictions, header=header, postfix=".stage3")			

	return


def main():
	csv_file_name = util.get_last_file_with_ending("stage2.csv")
	f = open(csv_file_name, "r")
	
	reader = csv.reader(f, delimiter=',')

	rows = []
	for row in reader:
		rows.append(row)

	eval(rows)
	return

if __name__ == "__main__":
	main()




