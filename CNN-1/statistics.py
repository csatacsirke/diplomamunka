import csv
import os
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from enum import Enum
import win32api
import argparse



from log import log
import util

csv_file_name_override = None


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
	line_width = 1
	plt.plot(fpr, tpr, color='darkorange',
			 lw=line_width, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=line_width, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.savefig(util.get_current_time_as_string() + "roc_curve.svg")
	plt.show()	

	#plt.savefig("roc_example.svg")
	#import code
	#code.interact(local=locals())


def normalize(array):
	array = np.subtract(array, np.min(array))
	array = np.divide(array, np.max(array))
	return array

def one_minus(array):
	array = np.subtract(1.0, array)
	return array

def mean_squared_error(a, b):
	return np.sum(np.square(a-b))/len(a)


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

def eval_errors(stage_3_results):
	tp = tn = fp = fn = 0
	for entry in stage_3_results:
		MSE, y, pred, file_name = entry
		

		y = int(y)
		pred = int(pred)

		if y == 0 and pred == 0 : tn += 1	
		if y == 0 and pred == 1 : fp += 1	
		if y == 1 and pred == 0 : fn += 1	
		if y == 1 and pred == 1 : tp += 1	
	log("true pos:", tp)
	log("true neg:", tn)
	log("false pos:", fp)
	log("false neg:", fn)



def calc_errors(mse_values, y_values, threshold):
	entries = zip(mse_values, y_values)
	tp = tn = fp = fn = 0

	for entry in entries:
		mse, y = entry
		#mse = float(mse)
		#y = int(y)

		pred = 1 if mse < threshold else 0

		if y == 0 and pred == 0 : tn += 1	
		if y == 0 and pred == 1 : fp += 1	
		if y == 1 and pred == 0 : fn += 1	
		if y == 1 and pred == 1 : tp += 1	

	return (fp, fn)


class ErrorTypes(Enum):
	true_positive = 1
	false_positive = 2
	true_negative = 3
	false_negative = 4
	wat = -1

# In statistical hypothesis testing, a type I error is the incorrect
# rejection of a true null hypothesis (also known as a "false positive" finding),
# while a type II error is incorrectly retaining a false null hypothesis
# (also known as a "false negative" finding).

def get_error_type(y, pred):
	if y == 0 and pred == 0 : return ErrorTypes.true_negative
	if y == 0 and pred == 1 : return ErrorTypes.false_positive
	if y == 1 and pred == 0 : return ErrorTypes.false_negative
	if y == 1 and pred == 1 : return ErrorTypes.true_positive
	return ErrorTypes.wat

def find_sweetspot(stage_2_results):
	
	mse_values = list(map(lambda entry: (lambda MSE, y, file_name: float(MSE))(*entry), stage_2_results))
	y_values = list(map(lambda entry: (lambda MSE, y, file_name: int(y))(*entry), stage_2_results))

	#print(y_values)

	min = np.min(mse_values)
	max = np.max(mse_values)
	steps = 1000

	log(min, " - ", max)
	


	for threshold in np.linspace(min, max, num=steps):
		(fp, fn) = calc_errors(mse_values, y_values, threshold)
		log(threshold, "->", fp, ", ", fn)
		

	return -1


def eval_autoencoder(stage_2_results):
	#find_sweetspot(stage_2_results)
	#return

	# record = (MSE, y, file_name)

	r = [list(a) for a in zip(*stage_2_results)]
	#r = list(zip(*stage_2_results))
	#print(r)

	MSE_list = r[0]
	y_list = r[1]

	
	y_list = np.array(y_list).astype(np.float)
	MSE_list = np.array(MSE_list).astype(np.float)

	
	#_r = map(stage_2_results





	tr = IntervalTransform(MSE_list)

	MSE_list_01 = tr.transform(MSE_list)
	#MSE_list = one_minus(normalize(MSE_list))

	print(MSE_list)
	#MSE_list = np.divide(MSE_list, np.max(MSE_list))
	#MSE_list = MSE_list / max(MSE_list)

	# TODO
	#ex_has_prob_hatar = 0.5
	#ex_has_prob_hatar = 0.0940293102095

	
	#ex_has_hatar = tr.inverse_transform(ex_has_prob_hatar)

	# todo
	ex_has_hatar = 0.0940293102095
	#log("hatar: ", ex_has_hatar)


	predictions = []
	for record in stage_2_results:
		# (MSE, y, file_name)
		MSE, y, file_name = record
		MSE = float(MSE)
		pred = 1 if MSE < ex_has_hatar else 0


		#log(MSE, "->", tr.transform(MSE))
		#log(MSE, "->", pred, "/", y)

		predictions.append((MSE, y, pred, file_name))
		#predictions = {
		#	"MSE": MSE,
		#	"y": y,
		#	"y_pred": pred,
		#	"file" : file_name
		#}


	
	predictions = filter(lambda entry: (lambda MSE, y, pred, file_name: "eredetibol.nyomtatott." not in file_name)(*entry), predictions)

	eval_errors(predictions)
	#result = tkMessageBox.askquestion("Save?", "Save results?", icon='warning')
	#if result == "yes":
	header = ("MSE", "y", "pred", "file_name")

	save_results = False
	if save_results:
		write_results_to_csv(predictions, header=header, postfix=".stage3")

	
	visualize_data = True
	if visualize_data:
		roc_auc_score = metrics.roc_auc_score(y_list, MSE_list_01)
		roc_curve = metrics.roc_curve(y_list, MSE_list_01)
		create_roc_curve_plot(roc_curve, roc_auc_score)

	return

def error_fn(entry):
	fpr, tpr, threshold = entry
	fnr = 1 - tpr
	return fpr + fnr

def find_min_error_rate(roc_curve):
	#fpr, tpr, thresholds = roc_curve

	zipped_curve = list(zip(*roc_curve))
	
	roc_min = min(zipped_curve, key=error_fn)

	fpr, tpr, threshold = roc_min
	fnr = 1 - tpr


	return (fpr, fnr, threshold)

def separate_predictions_by_true_classes(y_list, pred_list):

	pos = []
	neg = []
	for entry in zip(y_list, pred_list):
		y, pred = entry
		if y == 0:
			pos.append(pred)
		elif y == 1:
			neg.append(pred)
		else:
			log(y)
			assert(False)


	return (pos, neg)

def eval_predictor(stage_2_results):

	
	predicted_values = list(map(lambda entry: (lambda pred, y, file_name: float(pred))(*entry), stage_2_results))
	y_values = list(map(lambda entry: (lambda pred, y, file_name: int(y))(*entry), stage_2_results))


	roc_auc_score = metrics.roc_auc_score(y_values, predicted_values)
	roc_curve = metrics.roc_curve(y_values, predicted_values, drop_intermediate=False)

	roc_min = find_min_error_rate(roc_curve)

	log(roc_min)

	visualize_data = True
	if visualize_data:
		pos, neg = separate_predictions_by_true_classes(y_values, predicted_values)
		show_histograms([pos, neg], ["Hamis", "Eredeti"])
		create_roc_curve_plot(roc_curve, roc_auc_score)
		

	return

def show_histograms(functions, labels=None):

	bins = 40

	data_max = data_min = None
	for f in functions:
		f_max = max(f)
		data_max = max(f_max, data_max) if data_max is not None else f_max
		f_min = min(f)
		data_min = min(f_min, data_min) if data_min is not None else f_min
	
	
	#np.histogram(a, bins=10, range=None, normed=False, weights=None, density=None)[source]
	label_iterator = iter(labels) if labels is not None else None

	plt.figure()
	line_width = 1
	for f in functions:
		#hist, bin_edges = np.histogram(f, bins=bins, range=(data_min, data_max), normed=False, weights=None, density=None)
		#color='darkorange',
		#plt.plot(bin_edges[1:], hist, lw=line_width, label='h1' )
		label = None
		if label_iterator is not None:
			label = next(label_iterator)
		plt.hist(f, 100, normed=1, alpha=0.5, label=label, range=(data_min, data_max), log=True)
	#plt.xlim([0.0, 1.0])
	plt.xlim([data_min, data_max])
	plt.ylim([0.0, 1000.0])

	plt.xlabel("Modell kimenet")
	plt.ylabel("darab")
	#plt.title('')
	#plt.legend(loc="lower right")
	#plt.legend(loc="center")
	plt.legend(loc="upper center")
	plt.savefig(util.get_current_time_as_string() + "histogram.svg")
	plt.show()	


	return

def main():
	parser = argparse.ArgumentParser(description='Verdict SVM')
	parser.add_argument('csv_in', metavar='csv_in', type=str, nargs='?')
	
	args = parser.parse_args()
	csv_file_name = args.csv_in

	if csv_file_name is None:
		csv_file_name = util.get_last_file_with_ending("stage2.csv")
		
	
	if csv_file_name_override is not None:
		csv_file_name = csv_file_name_override

	f = open(csv_file_name, "r")
	
	reader = csv.reader(f, delimiter=',')

	rows = []
	for row in reader:
		rows.append(row)

	if "autoencoder" in csv_file_name:
		eval_fnc = eval_autoencoder
		#eval_autoencoder(rows)
	elif "predictor" in csv_file_name:
		eval_fnc = eval_predictor
		#eval_predictor(rows)
	else:
		log("missing metadatada in filename - trying predictor config")
		eval_fnc = eval_predictor
		pass

	eval_fnc(rows)

	
	return

#csv_file_name_override = "results_2017_12_11__11_21_27.predictor.stage2.csv"


if __name__ == "__main__":
	main()




