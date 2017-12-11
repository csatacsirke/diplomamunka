# import StringIO
import csv
import random as Random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
import sklearn
import argparse

# saját
from log import log
import statistics
from statistics import create_roc_curve_plot



# TODO 
"""
Balance-olni a validationt is?

Validation+test (80 10 10)  
	lehet random seed ( np.random.seed )
	
train -> tanitás ( több féle hiperparaméterrel) 
valid -> tesztelés 
	-> mohó kiválasztás, hogy melyik hiperparaméterre volt a legjobb
	-> train + valid on ujratanitás csak a legjobb hiperparamtéerrel
		-> tesztelés a teszt halmazon
hiperparaméterek:
	C param
	kernel
		degree
		gamma
hiperparaméterbeállítás:
	log skálán kipróbálni minden félét (pl 2 hatványok kicsitől nagyig)
	hyperopt ( python csomag )
		minden paraméterrel egy min és max + eloszlás
		param: egy függvény: {hiperparaméterek}, set1, set2 -> set2 hiba
			set1 en tanit, set2-n tesztel
			meg kell adni hogy melyik melyik (train / valid)
			ebből megkapjuk a legjobb hiperparamétert
			még egyszer meghivod (train + valid / test)-en
rontás kevesebb példával -> megnézni hogy függ a számuktól	
"""


use_two_images = True
randomly_swap_images = True
visualize = False
use_random_seed = True
train_ratio = 0.80


def zip_list(x):
	return list(zip(*x))

def readGroundTruth(row):
	line = row[0]
	if "copy" in line:
		return 1
	else:
		return 0

def read_valuable_params(row, valuable_param_indices):
	params = []
	for index in valuable_param_indices:
		params.append(float(row[index]))
	return params


def read_input_data(inputFile):

	valuable_param_indices = [18, 20, 22, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42]
	# f = StringIO.StringIO(scsv)
	
	# for row in reader:
	#     log '\t'.join(row)

	f = open(inputFile, "r")

	# lines = f.readlines()
	reader = csv.reader(f, delimiter=',')
	y = []
	x = []
	file_names = []

	for row in reader:

		if row[3] == "nofit":
			continue

		if(len(row) < max(valuable_param_indices)):
			log("fail - ", len(row), "/",  max(valuable_param_indices), row[0])
			continue

		
		x_row = read_valuable_params(row, valuable_param_indices)

		x.append(x_row)
		y.append(readGroundTruth(row))
		file_names.append(row[0])

	if use_two_images:
		params = (x, y, file_names)
		params = merge_twin_images(params)
		return params
	else:
		return x, y, file_names


def hamming_distance(s1, s2):
	"""Calculate the Hamming distance between two bit strings"""
	#assert len(s1) == len(s2)
	if not len(s1) == len(s2):
		#return 1000 # gányolás de gyorsan kellett
		return max(len(s1), len(s2))
	return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def merge_twin_images(params):
	x, y, file_names = params
	merged_params = []

	#records = list(zip(x, y, file_names))
	assert(len(x) == len(y))
	assert(len(x) == len(file_names))
	records = list(zip(x, y, file_names))
	
	enum = enumerate(records)

	record2 = None
	while True:
		
		iterator = next(enum, None)
		if iterator is None:
			break

		_, record1 = iterator
		if record1 is None:
			break

		if record2 is None:
			record2 = record1
			record1 = None
			continue

		#index1, record2 = next(enum, None)
		#if record2 is None:
		#	break
		
		if randomly_swap_images:
			# elvileg a két kép független egymástól, nem szabad hogy a 
			# kiértékelés a sorrendjüktől függjön
			rnd = np.random.randint(2) # [0,1]
			if rnd is 0:
				record1, record2 = record2, record1
		
		x1, y1, file_name1 = record1
		x2, y2, file_name2 = record2

		if hamming_distance(file_name1, file_name2) <= 2:
			assert(y1 == y2)
			new_record = (x1+x2, y1, (file_name1 + ";" + file_name2))
			merged_params.append(new_record)
			record2 = None
		else:
			log("kulonbozo file paros!?" )
			log(file_name1)
			log(file_name2)
			record2 = record1
			continue



	return list(zip(*merged_params))



def add_evaluation_coulumn(results):
	for index, row in enumerate(results):
		X, Y, Y_pred, file_names = row
		y_diff = "" if abs(Y - Y_pred) == 0 else "!!"
		new_row = (X, Y, Y_pred, y_diff, file_names)
		results[index] = new_row

#def convert_to_continuous(Y):
#	return list(map(lambda x: float(x), Y))



def convert_to_single_param(Y):
	#  [1, 0] -> [0] 
	#  [0, 1] -> [1] 
	def fnc(x):		
		first, second = x
		assert(abs((first+second)-1) < 0.01)
		return second
	return list(map(fnc, Y))


def calc_errors(y, pred):
	""" returns (fp, fn, tp, tn)"""
	# 0 -> eredeti
	# 1 -> hamis
	entries = zip(y, pred)
	tp = tn = fp = fn = 0

	for entry in entries:
		y, pred = entry

		if y == 0 and pred == 0 : tp += 1	
		if y == 0 and pred == 1 : fn += 1	
		if y == 1 and pred == 0 : fp += 1	
		if y == 1 and pred == 1 : tn += 1	

	return (fp, fn, tp, tn)


#def list_problematic_entries():

#	entries = zip(y, pred)
#	tp = tn = fp = fn = 0

#	for entry in entries:
#		y, pred = entry


#		#TODO
#		if y == 0 and pred > 0.5 : fp += 1	
#		if y == 1 and pred < 0.5 : fn += 1	

#	return (fp, fn)

#def random_split_data(X, Y, file_names):
def random_split_data(data):
	
	if not use_random_seed:
		np.random.seed(seed = 1337)
	#permutation = np.random.permutation(len(X))
	permutation = np.random.permutation(len(data))


	#data = list(zip(X, Y, file_names))
	data = list(map(lambda i: data[i], permutation))

	#X = list(map(lambda i: X[i], permutation))
	#Y = list(map(lambda i: Y[i], permutation))
	#file_names = list(map(lambda i: file_names[i], permutation))

	
	#test_offset = round( len(X) * train_ratio )
	test_offset = round( len(data) * train_ratio )


	train = data[0:test_offset]
	test = data[test_offset:]

	#X_train = np.array(x[0:test_offset])
	#Y_train = np.array(y[0:test_offset])

	#X_test = np.array(x[test_offset:])
	#Y_test = np.array(y[test_offset:])

	

	

	return (train, test)

def test(model, X, Y, file_names):

	Y_pred = model.predict(X)
	
	results = list(zip(X, Y, Y_pred, file_names))
	add_evaluation_coulumn(results)

	statistics.write_results_to_csv(results)


	Y_pred_proba = convert_to_single_param(model.predict_proba(X))
	
	# sklearn.metrics -> accuracy(ne), f1_score, roc_auc_score

	accuracy_score = sklearn.metrics.accuracy_score(Y, Y_pred, normalize=True, sample_weight=None)
	log("accuracy_score", accuracy_score)


	f1_score = sklearn.metrics.f1_score(Y, Y_pred)
	#f1_score = sklearn.metrics.f1_score(Y, Y_pred_proba)
	log("f1_score", f1_score)

	roc_auc_score = sklearn.metrics.roc_auc_score(Y, Y_pred_proba)
	log("roc_auc_score", roc_auc_score)
	
	fp, fn, tp, tn = calc_errors(Y, Y_pred)

	# false positive rate
	fpr = fp/len(Y)*100
	# false negative rate
	fnr = fn/len(Y)*100

	log("total samples: ", len(Y))

	log("false positive: ", fp, "(" + str(round(fpr,3)) + "%)" )
	log("false negative: ", fn, "(" + str(round(fnr,3)) + "%)" )

	params = model.get_params()
	#list_problematic_entries(Y, Y_pred_proba)
	
	if visualize:
		# fpr, tpr, thresholds 
		roc_curve = sklearn.metrics.roc_curve(Y, Y_pred_proba)

		create_roc_curve_plot(roc_curve, roc_auc_score)




def compute_class_weight_map(Y):
	class_weight = {k: v for (k, v) in zip([0, 1], compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=Y))}
	return class_weight


def create_and_fit_model(X, Y, kernel='rbf', C=1.0, gamma="auto", coef0=0.0):

	assert(len(X) == len(Y))



	#http://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html#sklearn.utils.class_weight.compute_class_weight
	#class_weight = {k: v for (k, v) in zip([0, 1], compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=Y))}
	class_weight = compute_class_weight_map(Y)
	
	model = SVC(
		C=C, 
		cache_size=200, 
		class_weight=class_weight, 
		coef0=coef0,
		decision_function_shape=None, 
		degree=2, 
		gamma=gamma, 
		kernel=kernel,
		max_iter=-1,
		probability=True, 
		random_state=None, 
		shrinking=True,
		tol=0.0001, 
		verbose=False)

	model.fit(X, Y) 

	return model


def print_percentage(i, N):
	print( "\r" + str((i+1)*100//N) + "%", end=""if i is not N-1 else "\n" )


def batch_fit_and_test(x, y, file_names, N = 200, kernel='rbf', C=1.0, gamma="auto", coef0=0.0):
	"""return (avg_fpr, avg_fnr)"""
	
	
	log("N={0}, C={1}, gamma={2}".format( N, C, gamma) )


	fpr_list = []
	fnr_list = []
	
	for i in range(N):
		print_percentage(i, N)

		train_data, test_data = random_split_data(zip_list((x, y, file_names)))

		X_train = np.array(list(map(lambda a: a[0], train_data)))
		Y_train = np.array(list(map(lambda a: a[1], train_data)))

		model = create_and_fit_model(X_train, Y_train, kernel=kernel, C=C, gamma=gamma, coef0=coef0)

	
		X_test = np.array(list(map(lambda a: a[0], test_data)))
		Y_test = np.array(list(map(lambda a: a[1], test_data)))
		file_names_test = list(map(lambda a: a[2], test_data))

		Y_pred = model.predict(X_test)

		fp, fn, tp, tn = calc_errors(Y_test, Y_pred)


		#n_pos = 0
		#n_neg = 0
		#for _y in Y_test:
		#	if _y == 0:
		#		n_pos += 1
		#	elif _y == 1:
		#		n_neg += 1
		#	else:
		#		log(_y)
		#		assert(False)
		
		fpr_list.append(fp/(fp+tn))
		fnr_list.append(fn/(fn+tp))

		#fpr_list.append(fp/(len(Y_test)))
		#fnr_list.append(fn/(len(Y_test)))
	
		pass

	log("total samples: ", len(x))
	log("average+deviation type 1 error rate:")
	avg_fpr = np.mean(fpr_list)
	var_fpr = np.std(fpr_list)
	log( round(avg_fpr, 5), round(var_fpr, 6) )

	log("average+deviation type 2 error rate:")
	avg_fnr = np.mean(fnr_list)
	var_fnr = np.std(fnr_list)
	log( round(avg_fnr, 5), round(var_fnr, 6) )
	
	
	return (avg_fpr, avg_fnr)


def train_and_test(x, y, file_names):
	
	
	#if not use_random_seed:
	#	np.random.seed(seed = 1337)
	#permutation = np.random.permutation(len(x))


	#x = list(map(lambda i: x[i], permutation))
	#y = list(map(lambda i: y[i], permutation))
	#file_names = list(map(lambda i: file_names[i], permutation))

	
	#test_offset = round( len(x) * train_ratio )



	#X_train = np.array(x[0:test_offset])
	#Y_train = np.array(y[0:test_offset])

	#X_test = np.array(x[test_offset:])
	#Y_test = np.array(y[test_offset:])

	# todo ahol hibázik megnézni miért rossz
	train_data, test_data = random_split_data(zip_list((x, y, file_names)))

	X_train = np.array(list(map(lambda a: a[0], train_data)))
	Y_train = np.array(list(map(lambda a: a[1], train_data)))

	log("creating model")
	# lowest loss: {'C': 6.250906239883302, 'gamma': 0.03357455814119452}
	C = 6.250906239883302
	gamma = 0.03357455814119452
	model = create_and_fit_model(X_train, Y_train, C = C, gamma=gamma)

	
	X_test = np.array(list(map(lambda a: a[0], test_data)))
	Y_test = np.array(list(map(lambda a: a[1], test_data)))
	file_names_test = list(map(lambda a: a[2], test_data))


	log("testing")
	test(model, X_test, Y_test, file_names_test)
	#test(model, x, y)


	# log(clf.predict([[-0.8, -1]]))
	


# http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
def finetune_with_hyperopt(x, y, file_names):
	from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
	
	low = np.log(1e-3)
	high = np.log(1e+3)

	fspace = {	
		'C': hp.loguniform('C', low, high),
		'gamma': hp.loguniform('gamma', low, high)
	}

	def f(params):
		C = params['C']
		gamma = params['gamma']
		fpr, fnr = batch_fit_and_test(x, y, file_names, N=200, gamma=gamma, C=C)
		val = fpr + fnr
		return {'loss': val, 'status': STATUS_OK}

	trials = Trials()
	best = fmin(fn=f, space=fspace, algo=tpe.suggest, max_evals=100, trials=trials)

	log('lowest loss:', best)

	log( 'trials:')
	for trial in trials.trials[:2]:
		log(trial)


#def finetune_hyperparams(x, y, file_names):

#	return

def main():
	parser = argparse.ArgumentParser(description='Verdict SVM')
	parser.add_argument('csv_in', metavar='csv_in', type=str, nargs='?')
	


	args = parser.parse_args()
	inputFile = args.csv_in

	if inputFile is None:
		inputFile = "jura\\2017.10.25\\Bpas-Verdict.csv"
		#inputFile = "jura\\2017.10.25\\Bpas-Merged.csv"
		
	

	log("reading params from: " + inputFile)
	log("use_two_images: ", use_two_images)
	x, y, file_names = read_input_data(inputFile)

	C = 6.250906239883302
	gamma = 0.03357455814119452
	
	batch_fit_and_test(x, y, file_names,C=C, gamma=gamma, N=2000)
	#train_and_test(x, y, file_names)
	return
	
	#inputFile = "jura\\2017.10.25\\Bpas-Merged.csv"


	

	#batch_fit_and_test(x, y, file_names)

	# lowest loss: {'C': 6.250906239883302, 'gamma': 0.03357455814119452}
	finetune_with_hyperopt(x, y, file_names)




if __name__ == "__main__":
	main()



