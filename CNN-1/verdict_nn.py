# import StringIO
import csv
import random as Random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
import sklearn


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


use_two_images = False
randomly_swap_images = True
visualize = False

def readGroundTruth(row):
	line = row[0]
	if "copy" in line:
		return 0
	else:
		return 1

def readSvmParams(row):
	params = []
	for index in valueableParamIndices:
		params.append(float(row[index]))
	return params


def readParams(inputFile):
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

		if(len(row) < max(valueableParamIndices)):
			log("fail - ", len(row), "/",  max(valueableParamIndices), row[0])
			continue

		
		x_row = readSvmParams(row)

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

def createModel(X, Y):

	assert(len(x) == len(y))



	#http://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html#sklearn.utils.class_weight.compute_class_weight
	# class weight: ha több az egyik osztáyl akk bekapcsolni "unbalanced"
	
	class_weight = {k: v for (k, v) in zip([0, 1], compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=Y))}
	
	clf = SVC(
		C=1., 
		cache_size=200, 
		class_weight=class_weight, 
		coef0=0.0,
		decision_function_shape=None, 
		degree=2, 
		gamma='auto', 
		kernel='rbf',
		max_iter=-1,
		probability=True, 
		random_state=None, 
		shrinking=True,
		tol=0.001, 
		verbose=False)


	clf.fit(X, Y) 

	return clf

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


def test(model, X, Y):

	Y_pred = model.predict(X)
	
	results = list(zip(X, Y, Y_pred, file_names))
	add_evaluation_coulumn(results)

	statistics.write_results_to_csv(results)


	Y_pred_proba = convert_to_single_param(model.predict_proba(X))
	
	# sklearn.metrics -> accuracy(ne), f1_score, roc_auc_score

	accuracy_score = sklearn.metrics.accuracy_score(Y, Y_pred, normalize=True, sample_weight=None)
	log("accuracy_score", accuracy_score)


	f1_score = sklearn.metrics.f1_score(Y, Y_pred)
	log("f1_score", f1_score)

	roc_auc_score = sklearn.metrics.roc_auc_score(Y, Y_pred_proba)
	log("roc_auc_score", roc_auc_score)
	
	if visualize:
		# fpr, tpr, thresholds 
		roc_curve = sklearn.metrics.roc_curve(Y, Y_pred_proba)

		create_roc_curve_plot(roc_curve, roc_auc_score)
	


if __name__ == "__main__":

	inputFile = "jura\\2017.10.25\\Bpas-Verdict.csv"

	valueableParamIndices = [18, 20, 22, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42]


	log("start")

	log("reading params")
	x, y, file_names = readParams(inputFile)

	np.random.seed(seed = 1337)
	permutation = np.random.permutation(len(x))


	x = list(map(lambda i: x[i], permutation))
	y = list(map(lambda i: y[i], permutation))
	file_names = list(map(lambda i: file_names[i], permutation))

	train_ratio = 0.8
	test_offset = round( len(x) * train_ratio )



	X_train = np.array(x[0:test_offset])
	Y_train = np.array(y[0:test_offset])

	X_test = np.array(x[test_offset:])
	Y_test = np.array(y[test_offset:])

	# todo ahol hibázik megnézni miért rossz


	log("creating model")
	model = createModel(X_train, Y_train)

	

	log("testing")
	#test(model, X_test, Y_test)
	test(model, x, y)


	# log(clf.predict([[-0.8, -1]]))


