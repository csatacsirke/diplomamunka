# import StringIO
import csv
import random as Random
import numpy as np
from sklearn.svm import SVC

# saját
from log import log
import evaluate


use_two_images = False
randomly_swap_images = True

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
		return 1000 # gányolás de gyorsan kellett
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

	#log(len(x), len(y))
	assert(len(x) == len(y))


	# X = X.reshape(1,-1) # valami deprecation warning miatt



	# TODO cache size : email hogy mia az
	# class weight: ha több az egyik osztáyl akk bekapcsolni "unbalanced"
	clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
		decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
		max_iter=-1, probability=False, random_state=None, shrinking=True,
		tol=0.001, verbose=False)


	clf.fit(X, Y) 

	return clf

def add_evaluation_coulumn(results):
	for index, row in enumerate(results):
		X, Y, Y_pred, file_names = row
		y_diff = "" if abs(Y - Y_pred) == 0 else "!!"
		new_row = (X, Y, Y_pred, y_diff, file_names)
		results[index] = new_row

def test(model, X, Y):

	Y_pred = model.predict(X)

	results = list(zip(X, Y, Y_pred, file_names))
	add_evaluation_coulumn(results)

	evaluate.write_results_to_csv(results)
	
	passed = 0
	total = 0
	for index in range(len(X)):
		if Y[index] == Y_pred[index]:
			passed += 1
		total += 1
		
	log("Success rate ", 100 * passed / total, "%")


	#passed = 0
	#total = 0
	#for index in range(len(X) - 1):
		
	#	# predict: le lehet tolni egyben
	#	oneEntry = X[index].reshape(1, -1)
	#	prediction = model.predict(oneEntry)
	#	groundTruth = Y[index]
	#	log(prediction, "(", groundTruth, ")")
	#	# sklearn.metrics -> accuracy(ne), f1_score, roc_auc_score
	#	if prediction == groundTruth:
	#		passed = passed + 1
	#	total = total + 1

	# for i in range(20):
	#     randomIndex = Random.randrange(numberOfSamples)

	#     oneEntry = X[randomIndex].reshape(1, -1)
	#     prediction = model.predict(oneEntry)
	#     groundTruth = Y[randomIndex]
	#     log(prediction, "(", groundTruth, ")")
	#     if prediction == groundTruth:
	#         passed = passed + 1
	#     total = total + 1
	


if __name__ == "__main__":

	inputFile = "jura\\2017.10.25\\Bpas-Verdict.csv"

	valueableParamIndices = [18, 20, 22, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42]


	log("start")

	log("reading params")
	x, y, file_names = readParams(inputFile)


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

	# todo ahol hibázik megnézni miért szar


	log("creating model")
	model = createModel(X_train, Y_train)

	

	log("testing")
	test(model, X_test, Y_test)


	# log(clf.predict([[-0.8, -1]]))


