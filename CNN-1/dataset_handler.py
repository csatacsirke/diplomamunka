import csv
import os
import numpy as np

from six.moves import cPickle as pickle


from log import log


def readGroundTruth(row):
    line = row[0]
    if "copy" in line:
        return 0
    else:
        return 1


def readInputParamsFromCsv(inputFile):
    
    f = open(inputFile, "r")

    # lines = f.readlines()
    reader = csv.reader(f, delimiter=',')
    ground_truths = []
    file_names = []
    # x = []

    for row in reader:
    
        if( len(row) < 1):
            print("fail - ", len(row))
            continue

        
        #x_row = readSvmParams(row)

        #x.append(x_row)
        file_name = row[0]
        file_names.append(file_name)
        ground_truths.append(readGroundTruth(row))


    return file_names, ground_truths







#training_file_names = file_names[0:validation_offset]
#training_ground_truths = ground_truths[0:validation_offset]
#validation_file_names = file_names[validation_offset:]
#validation_ground_truths = ground_truths[validation_offset:]
#test_file_names = file_names[test_offset:]
#test_ground_truths = ground_truths[test_offset:]

#pickle_file = os.path.join(data_root, 'dataset_separation.pickle')
pickle_file = "dataset_separation.pickle"
def read_input():
	base_dir = 'd:/diplomamunka/SpaceTicket_results/'
	input_csv = base_dir + 'Bpas-Verdict.csv'

	return readInputParamsFromCsv(input_csv)
	#file_names, ground_truths = readInputParamsFromCsv(input_csv)

def filter_fnc_pos(tuple):
	_, y = tuple
	return y > 0.5

def filter_fnc_neg(tuple):
	_, y = tuple
	return y < 0.5

def second_arg(tuple):
	_, y = tuple
	return y


def balance_input(x, y):
	samples = list(zip(x, y))


	positive_samples = list(filter(filter_fnc_pos, samples))
	negative_samples = list(filter(filter_fnc_neg, samples))

	
	m = max(len(positive_samples), len(negative_samples))
	length = 2 * m

	permutation = np.random.permutation(m)

	x_balanced = []
	y_balanced = []

	for index in permutation:
		
		x_balanced.append(positive_samples[index % len(positive_samples)][0])
		y_balanced.append(positive_samples[index % len(positive_samples)][1])

		assert(positive_samples[index % len(positive_samples)][1] == 1)

		x_balanced.append(negative_samples[index % len(negative_samples)][0])
		y_balanced.append(negative_samples[index % len(negative_samples)][1])

		assert(negative_samples[index % len(negative_samples)][1] == 0)

	return (x_balanced, y_balanced)

def create_new_random_dataset_separation():
	#global training_file_names, training_ground_truths
	#global validation_file_names, validation_ground_truths
	#global test_file_names, test_ground_truths

	#base_dir = 'd:/diplomamunka/SpaceTicket_results/'
	#input_csv = base_dir + 'Bpas-Verdict.csv'

	#file_names, ground_truths = readInputParamsFromCsv(input_csv)
	file_names, ground_truths = read_input()

	#file_names, ground_truths = balance_input(file_names, ground_truths)

	sample_count = len(file_names)

	permutation = np.random.permutation(sample_count)


	file_names = list(map(lambda x: file_names[x] , permutation))
	ground_truths = list(map(lambda x: ground_truths[x] , permutation))


	# TODO majd 'test set'
	training_ratio = 0.70
	validation_ratio = 0.15
	training_count = round(training_ratio * sample_count)
	validation_count = round(validation_ratio * sample_count)

	training_offset = 0
	validation_offset = training_count
	test_offset = validation_offset + validation_count

	training_file_names = file_names[0:validation_offset]
	training_ground_truths = ground_truths[0:validation_offset]
	validation_file_names = file_names[validation_offset:]
	validation_ground_truths = ground_truths[validation_offset:]
	test_file_names = file_names[test_offset:]
	test_ground_truths = ground_truths[test_offset:]

	assert( len(training_file_names) == len(training_ground_truths) ) 
	assert( len(validation_file_names) == len(validation_ground_truths) ) 


	training_file_names, training_ground_truths = balance_input(training_file_names, training_ground_truths)



	log("training samples: " + str(len(training_file_names)))
	log("validation samples: " + str(len(validation_file_names)))

	log("Created new dataset separation.")

	dataset = (training_file_names, training_ground_truths,
		validation_file_names, validation_ground_truths,
		test_file_names, test_ground_truths)

	save_dataset(dataset)

	return dataset


def load_dataset_or_create_new():
	#global training_file_names, training_ground_truths
	#global validation_file_names, validation_ground_truths
	#global test_file_names, test_ground_truths

	global pickle_file

	log("Loading dataset")

	if os.path.isfile(pickle_file):
	
		with open(pickle_file, 'rb') as f:
			save = pickle.load(f)
			training_file_names = save['train_dataset']
			training_ground_truths = save['train_labels']
			validation_file_names = save['valid_dataset']
			validation_ground_truths = save['valid_labels']
			test_file_names = save['test_dataset']
			test_ground_truths = save['test_labels']

			del save  # hint to help gc free up memory
			log("Loaded dataset")

			return (training_file_names, training_ground_truths,
				validation_file_names, validation_ground_truths,
				test_file_names, test_ground_truths)
		
	else :
		return create_new_random_dataset_separation()
	
	



def save_dataset(dataset):

	(training_file_names, training_ground_truths,
		validation_file_names, validation_ground_truths,
		test_file_names, test_ground_truths) = dataset
	#global training_file_names, training_ground_truths
	#global validation_file_names, validation_ground_truths
	#global test_file_names, test_ground_truths

	global pickle_file

	try:
		f = open(pickle_file, 'wb')
		save = {
			'train_dataset': training_file_names,
			'train_labels': training_ground_truths,
			'valid_dataset': validation_file_names,
			'valid_labels': validation_ground_truths,
			'test_dataset': test_file_names,
			'test_labels': test_ground_truths,
		}
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
		f.close()

		log("Saved dataset to " + pickle_file)
	except Exception as e:
		print('Unable to save data to', pickle_file, ':', e)
		raise





