import csv
import os
import glob
import numpy as np

from six.moves import cPickle as pickle


from log import log





def read_corners_from_csv_row(row : list):

	corners = []

	if(len(row) < 2):
		return corners
	

	last_col = row[len(row)-2]
	
	
	iterator = iter(last_col.split(';'))
	try:
		while True:
			coord_x_str = next(iterator)
			coord_y_str = next(iterator)

			coord_x = float(coord_x_str)
			coord_y = float(coord_y_str)

			corners.append([coord_x, coord_y])
	except StopIteration:
		pass

	return corners


def read_corners(input_csv):
	
	f = open(input_csv, "r")
	
	reader = csv.reader(f, delimiter=',')
	ground_truths = []
	file_names = []
	corners_list = []
	
	for row in reader:
		
		if( len(row) < 1):
		    print("fail - ", len(row))
		    continue
		
		
		#x_row = readSvmParams(row)
		
		#x.append(x_row)
		file_name = row[0]
		file_names.append(file_name)
		ground_truths.append(readGroundTruth(row))
		
		corners = read_corners_from_csv_row(row)
		corners_list.append(corners)

		
	return list(zip(file_names, corners_list))

def find_corners_in_list(list, file_name_to_find):
	for entry in list:
		file_name, corners = entry
		if file_name == file_name_to_find:
			return corners
	return None


pickle_file = "dataset_separation.pickle"

def readGroundTruth(row):
    line = row[0]
    if "copy" in line:
        return 0
    else:
        return 1


def read_full_input(inputFile, corners_list=None, zip_results=False, filter_nonexistent=True):
	if corners_list is None:
		corners_list = []
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

		if filter_nonexistent and not os.path.exists(file_name):
			continue

		file_names.append(file_name)
		ground_truths.append(readGroundTruth(row))
		if corners_list is not None:
			corners = read_corners_from_csv_row(row)
			corners_list.append(corners)

	if zip_results:
		return list(zip(file_names, ground_truths, corners_list))
		
	else:
		return file_names, ground_truths


def read_full_normalized_input_from_dir(dir):
	ground_truths = []
	file_names = []
	for filename in glob.iglob(dir + '/**/normalized*.png', recursive=True):
		
		file_names.append(filename)
		is_copy = 0 if "copy" in filename else 1
		ground_truths.append(is_copy)
		#print(filename, is_copy)
	return file_names, ground_truths


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

	log("Duplicated ", (len(x_balanced)-len(x)), "samples to balance the input")
	return (x_balanced, y_balanced)


def filter_offset_counterfeits(file_names, ground_truths):

	
	pairs = zip(file_names, ground_truths)

	pairs = list(filter(lambda entry: (lambda file_name, _: "eredetibol.nyomtatott." not in file_name)(*entry), pairs))


	filtered_file_names = list(map(lambda entry: (lambda fn, _: fn)(*entry), pairs))
	filtered_ground_truths = list(map(lambda entry: (lambda _, gt: gt)(*entry), pairs))


	assert(len(filtered_file_names) == len(filtered_ground_truths))

	log("filtered offsets: ", len(file_names) - len(filtered_file_names))

	return filtered_file_names, filtered_ground_truths


def create_new_random_dataset_separation_from_dir(dir):
	
	file_names, ground_truths = read_full_normalized_input_from_dir(dir)

	log("total samples: ", len(file_names))

	no_offset_counterfeit = True
	if no_offset_counterfeit:
		file_names, ground_truths = filter_offset_counterfeits(file_names, ground_truths)


	#file_names, ground_truths = balance_input(file_names, ground_truths)

	sample_count = len(file_names)

	permutation = np.random.permutation(sample_count)


	file_names = list(map(lambda x: file_names[x] , permutation))
	ground_truths = list(map(lambda x: ground_truths[x] , permutation))

	
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


def create_new_random_dataset_separation(input_file):
	
	file_names, ground_truths = read_full_input(input_file)

	log("total samples: ", len(file_names))

	no_offset_counterfeit = True
	if no_offset_counterfeit:
		file_names, ground_truths = filter_offset_counterfeits(file_names, ground_truths)


	#file_names, ground_truths = balance_input(file_names, ground_truths)

	sample_count = len(file_names)

	permutation = np.random.permutation(sample_count)


	file_names = list(map(lambda x: file_names[x] , permutation))
	ground_truths = list(map(lambda x: ground_truths[x] , permutation))

	
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

def load_dataset_or_create_new_from_dir(dir):
	global pickle_file

	log("Loading dataset")

	if os.path.isfile(pickle_file):
	
		with open(pickle_file, 'rb') as f:
			saved_data = pickle.load(f)
			training_file_names = saved_data['train_dataset']
			training_ground_truths = saved_data['train_labels']
			validation_file_names = saved_data['valid_dataset']
			validation_ground_truths = saved_data['valid_labels']
			test_file_names = saved_data['test_dataset']
			test_ground_truths = saved_data['test_labels']


			del saved_data  # hint to help gc free up memory
			log("Loaded dataset")

			return (training_file_names, training_ground_truths,
				validation_file_names, validation_ground_truths,
				test_file_names, test_ground_truths)
		
	else :
		#return create_new_random_dataset_separation(csv_file_name)
		return create_new_random_dataset_separation_from_dir(dir)
	

def load_dataset_or_create_new(csv_file_name):
	
	global pickle_file

	log("Loading dataset")

	if os.path.isfile(pickle_file):
	
		with open(pickle_file, 'rb') as f:
			saved_data = pickle.load(f)
			training_file_names = saved_data['train_dataset']
			training_ground_truths = saved_data['train_labels']
			validation_file_names = saved_data['valid_dataset']
			validation_ground_truths = saved_data['valid_labels']
			test_file_names = saved_data['test_dataset']
			test_ground_truths = saved_data['test_labels']


			del saved_data  # hint to help gc free up memory
			log("Loaded dataset")

			return (training_file_names, training_ground_truths,
				validation_file_names, validation_ground_truths,
				test_file_names, test_ground_truths)
		
	else :
		return create_new_random_dataset_separation(csv_file_name)
	
	



def save_dataset(dataset):

	(training_file_names, training_ground_truths,
		validation_file_names, validation_ground_truths,
		test_file_names, test_ground_truths) = dataset
	
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



def main():
	#default_input_file_name = 'jura/11.14/Bpas-Verdict.csv'
	#create_new_random_dataset_separation(default_input_file_name)

	read_full_normalized_input_from_dir("d:/Diplomamunka/SpaceTicket/")

	return

if __name__ == "__main__":
	main()
	



