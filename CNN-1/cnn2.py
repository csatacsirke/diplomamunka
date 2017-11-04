import csv
import numpy as np
import sys

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


# saját dolgok
from log import log
from log import set_log_file_path
from util import get_current_time_as_string
from util import get_last_weight_file

import dataset_handler
import evaluate


########################################################################################
# TODO
# kb 1600x1600 as képeink vannak, kicsinyiteni ha nem muszáj nem akarunk
# valszeg majd azt kéne csinálni hogy egy fix méretet középröl kivágunk
"""
 1) kicsinyitás ( downscale ) !!!!
 2) 
	random kivágás
	fix kivágás
		-középről
		-sarkokról
		
		
debug: 1-2 képre overfit a trainen
"""

img_width, img_height = 200, 200


# tanulási paraméterek
epochs = 5
batch_size = 16
# batch_size = 16

use_autoencoder = True
recalculate = False
evaluate_only = True
weights_only = False

# Ezt töltsd ki ha szeretnéd betölteni a munkamenetet
weight_file = None
if not recalculate:
	weight_file = get_last_weight_file()

	

if evaluate_only:
	assert(recalculate == False)
	recalculate = False

	
fnCopy = "d:\\diplomamunka\\Temp\\kicsi\\SPACTICK_2017_09_18_15_43_35_942_Marci.eredeti_1___.png"
fnOrig = "d:\\diplomamunka\\Temp\\kicsi\\SPACTICK_2017_10_09_16_02_27_938_Marci.fenymasolt.copy_0___.png"

#weight_file = "weights/2017_10_17__12_28_56.h5"



########################################################################################


#result_dir = "weights/"
# ha van rendes working directory akkor erre nincs szükség
#result_dir = "" 
#log_file_path = result_dir + get_current_time_as_string() + "_log" + ".txt"
#set_log_file_path(log_file_path)

log("Loading keras - this may be slow...")


import tensorflow as tf

# TODO megcsinálni
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import History
from keras.models import load_model

import keras.preprocessing.image as KerasImage
from PIL import Image as PilImage


# a rendszer nem engedi lefoglalni az egész memoriát, viszont a 
# tensorflow megprobalja
# kb 0.3 ra mindig jó volt
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


# ez amúgy jó eséllyel nem kell mert alapból channels last, 
# és ehhez alkalmazkodtunk
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

log("Done loading keras")


log("Start")

# globalis változok
# todo majd a végén ezeket pl command line paraméternek


#TODO EZT VISSZA

(training_file_names, training_ground_truths,
	validation_file_names, validation_ground_truths,
	test_file_names, test_ground_truths) = dataset_handler.load_dataset_or_create_new()




def load_image_as_array(file_name):
	image = KerasImage.load_img(file_name)
	image = processImage(image)
	image_array = KerasImage.img_to_array(image)
	return image_array

def processImage(image: PilImage) -> PilImage:
	
	dx = np.random.randint(-image.width//4, image.width//4+1)
	dy = np.random.randint(-image.height//4, image.height//4+1)

	
	center_x = image.width // 2 + dx
	center_y = image.height // 2 + dy
	# TODO randomizálni kicsit a centert

	left = center_x - img_width // 2
	top = center_y - img_height // 2
	right = left + img_width
	bottom = top + img_height
	
	image = image.crop((left, top, right, bottom))

	assert(image.width == img_width and image.height == img_height)
	
	return image



def create_single_sample_generator(file_names, ground_truths):
	assert(len(ground_truths) == len(ground_truths))
	count = len(file_names)
	while True:
		permutation = np.random.permutation(count)
		for i in permutation:
			# image = KerasImage.load_img(file_names[i], target_size=(img_width,img_height))
			image = KerasImage.load_img(file_names[i])
			image = processImage(image)
			image_array = KerasImage.img_to_array(image)
			# a modell 4d array-t vár, ezért be kell csomagolni
			flat = False
			if flat:
				image_array = np.array(image_array).reshape((1, img_width, img_height, 3))
				ground_truth = np.array(ground_truths[i]).reshape((1,1))
				yield image_array, ground_truth
			else:
				yield image_array, ground_truths[i]
			#image_array = image_array.reshape((1, img_width, img_height, 3))
			#image_array = np.array(image_array)
			
			#yield image_array, ground_truths[i]


# globalis parameter a batch_size
def create_generator(file_names, ground_truths):
	assert(len(ground_truths) == len(ground_truths))
	single_sample_generator =  create_single_sample_generator(file_names, ground_truths)

	while True:
		x = np.empty([batch_size, img_width, img_height, 3])
		y = np.empty([batch_size, 1])

		for i in range(batch_size):
			# todo , ez overkill, vagy ne generátor, vagy akkor a generátor legyen ujrahasználva
			if use_autoencoder:
				ground_truth = 0
				while ground_truth == 0:
					image_array, ground_truth = next(single_sample_generator)
				x[i] = image_array
				y[i] = ground_truth
			else:
				image_array, ground_truth = next(single_sample_generator)
				x[i] = image_array
				y[i] = ground_truth
			#np.append(x, image_array, axis=0)
			#np.append(y, ground_truth, axis=0)
		if use_autoencoder:
			yield x,x
		else:
			yield x,y




## nem volt jó mert 1 méretü batcheket csinált
#def __deprecated_create_generator(file_names, ground_truths):
#	assert(len(ground_truths) == len(ground_truths))
#	count = len(file_names)
#	while True:
#		permutation = np.random.permutation(count)
#		for i in permutation:
#			# image = KerasImage.load_img(file_names[i], target_size=(img_width,img_height))
#			image = KerasImage.load_img(file_names[i])
#			image = processImage(image)
#			image_array = KerasImage.img_to_array(image)
#			# a modell 4d array-t vár, ezért be kell csomagolni
#			image_array = np.array(image_array).reshape((1, img_width, img_height, 3))
#			ground_truth = np.array(ground_truths[i]).reshape((1,1))
#			#image_array = image_array.reshape((1, img_width, img_height, 3))
#			#image_array = np.array(image_array)
#			yield image_array, ground_truth

def beautfy_result(results_row):
	"""
	Alapból a str(np.array()) valami ilyet ir ki:
	[[0]] egy sima 0 helyett
	"""
	#y_pred = round(y_pred.flatten().flatten().tolist()[0])
	file_name, y, y_pred = results_row
	y_pred = round(y_pred.flatten().flatten().tolist()[0])
	return (file_name, y, y_pred)



def evaluate():
	assert(weight_file is not None)

	log("Evaluating...")
	#validation_file_names, validation_ground_truths,
	#test_generator = create_generator(test_file_names, test_ground_truths)
	eval_full_dataset = True
	if eval_full_dataset:
		file_names, ground_truths = dataset_handler.read_input();
	else:
		file_names = test_file_names
		ground_truths = test_ground_truths

	#length = len(test_file_names)
	length = len(file_names)

	results = []

	for i in range(length):
		file_name = file_names[i]
		#file_name = test_file_names[i]
		x = load_image_as_array(file_name).reshape((1, img_width, img_height, 3))
		#y = test_ground_truths[i]
		y = ground_truths[i]
		y_pred = model.predict(x)

		results.append((file_name, y, y_pred))


	# Alapból a str(np.array()) valami ilyet ir ki:
	# [[0]] egy sima 0 helyett
	results = map(beautify_results, results)
	header = ("file", "igazság", "jóslat")
	evaluate.write_results_to_csv(results, header=header)


# https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def rgb2gray(rgb):
	
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	#gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	gray = 0.3333 * r + 0.3333 * g + 0.3333 * b
	
	return gray


def evaluate_autoencoder():
	
	#gen = create_generator([fnOrig], [1])
	
	while True:
		sample = next(training_generator)
		x, y = sample

		prediction = model.predict(x)
		log(np.shape(prediction[0]))

		fig = plt.figure()

		subplot = fig.add_subplot(1,2,1)
		gray = rgb2gray(prediction[0])
		plt.imshow(gray)

		subplot = fig.add_subplot(1,2,2)
		plt.imshow(x[0])

		plt.show()
		#plt.waitforbuttonpress()


	return


# modell épités
log("Building model")


"""
float16 ot meg lehet próbálni
conv2 + dense:
	kernel_regularizer
	keras.regularizers.l2(0.001) % a hiperparamétert is be kell még  löni
	
	autoencoder:
		conv2dtranspose
		loss: squared error
		tanitás csak az eredetiken
		érdemes kézzel megnézni hogy a predict milyen képet generál
	
"""


def build_autoencoder_model():
	model = Sequential()
	"""
	model.add(Conv2D(64, (11, 11), input_shape=input_shape, padding="same", data_format="channels_last"))
	model.add(Conv2DTranspose(3, (11, 11), padding="same", data_format="channels_last"))
	"""
	
	log("input shape: ", input_shape)

	model.add(Conv2D(64, (5, 5), input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (5, 5)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Conv2D(256, (5, 5)))
	model.add(Activation('relu'))

	#model.add(MaxPooling2D(pool_size=(2, 2)))
	
	#####
	

	model.add(Conv2DTranspose(64, (5, 5)))
	model.add(Activation('relu'))
	model.add(UpSampling2D(size=(2, 2)))
	

	model.add(Conv2DTranspose(64, (5, 5)))
	model.add(Activation('relu'))
	model.add(UpSampling2D(size=(2, 2)))
	
	model.add(Conv2DTranspose(3, (5, 5)))

	#model.add(ZeroPadding2D())
	#model.add(Activation('relu'))
	#model.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
	
	#####

	model.compile(loss='mean_squared_error',
				  optimizer='adam', # ADAM
				  metrics=['accuracy']) 


	shape = model.get_output_shape_at(0)
	log(shape)
	#log(K.int_shape(last_index - 1))
	log("")
	log("")
	log("")
	return model


def build_model():
	model = Sequential()
	model.add(Conv2D(64, (11, 11), input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (7, 7)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Conv2D(256, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# 64 kicsi + lehet több dense egymás után
	model.add(Flatten())
	model.add(Dense(256))
	model.add(Activation('relu'))

	model.add(Dense(256))
	model.add(Activation('relu'))

	model.add(Dense(256))
	model.add(Activation('relu'))


	# lehet dropout a convok előtt is ( mindegyik előtt)
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
				  optimizer='rmsprop', # ADAM
				  metrics=['accuracy']) 

	return model

def load_or_build_model():
	model = None
	if weight_file is not None:
		try:
			model = load_model(weight_file)
			log("Successfully loaded full model: ", weight_file, "\r\n")
		except ValueError:
			log("Invalid model file (probably only saved weights)", weight_file, "\r\n")
			model = None

	if model is None:
		if use_autoencoder:
			model = build_autoencoder_model()
		else:
			model = build_model()


		if weight_file is not None:
			log("Loading weights from: " + weight_file)
			model.load_weights(weight_file)
	return model


training_generator = create_generator(training_file_names, training_ground_truths)
validation_generator = create_generator(validation_file_names, validation_ground_truths)


#if False:	
	
#	#fnCopy = "d:\\diplomamunka\\spaceticket\\htc\\copy_sok_fit_hiba\\SPACTICK_2017_10_09_16_02_27_938_Marci.fenymasolt.copy_0.png"
#	#fnOrig = "d:\\diplomamunka\\spaceticket\\htc\\Eredeti\\SPACTICK_2017_09_18_15_43_35_942_Marci.eredeti_1.png"

#	training_file_names2 = [fnCopy, fnOrig]
#	training_ground_truths2 = [0, 1]

#	training_file_names2 = [fnOrig]
#	training_ground_truths2 = [1]


#	training_generator = create_generator(training_file_names2, training_ground_truths2)



#debug = next(create_generator(training_file_names, training_ground_truths))
#debug2 = next(create_generator(validation_file_names, validation_ground_truths))

nb_train_samples = len(training_file_names)
nb_validation_samples = len(validation_file_names)

steps_per_epoch = nb_train_samples // batch_size

#validation_steps = nb_validation_samples // batch_size
validation_steps = nb_validation_samples // batch_size


#d1 = create_generator(training_file_names, training_ground_truths) 
#d2 = __deprecated_create_generator(training_file_names, training_ground_truths) 

#print(np.shape( next(d1)[0] ))
#print(np.shape( next(d2)[0] ))

#sys.exit()






model = load_or_build_model()


if evaluate_only:
	log("Evaluating")
	if use_autoencoder:
		evaluate_autoencoder()
	else:
		evaluate()
	sys.exit()
	

log("Fitting")
while True:
	"""
	callbacks:
		early stopping 
		terminate on nan
		history
		modell checkpoint
	itt is használni a class weightset
	"""
	# todo
	callbacks = []

	history = model.fit_generator(
		training_generator,
		steps_per_epoch=steps_per_epoch,
		epochs=epochs,
		validation_data=validation_generator,
		validation_steps=validation_steps,
		callbacks=callbacks)


	
	if weights_only:
		file_name = get_current_time_as_string() + '.h5'
		model.save_weights(file_name)
	else:
		file_name = get_current_time_as_string() + '.full_model.h5'
		model.save(file_name)

	
	log("Saved weights to : " + file_name)

	#  ez legyen callbackban
	log(str(history.history['val_loss']))
	log(str(history.history['val_acc']))
	log("=============================================")
	
	#if use_autoencoder:
	#	evaluate_autoencoder()

print("vege");



##################################################################################################################






#log("Fitting[Test]")
#first, first_y = next(training_generator)
#first = np.array(first).reshape((1, img_width, img_height, 3))
#first_y = np.array(first_y).reshape((1,1))
#model.fit(first, first_y)
#log("Fitting[Test] - done")
#sys.exit()






