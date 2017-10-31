import csv
import numpy as np
import sys

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

img_width, img_height = 500, 500


# tanulási paraméterek
epochs = 30
batch_size = 16
# batch_size = 16



# Ezt töltsd ki ha szeretnéd betölteni a munkamenetet
weight_file = None
weight_file = get_last_weight_file()
#weight_file = "weights/2017_10_17__12_28_56.h5"

evaluate_only = False

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
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import History

import keras.preprocessing.image as KerasImage
from PIL import Image as PilImage


# a rendszer nem engedi lefoglalni az egész memoriát, viszont a 
# tensorflow megprobalja
# kb 0.3 ra mindig jó volt
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


log("Done loading keras")


log("Start")

# globalis változok
# todo majd a végén ezeket pl command line paraméternek


#TODO EZT VISSZA

(training_file_names, training_ground_truths,
	validation_file_names, validation_ground_truths,
	test_file_names, test_ground_truths) = dataset_handler.load_dataset_or_create_new()

#datagen_training.
#datagen_training.flow(

#KerasImage.load_img(path)

#training_file_names = file_names[0:training_count]
#training_ground_truths = ground_truths[0:training_count]





#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3




# ezt a részt hanagoljuk, mert nem fért bele a memóriába
#log("Reading images")

#training_images = list(map(lambda file_name: KerasImage.load_img(file_name, target_size=(img_width,img_height)), training_file_names))
#validation_images = list(map(lambda file_name: KerasImage.load_img(file_name, target_size=(img_width,img_height)), validation_file_names))

#log("Done reading images")


#image = KerasImage.load_img(file_names[0], target_size=(img_width,img_height))
#image_array = KerasImage.img_to_array(image)


#log("teszt")



def load_image_as_array(file_name):
	image = KerasImage.load_img(file_name)
	image = processImage(image)
	image_array = KerasImage.img_to_array(image)
	return image_array

def processImage(image: PilImage) -> PilImage:
	
	center_x = image.width // 2
	center_y = image.height // 2
	# TODO randomizálni kicsit a centert

	left = center_x - img_width // 2
	top = center_y - img_height // 2
	right = left + img_width
	bottom = top + img_height
	
	image = image.crop((left, top, right, bottom))

	assert(image.width == img_width and image.height == img_height)
	
	return image



def single_sample_generator(file_names, ground_truths):
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

	while True:
		x = np.empty([batch_size, img_width, img_height, 3])
		y = np.empty([batch_size, 1])

		for i in range(batch_size):
			image_array, ground_truth = next(single_sample_generator(file_names, ground_truths))
			x[i] = image_array
			y[i] = ground_truth
			#np.append(x, image_array, axis=0)
			#np.append(y, ground_truth, axis=0)
		yield x,y



# nem volt jó mert 1 méretü batcheket csinált
def __deprecated_create_generator(file_names, ground_truths):
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
			image_array = np.array(image_array).reshape((1, img_width, img_height, 3))
			ground_truth = np.array(ground_truths[i]).reshape((1,1))
			#image_array = image_array.reshape((1, img_width, img_height, 3))
			#image_array = np.array(image_array)
			yield image_array, ground_truth

def beautfy_result(results_row):
	"""
	Alapból a str(np.array()) valami ilyet ir ki:
	[[0]] egy sima 0 helyett
	"""
	#y_pred = round(y_pred.flatten().flatten().tolist()[0])
	file_name, y, y_pred = results_row
	y_pred = round(y_pred.flatten().flatten().tolist()[0])
	return (file_name, y, y_pred)


training_generator = create_generator(training_file_names, training_ground_truths)
validation_generator = create_generator(validation_file_names, validation_ground_truths)

#debug = next(create_generator(training_file_names, training_ground_truths))
#debug2 = next(create_generator(validation_file_names, validation_ground_truths))

nb_train_samples = len(training_file_names)
nb_validation_samples = len(validation_file_names)


#d1 = create_generator(training_file_names, training_ground_truths) 
#d2 = __deprecated_create_generator(training_file_names, training_ground_truths) 

#print(np.shape( next(d1)[0] ))
#print(np.shape( next(d2)[0] ))

#sys.exit()




# modell épités
log("Building model")

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

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

model = Sequential()
model.add(Conv2D(32, (11, 11), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (7, 7)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 64 kicsi + lehet több dense egymás után
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
# lehet a convok előtt is ( mindegyik előtt)
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop', # ADAM
              metrics=['accuracy']) 






#log("Fitting[Test]")
#first, first_y = next(training_generator)
#first = np.array(first).reshape((1, img_width, img_height, 3))
#first_y = np.array(first_y).reshape((1,1))
#model.fit(first, first_y)
#log("Fitting[Test] - done")
#sys.exit()


log("Fitting")



#validation_steps = nb_validation_samples // batch_size
validation_steps = nb_validation_samples // batch_size

if weight_file is not None:
	log("loading weights from: " + weight_file)
	model.load_weights(weight_file)




if evaluate_only:
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


	sys.exit()
	

#model.save_weights('first_try.h5')
while True:
	"""
	callbacks:
		early stopping 
		terminate on nan
		history
		modell checkpoint
	itt is használni a class weightset
	"""
	history = model.fit_generator(
		training_generator,
		steps_per_epoch=nb_train_samples // batch_size,
		epochs=epochs,
		validation_data=validation_generator,
		validation_steps=validation_steps)


	file_name = get_current_time_as_string() + '.h5'
	model.save_weights(file_name)
	#log("Saved to : " + file_name)

	#  ez legyen callbackban
	log(str(history.history['val_loss']))
	log(str(history.history['val_acc']))
	log("=============================================")
	
	

print("vege");



##################################################################################################################









