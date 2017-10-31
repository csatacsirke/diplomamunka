import csv
import numpy as np



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





input_csv = 'd:/diplomamunka/SpaceTicket_results/Bpas-Verdict.csv'

file_names, ground_truths = readInputParamsFromCsv(input_csv)

sample_count = len(file_names)

permutation = np.random.permutation(sample_count)


file_names = list(map(lambda x: file_names[x] , permutation))
ground_truths = list(map(lambda x: ground_truths[x] , permutation))


# TODO majd 'test set'
training_ratio = 0.80
training_count = round(training_ratio * sample_count)

training_file_names = file_names[0:training_count]
training_ground_truths = ground_truths[0:training_count]
validation_file_names = file_names[training_count:]
validation_ground_truths = ground_truths[training_count:]


print("vege");
dummy = read();





from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K





# dimensions of our images.
img_width, img_height = 2000, 2000

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')







