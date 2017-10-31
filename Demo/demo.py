# import StringIO
import csv
import random as Random
import numpy as np
from sklearn.svm import SVC



# inputFile = "f:\\xx\\kimenet\\Iqr-Verdict_kis_adag.csv"
inputFile = "f:\\xx\\kimenet\\Iqr-Verdict.csv"

valueableParamIndices = [24, 26, 28, 30, 32, 34, 37, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 62, 64, 66, 68, 70, 71, 72, 73, 74, 75, 76, 79, 81, 82,83,84,85,86, 89, 90, 92, 93, 94, 97, 98, 99, 100, 101, 102,103, 106, 107,108, 109, 110, 111, 112, 115, 116, 117, 118, 119 ,120, 121, 124, 125, 126, 127, 128]


print("start")

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
    #     print '\t'.join(row)

    f = open(inputFile, "r")

    # lines = f.readlines()
    reader = csv.reader(f, delimiter=',')
    y = []
    x = []

    for row in reader:
        # for cell in row:
        #     print(cell)

        # print(row[124])
        
        # if( len(row) < max(valueableParamIndices)) :
        if( len(row) < 131):
            print("fail - ", len(row), "/",  max(valueableParamIndices), row[0])
            continue

        
        x_row = readSvmParams(row)

        x.append(x_row)
        y.append(readGroundTruth(row))

        # print(len(x_row))

        # print(x)
        # print(y)

        # return
        # print(reader)
        # for row in reader:
        #     myStr  = ""
        #     for cell in row:
        #         myStr.join(cell)
        #     print(row)
            # decode UTF-8 back to Unicode, cell by cell:
            # valami =  [unicode(cell, 'utf-8') for cell in row]
            # print(valami)
        # myList = list(myArray)
        # for entry in reader:
            # print(entry)
        # print(join(myList))
        

       

    return x, y




def createModel(X, Y):

    print(len(x), len(y))

    # Y = Y.reshape(-1, 1)
    # print(X)

    # X = X.reshape(1,-1)  # valami deprecation warning miatt

    #X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    #y = np.array([1, 1, 2, 2])

    #X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    #y = np.array([1, 1, 2, 2])

    # TODO cache size : email hogy mia az
    # class weight: ha több az egyik osztáyl akk bekapcsolni "unbalanced"
    clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)


    clf.fit(X, Y) 

    return clf

def test(X, Y):
    passed = 0
    total = 0
    for index in range(len(X)-1):
        
        # predict: le lehet tolni egyben
        oneEntry = X[index].reshape(1, -1)
        prediction = model.predict(oneEntry)
        groundTruth = Y[index]
        print(prediction, "(", groundTruth, ")")
        # sklearn.metrics -> accuracy(ne), f1_score, roc_auc_score
        if prediction == groundTruth:
            passed = passed + 1
        total = total + 1

    # for i in range(20):
    #     randomIndex = Random.randrange(numberOfSamples)

    #     oneEntry = X[randomIndex].reshape(1, -1)
    #     prediction = model.predict(oneEntry)
    #     groundTruth = Y[randomIndex]
    #     print(prediction, "(", groundTruth, ")")
    #     if prediction == groundTruth:
    #         passed = passed + 1
    #     total = total + 1 
    
    print("Success rate ", 100*passed/total, "%")


# readParams(inputFile)
print("reading params")
x, y = readParams(inputFile)

x_train = []
y_train = []

x_test = []
y_test = []
# a szám legyen fix numpy.random.permutation
for i in range(len(x)-1):
    random = Random.random()
    if random < 0.1:
        x_test.append(x[i])
        y_test.append(y[i])
    else:
        x_train.append(x[i])
        y_train.append(y[i])

X_train = np.array(x_train)
Y_train = np.array(y_train)

X_test = np.array(x_test)
Y_test = np.array(y_test)
# todo ahol hibázik megnézni miért szar

numberOfSamples = len(x)


print("creating model")
model = createModel(X_train, Y_train)

print("testing")
test(X_test, Y_test)


# print(clf.predict([[-0.8, -1]]))
