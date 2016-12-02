import csv
import random
import math
import operator

# prepare data
attributes =[]

#Indicate the metric type of each attribute
# 1 ~ Discrete/Symbolic
# 2 ~ Continuous
# 3 ~ Ordered/Symbolic
#metricAttributes=[3,3,3,3,3,3]
#metricAttributes=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
metricAttributes=[2,2,2,2,2,2,2]

#For car dataset
mydict = [{'vhigh': 1, 'high': 2, 'med': 3, 'low': 4}, {'vhigh': 1, 'high': 2, 'med': 3, 'low': 4}, {'2': 1, '3': 2, '4': 3, '5more': 4}, {'2': 1, '4': 2, 'more': 3}, {'small': 1, 'med': 2, 'big': 3}, {'low': 1, 'med': 2, 'high': 3}]
#mykeys = ['three', 'one','ten']
#newList={k:mydict[k] for k in mykeys if k in mydict}

#print mydict[1]['low']

trainingSet=[]
testSet=[]
split = 0.67


def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for i in dataset[0]:
            attributes.append(i)
            

        for x in range(1,len(dataset)-1):
            for y in range(len(attributes)-1):
                if(metricAttributes[y] == 3):
                    dataset[x][y] = mydict[y][str(dataset[x][y])]
                elif(metricAttributes[y] == 1):
                    dataset[x][y]
                else:
                    if (isfloat(dataset[x][y])):
                        dataset[x][y] = float(dataset[x][y])
                    else:
                        dataset[x][y] = 0
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length, metricAttributes, weight):
    distance = 0
    for x in range(length):
        if (metricAttributes[x] == 1):
            if (instance1[x]==instance2[x]):
                distance += 0
            else:
                distance += 1*weight[x]
        if (metricAttributes[x] == 2):
            distance += weight[x]*pow((instance1[x] - instance2[x]), 2)
        if (metricAttributes[x] == 3):
            distance += weight[x]*abs(instance1[x] - instance2[x])
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k, metricAttributes, weight):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length, metricAttributes, weight)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0
    
def classifier():
    trainingSet=[]
    testSet=[]
    split = 0.8
    loadDataset('D:\Study in UNB\Machine Learning\Assignment\data\ecoli.csv', split, trainingSet, testSet)
    print '\nTrain set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))
    
    
    #Indicate the target class
    target = attributes[-1]
    
    
    #Calculate the weight of each attribute
    #weight = 1/float(len(metricAttributes))
    #weight=[.25,.25,.25,.25]
    weight=[]
    for i in range(len(metricAttributes)):
        weight.append(1/float(len(metricAttributes)))
    
    
    
    # generate predictions
    predictions=[]
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k, metricAttributes, weight)
        result = getResponse(neighbors)
        predictions.append(result)
        #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    
    return accuracy/100

def main():
    acc = []
    avg_acc = 0
    for i in range(10):
        attributes[:] = []
        acc.append(classifier())
        avg_acc = avg_acc + acc[i]
    avg_acc = avg_acc/10
    
    print '\nAverage Accuracy: %.4f' % avg_acc
    
    std = 0
    for each in acc:
        std = std + (each-avg_acc)*(each-avg_acc)
        
    print "\nStandard deviation: %.4f" % math.sqrt(std/len(acc)) 

if __name__ == "__main__":
    main()