import csv
import math
import random
import itertools
import operator

# Implement your decision tree below
# Used the ID3 algorithm to implement the Decision Tree

# Class used for learning and building the Decision Tree using the given Training Set
class DecisionTree():
    tree = {}

    def learn(self, training_set, attributes, target):
        self.tree = build_tree(training_set, attributes, target)


# Class Node which will be used while classify a test-instance using the tree which was built earlier
class Node():
    value = ""
    children = []

    def __init__(self, val, dictionary):
        self.value = val
        if (isinstance(dictionary, dict)):
            self.children = dictionary.keys()

# A function to read csv data
def read_csv_file(filename):
    data = []
    with open(filename, 'r+') as csvfile:
        content = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in content:
            #data.append([item.strip() for item in line])
            data.append(tuple(line))

    return data

# Majority Function which tells which class has more entries in given data-set
def majorClass(attributes, data, target):

    freq = {}
    index = attributes.index(target)

    for tuple in data:
        if (freq.has_key(tuple[index])):
            freq[tuple[index]] += 1 
        else:
            freq[tuple[index]] = 1

    max = 0
    major = ""

    for key in freq.keys():
        if freq[key]>max:
            max = freq[key]
            major = key

    return major


# Calculates the entropy of the data given the target attribute
def entropy(attributes, data, targetAttr):

    freq = {}
    dataEntropy = 0.0

    i = 0
    for entry in attributes:
        if (targetAttr == entry):
            break
        i = i + 1

    i = i - 1

    for entry in data:
        if (freq.has_key(entry[i])):
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]]  = 1.0

    for freq in freq.values():
        dataEntropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
        
    return dataEntropy


# Calculates the information gain (reduction in entropy) in the data when a particular attribute is chosen for splitting the data.
def info_gain(attributes, data, attr, targetAttr):

    freq = {}
    subsetEntropy = 0.0
    i = attributes.index(attr)

    for entry in data:
        if (freq.has_key(entry[i])):
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]]  = 1.0

    for val in freq.keys():
        valProb        = freq[val] / sum(freq.values())
        dataSubset     = [entry for entry in data if entry[i] == val]
        subsetEntropy += valProb * entropy(attributes, dataSubset, targetAttr)

    return (entropy(attributes, data, targetAttr) - subsetEntropy)


# This function chooses the attribute among the remaining attributes which has the maximum information gain.
def attr_choose(data, attributes, target):

    best = attributes[0]
    maxGain = 0;

    for attr in attributes:
        newGain = info_gain(attributes, data, attr, target) 
        if newGain>maxGain:
            maxGain = newGain
            best = attr

    return best


# This function will get unique values for that particular attribute from the given data
def get_values(data, attributes, attr):

    index = attributes.index(attr)
    values = []

    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])

    return values

# This function will get all the rows of the data where the chosen "best" attribute has a value "val"
def get_data(data, attributes, best, val):

    new_data = [[]]
    index = attributes.index(best)

    for entry in data:
        if (entry[index] == val):
            newEntry = []
            for i in range(0,len(entry)):
                if(i != index):
                    newEntry.append(entry[i])
            new_data.append(newEntry)

    new_data.remove([])    
    return new_data


# This function is used to build the decision tree using the given data, attributes and the target attributes. It returns the decision tree in the end.
def build_tree(data, attributes, target):

    data = data[:]
    vals = [record[attributes.index(target)] for record in data]
    default = majorClass(attributes, data, target)

    if not data or (len(attributes) - 1) <= 0:
        return default
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        best = attr_choose(data, attributes, target)
        tree = {best:{}}
    
        for val in get_values(data, attributes, best):
            new_data = get_data(data, attributes, best, val)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = build_tree(new_data, newAttr, target)
            tree[best][val] = subtree
    
    return tree

def print_tree(tree, str):
    """
    This function recursively crawls through the d-tree and prints it out in a
    more readable format than a straight print of the Python dict object.
    """
    
    
    if type(tree) == dict:
        print "%s%s" % (str, tree.keys()[0])
        #print "%s\t|=%s%s" % (str, item, "-\\")
        
        for item in tree.values()[0].keys():
            print "%s\t|=%s%s" % (str, item, "-\\")
            #print "=%s%s" % (item, "-\\")
            print_tree(tree.values()[0][item], str + "\t|\t|")
    else: #printing leaves
        print "%s->%s" % (str, tree)
        #print "|\t",
        '''
        if (count<>0):
            for i in range(count*2-1): 
                print "|\t",
            count = 0'''

# This function runs the decision tree algorithm. It parses the file for the data-set, and then it runs the 10-fold cross-validation. It also classifies a test-instance and later compute the average accurracy
# Improvements Used: 
# 1. Discrete Splitting for attributes "age" and "fnlwght"
# 2. Random-ness: Random Shuffle of the data before Cross-Validation

# This function create the cross-validation training and test dataset
def create_dataset(k):
    lines = []
    #with open("D:\Study in UNB\Machine Learning\Assignment\data\mushroom.csv") as csvfile:
    with open("D:\Study in UNB\Machine Learning\Assignment\data\cancer.csv") as csvfile:
        for line in csv.reader(csvfile, delimiter=","):
            lines.append(line)
    
    attributes = []
    
    for i in lines[0]:
        attributes.append(i)
    
    del lines[0]
    
    #print attributes
        
    for j in range(k):
        random.shuffle(lines)
        training_set = [x for i, x in enumerate(lines) if i % k != j]
        testing_set = [x for i, x in enumerate(lines) if i % k == j]
    
    #print training_set
    #print testing_set
    
    return (attributes, training_set, testing_set)
    

# This function create the small subset of training and test data with the smaller number of feature for each random tree        
def create_smaller_subset(Num, Final_att = [], Final_train_subset = [], Final_test_subset = []):

    new_tuple = create_dataset(5)
    attributes = new_tuple[0]
    training_set = new_tuple[1]
    testing_set = new_tuple[2]
    
    #print attributes
    #print training_set
    #print testing_set
       
    #Number_of_Tree = random.randint(2, 10)
    for k in range(Num):
        #Random_Features = random.sample(range(0, 22),random.randint(2, 22)) #Mushroom dataset
        Random_Features = random.sample(range(0, 9),random.randint(2, 9))
        #print Random_Features 
        training_subset = [] 
        testing_subset = []
        att_subset =[]
        
        for j in range(len(Random_Features)):
            att_subset.append(attributes[Random_Features[j]])
        att_subset.append(attributes[-1])
        print "Selected features of Small Tree "+str(k)+": "+str(att_subset)
        Final_att.append(att_subset)
        
                
        for line in training_set:
            selectedFeatures = []
            for j in range(len(Random_Features)):
                selectedFeatures.append(line[Random_Features[j]])
            selectedFeatures.append(line[-1])
            training_subset.append(selectedFeatures)
        #print training_subset
        Final_train_subset.append(training_subset)
        #print type (Final_train_subset)
            
        for line in testing_set:
            selectedFeatures = []
            for j in range(len(Random_Features)):
                selectedFeatures.append(line[Random_Features[j]])
            selectedFeatures.append(line[-1])
            testing_subset.append(selectedFeatures)
        #print testing_subset
        Final_test_subset.append(testing_subset)
 
#def most_common(lst):
    #return max(set(lst), key=lst.count)
def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

def random_forest():
       
    tree = [DecisionTree() for i in range(20)]

    '''
    lines = []
    with open("D:\Study in UNB\Machine Learning\Assignment\data\car.csv") as csvfile:
        for line in csv.reader(csvfile, delimiter=","):
            lines.append(line)'''
    
    Number_of_Tree = random.randint(2, 20)
    #array_Random_Features = []
    print "Number of tree: "+ str(Number_of_Tree)  
    
    att=[]
    train=[]
    test=[]
    final_results = []
    
    create_smaller_subset(Number_of_Tree,att,train,test)
    
    for k in range(Number_of_Tree):
        
         #Indicate the target class
        target = att[k][-1]
        print "Number of training records: %d" % len(train[k])
        
        tree[k] = DecisionTree()
        tree[k].learn(train[k], att[k], target)
        #print_tree(tree[k].tree.copy(), "|")
        
        results =[]
        result_target = []
        
    #Test data and deliver the testing results of sub tree
        #print len(test[k])
    
        for entry in test[k]:
            tempDict = tree[k].tree.copy()
            result = ""
            while(isinstance(tempDict, dict)):
                root = Node(tempDict.keys()[0], tempDict[tempDict.keys()[0]])
                tempDict = tempDict[tempDict.keys()[0]]
                index = att[k].index(root.value)
                value = entry[index]
                if(value in tempDict.keys()):
                    child = Node(value, tempDict[value])
                    result = tempDict[value]
                    tempDict = tempDict[value]
                else:
                    result = "Null"
                    result_target.append(result)
                    break
            if result != "Null":
                results.append(result == entry[-1])
                result_target.append(result)
                
        #print results
        final_results.append(result_target)
                    
        accuracy = float(results.count(True))/float(len(results))
        print "Accuracy of Small Tree "+ str(k)+" :" + str(accuracy)
        
    #for i in final_results:
        #print len(i), i
    
    #Make the majority vote
    classified_random_forest=[]
    for i in range(len(final_results[1])):
        a = []
        for k in range(Number_of_Tree):
            #for t in test[k]:
            #    print t[-1],
            #print final_results[k]
            a.append(final_results[k][i])
        #print a
        #print most_common(a)
        classified_random_forest.append(str(most_common(a)))
    
    # Compare final result with test set and calculate the accuracy rate.
    #print classified_random_forest
    
    acc = []
    testSet = test[0]
    print "\n Result after majority voting:"
    for i in range(len(classified_random_forest)):
        print('> predicted=' + repr(classified_random_forest[i]) + ', actual=' + repr(testSet[i][-1]))
        if (classified_random_forest[i]==testSet[i][-1]):
            acc.append(True)
        else:
            acc.append(False)
    
    accuracy = float(acc.count(True))/float(len(acc))
    print "Accuracy:" + str(accuracy)
    
    return accuracy
            
def main():
    acc = []
    avg_acc = 0
    for i in range(10):
        acc.append(random_forest())
        avg_acc = avg_acc + acc[i]
    avg_acc = avg_acc/10
    
    print '\nAverage Accuracy: %.4f' % avg_acc
    
    std = 0
    for each in acc:
        std = std + (each-avg_acc)*(each-avg_acc)
        
    print "\nStandard deviation: %.4f" % math.sqrt(std/len(acc)) 

    
    
if __name__ == "__main__":
    main()