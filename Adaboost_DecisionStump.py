import csv
import itertools
import random
import math

# Function to load data
def load_our_csv(filename):
    with open(filename) as f:
        data_reader = csv.reader(f)
        data = list(data_reader)
    # Strip zero rows at end
    while len(data[-1]) == 0:
        data.pop()
    # Strip white space
    data = [ [x.rstrip().strip() for x in row] for row in data]
    return data

data = load_our_csv("D:/Study in UNB/Machine Learning/Assignment/data/mushroom.csv")    

#print(data[0])
del data[0]

#Function to check whether feature is a numeric value
def is_numeric(column):
    for x in column:
        try:
            int(x)
        except ValueError:
            return False
    return True

#Function to process the final class column
def process_final_column(entry):
    """Assume the class is the final entry in each row."""
    parsedict = {"p":-1, "e":1}
    return parsedict[entry]

#Function to change value of features to binary array
def to_binary(entry, choices):
    return [ (0 if entry != choice else 1) for choice in choices]

if len( set( len(row) for row in data ) ) > 1:
    raise Exception("Rows have different lengths!")
num_columns = len(data[0])
size_continuous_columns = dict()
choices_for_columns = dict()
for column_number in range(num_columns):
    column = [ row[column_number] for row in data ]
    if is_numeric(column):
        size_continuous_columns[column_number] = len(set(column))
    else:
        choices_for_columns[column_number] = list(set(column))
        #if len(set(column)) <= 1:
            #raise Exception("Column {} seems to be constant!".format(column_number))
#print("Choices in continuous data:", list(size_continuous_columns.values()) )
#if min(size_continuous_columns.values()) <= 6:
    #raise Exception("There is a 'continuous' column which looks discrete!")
#for col in choices_for_columns:
#    print("Column {} -->".format(col), choices_for_columns[col])
    
data_processed = []
for row in data:
    newrow = []
    for cn, entry in enumerate(row[:-1]):
        if cn in size_continuous_columns:
            newrow.append( int(entry) )
        else:
            newrow.extend( to_binary(entry, choices_for_columns[cn]) )
    newrow.append( process_final_column(row[-1]) )
    data_processed.append( newrow )

#print(data_processed[0])
#print(data[0])





#This class is used to build a decision stump
class DecisionStump:
    def __init__(self, data, realclasses):
        """data is a list of values, and realclasses is a list of +/-1 indicating which class example i comes from.
        Initialises self.x0 so that the classifier given by i -> -1 iff x[i] <= self.x0 obtains the highest
        rate of correct classification.  self.correct is set to count of correctly classified data."""
        choices = set(data)
        choices.add( min(choices) - 1)
        self.x0 = min(choices, key = lambda choice : self.number_correct(data, realclasses, choice) )
        self.correct = len(data) - self.number_correct(data, realclasses, self.x0)

    def number_correct(self, data, realclasses, x0):
        return ( sum( cl != -1 for x, cl in zip(data, realclasses) if x <= x0)
                      + sum( cl != 1 for x, cl in zip(data, realclasses) if x > x0) )

    def classify(self, entry):
        return -1 if entry <= self.x0 else 1

#d = DecisionStump([8,2,4,5,5], [-1,1,1,-1,1])
#print d.x0, d.correct


#This class is used to build decision stump along with data
class DecisionStumpFromData:
    def __init__(self, data):
        """data is a list of rows, each row being a list of features ending with the class +/-1."""
        """Sets self.x0, self.column, self.negated=True/False."""
        num_columns = len(data[0]) - 1
        #print num_columns
        choices = itertools.product(range(num_columns), [True, False])
        #print itertools.product(range(num_columns), [True, False])
        classes = [row[-1] for row in data]
        #print classes
        results = []
        for (col, negated) in choices:
            column = [ row[col] for row in data ]
            #print column
            if negated:
                column = [-x for x in column]
                #print column
            d = DecisionStump(column, classes)
            results.append( (col, negated, d.x0, d.correct) )
        self.column, self.negated, self.x0, _ = max(results, key = lambda tup : tup[3])
        
    def classify(self, row):
        entry = row[self.column]
        if self.negated:
            entry = -entry
        return -1 if entry <= self.x0 else 1
    
    def __repr__(self):
        return "DecisionStumpFromData(x0={}, column={}, negated={})".format(self.x0, self.column, self.negated)
    

# with open("D:/Study in UNB/Machine Learning/Assignment/data/adult_data_processed.csv") as f:
#     data = [ [ int(x) for x in row] for row in csv.reader(f) ]
# print(data[0])

# import random
# d = DecisionStumpFromData(random.sample(data, 500))
# print len(data)
# print sum( d.classify(row) == row[-1] for row in data)
# print (sum( d.classify(row) == row[-1] for row in data)/float(len(data)))
# print d


#This class is use to adjust the weights
class WeightedReSampler:
    def __init__(self, data, weights):
        """data is a list, and weights should be a list of non-negative but unnormalised density."""
        self.data = list(zip(data, weights))
        self.data.sort(key = lambda pair : pair[1])
        self.normalisation = sum(p for _,p in self.data)
        
    def sample(self):
        prob = random.random() * self.normalisation
        cumulative = 0
        for x, p in self.data:
            cumulative += p
            if cumulative >= prob:
                return x
        return self.data[-1][0]
    
# import collections
# values = [1,2,3,4,5]
# prob = [1,10,2,5,13]
# w = WeightedReSampler(values, prob)
# c = collections.Counter(w.sample() for _ in range(100000))
# for x, p in zip(values,prob):
#     print(x,c[x]/100000., p/float(sum(prob)))
    


def WLDecisionStump(samples):
    """Input: samples is list of input rows in our format.
    Output should be an object which supports a `classify` method."""
    return DecisionStumpFromData(samples)

#This class is using adaboost for decision stump
class AdaBoost:
    def __init__(self, weaklearner, data, classes):
        """weaklearner should be a function `DecisionStumpFromData(samples)` which returns an object
        representing a learner and which has a method `classify` compatible with the data.
        `data` should be a list of data, and `classes` should be a list of the class each row is in."""
        self.data = data
        self.classes = classes
        self.weaklearner = weaklearner
        self.weights = [1.0] * len(data)
        self.weightsum = []
        
    def add(self, alpha, h):
        self.weightsum.append( (alpha, h) )
        
    def classify(self, row):
        x = sum( alpha * h.classify(row) for alpha, h in self.weightsum )
        return -1 if x <=0 else 1
    
    def fraction_correct(self):
        return sum( cl == self.classify(row) for cl, row in zip(self.classes, self.data) ) / float(len(self.data))
    
    def step(self):
        wrs = WeightedReSampler(self.data, self.weights)
        samples = [wrs.sample() for _ in range(500)] # 500 = Magic Number
        ht = self.weaklearner(samples)
        et = sum( w for w, row, cl in zip(self.weights, self.data, self.classes) if  cl != ht.classify(row) )
        et /= sum(self.weights)
        alphat = math.log( (1-et) / et ) / 2
        self.add(alphat, ht)
        self.weights = [ w * math.exp(-cl * ht.classify(row) * alphat) for row, cl, w in zip(self.data, self.classes, self.weights) ]
        return et



def main():    
    
    K = 5
    
    for k in range(K):
        random.shuffle(data_processed)
        training_set = [x for i, x in enumerate(data_processed) if i % K != k]
        test_set = [x for i, x in enumerate(data_processed) if i % K == k]
    
    #print training_set[0]
    
    ab = AdaBoost(WLDecisionStump, training_set, [row[-1] for row in training_set])
    for _ in range(20):
        print 'Round '+ str(_) +' => Error rate for current distribution:', ab.step()
        print "        => Currently being classified correctly:", ab.fraction_correct()
    
#     with open("adult_test_processed.csv") as f:
#         test_data = [ [ int(x) for x in row] for row in csv.reader(f) ]
#     print sum( row[-1] == ab.classify(row) for row in test_data ) / float(len(test_data))

    
    
    #d = DecisionStumpFromData(random.sample(training_set, 500))
    d = DecisionStumpFromData(training_set)
    print '\nDecisionStump without Adaboost - Accuracy = '+str(sum( d.classify(row) == row[-1] for row in test_set ) / float(len(test_set)))
    
    print '\nAfter using Adaboost - Accuracy = '+str(sum( row[-1] == ab.classify(row) for row in test_set ) / float(len(test_set)))
    
    
if __name__ == "__main__":
    main()    
