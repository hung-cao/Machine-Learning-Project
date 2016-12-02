import os
import sys
import math
import collections
import operator
import copy
import random
import csv


## Distribution ########################################################################################

class Distribution(object):

    def __init__(self):
        raise NotImplementedError("Subclasses should override.")

    @classmethod
    def mleEstimate(cls, points):
        raise NotImplementedError("Subclasses should override.")

    @classmethod
    def momEstimate(cls, points):
        raise NotImplementedError("Subclasses should override.")

## ContinuousDistribution ##############################################################################

class ContinuousDistribution(Distribution):

    def pdf(self, value):
        raise NotImplementedError("Subclasses should override.")

    def cdf(self, value):
        raise NotImplementedError("Subclasses should override.")

## Uniform #############################################################################################

class Uniform(ContinuousDistribution):

    def __init__(self, alpha, beta):
        if alpha == beta: raise ParametrizationError("alpha and beta cannot be equivalent")
        self.alpha = alpha
        self.beta = beta
        self.range = beta - alpha

    def pdf(self, value):
        if value < self.alpha or value > self.beta: return 0.0
        else: return 1.0 / self.range

    def cdf(self, value):
        if value < self.alpha: return 0.0
        elif value >= self.beta: return 1.0
        else: return (value - self.alpha) / self.range

    def __str__(self):
        return "Continuous Uniform distribution: alpha = %s, beta = %s" % (self.alpha, self.beta)

    @classmethod
    def mleEstimate(cls, points):
        return cls(min(points), max(points))

## Gaussian ############################################################################################

class Gaussian(ContinuousDistribution):

    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev
        if stdev == 0.0: raise ParametrizationError("standard deviation must be non-zero")
        if stdev < 0.0: raise ParametrizationError("standard deviation must be positive")
        self.variance = math.pow(stdev, 2.0)

    def pdf(self, value):
        numerator = math.exp(-math.pow(float(value - self.mean) / self.stdev, 2.0) / 2.0)
        denominator = math.sqrt(2 * math.pi * self.variance)
        return numerator / denominator

    def cdf(self, value):
        return 0.5 * (1.0 + math.erf((value - self.mean) / math.sqrt(2.0 * self.variance)))

    def __str__(self):
        return "Continuous Gaussian (Normal) distribution: mean = %s, standard deviation = %s" % (self.mean, self.stdev)

    @classmethod
    def mleEstimate(cls, points):
        numPoints = float(len(points))
        if numPoints <= 1: raise EstimationError("must provide at least 2 training points")

        mean = sum(points) / numPoints

        variance = 0.0
        for point in points:
            variance += math.pow(float(point) - mean, 2.0)
        variance /= (numPoints - 1.0)
        stdev = math.sqrt(variance)

        return cls(mean, stdev)

## TruncatedGaussian ##################################################################################

class TruncatedGaussian(ContinuousDistribution):

    def __init__(self, mean, stdev, alpha, beta):
        self.mean = mean
        if stdev == 0.0: raise ParametrizationError("standard deviation must be non-zero")
        if stdev < 0.0: raise ParametrizationError("standard deviation must be positive")
        self.stdev = stdev
        self.variance = math.pow(stdev, 2.0)
        self.alpha = alpha
        self.beta = beta

    def pdf(self, value):
        if self.alpha == self.beta or self.__phi(self.alpha) == self.__phi(self.beta):
            if value == self.alpha: return 1.0
            else: return 0.0
        else:
            numerator = math.exp(-math.pow((value - self.mean) / self.stdev, 2.0) / 2.0)
            denominator = math.sqrt(2 * math.pi) * self.stdev * (self.__phi(self.beta) - self.__phi(self.alpha))
            return numerator / denominator

    def cdf(self, value):
        if value < self.alpha or value > self.beta:
            return 0.0
        else:
            numerator = self.__phi((value - self.mean) / self.stdev) - self.__phi(self.alpha)
            denominator = self.__phi(self.beta) - self.__phi(self.alpha)
            return numerator / denominator

    def __str__(self):
        return "Continuous Truncated Gaussian (Normal) distribution: mean = %s, standard deviation = %s, lower bound = %s, upper bound = %s" % (self.mean, self.stdev, self.alpha, self.beta)

    @classmethod
    def mleEstimate(cls, points):
        numPoints = float(len(points))

        if numPoints <= 1: raise EstimationError("must provide at least 2 training points")

        mean = sum(points) / numPoints

        variance = 0.0
        for point in points:
            variance += math.pow(float(point) - mean, 2.0)
        variance /= (numPoints - 1.0)
        stdev = math.sqrt(variance)

        return cls(mean, stdev, min(points), max(points))

    def __phi(self, value):
        return 0.5 * (1.0 + math.erf((value - self.mean) / (self.stdev * math.sqrt(2.0))))

## LogNormal ###########################################################################################

class LogNormal(ContinuousDistribution):

    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev
        if stdev == 0.0: raise ParametrizationError("standard deviation must be non-zero")
        if stdev < 0.0: raise ParametrizationError("standard deviation must be positive")
        self.variance = math.pow(stdev, 2.0)

    def pdf(self, value):
        if value <= 0:
            return 0.0
        else:
            return math.exp(-math.pow(float(math.log(value) - self.mean) / self.stdev, 2.0) / 2.0) / (value * math.sqrt(2 * math.pi * self.variance))

    def cdf(self, value):
        return 0.5 + 0.5 * math.erf((math.log(value) - self.mean) / math.sqrt(2.0 * self.variance))

    def __str__(self):
        return "Continuous Log Normal distribution: mean = %s, standard deviation = %s" % (self.mean, self.stdev)

    @classmethod
    def mleEstimate(cls, points):
        numPoints = float(len(points))

        if numPoints <= 1: raise EstimationError("must provide at least 2 training points")

        mean = sum(math.log(float(point)) for point in points) / numPoints

        variance = 0.0
        for point in points:
            variance += math.pow(math.log(float(point)) - mean, 2.0)
        variance /= (numPoints - 1.0)
        stdev = math.sqrt(variance)

        return cls(mean, stdev)

## Exponential ########################################################################################

class Exponential(ContinuousDistribution):

    def __init__(self, lambdaa):
        # 2 "a"s to avoid confusion with "lambda" keyword
        self.lambdaa = lambdaa
    
    def mean(self):
        return 1.0 / self.lambdaa
    
    def variance(self):
        return 1.0 / pow(self.lambdaa, 2.0)

    def pdf(self, value):
        return self.lambdaa * math.exp(-self.lambdaa * value)

    def cdf(self, value):
        return 1.0 - math.exp(-self.lambdaa * value)

    def __str__(self):
        return "Continuous Exponential distribution: lamda = %s" % self.lambdaa

    @classmethod
    def mleEstimate(cls, points):
        if len(points) == 0: raise EstimationError("Must provide at least one point.")
        if min(points) < 0.0: raise EstimationError("Exponential distribution only supports non-negative values.")
        
        mean = float(sum(points)) / float(len(points))
        
        if mean == 0.0: raise ParametrizationError("Mean of points must be positive.")
        
        return cls(1.0 / mean)

## KernelDensityEstimate ##############################################################################

class KernelDensityEstimate(ContinuousDistribution):
    '''
        See this paper for more information about using Gaussian
        Kernal Density Estimation with the Naive Bayes Classifier:
        http://www.cs.iastate.edu/~honavar/bayes-continuous.pdf
    '''

    def __init__(self, observedPoints):
        self.observedPoints = observedPoints
        self.numObservedPoints = float(len(observedPoints))
        self.stdev = 1.0 / math.sqrt(self.numObservedPoints)

    def pdf(self, value):
        pdfValues = [self.__normalPdf(point, self.stdev, value) for point in self.observedPoints]
        return sum(pdfValues) / self.numObservedPoints

    def __normalPdf(self, mean, stdev, value):
        numerator = math.exp(-math.pow(float(value - mean) / stdev, 2.0) / 2.0)
        denominator = math.sqrt(2 * math.pi * math.pow(stdev, 2.0))
        return numerator / denominator

    def cdf(self, value):
        raise NotImplementedError("Not implemented")

    def __str__(self):
        return "Continuous Gaussian Kernel Density Estimate distribution"

    @classmethod
    def mleEstimate(cls, points):
        return cls(points)

## DiscreteDistribution ###############################################################################

class DiscreteDistribution(Distribution):

    def probability(self, value):
        raise NotImplementedError("Subclasses should override.")

## Uniform ############################################################################################

class DiscreteUniform(DiscreteDistribution):

    def __init__(self, alpha, beta):
        if alpha == beta: raise Exception("alpha and beta cannot be equivalent")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.prob = 1.0 / (self.beta - self.alpha)

    def probability(self, value):
        if value < self.alpha or value > self.beta: return 0.0
        else: return self.prob

    def __str__(self):
        return "Discrete Uniform distribution: alpha = %s, beta = %s" % (self.alpha, self.beta)

    @classmethod
    def mleEstimate(cls, points):
        return cls(min(points), max(points))

## Poissoin ###########################################################################################

class Poisson(DiscreteDistribution):

    def __init__(self, lambdaa):
        # 2 "a"s to avoid confusion with "lambda" keyword
        self.lambdaa = lambdaa

    def probability(self, value):
        try:
            first = float(math.pow(self.lambdaa, value)) / float(math.factorial(value))
            second = float(math.exp(-float(self.lambdaa)))
            return first * second
        except OverflowError as error:
            # this is an approximation to the probability of very unlikely events
            return 0.0

    def __str__(self):
        return "Discrete Poisson distribution: lamda = %s" % self.lambdaa

    @classmethod
    def mleEstimate(cls, points):
        mean = float(sum(points)) / float(len(points))
        return cls(mean)

## Multinomial #######################################################################################

class Multinomial(DiscreteDistribution):

    def __init__(self, categoryCounts, smoothingFactor = 1.0):
        self.categoryCounts = categoryCounts
        self.numPoints = float(sum(categoryCounts.values()))
        self.numCategories = float(len(categoryCounts))
        self.smoothingFactor = float(smoothingFactor)

    def probability(self, value):
        if not value in self.categoryCounts:
            return 0.0
        numerator = float(self.categoryCounts[value]) + self.smoothingFactor
        denominator = self.numPoints + self.numCategories * self.smoothingFactor
        return numerator / denominator

    def __str__(self):
        return "Discrete Multinomial distribution: buckets = %s" % self.categoryCounts

    @classmethod
    def mleEstimate(cls, points):
        categoryCounts = collections.Counter()
        for point in points:
            categoryCounts[point] += 1
        return cls(categoryCounts)

## Binary ############################################################################################

class Binary(Multinomial):

    def __init__(self, trueCount, falseCount, smoothingFactor = 1.0):
        categoryCounts = { True : trueCount, False : falseCount }
        Multinomial.__init__(self, categoryCounts, smoothingFactor)

    def __str__(self):
        return "Discrete Binary distribution: true count = %s, false count = %s" % (self.categoryCounts[True], self.categoryCounts[False])

    @classmethod
    def mleEstimate(cls, points, smoothingFactor = 1.0):
        trueCount = 0
        for point in points:
            if point: trueCount += 1
        falseCount = len(points) - trueCount
        return cls(trueCount, falseCount, smoothingFactor)

## Errors ############################################################################################

class EstimationError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class ParametrizationError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


## Feature ##############################################################################

class Feature(object):

    def __init__(self, name, distribution, value):
        self.name = name
        self.distribution = distribution
        self.value = value

    def __repr__(self):
        return self.name + " => " + str(self.value)
    
    def hashable(self):
        return (self.name, self.value)

    @classmethod
    def binary(cls, name):
        return cls(name, Binary, True)

## ExtractedFeature #####################################################################

class ExtractedFeature(Feature):

    def __init__(self, object):
        name = self.__class__.__name__
        distribution = self.distribution()
        value = self.extract(object)
        super(ExtractedFeature, self).__init__(name, distribution, value)

    def extract(self, object):
        # returns feature value corresponding to |object|
        raise NotImplementedError("Subclasses should override.")

    @classmethod
    def distribution(cls):
        # returns the distribution this feature conforms to
        raise NotImplementedError("Subclasses should override.")

## NaiveBayesClassifier #################################################################

class NaiveBayesClassifier(object):

    def __init__(self, featurizer = None):
        self.featurizer = featurizer
        self.priors = None
        self.distributions = None
    
    def featurize(self, object):
        if self.featurizer is None:
            raise Exception("If no featurizer is provided upon initialization, self.featurize must be overridden.")
        return self.featurizer(object)

    def train(self, objects, labels):
        featureValues = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
        distributionTypes = {}

        labelCounts = collections.Counter()

        for index, object in enumerate(objects):
            label = labels[index]
            labelCounts[label] += 1
            for feature in self.featurize(object):
                featureValues[label][feature.name].append(feature.value)
                distributionTypes[feature.name] = feature.distribution

        self.distributions = collections.defaultdict(lambda: {})
        for label in featureValues:
            for featureName in featureValues[label]:
                try:
                    values = featureValues[label][featureName]
                    if issubclass(distributionTypes[featureName], Binary):
                        trueCount = len([value for value in values if value])
                        # the absence of binary feature is treated as it having been present with a False value
                        falseCount = labelCounts[label] - trueCount
                        distribution = Binary(trueCount, falseCount)
                    else:
                        distribution = distributionTypes[featureName].mleEstimate(values)
                except EstimationError, ParametrizationError:
                    if issubclass(distributionTypes[featureName], Binary):
                        distribution = Binary(0, labelCounts[label])
                    elif issubclass(distributionTypes[featureName], DiscreteDistribution):
                        distribution = DiscreteUniform(-sys.maxint, sys.maxint)
                    else:
                        distribution = Uniform(-sys.float_info.max, sys.float_info.max)
                self.distributions[label][featureName] = distribution

        self.priors = collections.Counter()
        for label in labelCounts:
            # A label count can never be 0 because we only generate
            # a label count upon observing the first data point that
            # belongs to it. As a result, we don't worrying about
            # the argument to log being 0 here.
            self.priors[label] = math.log(labelCounts[label])

    def __labelWeights(self, object):
        features = self.featurize(object)

        labelWeights = copy.deepcopy(self.priors)

        for feature in features:
            for label in self.priors:
                if feature.name in self.distributions[label]:
                    distribution = self.distributions[label][feature.name]
                    if isinstance(distribution, DiscreteDistribution):
                        probability = distribution.probability(feature.value)
                    elif isinstance(distribution, ContinuousDistribution):
                        probability = distribution.pdf(feature.value)
                    else:
                        raise Exception("Naive Bayes Training Error: Invalid probability distribution")
            
                else:
                    if issubclass(feature.distribution, Binary):
                        distribution = Binary(0, self.priors[label])
                        probability = distribution.probability(feature.value)
                    else:
                        raise Exception("Naive Bayes Training Error: Non-binary features must be present for all training examples")

                if probability == 0.0: labelWeights[label] = float("-inf")
                else: labelWeights[label] += math.log(probability)

        return labelWeights
    
    def probability(self, object, label):
        labelWeights = self.__labelWeights(object)
        
        numerator = labelWeights[label]
        if numerator == float("-inf"): return 0.0
        
        denominator = 0.0
        minWeight = min(labelWeights.iteritems(), key=operator.itemgetter(1))[1]
        for label in labelWeights:
            weight = labelWeights[label]
            if minWeight < 0.0: weight /= (-minWeight)
            denominator += math.exp(weight)
        denominator = math.log(denominator)
        
        return math.exp(numerator - denominator)

    def probabilities(self, object):
        labelProbabilities = collections.Counter()
        for label in self.priors:
            labelProbabilities[label] = self.probability(object, label)
        return labelProbabilities

    def classify(self, object, costMatrix=None):
        if costMatrix is None:
            labelWeights = self.__labelWeights(object)
            return max(labelWeights.iteritems(), key=operator.itemgetter(1))[0]
        
        else:
            labelCosts = collections.Counter()
            labelProbabilities = self.probabilities(object)
            for predictedLabel in labelProbabilities:
                if predictedLabel not in costMatrix: raise Exception("Naive Bayes Prediction Error: Cost matrix does not include all labels.")
                cost = 0.0
                for actualLabel in labelProbabilities:
                    if actualLabel not in costMatrix: raise Exception("Naive Bayes Prediction Error: Cost matrix does not include all labels.")
                    cost += labelProbabilities[predictedLabel] * costMatrix[predictedLabel][actualLabel]
                labelCosts[predictedLabel] = cost
            return min(labelCosts.iteritems(), key=operator.itemgetter(1))[0]

    def accuracy(self, objects, goldLabels):
        if len(objects) == 0 or len(objects) != len(goldLabels):
            raise ValueError("Malformed data")
        
        numCorrect = 0
        for index, object in enumerate(objects):
            if self.classify(object) == goldLabels[index]:
                numCorrect += 1
                #print('> predicted=' + repr(self.classify(object)) + ', actual=' + repr(goldLabels[index]))
            #else:
                #print('> predicted=' + repr(self.classify(object)) + ', actual=' + repr(goldLabels[index]))
        return float(numCorrect) / float(len(objects))

def classifier():
        
    data = []
    target = []
    attributes = []
    
    with open('D:\Study in UNB\Machine Learning\Assignment\data\letter.csv', 'r+') as csvfile:
        content = csv.reader(csvfile, delimiter=',', quotechar='"')
        dataset = list(content)
        for i in dataset[0]:
            attributes.append(i)
        del dataset[0]
        
        random.shuffle(dataset)    
        
        for line in dataset:
            #data.append([item.strip() for item in line])
            data.append(line[:-1])
            target.append(line[-1])
    
    #print data
    #print target
    
    '''
    for i in range(1,len(dataset)):
        if random.random() < split:
            training_set[0].append(data[i])
            training_set[1].append(Class[i])
        else:
            test_set[0].append(data[i])
            test_set[1].append(Class[i])
        
    
    print training_set
    '''
    '''
    # The data set is described here: https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)
    raw_data = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data").text.strip()
    
    #print raw_data
    
    lines = raw_data.split("\n")
    #print lines
    value_matrix = [line.split() for line in lines]
    #for i in value_matrix:
    #    print i
    data_points = [values[:-1] for values in value_matrix]
    #for i in data_points:
    #    print i
    print len(data_points)
    #print data_points
    labels = [values[-1] for values in value_matrix]
    
    for i in labels:
        print i
    
    data_set_slice = len(data_points) / 2
    training_set = (data_points[:data_set_slice], labels[:data_set_slice])
    test_set = (data_points[data_set_slice:], labels[data_set_slice:])
    
    print training_set[0]
    #print training_set[1]
    '''
    data_set_slice = len(data) / 5
    test_set = (data[:data_set_slice], target[:data_set_slice])
    training_set = (data[data_set_slice:], target[data_set_slice:])
    
    print "Number of training samples:" + str(len(data[data_set_slice:]))
    print "Number of test samples:" + str(len(data[:data_set_slice]))
    
    def featurizer(data_point):
        '''return [ 
            nb.Feature("Checking account status", distributions.Multinomial, data_point[0]), # bucketed and therefore categorical
            nb.Feature("Duration in months", distributions.Exponential, float(data_point[1])), # continuous and probably follows a power law distribution
            nb.Feature("Credit history", distributions.Multinomial, data_point[2]), # categorical
            nb.Feature("Purpose", distributions.Multinomial, data_point[3]), # categorical
            nb.Feature("Credit amount", distributions.Gaussian, float(data_point[4])), # continuous and probably follows a normal distribution
            nb.Feature("Savings account status", distributions.Multinomial, data_point[5]), # bucketed and therefore categorical
            nb.Feature("Unemployment duration", distributions.Multinomial, data_point[6]), # bucketed and therefore categorical
            nb.Feature("Installment rate", distributions.Gaussian, float(data_point[7])), # continuous and probably follows a normal distribution
            nb.Feature("Personal status", distributions.Multinomial, data_point[8]), # categorical
            nb.Feature("Other debtors", distributions.Multinomial, data_point[9]), # categorical
            nb.Feature("Present residence", distributions.Exponential, float(data_point[10])), # continuous and probably follows a power law distribution
            nb.Feature("Property status", distributions.Multinomial, data_point[11]), # categorical
            nb.Feature("Age", distributions.Gaussian, float(data_point[12])), # continuous and probably follows a normal distribution
            nb.Feature("Other installment plans", distributions.Multinomial, data_point[13]), # categorical
            nb.Feature("Housing", distributions.Multinomial, data_point[14]), # categorical
            nb.Feature("Number of credit cards", distributions.Exponential, float(data_point[15])), # continuous and probably follows a power law distribution
            nb.Feature("Job", distributions.Multinomial, data_point[16]), # categorical
            nb.Feature("Number of people liable", distributions.Exponential, float(data_point[17])), # continuous and probably follows a power law distribution
            nb.Feature("Telephone", distributions.Multinomial, data_point[18]), # categorical
            nb.Feature("Foreign worker", distributions.Multinomial, data_point[19]) # categorical
            '''
        return [ 
            Feature("x-box", Multinomial, data_point[0]), 
            Feature("y-box", Multinomial, data_point[1]), 
            Feature("width", Multinomial, data_point[2]), 
            Feature("high", Multinomial, data_point[3]), 
            Feature("onpix", Multinomial, data_point[4]), 
            Feature("x-bar", Multinomial, data_point[5]), 
            Feature("y-bar", Multinomial, data_point[6]), 
            Feature("x2bar", Multinomial, data_point[7]), 
            Feature("y2bar", Multinomial, data_point[8]), 
            Feature("xybar", Multinomial, data_point[9]), 
            Feature("x2ybr", Multinomial, data_point[10]), 
            Feature("xy2br", Multinomial, data_point[11]), 
            Feature("x-ege", Multinomial, data_point[12]), 
            Feature("xegvy", Multinomial, data_point[13]), 
            Feature("y-ege", Multinomial, data_point[14]), 
            Feature("yegvx", Multinomial, data_point[15])
        ]    
    
    classify = NaiveBayesClassifier(featurizer)
    classify.train(training_set[0], training_set[1])
    #print featurizer
    print "Accuracy = %s" % classify.accuracy(test_set[0], test_set[1])
    
    return classify.accuracy(test_set[0], test_set[1])
    
def main():
    acc = []
    avg_acc = 0
    for i in range(10):
        #attributes[:] = []
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