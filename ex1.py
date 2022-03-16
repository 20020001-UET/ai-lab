import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import modelAlgorithm
from modelAlgorithm import extractDataFromCSV
from modelAlgorithm import PolynomialHypothesis
from modelAlgorithm import plotDataset

def default(str):
    return str + ' [Default: %default]'

def readCommand( argv ):
    """
    Processes the command used to run the experiment from the command line.
    """

    from optparse import OptionParser
    usageStr = """
    USAGE:          python3 ex1.py <option>
    EXAMPLES:   
                    (1) python3 ex1.py
                        - start the default experiment
    """

    parser = OptionParser(usageStr)

    parser.add_option('--train', dest='trainingSet', type='str',
                      help=default('the file\'s name contains training set'), default='train.csv')

    parser.add_option('--test', dest='testSet', type='str',
                      help=default('the file\'s name contains test set'), default='test.csv')

    parser.add_option('-a', '--alpha', dest='alpha', type='float', 
                      help=default('the learning rate'), default=0.01)

    parser.add_option('-b', '--batchSize', dest='batchSize', type='int',
                      help=default('the batch size'), default=1)

    parser.add_option('-i', '--iteration', dest='iteration', type='int',
                      help=default('the number of repetition of the Learning process'), default=100)

    parser.add_option('-d', '--degree', dest='degree', action='append', type='int',
                      help=default('the degree of polynomial'), default=[])

    parser.add_option('-r', '--randomSeed', dest='randomSeed', type='int',
                      help=default('the random seed'), default=10)

    parser.add_option('-f', '--featureScalingMethod', dest='featureScalingMethod', type='int',
                      help=default('the feature scaling METHOD TYPE in the model algorithm to use'), default=0)

    parser.add_option('-g', '--graphModeEnabled', action='store_true', dest='graphModeEnabled', 
                      help=default('Create graph from the predict data'), default=False)

    options, otherjunk = parser.parse_args(argv)

    if len(otherjunk) != 0:
        raise Exception('Command input not understood: ' + str(otherjunk)) 

    return options

def printOption(options):
    print('Experiment 1 - Polynomial Hypothesis')
    print('Option:')

    args = dict()

    # Get option arguments
    args['training file name'] = options.trainingSet
    args['test file name'] = options.testSet
    args['learning rate'] = options.alpha
    args['batch size'] = options.batchSize
    args['iteration'] = options.iteration
    args['degree'] = options.degree
    args['random seed'] = options.randomSeed
    args['FEATURE SCALING method'] = options.featureScalingMethod
    args['GRAPH MODE enabled'] = options.graphModeEnabled

    for optionName, optionValue in args.items():
        line = '{:<24} = {:<8}'.format(str(optionName), str(optionValue))
        print('\t'+line)

def runExperiment(options):
    print('Process: ')
    header = '{:>8} {:>12} {:<72}'.format('Degree', 'R squared', 'Theta')
    print('\t' + header)
    
    # Initialize random with seed
    np.random.seed(options.randomSeed)
        
    # Read data set
    t_train, y_train, training_set = extractDataFromCSV(options.trainingSet)
    t_test, y_test, test_set = extractDataFromCSV(options.testSet)

    # Prepare the arguments of model
    alpha = options.alpha
    batch_size = options.batchSize
    iteration = options.iteration
    degree = options.degree
    feature_scaling_method = options.featureScalingMethod

    # Run the Experiment
    if len(degree) == 0:
        degree.append(3)
    if len(degree) == 2:
        n, m = degree
        degree = range(n, m+1)
    for current_degree in degree:
        # Build model
        model = PolynomialHypothesis(training_set.copy(), alpha, batch_size, iteration, current_degree, feature_scaling_method)
        model.training()

        # Print model result
        import textwrap
        print('\t' + '{:>8} {:>12.8f} '.format(str(current_degree), model.rSquared(test_set.copy()))
                + textwrap.fill(str(model.theta), 72, subsequent_indent='\t{:<22}'.format(' ')))

        if not options.graphModeEnabled:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Polynomial Hypothesis with Degree = ' + str(current_degree))
        ax1.set_title('Training set')
        plotDataset(ax1, training_set.copy(), model)
        ax2.set_title('Test set')
        plotDataset(ax2, test_set.copy(), model)
        plt.show()

        del model

    print('Experiment done!')

    return

if __name__ == '__main__':

    options = readCommand( sys.argv[1:] )

    printOption(options)
    runExperiment(options)


