# ex1.py
# ------------------
# Licensing Information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to 20020001-UET (Github).
#
# Attribution Information: This experiment was developed at UET Artificial
# Intelligence Laboratory (AI Lab).

"""
ex1.py is a experiment implementation using modelAlgorithm to build up 
model and testing it. This is practice using python3 to run.

To using this module, simply import ex1.py in your python code.
Thank you for reading!

"""

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
                        - start the default experiment.
                    (2) python3 ex1.py --degree MIN_DEGREE --degree MAX_DEGREE
                        - start simulations of Polynomial Hypothesis Model with
                        degree increase from MIN_DEGREE to MAX_DEGREE.
                    (3) python3 ex1.py -f FS_METHOD
                        - start a simulation of Polynomial Hypothesis Model and
                        use Feature Scaling:
                            FS_METHOD=0 [Default] -> None
                            FS_METHOD=1 -> Min-Max Normalization
                            FS_METHOD=2 -> Mean Normalization
                    (4) python3 ex1.py -g
                        - enabled GRAPH MODE -> it means after building the
                        Model, a graph of plotting data from Training set and
                        Test set it shows up on screen.
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

    parser.add_option('-l', '--lambda', dest='Lambda', type='float',
                      help=default('the value of lambda which using for regularization'), default=0)

    parser.add_option('-g', '--graphModeEnabled', action='store_true', dest='graphModeEnabled', 
                      help=default('Create graph from the predict data'), default=False)

    parser.add_option('-o', '--outputEnabled', action='store_true', dest='outputEnabled', 
                      help=default('Create output file csv & config file'), default=False)

    options, otherjunk = parser.parse_args(argv)

    if len(otherjunk) != 0:
        raise Exception('Command input not understood: ' + str(otherjunk)) 

    return options

def extractArgs(options):

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
    args['lambda (Regularization)'] = options.Lambda
    args['GRAPH MODE enabled'] = options.graphModeEnabled
    args['OUTPUT enabled'] = options.outputEnabled

    return args

def printOption(options):
    print('Experiment 1 - Polynomial Hypothesis')
    print('Option:')

    args = extractArgs(options)

    for optionName, optionValue in args.items():
        line = '{:<24} = {:<8}'.format(str(optionName), str(optionValue))
        print('\t'+line)

def runExperiment(options):
    print('Process: ')
    header = '{:>8} {:>12} {:<72}'.format('Degree', 'R squared', 'Theta')
    print('\t' + header)
        
    # Read data set
    t_train, y_train, training_set = extractDataFromCSV(options.trainingSet)
    t_test, y_test, test_set = extractDataFromCSV(options.testSet)

    # Prepare the arguments of model
    alpha = options.alpha
    batch_size = options.batchSize
    iteration = options.iteration
    degree = options.degree
    feature_scaling_method = options.featureScalingMethod
    Lambda = options.Lambda

    # Output initialize
    output = []

    # Run the Experiment
    if len(degree) == 0:
        degree.append(3)

    if len(degree) == 2:
        n, m = degree
        degree = range(n, m+1)
     
    for current_degree in degree:
        # Initialize random with seed    
        np.random.seed(options.randomSeed)

        # Build model
        model = PolynomialHypothesis(training_set, alpha, batch_size, iteration, current_degree, feature_scaling_method, Lambda)
        model.training()

        # Calculate output
        cDegree = current_degree
        rSquared = model.rSquared(test_set.copy())
        cTheta = model.theta

        output.append([cDegree, rSquared, cTheta])

        # Print model result
        import textwrap
        print('\t' + '{:>8} {:>12.2E} '.format(str(cDegree), rSquared)
                + textwrap.fill(str(cTheta), 72, subsequent_indent='\t{:<22}'.format(' ')))

        if not options.graphModeEnabled:
            del model
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Polynomial Hypothesis with Degree = ' + str(current_degree))
        ax1.set_title('Training set')
        plotDataset(ax1, training_set.copy(), model)
        ax2.set_title('Test set')
        plotDataset(ax2, test_set.copy(), model)
        plt.show()

        del model

    # Create experiment name
    exp_name = 'Polynomial Hypothesis - Degree ' + str(degree)

    # Save the output value
    if options.outputEnabled:
        saveOutput(output, options, exp_name)


    print('Experiment has done!')

    return

def saveOutput(output, options, name):
    # Output value
    outputData = pd.DataFrame(output, columns=['Degree','R Squared','Theta'])
    outputData.to_csv(name+'.csv')

    # Experiment options
    outputFile = open(name+'.txt', 'w')
    args = extractArgs(options)

    for optionName, optionValue in args.items():
        line = '{:<24} = {:<8}'.format(str(optionName), str(optionValue))
        outputFile.write('\t'+line+'\n')

    outputFile.close()

if __name__ == '__main__':

    options = readCommand( sys.argv[1:] )

    printOption(options)
    runExperiment(options)


