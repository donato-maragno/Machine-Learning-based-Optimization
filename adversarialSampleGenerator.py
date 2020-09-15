import pickle
import random
import math
from keras.engine.saving import load_model
from pyomo import environ
from pyomo.environ import *
from pyomo.opt import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

'''
    adversarialGen() is a function that returns the optimal solution given a certain scenario (.dat) and
    a palatability_limit value. Every time a solution is found, it is stored in a dataset that is used to retrain the 
    ANN once we store a solution for each of the palatability_limit values (from 1 to 9).
    A for loop has been used to iterate over different palatability_limit values. 
'''
def adversarialGen():
    # Features of the model
    features_name = ['Beans', 'Bulgur', 'Cheese', 'Fish', 'Meat', 'Corn-soya blend (CSB)', 'Dates',
                     'Dried skim milk (enriched) (DSM)', 'Milk', 'Salt', 'Lentils', 'Maize', 'Maize meal', 'Chickpeas', 'Rice',
                     'Sorghum/millet', 'Soya-fortified bulgur wheat', 'Soya-fortified maize meal',
                     'Soya-fortified sorghum grits', 'Soya-fortified wheat flour', 'Sugar', 'Oil', 'Wheat',
                     'Wheat flour', 'Wheat-soya blend (WSB)']

    # Importing the pre-trained models
    # file_lr = 'modelsOL/lr_OL.sav'
    # linear_reg = pickle.load(open(file_lr, 'rb'))
    file_ANN_simple = 'modelsOL/simple2_OL.h5'
    ANN = load_model(file_ANN_simple)
    weights_ANN = pd.DataFrame(ANN.get_weights()[0], index=features_name)
    # weights = pd.DataFrame(linear_reg.coef_, columns=features_name)
    # intercept = linear_reg.intercept_

    '''
        It is used to the difene the context and structure of the problem.
        An instance is created later with the '.dat' file.
    '''
    model = AbstractModel()

    # Foods
    model.F = Set()
    # Nutrients
    model.N = Set()
    # Cereals & grains
    model.G1 = Set()
    # Pulses & Vegetables
    model.G2 = Set()
    # Oil & Fats
    model.G3 = Set()
    # Mixed & Blended Foods
    model.G4 = Set()
    # Meat & Fish & Dairy
    model.G5 = Set()

    # Cost of each food
    model.c = Param(model.F, within=PositiveReals)
    # Amount of nutrient in each food
    model.a = Param(model.F, model.N, within=NonNegativeReals)
    # Lower bound on each nutrient
    model.Nmin = Param(model.N, within=NonNegativeReals, default=0.0)
    # Volume per serving of food
    model.V = Param(model.F, within=PositiveReals)
    # Upper and lower bounds on each group
    model.maxG1 = Param(within=NonNegativeReals)
    model.minG1 = Param(within=NonNegativeReals)
    model.maxG2 = Param(within=NonNegativeReals)
    model.minG2 = Param(within=NonNegativeReals)
    model.maxG3 = Param(within=NonNegativeReals)
    model.minG3 = Param(within=NonNegativeReals)
    model.maxG4 = Param(within=NonNegativeReals)
    model.minG4 = Param(within=NonNegativeReals)
    model.maxG5 = Param(within=NonNegativeReals)
    model.minG5 = Param(within=NonNegativeReals)
    # Decision variable
    model.x = Var(model.F, within=NonNegativeReals)
    # Slack variables
    model.Sl = Var(model.N, within=NonNegativeReals)

    # Extra variables to control the palatability score (working in progress)
    # model.Sg1 = Var(within=Reals) ###
    # model.Sg2 = Var(within=Reals) ###
    # model.Sg3 = Var(within=Reals) ###
    # model.Sg4 = Var(within=Reals) ###
    # model.Sg5 = Var(within=Reals) ###

    # Minimize the cost of food that is consumed
    def cost_rule1(model):
        return sum(model.c[i] * model.x[i] / 10000 for i in model.F)

    # Minimize the value of the Sl[j]s
    def cost_rule2(model):
        return sum(model.Sl[i] for i in model.N)

    # Limit nutrient consumption for each nutrient
    def nutrient_rule(model, j):
        value = sum(model.a[i, j] * model.x[i] for i in model.F)
        return value >= model.Nmin[j] * (1 - model.Sl[j])

    def slackNut_rule(model, i):
        return model.Sl[i] <= 1

    '''
        Limiting the amount of Food for each group
    '''
    def G1_rule(model):
        valueG1 = sum(model.x[i] * 100 for i in model.G1)
        return environ.inequality(model.minG1, valueG1, model.maxG1)

    def G2_rule(model):
        valueG2 = sum(model.x[i] * 100 for i in model.G2)
        return environ.inequality(model.minG2, valueG2, model.maxG2)

    def G3_rule(model):
        valueG3 = sum(model.x[i] * 100 for i in model.G3)
        return environ.inequality(model.minG3, valueG3, model.maxG3)

    def G4_rule(model):
        valueG4 = sum(model.x[i] * 100 for i in model.G4)
        return environ.inequality(model.minG4, valueG4, model.maxG4)

    def G5_rule(model):
        valueG5 = sum(model.x[i] * 100 for i in model.G5)
        return environ.inequality(model.minG5, valueG5, model.maxG5)

    def salt_rule(model):
        return environ.inequality(0.05, model.x['Salt'], 0.15)

    def sugar_rule(model):
        return environ.inequality(0.00, model.x['Sugar'], 0.05)

    ''' 
        Palatability constraint.
        Linear model: LINEAR REGRESSION
    '''
    # def linear_palatability_rule(model):
    #     global counter
    #     palatability_limit = counter
    #     palatability_score = sum(model.x[i]*weights[i][0] for i in model.F) + intercept[0]
    #     return palatability_score >= palatability_limit

    '''
        Palatability constraint.
        Non-linear model: NEURAL NETWORK with 2 hidden layers composed by 5 and 2 nodes from left to right.
        Activation functions: tanh and sigmoid for the last layer
        Loss function: MSE
    '''
    def nonlinear_palatability_rule(model):
        global counter
        palatability_limit = counter
        palatability_score = 1/(1+environ.exp(-
                                              (pyomo.core.expr.tanh(
            pyomo.core.expr.tanh(sum(model.x[i] * weights_ANN.iloc[:, 0][i] for i in model.F) + ANN.get_weights()[1][0])*ANN.get_weights()[2][:,0][0]
        + pyomo.core.expr.tanh(sum(model.x[i] * weights_ANN.iloc[:, 1][i] for i in model.F) + ANN.get_weights()[1][1])*ANN.get_weights()[2][:,0][1]
        + pyomo.core.expr.tanh(sum(model.x[i] * weights_ANN.iloc[:, 2][i] for i in model.F) + ANN.get_weights()[1][2])*ANN.get_weights()[2][:,0][2]
        + pyomo.core.expr.tanh(sum(model.x[i] * weights_ANN.iloc[:, 3][i] for i in model.F) + ANN.get_weights()[1][3])*ANN.get_weights()[2][:,0][3]
        + pyomo.core.expr.tanh(sum(model.x[i] * weights_ANN.iloc[:, 4][i] for i in model.F) + ANN.get_weights()[1][4])*ANN.get_weights()[2][:,0][4]
        + ANN.get_weights()[3][0])*ANN.get_weights()[4][0][0]
                                     + (pyomo.core.expr.tanh(
            pyomo.core.expr.tanh(sum(model.x[i] * weights_ANN.iloc[:, 0][i] for i in model.F) + ANN.get_weights()[1][0])*ANN.get_weights()[2][:,1][0]
        + pyomo.core.expr.tanh(sum(model.x[i] * weights_ANN.iloc[:, 1][i] for i in model.F) + ANN.get_weights()[1][1])*ANN.get_weights()[2][:,1][1]
        + pyomo.core.expr.tanh(sum(model.x[i] * weights_ANN.iloc[:, 2][i] for i in model.F) + ANN.get_weights()[1][2])*ANN.get_weights()[2][:,1][2]
        + pyomo.core.expr.tanh(sum(model.x[i] * weights_ANN.iloc[:, 3][i] for i in model.F) + ANN.get_weights()[1][3])*ANN.get_weights()[2][:,1][3]
        + pyomo.core.expr.tanh(sum(model.x[i] * weights_ANN.iloc[:, 4][i] for i in model.F) + ANN.get_weights()[1][4])*ANN.get_weights()[2][:,1][4]
            + ANN.get_weights()[3][1])*ANN.get_weights()[4][1][0])
                        + ANN.get_weights()[5][0])
                                              ))
        return palatability_score*10 >= palatability_limit

    ###############################################################
    # def slack1_rule(model):
    #     return sum(model.x[i] * 100 for i in model.G1) - model.Sg1 == (model.minG1 + model.maxG1) / 2
    #
    # def slack2_rule(model):
    #     return sum(model.x[i] * 100 for i in model.G2) - model.Sg2 == (model.minG2 + model.maxG2) / 2
    #
    # def slack3_rule(model):
    #     return sum(model.x[i] * 100 for i in model.G3) - model.Sg3 == (model.minG3 + model.maxG3) / 2
    #
    # def slack4_rule(model):
    #     return sum(model.x[i] * 100 for i in model.G4) - model.Sg4 == (model.minG4 + model.maxG4) / 2
    #
    # def slack5_rule(model):
    #     return sum(model.x[i] * 100 for i in model.G5) - model.Sg5 == (model.minG5 + model.maxG5) / 2
    #
    # def palatability_formula_rule(model):
    #     global counter
    #     weighted_palatability = environ.sqrt(model.Sg1 ** 2 + (2.5 * model.Sg2) ** 2
    #                                                + (10 * model.Sg3) ** 2 + (4.2 * model.Sg4) ** 2
    #                                                + (6.25 * model.Sg5) ** 2)
    #     return environ.inequality((counter)*28, 279.96-weighted_palatability, (counter + 0.9)*28)
    #
    # model.palatability_formula = Constraint(rule=palatability_formula_rule)
    # model.slack_const1 = Constraint(rule=slack1_rule)
    # model.slack_const2 = Constraint(rule=slack2_rule)
    # model.slack_const3 = Constraint(rule=slack3_rule)
    # model.slack_const4 = Constraint(rule=slack4_rule)
    # model.slack_const5 = Constraint(rule=slack5_rule)
    ########################################################################
    # def nonlinear_palatability_rule(model):
    #     limit = random.randint(1, 9)
    #     palatability_score = 1 / (1 + environ.exp(-
    #                                               (pyomo.core.expr.tanh(
    #                                                   pyomo.core.expr.tanh((sum(model.x[i] for i in model.G1) *
    #                                                                         weights_ANN.iloc[:, 0][0] + sum(
    #                                                               model.x[i] for i in model.G2) *
    #                                                                         weights_ANN.iloc[:, 0][1] + sum(
    #                                                               model.x[i] for i in model.G3) *
    #                                                                         weights_ANN.iloc[:, 0][2] + sum(
    #                                                               model.x[i] for i in model.G4) *
    #                                                                         weights_ANN.iloc[:, 0][3] + sum(
    #                                                               model.x[i] for i in model.G5) *
    #                                                                         weights_ANN.iloc[:, 0][4]) +
    #                                                                        ANN.get_weights()[1][0]) *
    #                                                   ANN.get_weights()[2][:, 0][0]
    #                                                   + pyomo.core.expr.tanh((sum(model.x[i] for i in model.G1) *
    #                                                                           weights_ANN.iloc[:, 1][0] + sum(
    #                                                               model.x[i] for i in model.G2) *
    #                                                                           weights_ANN.iloc[:, 1][1] + sum(
    #                                                               model.x[i] for i in model.G3) *
    #                                                                           weights_ANN.iloc[:, 1][2] + sum(
    #                                                               model.x[i] for i in model.G4) *
    #                                                                           weights_ANN.iloc[:, 1][3] + sum(
    #                                                               model.x[i] for i in model.G5) *
    #                                                                           weights_ANN.iloc[:, 1][4]) +
    #                                                                          ANN.get_weights()[1][1]) *
    #                                                   ANN.get_weights()[2][:, 0][1]
    #                                                   + pyomo.core.expr.tanh((sum(model.x[i] for i in model.G1) *
    #                                                                           weights_ANN.iloc[:, 2][0] + sum(
    #                                                               model.x[i] for i in model.G2) *
    #                                                                           weights_ANN.iloc[:, 2][1] + sum(
    #                                                               model.x[i] for i in model.G3) *
    #                                                                           weights_ANN.iloc[:, 2][2] + sum(
    #                                                               model.x[i] for i in model.G4) *
    #                                                                           weights_ANN.iloc[:, 2][3] + sum(
    #                                                               model.x[i] for i in model.G5) *
    #                                                                           weights_ANN.iloc[:, 2][4]) +
    #                                                                          ANN.get_weights()[1][2]) *
    #                                                   ANN.get_weights()[2][:, 0][2]
    #                                                   + pyomo.core.expr.tanh((sum(model.x[i] for i in model.G1) *
    #                                                                           weights_ANN.iloc[:, 3][0] + sum(
    #                                                               model.x[i] for i in model.G2) *
    #                                                                           weights_ANN.iloc[:, 3][1] + sum(
    #                                                               model.x[i] for i in model.G3) *
    #                                                                           weights_ANN.iloc[:, 3][2] + sum(
    #                                                               model.x[i] for i in model.G4) *
    #                                                                           weights_ANN.iloc[:, 3][3] + sum(
    #                                                               model.x[i] for i in model.G5) *
    #                                                                           weights_ANN.iloc[:, 3][4]) +
    #                                                                          ANN.get_weights()[1][3]) *
    #                                                   ANN.get_weights()[2][:, 0][3]
    #                                                   + pyomo.core.expr.tanh((sum(model.x[i] for i in model.G1) *
    #                                                                           weights_ANN.iloc[:, 4][0] + sum(
    #                                                               model.x[i] for i in model.G2) *
    #                                                                           weights_ANN.iloc[:, 4][1] + sum(
    #                                                               model.x[i] for i in model.G3) *
    #                                                                           weights_ANN.iloc[:, 4][2] + sum(
    #                                                               model.x[i] for i in model.G4) *
    #                                                                           weights_ANN.iloc[:, 4][3] + sum(
    #                                                               model.x[i] for i in model.G5) *
    #                                                                           weights_ANN.iloc[:, 4][4]) +
    #                                                                          ANN.get_weights()[1][4]) *
    #                                                   ANN.get_weights()[2][:, 0][4]
    #                                                   + ANN.get_weights()[3][0]) * ANN.get_weights()[4][0][0]
    #                                                + (pyomo.core.expr.tanh(
    #                                                           pyomo.core.expr.tanh((sum(model.x[i] for i in model.G1) *
    #                                                                                 weights_ANN.iloc[:, 0][0] + sum(
    #                                                                       model.x[i] for i in model.G2) *
    #                                                                                 weights_ANN.iloc[:, 0][1] + sum(
    #                                                                       model.x[i] for i in model.G3) *
    #                                                                                 weights_ANN.iloc[:, 0][2] + sum(
    #                                                                       model.x[i] for i in model.G4) *
    #                                                                                 weights_ANN.iloc[:, 0][3] + sum(
    #                                                                       model.x[i] for i in model.G5) *
    #                                                                                 weights_ANN.iloc[:, 0][4]) +
    #                                                                                ANN.get_weights()[1][0]) *
    #                                                           ANN.get_weights()[2][:, 1][0]
    #                                                           + pyomo.core.expr.tanh((sum(
    #                                                               model.x[i] for i in model.G1) *
    #                                                                                   weights_ANN.iloc[:, 1][0] + sum(
    #                                                                       model.x[i] for i in model.G2) *
    #                                                                                   weights_ANN.iloc[:, 1][1] + sum(
    #                                                                       model.x[i] for i in model.G3) *
    #                                                                                   weights_ANN.iloc[:, 1][2] + sum(
    #                                                                       model.x[i] for i in model.G4) *
    #                                                                                   weights_ANN.iloc[:, 1][3] + sum(
    #                                                                       model.x[i] for i in model.G5) *
    #                                                                                   weights_ANN.iloc[:, 1][4]) +
    #                                                                                  ANN.get_weights()[1][1]) *
    #                                                           ANN.get_weights()[2][:, 1][1]
    #                                                           + pyomo.core.expr.tanh((sum(
    #                                                               model.x[i] for i in model.G1) *
    #                                                                                   weights_ANN.iloc[:, 2][0] + sum(
    #                                                                       model.x[i] for i in model.G2) *
    #                                                                                   weights_ANN.iloc[:, 2][1] + sum(
    #                                                                       model.x[i] for i in model.G3) *
    #                                                                                   weights_ANN.iloc[:, 2][2] + sum(
    #                                                                       model.x[i] for i in model.G4) *
    #                                                                                   weights_ANN.iloc[:, 2][3] + sum(
    #                                                                       model.x[i] for i in model.G5) *
    #                                                                                   weights_ANN.iloc[:, 2][4]) +
    #                                                                                  ANN.get_weights()[1][2]) *
    #                                                           ANN.get_weights()[2][:, 1][2]
    #                                                           + pyomo.core.expr.tanh((sum(
    #                                                               model.x[i] for i in model.G1) *
    #                                                                                   weights_ANN.iloc[:, 3][0] + sum(
    #                                                                       model.x[i] for i in model.G2) *
    #                                                                                   weights_ANN.iloc[:, 3][1] + sum(
    #                                                                       model.x[i] for i in model.G3) *
    #                                                                                   weights_ANN.iloc[:, 3][2] + sum(
    #                                                                       model.x[i] for i in model.G4) *
    #                                                                                   weights_ANN.iloc[:, 3][3] + sum(
    #                                                                       model.x[i] for i in model.G5) *
    #                                                                                   weights_ANN.iloc[:, 3][4]) +
    #                                                                                  ANN.get_weights()[1][3]) *
    #                                                           ANN.get_weights()[2][:, 1][3]
    #                                                           + pyomo.core.expr.tanh((sum(
    #                                                               model.x[i] for i in model.G1) *
    #                                                                                   weights_ANN.iloc[:, 4][0] + sum(
    #                                                                       model.x[i] for i in model.G2) *
    #                                                                                   weights_ANN.iloc[:, 4][1] + sum(
    #                                                                       model.x[i] for i in model.G3) *
    #                                                                                   weights_ANN.iloc[:, 4][2] + sum(
    #                                                                       model.x[i] for i in model.G4) *
    #                                                                                   weights_ANN.iloc[:, 4][3] + sum(
    #                                                                       model.x[i] for i in model.G5) *
    #                                                                                   weights_ANN.iloc[:, 4][4]) +
    #                                                                                  ANN.get_weights()[1][4]) *
    #                                                           ANN.get_weights()[2][:, 1][4]
    #                                                           + ANN.get_weights()[3][1]) * ANN.get_weights()[4][1][0])
    #                                                + ANN.get_weights()[5][0])
    #                                               ))
    #     return palatability_score * 10 >= 5

    ''' 
        Call to constraints and O.F.
    '''
    model.cost = Objective(rule=cost_rule2)
    model.nutrient_limit = Constraint(model.N, rule=nutrient_rule)
    model.slackNut_limit = Constraint(model.N, rule=slackNut_rule)

    model.G1_limit = Constraint(rule=G1_rule)
    model.G2_limit = Constraint(rule=G2_rule)
    model.G3_limit = Constraint(rule=G3_rule)
    model.G4_limit = Constraint(rule=G4_rule)
    model.G5_limit = Constraint(rule=G5_rule)
    model.salt_limit = Constraint(rule=salt_rule)
    model.sugar_limit = Constraint(rule=sugar_rule)

    '''
        'nonlinear_palatability_rule' uses the ANN.
        'linear_palatabilit_rule' uses a linear regression
    '''
    model.palatability_limit = Constraint(rule=nonlinear_palatability_rule)
    # model.palatability_limit = Constraint(rule=linear_palatability_rule)

    '''
        Instantiate the model using the .dat file
        The GNU Linear Programming Kit is a software package intended for solving large-scale linear programming,
        mixed integer programming, and other related problems.
    '''
    instance = model.create_instance('WFPdietOriginal.dat')
    solver_manager = environ.SolverManagerFactory('neos')
    results = solver_manager.solve(instance, opt='knitro')
    # opt = SolverFactory('glpk')
    # results = opt.solve(instance)
    # instance.display()
    if (results.Solver._list[0]['Termination condition'].key != 'optimal'):
        print(results.Solver._list[0]['Termination condition'].key)
        return 0

    # Calculation of the Palatability score with the deterministic formula
    Sg1 = abs(sum(instance.x[i].value for i in instance.G1) * 100 - (instance.minG1 + instance.maxG1) / 2)
    Sg2 = abs(sum(instance.x[i].value for i in instance.G2) * 100 - (instance.minG2 + instance.maxG2) / 2)
    Sg3 = abs(sum(instance.x[i].value for i in instance.G3) * 100 - (instance.minG3 + instance.maxG3) / 2)
    Sg4 = abs(sum(instance.x[i].value for i in instance.G4) * 100 - (instance.minG4 + instance.maxG4) / 2)
    Sg5 = abs(sum(instance.x[i].value for i in instance.G5) * 100 - (instance.minG5 + instance.maxG5) / 2)

    palatability = np.round(math.sqrt(Sg1 ** 2 + Sg2 ** 2
                                      + Sg3 ** 2 + Sg4 ** 2
                                      + Sg5 ** 2), 2)
    weighted_palatability = np.round(math.sqrt(Sg1 ** 2 + (2.5 * Sg2) ** 2
                                               + (10 * Sg3) ** 2 + (4.2 * Sg4) ** 2
                                               + (6.25 * Sg5) ** 2), 2)


    print('real palatability:', 279.96-weighted_palatability)

    '''
        Saving of the solution
    '''
    global counter
    print('writing on the dataset')
    f = open("Datasets/datasetOL50it.csv", "a")
    row = ""
    for i in instance.F:
        if i != 'Wheat-soya blend (WSB)':
            row += str(np.round(instance.x[i].value, 2)) + ','
        else:
            row += str(np.round(instance.x[i].value, 2)) + ',' + str(float(weighted_palatability)) + '\n'

    f.write(row)
    f.close()
    print(results.Solver._list[0]['Termination condition'])
    return 0


file_ANN_simple = 'modelsOL/simple2_OL.h5'
# file_lr = 'modelsOL/lr_OL.sav'
counter = 10
for i in range(0, 1000):

    # every 9 calls to adversarialGen() the ML model is adjusted to consider also adversarial solutions
    if counter % 10 == 0:
        counter = 0.5
        model = load_model(file_ANN_simple)
        # layer1 = model.layers[0]
        # layer1.trainable = False
        dataset = pd.read_csv('Datasets/datasetOL50it.csv')  # Critical choice
        max_palatability = dataset['lable'].max()
        dataset['lable'] = max_palatability - dataset['lable']

        features_name = ['Beans', 'Bulgur', 'Cheese', 'Fish', 'Meat', 'Corn-soya blend (CSB)', 'Dates',
                         'Dried skim milk (enriched) (DSM)', 'Milk', 'Salt', 'Lentils', 'Maize', 'Maize meal',
                         'Chickpeas','Rice','Sorghum/millet', 'Soya-fortified bulgur wheat',
                         'Soya-fortified maize meal', 'Soya-fortified sorghum grits', 'Soya-fortified wheat flour',
                         'Sugar', 'Oil', 'Wheat','Wheat flour','Wheat-soya blend (WSB)']

        target_name = ['lable']
        sc_t_ANN = MinMaxScaler(feature_range=(0, 1))
        # sc_t_LR = MinMaxScaler(feature_range=(0, 10))
        dataset[target_name] = sc_t_ANN.fit_transform((dataset[target_name]))
        X = dataset[features_name]
        y = dataset[target_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        # model.compile(loss='mse', optimizer='Adam', metrics=['mse', 'mae'])
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        # clf = LinearRegression().fit(X_train, y_train)
        print('Model trained')
        model.save('modelsOL/simple2_OL.h5')
        # pickle.dump(clf, open(file_lr, 'wb'))
    print('palatability_limit:', counter)
    a = adversarialGen()
    counter += 0.5
