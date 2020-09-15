import time
import math
from keras.engine.saving import load_model
from pyomo import environ
from pyomo.environ import *
from pyomo.opt import *
import pickle
import numpy as np
import pandas as pd

# The features used are the amount of food for each type
features_name = ['Beans','Bulgur','Cheese','Fish','Meat','Corn-soya blend (CSB)','Dates'
                 ,'Dried skim milk (enriched) (DSM)','Milk','Salt','Lentils','Maize','Maize meal','Chickpeas','Rice',
                 'Sorghum/millet','Soya-fortified bulgur wheat','Soya-fortified maize meal',
                 'Soya-fortified sorghum grits','Soya-fortified wheat flour','Sugar','Oil','Wheat','Wheat flour',
                 'Wheat-soya blend (WSB)']

# Importing the pre-trained models
file_lr = 'models/linearRegression_W.sav'
linear_reg = pickle.load(open(file_lr, 'rb'))
file_ANN_simple = 'Gmodels/simple2.h5'
ANN = load_model(file_ANN_simple)
weights_ANN = pd.DataFrame(ANN.get_weights()[0], index=features_name)
weights = pd.DataFrame(linear_reg.coef_, columns=features_name)
intercept = linear_reg.intercept_

'''
    diet_model() is a function that returns the optimal solution given a certain scenario (.dat) and
    a palatability_limit value. 
    A for loop has been used to iterate over different palatability_limit values.
'''
def diet_model():
    model = AbstractModel()  # An instance is created later with the '.dat' file

    '''
        Sets initialization
    '''
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
    # Mix of foods for extra constraints
    model.G0 = Set()

    '''
        Parameters initialization
    '''
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

    '''
        Variables initialization
    '''
    # Decision variable
    model.x = Var(model.F, within=NonNegativeReals)
    # Slack variables
    model.Sl = Var(model.N, within=NonNegativeReals)

    '''
        OBJECTIVE FUNCTION: Minimize the value of the Sl[j]s
    '''
    # Minimize the cost of food that is consumed
    # def cost_rule1(model):
    #     return sum(model.c[i] * model.x[i] / 10000 for i in model.F)

    # Minimize the value of the Sl[j]s
    def cost_rule(model):
        return sum(model.Sl[i] for i in model.N)
    def cost_rule2(model):
        return sum(model.c[i] * model.x[i] / 10000 for i in model.F)

    '''
        CONSTRAINTS
    '''
    # Limit nutrient consumption for each nutrient
    def nutrient_rule(model, j):
        value = sum(model.a[i, j] * model.x[i] for i in model.F)
        return value >= model.Nmin[j] * (1 - model.Sl[j])

    # Limit the Slack variable value to be less than 1
    def slackNut_rule(model, i):
        return model.Sl[i] <= 1

    '''
        To limit the amount of Food for each group
    '''
    def G1_rule(model):
        valueG1 = sum(model.x[i] * 100 for i in model.G1)
        return environ.inequality(model.minG1, valueG1, model.maxG1)
        # return environ.inequality(375, valueG1, model.maxG1)

    def G2_rule(model):
        valueG2 = sum(model.x[i] * 100 for i in model.G2)
        return environ.inequality(model.minG2, valueG2, model.maxG2)
        # return environ.inequality(80, valueG2, model.maxG2)

    def G3_rule(model):
        valueG3 = sum(model.x[i] * 100 for i in model.G3)
        return environ.inequality(model.minG3, valueG3, model.maxG3)
        # return environ.inequality(27.5, valueG3, model.maxG3)
    def G4_rule(model):
        valueG4 = sum(model.x[i] * 100 for i in model.G4)
        return environ.inequality(model.minG4, valueG4, model.maxG4)
        # return environ.inequality(30, valueG4, model.maxG4)

    def G5_rule(model):
        valueG5 = sum(model.x[i] * 100 for i in model.G5)
        return environ.inequality(model.minG5, valueG5, model.maxG5)
        # return environ.inequality(20, valueG5, model.maxG5)

    def salt_rule(model):
        return environ.inequality(0.05, model.x['Salt'], 0.15)

    def sugar_rule(model):
        return environ.inequality(0.00, model.x['Sugar'], 0.05)

    ''' 
        Palatability constraint.
        Linear model: LINEAR REGRESSION
    '''
    def linear_palatability_rule(model):
        global palatability_limit
        palatability_score = sum(model.x[i]*weights[i][0] for i in model.F) + intercept[0]
        return palatability_score >= palatability_limit


    '''
        Palatability constraint.
        Non-linear model: NEURAL NETWORK with 2 hidden layers composed by 5 and 2 nodes from left to right.
        Activation functions: tanh and sigmoid for the last layer
    '''
    def nonlinear_palatability_rule(model):
        global palatability_limit
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

    # Simple ANN
    # def nonlinear_palatability_rule(model):
    #     palatability_score = 1/(1+environ.exp(-
    #                                           (pyomo.core.expr.tanh(
    #         pyomo.core.expr.tanh((sum(model.x[i] for i in model.G1) * weights_ANN.iloc[:, 0][0] + sum(model.x[i] for i in model.G2) * weights_ANN.iloc[:, 0][1] + sum(model.x[i] for i in model.G3) * weights_ANN.iloc[:, 0][2] + sum(model.x[i] for i in model.G4) * weights_ANN.iloc[:, 0][3] + sum(model.x[i] for i in model.G5) * weights_ANN.iloc[:, 0][4]) + ANN.get_weights()[1][0])*ANN.get_weights()[2][:,0][0]
    #     + pyomo.core.expr.tanh((sum(model.x[i] for i in model.G1) * weights_ANN.iloc[:, 1][0] + sum(model.x[i] for i in model.G2) * weights_ANN.iloc[:, 1][1] + sum(model.x[i] for i in model.G3) * weights_ANN.iloc[:, 1][2] + sum(model.x[i] for i in model.G4) * weights_ANN.iloc[:, 1][3] + sum(model.x[i] for i in model.G5) * weights_ANN.iloc[:, 1][4]) + ANN.get_weights()[1][1])*ANN.get_weights()[2][:,0][1]
    #     + pyomo.core.expr.tanh((sum(model.x[i] for i in model.G1) * weights_ANN.iloc[:, 2][0] + sum(model.x[i] for i in model.G2) * weights_ANN.iloc[:, 2][1] + sum(model.x[i] for i in model.G3) * weights_ANN.iloc[:, 2][2] + sum(model.x[i] for i in model.G4) * weights_ANN.iloc[:, 2][3] + sum(model.x[i] for i in model.G5) * weights_ANN.iloc[:, 2][4]) + ANN.get_weights()[1][2])*ANN.get_weights()[2][:,0][2]
    #     + pyomo.core.expr.tanh((sum(model.x[i] for i in model.G1) * weights_ANN.iloc[:, 3][0] + sum(model.x[i] for i in model.G2) * weights_ANN.iloc[:, 3][1] + sum(model.x[i] for i in model.G3) * weights_ANN.iloc[:, 3][2] + sum(model.x[i] for i in model.G4) * weights_ANN.iloc[:, 3][3] + sum(model.x[i] for i in model.G5) * weights_ANN.iloc[:, 3][4]) + ANN.get_weights()[1][3])*ANN.get_weights()[2][:,0][3]
    #     + pyomo.core.expr.tanh((sum(model.x[i] for i in model.G1) * weights_ANN.iloc[:, 4][0] + sum(model.x[i] for i in model.G2) * weights_ANN.iloc[:, 4][1] + sum(model.x[i] for i in model.G3) * weights_ANN.iloc[:, 4][2] + sum(model.x[i] for i in model.G4) * weights_ANN.iloc[:, 4][3] + sum(model.x[i] for i in model.G5) * weights_ANN.iloc[:, 4][4]) + ANN.get_weights()[1][4])*ANN.get_weights()[2][:,0][4]
    #     + ANN.get_weights()[3][0])*ANN.get_weights()[4][0][0]
    #                                  + (pyomo.core.expr.tanh(
    #         pyomo.core.expr.tanh((sum(model.x[i] for i in model.G1) * weights_ANN.iloc[:, 0][0] + sum(model.x[i] for i in model.G2) * weights_ANN.iloc[:, 0][1] + sum(model.x[i] for i in model.G3) * weights_ANN.iloc[:, 0][2] + sum(model.x[i] for i in model.G4) * weights_ANN.iloc[:, 0][3] + sum(model.x[i] for i in model.G5) * weights_ANN.iloc[:, 0][4]) + ANN.get_weights()[1][0]) * ANN.get_weights()[2][:,1][0]
    #     + pyomo.core.expr.tanh((sum(model.x[i] for i in model.G1) * weights_ANN.iloc[:, 1][0] + sum(model.x[i] for i in model.G2) * weights_ANN.iloc[:, 1][1] + sum(model.x[i] for i in model.G3) * weights_ANN.iloc[:, 1][2] + sum(model.x[i] for i in model.G4) * weights_ANN.iloc[:, 1][3] + sum(model.x[i] for i in model.G5) * weights_ANN.iloc[:, 1][4]) + ANN.get_weights()[1][1]) * ANN.get_weights()[2][:,1][1]
    #     + pyomo.core.expr.tanh((sum(model.x[i] for i in model.G1) * weights_ANN.iloc[:, 2][0] + sum(model.x[i] for i in model.G2) * weights_ANN.iloc[:, 2][1] + sum(model.x[i] for i in model.G3) * weights_ANN.iloc[:, 2][2] + sum(model.x[i] for i in model.G4) * weights_ANN.iloc[:, 2][3] + sum(model.x[i] for i in model.G5)* weights_ANN.iloc[:, 2][4]) + ANN.get_weights()[1][2]) * ANN.get_weights()[2][:,1][2]
    #     + pyomo.core.expr.tanh((sum(model.x[i] for i in model.G1) * weights_ANN.iloc[:, 3][0] + sum(model.x[i] for i in model.G2) * weights_ANN.iloc[:, 3][1] + sum(model.x[i] for i in model.G3) * weights_ANN.iloc[:, 3][2] + sum(model.x[i] for i in model.G4) * weights_ANN.iloc[:, 3][3] + sum(model.x[i] for i in model.G5) * weights_ANN.iloc[:, 3][4]) + ANN.get_weights()[1][3]) * ANN.get_weights()[2][:,1][3]
    #     + pyomo.core.expr.tanh((sum(model.x[i] for i in model.G1) * weights_ANN.iloc[:, 4][0] + sum(model.x[i] for i in model.G2) * weights_ANN.iloc[:, 4][1] + sum(model.x[i] for i in model.G3) * weights_ANN.iloc[:, 4][2] + sum(model.x[i] for i in model.G4) * weights_ANN.iloc[:, 4][3] + sum(model.x[i] for i in model.G5) * weights_ANN.iloc[:, 4][4]) + ANN.get_weights()[1][4]) * ANN.get_weights()[2][:,1][4]
    #         + ANN.get_weights()[3][1])*ANN.get_weights()[4][1][0])
    #                     + ANN.get_weights()[5][0])
    #                                           ))
    #     return palatability_score*10 >= 5


    ''' 
        Call to constraints and O.F.
    '''
    model.cost = Objective(rule=cost_rule, sense=minimize)

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
    instance = model.create_instance('WFPdietOriginal.dat', report_timing=False)

    start = time.time()
    solver_manager = environ.SolverManagerFactory('neos')
    # results = solver_manager.solve(instance, opt='knitro', options='algorithm=4 debug=1 outlev=3')  # for the non linear palatability constraint
    results = solver_manager.solve(instance, opt='knitro', tee=True)
    end = time.time()
    print("Time:", end - start)
    # opt = SolverFactory('ipopt')  # for the  linear palatability constraint
    # opt.options["interior"] = ""
    # results = opt.solve(instance)
    # instance.display()
    # instance.pprint()
    # Calculation of the Palatability score with the deterministic formula
    Sg1 = (sum(instance.x[i].value for i in instance.G1)*100 - (instance.minG1 + instance.maxG1) / 2)
    Sg2 = (sum(instance.x[i].value for i in instance.G2)*100 - (instance.minG2 + instance.maxG2) / 2)
    Sg3 = (sum(instance.x[i].value for i in instance.G3)*100 - (instance.minG3 + instance.maxG3) / 2)
    Sg4 = (sum(instance.x[i].value for i in instance.G4)*100 - (instance.minG4 + instance.maxG4) / 2)
    Sg5 = (sum(instance.x[i].value for i in instance.G5)*100 - (instance.minG5 + instance.maxG5) / 2)

    print(Sg1, Sg2, Sg3, Sg4, Sg5)
    max_palatability = 279.96
    max_palatability_nonW = 139.69
    weighted_palatability = np.round(math.sqrt(Sg1 ** 2 + (2.5 * Sg2) ** 2
                                               + (10 * Sg3) ** 2 + (4.2 * Sg4) ** 2
                                               + (6.25 * Sg5) ** 2), 2)
    nonweighted_palatability = np.round(math.sqrt(Sg1 ** 2 + (Sg2) ** 2
                                               + (Sg3) ** 2 + (Sg4) ** 2
                                               + (Sg5) ** 2), 2)
    palatability = max(max_palatability - weighted_palatability, 0)

    # palatability = 1/(1+environ.exp(-(pyomo.core.expr.tanh(
    #         pyomo.core.expr.tanh(sum(instance.x[i].value * weights_ANN.iloc[:, 0][i] for i in instance.F) + ANN.get_weights()[1][0])*ANN.get_weights()[2][:,0][0]
    #     + pyomo.core.expr.tanh(sum(instance.x[i].value * weights_ANN.iloc[:, 1][i] for i in instance.F) + ANN.get_weights()[1][1])*ANN.get_weights()[2][:,0][1]
    #     + pyomo.core.expr.tanh(sum(instance.x[i].value * weights_ANN.iloc[:, 2][i] for i in instance.F) + ANN.get_weights()[1][2])*ANN.get_weights()[2][:,0][2]
    #     + pyomo.core.expr.tanh(sum(instance.x[i].value * weights_ANN.iloc[:, 3][i] for i in instance.F) + ANN.get_weights()[1][3])*ANN.get_weights()[2][:,0][3]
    #     + pyomo.core.expr.tanh(sum(instance.x[i].value * weights_ANN.iloc[:, 4][i] for i in instance.F) + ANN.get_weights()[1][4])*ANN.get_weights()[2][:,0][4]
    #         + ANN.get_weights()[3][0])*ANN.get_weights()[4][0][0]
    #                                  + (pyomo.core.expr.tanh(
    #         pyomo.core.expr.tanh(sum(instance.x[i].value * weights_ANN.iloc[:, 0][i] for i in instance.F) + ANN.get_weights()[1][0])*ANN.get_weights()[2][:,1][0]
    #     + pyomo.core.expr.tanh(sum(instance.x[i].value * weights_ANN.iloc[:, 1][i] for i in instance.F) + ANN.get_weights()[1][1])*ANN.get_weights()[2][:,1][1]
    #     + pyomo.core.expr.tanh(sum(instance.x[i].value * weights_ANN.iloc[:, 2][i] for i in instance.F) + ANN.get_weights()[1][2])*ANN.get_weights()[2][:,1][2]
    #     + pyomo.core.expr.tanh(sum(instance.x[i].value * weights_ANN.iloc[:, 3][i] for i in instance.F) + ANN.get_weights()[1][3])*ANN.get_weights()[2][:,1][3]
    #     + pyomo.core.expr.tanh(sum(instance.x[i].value * weights_ANN.iloc[:, 4][i] for i in instance.F) + ANN.get_weights()[1][4])*ANN.get_weights()[2][:,1][4]
    #         + ANN.get_weights()[3][1])*ANN.get_weights()[4][1][0])
    #                                  + ANN.get_weights()[5][0])
    #                                           ))

    # palatability_value = 1 / (1 + environ.exp(-
    #                                           (pyomo.core.expr.tanh(
    #                                               pyomo.core.expr.tanh((sum(instance.x[i].value for i in instance.G1) *
    #                                                                     weights_ANN.iloc[:, 0][0] + sum(
    #                                                           instance.x[i].value for i in instance.G2) * weights_ANN.iloc[:, 0][
    #                                                                         1] + sum(instance.x[i].value for i in instance.G3) *
    #                                                                     weights_ANN.iloc[:, 0][2] + sum(
    #                                                           instance.x[i].value for i in instance.G4) * weights_ANN.iloc[:, 0][
    #                                                                         3] + sum(instance.x[i].value for i in instance.G5) *
    #                                                                     weights_ANN.iloc[:, 0][4]) + ANN.get_weights()[1][
    #                                                                        0]) * ANN.get_weights()[2][:, 0][0]
    #                                               + pyomo.core.expr.tanh((sum(instance.x[i].value  for i in instance.G1) *
    #                                                                       weights_ANN.iloc[:, 1][0] + sum(
    #                                                           instance.x[i].value for i in instance.G2) * weights_ANN.iloc[:, 1][
    #                                                                           1] + sum(instance.x[i].value for i in instance.G3) *
    #                                                                       weights_ANN.iloc[:, 1][2] + sum(
    #                                                           instance.x[i].value for i in instance.G4) * weights_ANN.iloc[:, 1][
    #                                                                           3] + sum(instance.x[i].value for i in instance.G5) *
    #                                                                       weights_ANN.iloc[:, 1][4]) + ANN.get_weights()[1][
    #                                                                          1]) * ANN.get_weights()[2][:, 0][1]
    #                                               + pyomo.core.expr.tanh((sum(instance.x[i].value for i in instance.G1) *
    #                                                                       weights_ANN.iloc[:, 2][0] + sum(
    #                                                           instance.x[i].value  for i in instance.G2) * weights_ANN.iloc[:, 2][
    #                                                                           1] + sum(instance.x[i].value for i in instance.G3) *
    #                                                                       weights_ANN.iloc[:, 2][2] + sum(
    #                                                           instance.x[i].value  for i in instance.G4) * weights_ANN.iloc[:, 2][
    #                                                                           3] + sum(instance.x[i].value for i in instance.G5) *
    #                                                                       weights_ANN.iloc[:, 2][4]) + ANN.get_weights()[1][
    #                                                                          2]) * ANN.get_weights()[2][:, 0][2]
    #                                               + pyomo.core.expr.tanh((sum(instance.x[i].value for i in instance.G1) *
    #                                                                       weights_ANN.iloc[:, 3][0] + sum(
    #                                                           instance.x[i].value for i in instance.G2) * weights_ANN.iloc[:, 3][
    #                                                                           1] + sum(instance.x[i].value for i in instance.G3) *
    #                                                                       weights_ANN.iloc[:, 3][2] + sum(
    #                                                           instance.x[i].value for i in instance.G4) * weights_ANN.iloc[:, 3][
    #                                                                           3] + sum(instance.x[i].value for i in instance.G5) *
    #                                                                       weights_ANN.iloc[:, 3][4]) + ANN.get_weights()[1][
    #                                                                          3]) * ANN.get_weights()[2][:, 0][3]
    #                                               + pyomo.core.expr.tanh((sum(instance.x[i].value for i in instance.G1) *
    #                                                                       weights_ANN.iloc[:, 4][0] + sum(
    #                                                           instance.x[i].value for i in instance.G2) * weights_ANN.iloc[:, 4][
    #                                                                           1] + sum(instance.x[i].value for i in instance.G3) *
    #                                                                       weights_ANN.iloc[:, 4][2] + sum(
    #                                                           instance.x[i].value for i in instance.G4) * weights_ANN.iloc[:, 4][
    #                                                                           3] + sum(instance.x[i].value for i in instance.G5) *
    #                                                                       weights_ANN.iloc[:, 4][4]) + ANN.get_weights()[1][
    #                                                                          4]) * ANN.get_weights()[2][:, 0][4]
    #                                               + ANN.get_weights()[3][0]) * ANN.get_weights()[4][0][0]
    #                                            + (pyomo.core.expr.tanh(
    #                                                       pyomo.core.expr.tanh((sum(instance.x[i].value for i in instance.G1) *
    #                                                                             weights_ANN.iloc[:, 0][0] + sum(
    #                                                                   instance.x[i].value for i in instance.G2) *
    #                                                                             weights_ANN.iloc[:, 0][1] + sum(
    #                                                                   instance.x[i].value for i in instance.G3) *
    #                                                                             weights_ANN.iloc[:, 0][2] + sum(
    #                                                                   instance.x[i].value for i in instance.G4) *
    #                                                                             weights_ANN.iloc[:, 0][3] + sum(
    #                                                                   instance.x[i].value for i in instance.G5) *
    #                                                                             weights_ANN.iloc[:, 0][4]) +
    #                                                                            ANN.get_weights()[1][0]) *
    #                                                       ANN.get_weights()[2][:, 1][0]
    #                                                       + pyomo.core.expr.tanh((sum(instance.x[i].value for i in instance.G1) *
    #                                                                               weights_ANN.iloc[:, 1][0] + sum(
    #                                                                   instance.x[i].value for i in instance.G2) *
    #                                                                               weights_ANN.iloc[:, 1][1] + sum(
    #                                                                   instance.x[i].value for i in instance.G3) *
    #                                                                               weights_ANN.iloc[:, 1][2] + sum(
    #                                                                   instance.x[i].value for i in instance.G4) *
    #                                                                               weights_ANN.iloc[:, 1][3] + sum(
    #                                                                   instance.x[i].value for i in instance.G5) *
    #                                                                               weights_ANN.iloc[:, 1][4]) +
    #                                                                              ANN.get_weights()[1][1]) *
    #                                                       ANN.get_weights()[2][:, 1][1]
    #                                                       + pyomo.core.expr.tanh((sum(instance.x[i].value for i in instance.G1) *
    #                                                                               weights_ANN.iloc[:, 2][0] + sum(
    #                                                                   instance.x[i].value for i in instance.G2) *
    #                                                                               weights_ANN.iloc[:, 2][1] + sum(
    #                                                                   instance.x[i].value for i in instance.G3) *
    #                                                                               weights_ANN.iloc[:, 2][2] + sum(
    #                                                                   instance.x[i].value for i in instance.G4) *
    #                                                                               weights_ANN.iloc[:, 2][3] + sum(
    #                                                                   instance.x[i].value for i in instance.G5) *
    #                                                                               weights_ANN.iloc[:, 2][4]) +
    #                                                                              ANN.get_weights()[1][2]) *
    #                                                       ANN.get_weights()[2][:, 1][2]
    #                                                       + pyomo.core.expr.tanh((sum(instance.x[i].value for i in instance.G1) *
    #                                                                               weights_ANN.iloc[:, 3][0] + sum(
    #                                                                   instance.x[i].value for i in instance.G2) *
    #                                                                               weights_ANN.iloc[:, 3][1] + sum(
    #                                                                   instance.x[i].value for i in instance.G3) *
    #                                                                               weights_ANN.iloc[:, 3][2] + sum(
    #                                                                   instance.x[i].value for i in instance.G4) *
    #                                                                               weights_ANN.iloc[:, 3][3] + sum(
    #                                                                   instance.x[i].value for i in instance.G5) *
    #                                                                               weights_ANN.iloc[:, 3][4]) +
    #                                                                              ANN.get_weights()[1][3]) *
    #                                                       ANN.get_weights()[2][:, 1][3]
    #                                                       + pyomo.core.expr.tanh((sum(instance.x[i].value for i in instance.G1) *
    #                                                                               weights_ANN.iloc[:, 4][0] + sum(
    #                                                                   instance.x[i].value for i in instance.G2) *
    #                                                                               weights_ANN.iloc[:, 4][1] + sum(
    #                                                                   instance.x[i].value for i in instance.G3) *
    #                                                                               weights_ANN.iloc[:, 4][2] + sum(
    #                                                                   instance.x[i].value for i in instance.G4) *
    #                                                                               weights_ANN.iloc[:, 4][3] + sum(
    #                                                                   instance.x[i].value for i in instance.G5) *
    #                                                                               weights_ANN.iloc[:, 4][4]) +
    #                                                                              ANN.get_weights()[1][4]) *
    #                                                       ANN.get_weights()[2][:, 1][4]
    #                                                       + ANN.get_weights()[3][1]) * ANN.get_weights()[4][1][0])
    #                                            + ANN.get_weights()[5][0])
    #                                           ))
    # score_comparison = ANN.predict(np.array([[sum(instance.x[i].value for i in instance.G1),
    #                                           sum(instance.x[i].value for i in instance.G2),
    #                                           sum(instance.x[i].value for i in instance.G3),
    #                                           sum(instance.x[i].value for i in instance.G4),
    #                                           sum(instance.x[i].value for i in instance.G5)]]))

    '''
        Saving of the optimal solution
    '''
    solution = []
    print('SOLUTION: ')
    for i in instance.F:
    #     f = open("solutions.csv", "a")
    #     row = ""
        solution.append(instance.x[i].value)
    #     row += str(instance.x[i].value) + '\n'
    #     # row += i + '\n'
    #     f.write(row)
    #     f.close()
    #     # print(i, np.round(instance.x[i].value, 2))
    #     # if(instance.x[i].value > 0.00001):
    #         # print(i, instance.x[i].value)
    # # To print variables values and obj function
    # for v in instance.component_objects(Var, active=True):
    #     varobject = getattr(instance, str(v))
    #     cost = 0
    #     if (varobject.name == 'Sl'):
    #         for index in varobject:
    #             cost += varobject[index].value
    #             # print("   ", index, varobject[index].value)
    # print(cost)

    # f = open("solutions.csv", "a")
    # f.write('\n')
    # f.close()
    print(solution)
    print('PALATABILITY', palatability)
    print('THE RESULT IS:', results.Solver._list[0]['Termination condition'])
i = 0.5
while(i!=9.5):
    i += 0.5
    palatability_limit = i
    print('Limit:', i)
    diet_model()
# diet_model()





