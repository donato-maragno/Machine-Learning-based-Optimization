from pyomo import environ
from pyomo.environ import *
from pyomo.opt import *
import numpy as np
import random
import math

'''
Description: in the for loop there is a call to diet_model() to solve the model optimally.
The model in question has always the some structure with some parameters randomly generated in order to
obtain every time a different solution - food basket - with different palatability values.
'''


def diet_model():
    model = AbstractModel()  # an abstract model is initialized. An instance is created later with the '.dat' file

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
    # Volume per serving of food
    model.V = Param(model.F, within=PositiveReals)

    '''
        Variables initialization
    '''
    # Decision variable
    model.x = Var(model.F, within=NonNegativeReals)
    # Slack variables
    model.Sg1 = Var(within=Reals)
    model.Sg2 = Var(within=Reals)
    model.Sg3 = Var(within=Reals)
    model.Sg4 = Var(within=Reals)
    model.Sg5 = Var(within=Reals)
    model.Sl = Var(model.N, within=NonNegativeReals)

    '''
        OBJECTIVE FUNCTION: Minimize the value of the Sl[j]s
    '''
    def cost_rule(model):
        return sum(model.Sl[i] for i in model.N)

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

    '''
        Impose the sum of foods per group minus the Sgi (auxiliary var) to equal the mid point of each group range
        the mid point is considered the optimal point
    '''

    def slack1_rule(model):
        return sum(model.x[i] * 100 for i in model.G1) - model.Sg1 == (model.minG1 + model.maxG1) / 2

    def slack2_rule(model):
        return sum(model.x[i] * 100 for i in model.G2) - model.Sg2 == (model.minG2 + model.maxG2) / 2

    def slack3_rule(model):
        return sum(model.x[i] * 100 for i in model.G3) - model.Sg3 == (model.minG3 + model.maxG3) / 2

    def slack4_rule(model):
        return sum(model.x[i] * 100 for i in model.G4) - model.Sg4 == (model.minG4 + model.maxG4) / 2

    def slack5_rule(model):
        return sum(model.x[i] * 100 for i in model.G5) - model.Sg5 == (model.minG5 + model.maxG5) / 2

    def palatability_rule(model):
        global counter
        weighted_palatability = environ.sqrt(model.Sg1 ** 2 + (2.5 * model.Sg2) ** 2
                                                   + (10 * model.Sg3) ** 2 + (4.2 * model.Sg4) ** 2
                                                   + (6.25 * model.Sg5) ** 2)
        return 279.96-weighted_palatability >= counter*28

    # To limit the amount of salt and sugar (I made it up)
    def salt_rule(model):
        return environ.inequality(0.05, model.x['Salt'], 1)

    def sugar_rule(model):
        return environ.inequality(0.00, model.x['Sugar'], 0.05)


    ''' 
        Call to constraints and O.F.
    '''
    model.cost = Objective(rule=cost_rule)
    model.nutrient_limit = Constraint(model.N, rule=nutrient_rule)
    model.slackNut_limit = Constraint(model.N, rule=slackNut_rule)

    model.G1_limit = Constraint(rule=G1_rule)
    model.G2_limit = Constraint(rule=G2_rule)
    model.G3_limit = Constraint(rule=G3_rule)
    model.G4_limit = Constraint(rule=G4_rule)
    model.G5_limit = Constraint(rule=G5_rule)
    model.salt_limit = Constraint(rule=salt_rule)
    model.sugar_limit = Constraint(rule=sugar_rule)
    model.slack_const1 = Constraint(rule=slack1_rule)
    model.slack_const2 = Constraint(rule=slack2_rule)
    model.slack_const3 = Constraint(rule=slack3_rule)
    model.slack_const4 = Constraint(rule=slack4_rule)
    model.slack_const5 = Constraint(rule=slack5_rule)

    # model.palatability_limit = Constraint(rule=palatability_rule)


    '''
        Instantiate the model using the .dat file
        The GNU Linear Programming Kit is a software package intended for solving large-scale linear programming,
        mixed integer programming, and other related problems.
    '''
    instance = model.create_instance('WFPdietOriginal.dat')
    solver_manager = environ.SolverManagerFactory('neos')
    results = solver_manager.solve(instance, opt='knitro')  # for the non linear palatability constraint
    if (results.Solver._list[0]['Termination condition'].key == 'infeasible'):
        return 0
    # instance.pprint()  # to display the initial model structure before the optimization
    # instance.display()  # to display the solution with constraints and OF
    if instance.Sg5.value is None:
        print('var Sg5 null')
        return 1
    '''
        Deterministic palatability function, weighted and non-weighted 
    '''
    palatability = np.round(math.sqrt(instance.Sg1.value ** 2 + instance.Sg2.value ** 2
                                      + instance.Sg3.value ** 2 + instance.Sg4.value ** 2
                                      + instance.Sg5.value ** 2), 2)
    weighted_palatability = np.round(math.sqrt(instance.Sg1.value ** 2 + (2.5 * instance.Sg2.value) ** 2
                                               + (10 * instance.Sg3.value) ** 2 + (4.2 * instance.Sg4.value) ** 2
                                               + (6.25 * instance.Sg5.value) ** 2), 2)

    '''
        Saving of the optimal solution
    '''
    solution = []
    print('SOLUTION: ')
    for i in instance.F:
        f = open("solutions.csv", "a")
        row = ""
        solution.append(instance.x[i].value)
        row += str(instance.x[i].value) + '\n'
        # row += i + '\n'
        f.write(row)
        f.close()
        # print(i, np.round(instance.x[i].value, 2))
        # if(instance.x[i].value > 0.00001):
            # print(i, instance.x[i].value)
    # To print variables values and obj function
    for v in instance.component_objects(Var, active=True):
        varobject = getattr(instance, str(v))
        cost = 0
        if (varobject.name == 'Sl'):
            for index in varobject:
                cost += varobject[index].value
                # print("   ", index, varobject[index].value)
    print(cost)

    f = open("solutions.csv", "a")
    f.write('\n')
    f.close()
    print('PALATABILITY', 279.96-weighted_palatability)
    print('THE RESULT IS:', results.Solver._list[0]['Termination condition'])

    return 0


counter = 0
while(counter!=9.5):
    counter += 0.5
    print('Limit:', counter)
    a = diet_model()

