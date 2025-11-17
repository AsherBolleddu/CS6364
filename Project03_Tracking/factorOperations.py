# factorOperations.py
# -------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from typing import List
from bayesNet import Factor
import functools
from util import raiseNotDefined


def joinFactorsByVariableWithCallTracking(callTrackingList=None):

    def joinFactorsByVariable(factors: List[Factor], joinVariable: str):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin = [
            factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [
            factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len(
            [factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", factor)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +
                             "joinVariable: " + str(joinVariable) + "\n" +
                             ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))

        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable


joinFactorsByVariable = joinFactorsByVariableWithCallTracking()

########### ########### ###########
########### QUESTION 2  ###########
########### ########### ###########


def joinFactors(factors: List[Factor]):
    """
    Input factors is a list of factors.  

    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables())
                           for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factor)
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                             + "unconditionedVariables: " + str(intersect) +
                             "\nappear in more than one input factor.\n" +
                             "Input factors: \n" +
                             "\n".join(map(str, factors)))

    "*** YOUR CODE HERE ***"
    # Convert to list if it's not already (handles dict_values case)
    if not isinstance(factors, list):
        factors = list(factors)

    # Handle edge case: if no factors provided, return None
    if not factors or len(factors) == 0:
        return None

    # Handle edge case: if only one factor, return a copy of it
    if len(factors) == 1:
        factor = factors[0]
        # Create a new factor with the same variables and probabilities
        newFactor = Factor(factor.unconditionedVariables(),
                           factor.conditionedVariables(),
                           factor.variableDomainsDict())
        for assignmentDict in factor.getAllPossibleAssignmentDicts():
            newFactor.setProbability(
                assignmentDict, factor.getProbability(assignmentDict))
        return newFactor

    # Collect all unconditioned and conditioned variables from all factors
    allUnconditionedVariables = set()
    allConditionedVariables = set()

    for factor in factors:
        allUnconditionedVariables = allUnconditionedVariables.union(
            factor.unconditionedVariables())
        allConditionedVariables = allConditionedVariables.union(
            factor.conditionedVariables())

    # The final unconditioned variables are ALL variables that appear as unconditioned in any factor
    # The final conditioned variables are ONLY those that appear ONLY as conditioned (never unconditioned)
    finalUnconditionedVariables = allUnconditionedVariables
    finalConditionedVariables = allConditionedVariables - allUnconditionedVariables

    # Get the variableDomainsDict from the first factor (they're all the same)
    variableDomainsDict = factors[0].variableDomainsDict()

    # Create the new joined factor
    joinedFactor = Factor(finalUnconditionedVariables,
                          finalConditionedVariables,
                          variableDomainsDict)

    # For each possible assignment in the joined factor,
    # calculate the product of probabilities from all input factors
    for assignmentDict in joinedFactor.getAllPossibleAssignmentDicts():
        # Calculate the product of probabilities from all factors
        productProb = 1.0
        for factor in factors:
            # Each factor will extract only the variables it cares about from assignmentDict
            productProb *= factor.getProbability(assignmentDict)

        # Set the probability in the joined factor
        joinedFactor.setProbability(assignmentDict, productProb)

    return joinedFactor
    "*** END YOUR CODE HERE ***"

########### ########### ###########
########### QUESTION 3  ###########
########### ########### ###########


def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor: Factor, eliminationVariable: str):
        """
        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.

        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Elimination variable is not an unconditioned variable "
                             + "in this factor\n" +
                             "eliminationVariable: " + str(eliminationVariable) +
                             "\nunconditionedVariables:" + str(factor.unconditionedVariables()))

        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Factor has only one unconditioned variable, so you "
                             + "can't eliminate \nthat variable.\n" +
                             "eliminationVariable:" + str(eliminationVariable) + "\n" +
                             "unconditionedVariables: " + str(factor.unconditionedVariables()))

        "*** YOUR CODE HERE ***"
        # Get the new sets of variables after elimination
        newUnconditionedVariables = factor.unconditionedVariables() - \
            {eliminationVariable}
        newConditionedVariables = factor.conditionedVariables()

        # Get the variable domains
        variableDomainsDict = factor.variableDomainsDict()

        # Create the new factor with the elimination variable removed
        newFactor = Factor(newUnconditionedVariables,
                           newConditionedVariables,
                           variableDomainsDict)

        # For each possible assignment in the new factor (without the elimination variable),
        # sum up the probabilities over all values of the elimination variable
        for assignmentDict in newFactor.getAllPossibleAssignmentDicts():
            # Sum probabilities over all values of the elimination variable
            sumProb = 0.0

            # Iterate through all possible values of the elimination variable
            for elimValue in variableDomainsDict[eliminationVariable]:
                # Create an extended assignment that includes the elimination variable
                extendedAssignment = dict(assignmentDict)
                extendedAssignment[eliminationVariable] = elimValue

                # Add the probability for this assignment
                sumProb += factor.getProbability(extendedAssignment)

            # Set the summed probability in the new factor
            newFactor.setProbability(assignmentDict, sumProb)

        return newFactor
        "*** END YOUR CODE HERE ***"

    return eliminate


eliminate = eliminateWithCallTracking()
