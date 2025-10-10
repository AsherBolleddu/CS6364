# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        # Start with the game score
        score = successorGameState.getScore()

        # 1. Handle ghost distances
        ghost_distances = []
        for ghost_state in newGhostStates:
            ghost_pos = ghost_state.getPosition()
            distance = manhattanDistance(newPos, ghost_pos)
            ghost_distances.append(distance)

        # Find the nearest ghost
        min_ghost_distance = min(
            ghost_distances) if ghost_distances else float('inf')

        # Penalize being too close to ghosts (unless they're scared)
        if min_ghost_distance > 0:
            if min_ghost_distance <= 2 and newScaredTimes[ghost_distances.index(min_ghost_distance)] == 0:
                # Very dangerous - heavily penalize
                score -= 500
            elif min_ghost_distance <= 3 and newScaredTimes[ghost_distances.index(min_ghost_distance)] == 0:
                # Still dangerous
                score -= 50
        else:
            # Ghost is on top of Pacman - game over
            score -= 1000

        # 2. Handle food distances
        food_list = newFood.asList()

        if food_list:
            # Find distances to all food
            food_distances = [manhattanDistance(
                newPos, food) for food in food_list]

            # Reward being close to the nearest food
            min_food_distance = min(food_distances)
            if min_food_distance > 0:
                # Use reciprocal as suggested in the instructions
                score += 10.0 / min_food_distance

            # Small penalty for remaining food (encourages eating)
            score -= 4 * len(food_list)

        # 3. Check if this action eats food
        current_food_list = currentGameState.getFood().asList()
        if len(current_food_list) > len(food_list):
            # We ate food! Big bonus
            score += 100

        # 4. Bonus for scared ghosts nearby (hunting opportunity)
        for i, scared_time in enumerate(newScaredTimes):
            if scared_time > 0:
                ghost_pos = newGhostStates[i].getPosition()
                distance = manhattanDistance(newPos, ghost_pos)
                if distance <= scared_time:  # Can we reach it while scared?
                    score += 200.0 / (distance + 1)

        # 5. Penalize stopping (encourage movement)
        if action == Directions.STOP:
            score -= 10

        # 6. Handle capsules
        capsules = successorGameState.getCapsules()
        if capsules:
            capsule_distances = [manhattanDistance(
                newPos, capsule) for capsule in capsules]
            min_capsule_distance = min(capsule_distances)
            # Encourage moving toward capsules
            score += 5.0 / (min_capsule_distance + 1)

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """

        def minimax(state, depthRemaining, agentIndex):
            """
            Returns the minimax value for the given state.
            For the root Pacman node only: returns (value, action)
            For all other nodes: returns value
            """
            # Check terminal states
            if state.isWin() or state.isLose() or depthRemaining == 0:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            legalActions = state.getLegalActions(agentIndex)

            # No legal actions available
            if not legalActions:
                return self.evaluationFunction(state)

            # Calculate next agent and depth
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depthRemaining - 1 if nextAgent == 0 else depthRemaining

            # Pacman's turn (maximizer)
            if agentIndex == 0:
                maxEval = float('-inf')
                bestAction = None

                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    eval = minimax(successor, nextDepth, nextAgent)

                    # Handle tuple return from recursive calls
                    if isinstance(eval, tuple):
                        eval = eval[0]

                    if eval > maxEval:
                        maxEval = eval
                        bestAction = action

                # Return action for root node
                if depthRemaining == self.depth:
                    return bestAction
                else:
                    return maxEval

            # Ghost's turn (minimizer)
            else:
                minEval = float('inf')

                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    eval = minimax(successor, nextDepth, nextAgent)

                    # Handle tuple return from recursive calls
                    if isinstance(eval, tuple):
                        eval = eval[0]

                    minEval = min(minEval, eval)

                return minEval

        # Start the minimax algorithm
        return minimax(gameState, self.depth, 0)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)

    This implementation follows the pseudocode structure:
    - α: MAX's best option on path to root
    - β: MIN's best option on path to root
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def maxValue(state, depth, alpha, beta):
            """
            Max-value function for Pacman with alpha-beta pruning.

            Pseudocode:
            def max-value(state, α, β):
                initialize v = -∞
                for each successor of state:
                    v = max(v, value(successor, α, β))
                    if v > β return v  # Prune
                    α = max(α, v)
                return v
            """
            # Terminal test
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None

            v = float('-inf')
            bestAction = None

            # For each action available to Pacman
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)

                # Get value from min layer (first ghost)
                minVal = minValue(successor, depth, 1, alpha, beta)

                # Update maximum value and best action
                if minVal > v:
                    v = minVal
                    bestAction = action

                # Pruning condition for MAX node
                if v > beta:
                    return v, bestAction

                # Update alpha (MAX's best guarantee)
                alpha = max(alpha, v)

            return v, bestAction

        def minValue(state, depth, agentIndex, alpha, beta):
            """
            Min-value function for ghosts with alpha-beta pruning.

            Pseudocode:
            def min-value(state, α, β):
                initialize v = +∞
                for each successor of state:
                    v = min(v, value(successor, α, β))
                    if v < α return v  # Prune
                    β = min(β, v)
                return v
            """
            # Terminal test
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = float('inf')
            numAgents = state.getNumAgents()

            # For each action available to this ghost
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)

                # Determine next agent
                if agentIndex == numAgents - 1:
                    # Last ghost - go back to Pacman (next depth)
                    value, _ = maxValue(successor, depth - 1, alpha, beta)
                else:
                    # Next ghost at same depth
                    value = minValue(successor, depth,
                                     agentIndex + 1, alpha, beta)

                # Update minimum value
                v = min(v, value)

                # Pruning condition for MIN node
                if v < alpha:
                    return v

                # Update beta (MIN's best guarantee)
                beta = min(beta, v)

            return v

        # Initial call with α = -∞, β = +∞
        _, action = maxValue(gameState, self.depth,
                             float('-inf'), float('inf'))
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        def getValue(state, depth, agentIndex):
            """
            Returns the value of a state using expectimax.
            - Terminal states: return evaluation
            - Max nodes (Pacman): return max value
            - Chance nodes (Ghosts): return expected value
            """
            # Terminal state check
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            # Pacman's turn (maximize)
            if agentIndex == 0:
                return maxValue(state, depth)
            # Ghost's turn (expected value)
            else:
                return expValue(state, depth, agentIndex)

        def maxValue(state, depth):
            """Max value for Pacman"""
            v = float('-inf')
            bestAction = None

            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                # Next agent is ghost 1
                successorValue = getValue(successor, depth, 1)

                if successorValue > v:
                    v = successorValue
                    bestAction = action

            # Return action at root, value otherwise
            if depth == self.depth:
                return bestAction
            return v

        def expValue(state, depth, agentIndex):
            """Expected value for ghosts (uniform random policy)"""
            v = 0
            legalActions = state.getLegalActions(agentIndex)

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)

                # Determine next agent
                if agentIndex == state.getNumAgents() - 1:
                    # Last ghost, next is Pacman with depth-1
                    successorValue = getValue(successor, depth - 1, 0)
                else:
                    # Next ghost
                    successorValue = getValue(successor, depth, agentIndex + 1)

                v += successorValue

            # Return average (expected value with uniform probability)
            return v / len(legalActions)

        # Start search
        return getValue(gameState, self.depth, 0)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Balanced evaluation function that:
    1. Heavily weights game score as base
    2. Uses reciprocal of minimum food distance for strong attraction
    3. Penalizes remaining food to encourage completion
    4. Avoids active ghosts with distance-based penalties
    5. Aggressively hunts scared ghosts
    6. Values capsules based on context
    """

    # Win/Lose states
    if currentGameState.isWin():
        return 10000
    if currentGameState.isLose():
        return -10000

    # Get state info
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # Start with score (important - includes eating food/ghosts)
    value = currentGameState.getScore()

    # Food evaluation
    if food:
        minFoodDist = min(manhattanDistance(pos, f) for f in food)
        # Strong reciprocal attraction to nearest food
        value += 10.0 / (minFoodDist + 1)  # +1 to avoid division by zero
        # Penalty for remaining food
        value -= 2 * len(food)

    # Ghost evaluation
    for ghost in ghosts:
        dist = manhattanDistance(pos, ghost.getPosition())

        if ghost.scaredTimer > 0:
            # Hunt scared ghosts
            if dist <= ghost.scaredTimer:
                value += 200.0 / (dist + 1)
        else:
            # Avoid active ghosts
            if dist <= 1:
                value -= 200
            elif dist <= 2:
                value -= 50

    # Capsule evaluation
    if capsules:
        minCapsuleDist = min(manhattanDistance(pos, c) for c in capsules)
        value += 5.0 / (minCapsuleDist + 1)

    return value


# Abbreviation
better = betterEvaluationFunction
