# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    start = problem.getStartState() # Get the start state
    fringe = util.Stack() # DFS is implemented with stack
    visited = set() # Keep track of the nodes visited so far

    fringe.push((start, [])) # Initialize stack with the start state and empty path
    
    while not fringe.isEmpty():
        state, path = fringe.pop() # Get the current state and path

        if problem.isGoalState(state):
            return path
        
        if state not in visited:
            visited.add(state)

        # Get the next state and next action (string that corresponds to North, South, East, West) from the next connected Node
        for (nextState, action, stepCost) in problem.getSuccessors(state):
            if nextState not in visited: 
                nextPath = path + [action] # Add the next path to the current path
                fringe.push((nextState, nextPath))

    return []

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState() # Get the start state
    fringe = util.Queue() # BFS is implemented with Queue
    visited = {start} # Keep track of the nodes visited so far, intialized with start state
    
    fringe.push((start, [])) # Initialize queue with start state and empty path

    while not fringe.isEmpty():
        state, path = fringe.pop() # Get the current state and path

        if problem.isGoalState(state):
            return path
        
        # get the next state and next action from the next connected Node
        for (nextState, action, stepCost) in problem.getSuccessors(state):
            if nextState not in visited:
                nextPath = path + [action] # Add the next path to the current path
                fringe.push((nextState, nextPath))
                visited.add(nextState)
    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    fringe = util.PriorityQueue() # Queue now has weight on it
    bestCost = {start: 0} # Necessary to keep track of the cost, cost is 0 for the start state

    fringe.push((start, [], 0), 0) # ((Start state, path, cost) priority) 
    
    while not fringe.isEmpty():
        state, path, cost = fringe.pop()

        # Checks to see if the current cost is greater than the existing cost in the map if the current state is in there, if so we can just skip
        if state in bestCost and cost > bestCost[state]: 
            continue

        bestCost[state] = cost # Add the current state with the current cost

        if problem.isGoalState(state):
            return path
        
        
        for (nextState, action, stepCost) in problem.getSuccessors(state):
            newCost = cost + stepCost # Calculate the new cost with the next edge (stepCost)
            nextPath = path + [action] 
            if nextState not in bestCost or newCost < bestCost[nextState]:
                fringe.push((nextState, nextPath, newCost), newCost)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    fringe = util.PriorityQueue()
    bestCost = {start: 0} 
    # Heuristic h(n) estimates the cost from current node to the goal
    fringe.push((start, [], 0), heuristic(start, problem)) 
    while not fringe.isEmpty():
        state, path, cost = fringe.pop()

        if state in bestCost and cost > bestCost[state]:
            continue

        bestCost[state] = cost

        if problem.isGoalState(state):
            return path
        
        for (nextState, action, stepCost) in problem.getSuccessors(state):
            newCost = cost + stepCost
            nextPath = path + [action]
            nextHeuristic = newCost + heuristic(nextState, problem) # Get next estimated cost

            if nextState not in bestCost or newCost < bestCost[nextState]:
                fringe.push((nextState, nextPath, newCost), nextHeuristic)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
