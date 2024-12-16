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
import datetime
from util import*

import heapq

class PriorityQueue:
    """Priority Queue implementation using a heap."""
    def __init__(self):
        self.heap = []

    def push(self, item, priority):
        heapq.heappush(self.heap, (priority, item))

    def pop(self):
        return heapq.heappop(self.heap)[1]

    def isEmpty(self):
        return len(self.heap) == 0
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

def depthFirstSearch(problem):

    # Start measuring time to analyze the runtime of the algorithm
    start_time = time.time()
    # Get the starting state
    start_state = problem.getStartState()

    # If the start state is the goal, return an empty path (no actions needed)
    if problem.isGoalState(start_state):
        return []

    # Initialize a stack for DFS
    frontier = Stack()  # Stack to store (state, path) tuples
    frontier.push((start_state, []))  # Start with the initial state and an empty path

    # Set to keep track of explored states
    explored = set()

    while not frontier.isEmpty():
        # Pop the current state and the path leading to it
        curr_state, curr_path = frontier.pop()

        # If the current state is a goal state, return the path
        if problem.isGoalState(curr_state):
            # Stop the timer when the goal is found
            end_time = time.time()
            # Print the running time of the search
            print(f"Running Time: {end_time - start_time:.6f} seconds")
            return curr_path

        # If the current state has not been explored yet
        if curr_state not in explored:
            explored.add(curr_state)  # Mark the state as explored

            # Expand the current state and push successors onto the stack
            for successor, action, step_cost in problem.getSuccessors(curr_state):
                if successor not in explored:
                    # Add the successor to the stack with the updated path
                    frontier.push((successor, curr_path + [action]))

    # Return an empty list if no solution is found
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    """ Search the shallowest nodes in the search tree first. """
     # Start measuring time to analyze the runtime of the algorithm
    start_time = time.time()
    currPath = []           # The path that is popped from the frontier in each loop
    currState =  problem.getStartState()    # The state(position) that is popped for the frontier in each loop
    print(f"currState: {currState}")
    if problem.isGoalState(currState):     # Checking if the start state is also a goal state
        return currPath

    frontier = Queue()
    frontier.push( (currState, currPath) )    # Insert just the start state, in order to pop it first
    explored = set()
    while not frontier.isEmpty():
        currState, currPath = frontier.pop()    # Popping a state and the corresponding path
        # To pass autograder.py question2:
        if problem.isGoalState(currState):
            # Stop the timer when the goal is found
            end_time = time.time()
            # Print the running time of the search
            print(f"Running Time: {end_time - start_time:.6f} seconds")
            return currPath
        explored.add(currState)
        frontierStates = [ t[0] for t in frontier.list ]
        for s in problem.getSuccessors(currState):
            if s[0] not in explored and s[0] not in frontierStates:
                # Lecture code:
                # if problem.isGoalState(s[0]):
                #     return currPath + [s[1]]
               frontier.push( (s[0], currPath + [s[1]]) )      # Adding the successor and its path to the frontier       
    print(f"Running Time: {end_time - start_time:.6f} seconds")
    return []       # If this point is reached, a solution could not be found.

def uniformCostSearch(problem):
    # Start measuring time to analyze the runtime of the algorithm
    start_time = time.time()
    
    # Initialize the frontier as a priority queue where elements are ordered by cost
    frontier = PriorityQueue()
    
    # Get the start state of the problem
    start_state = problem.getStartState()
    
    # Push the start state into the frontier with an empty path and a cost of 0
    frontier.push((start_state, []), 0)
    
    # A set to keep track of explored nodes to avoid revisiting them
    explored = set()
    
    # A dictionary to store the minimum cost to reach each state
    cost_so_far = {start_state: 0}
    
    # Loop until there are no nodes left to explore in the frontier
    while not frontier.isEmpty():
        # Pop the node with the lowest cost from the frontier
        curr_state, curr_path = frontier.pop()
        
        # Debugging statement: Print the current state, path, and cost
        print(f"Exploring: {curr_state}, Path: {curr_path}, Cost: {cost_so_far[curr_state]}")
        
        # Check if the current state is the goal state
        if problem.isGoalState(curr_state):
            # Stop the timer when the goal is found
            end_time = time.time()
            
            # Debugging output: Print the goal state found
            print(f"Goal found: {curr_state}")
            
            # Print the running time of the search
            print(f"Running Time: {end_time - start_time:.6f} seconds")
            
            # Return the path to the goal state
            return curr_path
        
        # If the current state has not been explored yet
        if curr_state not in explored:
            # Mark the current state as explored
            explored.add(curr_state)
            
            # For each successor (child node) of the current state
            for successor, action, step_cost in problem.getSuccessors(curr_state):
                # Calculate the total cost to reach the successor
                new_cost = cost_so_far[curr_state] + step_cost
                
                # If the successor is not in cost_so_far or if the new cost is lower than the existing cost
                if successor not in cost_so_far or new_cost < cost_so_far[successor]:
                    # Update the cost to reach this successor
                    cost_so_far[successor] = new_cost
                    
                    # Push the successor into the frontier with its associated path and new cost as priority
                    frontier.push((successor, curr_path + [action]), new_cost)
                    
                    # Debugging statement: Print the successor, its updated path, and new cost
                    print(f"Pushing: {successor}, New Path: {curr_path + [action]}, New Cost: {new_cost}")
    
    # If no solution is found, return an empty list
    return []

    

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
