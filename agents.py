import random
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, List

import numpy as np
import torch
from torch import nn

from go_search_problem import GoProblem, GoState, Action, HeuristicGoProblem
from heuristic_go_problems import GoProblemSimpleHeuristic, GoProblemLearnedHeuristic
from models import ValueNetwork, load_model

MAXIMIZER = 0
MIMIZER = 1

class GameAgent(ABC):
    """Abstract base class for all Go game agents."""
    
    @abstractmethod
    def get_move(self, state: GoState, time_limit: float) -> Action:
        """Get the best move for the given state within the time limit.
        
        Args:
            state: Current game state
            time_limit: Maximum time in seconds to spend on this move
            
        Returns:
            Action index representing the chosen move
        """
        pass

    def reset(self):
        """Reset any internal state of the agent if necessary.
            Called after a new game is started.
        """
        pass


class RandomAgent(GameAgent):
    # An Agent that makes random moves

    def __init__(self):
        self.search_problem = GoProblem()

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get random move for a given state
        """
        actions = self.search_problem.get_available_actions(game_state)
        return random.choice(actions)

    def __str__(self):
        return "RandomAgent"


class GreedyAgent(GameAgent):
    def __init__(self, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get move of agent for given game state.
        Greedy agent looks one step ahead with the provided heuristic and chooses the best available action
        (Greedy agent does not consider remaining time)

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        """
        # Create new GoSearchProblem with provided heuristic
        search_problem = self.search_problem

        # Player 0 is maximizing
        if game_state.player_to_move() == MAXIMIZER:
            best_value = -float('inf')
        else:
            best_value = float('inf')
        best_action = None

        # Get Available actions
        actions = search_problem.get_available_actions(game_state)

        # Compare heuristic of every reachable next state
        for action in actions:
            new_state = search_problem.transition(game_state, action)
            value = search_problem.heuristic(new_state, new_state.player_to_move())
            if game_state.player_to_move() == MAXIMIZER:
                if value > best_value:
                    best_value = value
                    best_action = action
            else:
                if value < best_value:
                    best_value = value
                    best_action = action

        # Return best available action
        return best_action

    def __str__(self):
        """
        Description of agent (Greedy + heuristic/search problem used)
        """
        return "GreedyAgent + " + str(self.search_problem)

#############################################
# 
#
# Part 1: Basic Adversarial Search Algorithms
#
#
#############################################

class MinimaxAgent(GameAgent):
    def __init__(self, depth_cutoff=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth_cutoff
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using minimax algorithm

        MinimaxAgents should not consider time limit, they simply search to their specified depth_cutoff
        If your agent is running out of time, you should use a shorter cutoff depth
        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO Part 1: implement get_move method of MinimaxAgent
        best_action, _ = self.minimax_helper(self.search_problem, game_state, 0, self.depth)


        return best_action

    def minimax_helper(self, asp: HeuristicGoProblem, state: GoState, depth: int, cutoff_depth: float) -> Tuple[Action, float]:
        """
        Helper function for minimax that performs the recursive search.

        Args:
            asp: The adversarial search problem.
            depth: Current depth in the search tree.
            cutoff_depth: Maximum search depth (0 = start state, 1 = one move ahead).
            stats: Dictionary to track search statistics.
        
        Returns:
            The minimax value of the current state.
        """
        if asp.is_terminal_state(state):
            return None, 1000 * asp.get_result(state)

        if depth == cutoff_depth:
            return None, asp.heuristic(state, state.player_to_move())
        
        # Determine if the current player is the maximizing player (player 0) or the minimizing player (player 1)
        if state.player_to_move() == 0:
            return self.max_minimax(asp, state, depth, cutoff_depth)
        else:
            return self.min_minimax(asp, state, depth, cutoff_depth)
        
    def max_minimax(self, asp: HeuristicGoProblem, state: GoState, depth: int, cutoff_depth: float = float('inf')) -> Tuple[Action, float]:
        """
        Helper function for minimax that computes the maximum value for the maximizing player.

        args:
            asp: The adversarial search problem.
            state: The current game state.
            depth: Current depth in the search tree.
            stats: Dictionary to track search statistics.
            cutoff_depth: Maximum search depth (0 = start state, 1 = one move ahead).

        """
        v = float('-inf')
        best_action = None

        for action in asp.get_available_actions(state):
            _, value = self.minimax_helper(asp, asp.transition(state, action), depth + 1, cutoff_depth)

            if value > v:
                v = value
                best_action = action

        return best_action, v


    def min_minimax(self, asp: HeuristicGoProblem, state: GoState, depth: int, cutoff_depth: float = float('inf')) -> Tuple[Action, float]:
        """
        Helper function for minimax that computes the minimum value for the minimizing player.

        args:
            asp: The adversarial search problem.
            state: The current game state.
            depth: Current depth in the search tree.
            stats: Dictionary to track search statistics.
            cutoff_depth: Maximum search depth (0 = start state, 1 = one move ahead).
        """
        v = float('inf')
        best_action = None

        for action in asp.get_available_actions(state):
            _, value = self.minimax_helper(asp, asp.transition(state, action), depth + 1, cutoff_depth)

            if value < v:
                v = value
                best_action = action

        return best_action, v
            

    def __str__(self):
        return f"MinimaxAgent w/ depth {self.depth} + " + str(self.search_problem)
    



class AlphaBetaAgent(GameAgent):
    def __init__(self, depth_cutoff=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth_cutoff
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using alpha-beta algorithm

        AlphaBetaAgents should not consider time limit, they simply search to their specified depth_cutoff
        If your agent is running out of time, you should use a shorter cutoff depth

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO Part 1: implement get_move algorithm of AlphaBeta Agent
        alpha = float('-inf')
        beta = float('inf')

        best_action, _ = self.ab_helper(self.search_problem, game_state, 0, alpha, beta, self.depth)

        return best_action
    

    def ab_helper(self, asp: HeuristicGoProblem, state: GoState, depth: int, alpha: float, beta: float, cutoff_depth: float) -> Tuple[Action, float]:
        """
        Helper function for minimax that performs the recursive search.

        Args:
            asp: The adversarial search problem.
            depth: Current depth in the search tree.
            cutoff_depth: Maximum search depth (0 = start state, 1 = one move ahead).
            stats: Dictionary to track search statistics.
            alpha: The best value that the maximizing player can guarantee at this level or above.
            beta: The best value that the minimizing player can guarantee at this level or above.
        
        Returns:
            The minimax value of the current state.
        """
        if asp.is_terminal_state(state):
            return None, 1000*asp.get_result(state)
        
        if depth == cutoff_depth:
            return None, asp.heuristic(state, state.player_to_move())


        if state.player_to_move() == 0:
            return self.max_ab(asp, state, depth, alpha, beta, cutoff_depth)
        else:
            return self.min_ab(asp, state, depth, alpha, beta, cutoff_depth)

        

    def max_ab(self, asp: HeuristicGoProblem, state: GoState, depth: int, alpha: float, beta: float, cutoff_depth: float = float('inf')) -> Tuple[Action, float]:
        """
        Helper function for minimax that computes the maximum value for the maximizing player.

        args:
            asp: The adversarial search problem.
            state: The current game state.
            depth: Current depth in the search tree.
            stats: Dictionary to track search statistics.
            alpha: The best value that the maximizing player can guarantee at this level or above.
            beta: The best value that the minimizing player can guarantee at this level or above.
            cutoff_depth: Maximum search depth (0 = start state, 1 = one move ahead).
        """
        v = float('-inf')
        best_action = None

        for action in asp.get_available_actions(state):
            _, value = self.ab_helper(asp, asp.transition(state, action), depth + 1, alpha, beta, cutoff_depth)

            if value > v:
                v = value
                best_action = action

            if value >= beta:
                return action, value
            else:
                alpha = max(alpha, v)

        return best_action, v


    def min_ab(self, asp: HeuristicGoProblem, state: GoState, depth: int, alpha: float, beta: float, cutoff_depth: float = float('inf')) -> Tuple[Action, float]:
        """
        Helper function for minimax that computes the minimum value for the minimizing player.

        args:
            asp: The adversarial search problem.
            state: The current game state.
            depth: Current depth in the search tree.
            stats: Dictionary to track search statistics.
            alpha: The best value that the maximizing player can guarantee at this level or above.
            beta: The best value that the minimizing player can guarantee at this level or above.
            cutoff_depth: Maximum search depth (0 = start state, 1 = one move ahead).
        """
        v = float('inf')
        best_action = None

        for action in asp.get_available_actions(state):
            _, value = self.ab_helper(asp, asp.transition(state, action), depth + 1, alpha, beta, cutoff_depth)

            if value < v:
                best_action = action
                v = value

            if value <= alpha:
                return action, value
            else:
                beta = min(beta, v)

        return best_action, v

    def __str__(self):
        return f"AlphaBeta w/ depth {self.depth} + " + str(self.search_problem)
    


def get_features(game_state: GoState):
    """
    Map a game state to a list of features.

    Some useful functions from game_state include:
        game_state.size: size of the board
        get_pieces_coordinates(player_index): get coordinates of all pieces of a player (0 or 1)
        get_pieces_array(player_index): get a 2D array of pieces of a player (0 or 1)
        
        get_board(): get a 2D array of the board with 4 channels (player 0, player 1, empty, and player to move). 4 channels means the array will be of size 4 x n x n
    
        Descriptions of these methods can be found in the GoState

    Input:
        game_state: GoState to encode into a fixed size list of features
    Output:
        features: list of features
    """

    board_size = game_state.size
    # TODO: Encode game_state into a list of features
    features = []
    board = game_state.get_board()
    black_pieces = board[0].reshape(board_size * board_size)
    white_pieces = board[1].reshape(board_size * board_size)
    player = game_state.player_to_move()

    # stone difference
    stone_diff = np.sum(black_pieces) - np.sum(white_pieces)

    legal = np.zeros(26)
    for action in game_state.legal_actions():
        legal[action] = 1

    features = np.concatenate((black_pieces, white_pieces, [player], [stone_diff], legal))


    return features


def create_value_agent_from_model():
    """
    Create agent object from saved model. This (or other methods like this) will be how your agents will be created in gradescope and in the final tournament.
    """

    model_path = "value_model.pt"
    # TODO: Update number of features for your own encoding size
    feature_size = 78
    model = load_model(model_path, ValueNetwork(feature_size))
    heuristic_search_problem = GoProblemLearnedHeuristic(model)

    # TODO: Try with other heuristic agents (IDS/AB/Minimax)
    learned_agent = GreedyAgent(heuristic_search_problem)

    return learned_agent


################################################
#
# Part 2: Advanced Adversarial Search Algorithms
#
################################################

class IterativeDeepeningAgent(GameAgent):
    def __init__(self, cutoff_time=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.cutoff_time = cutoff_time
        self.search_problem = search_problem

    def get_move(self, game_state, time_limit):
        """
        Get move of agent for given game state using iterative deepening algorithm (+ alpha-beta).
        Iterative deepening is a search algorithm that repeatedly searches for a solution to a problem,
        increasing the depth of the search with each iteration.

        The advantage of iterative deepening is that you can stop the search based on the time limit, rather than depth.
        The recommended approach is to modify your implementation of Alpha-beta to stop when the time limit is reached
        and run IDS on that modified version.

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO Part 2: implement get_move algorithm of IterativeDeepeningAgent
        best_move = random.choice(self.search_problem.get_available_actions(game_state))
        depth = 1
        time_end = time.time() + min(time_limit, self.cutoff_time) - 0.05

        while time.time() < time_end:
            move, _ = self.ab_helper(self.search_problem, game_state, 0, float('-inf'), float('inf'), depth, time_end)
            if move is not None:
                best_move = move
            else:
                break
            depth += 1

        return best_move
    

    def ab_helper(self, asp: HeuristicGoProblem, state: GoState, depth: int, alpha: float, beta: float, cutoff_depth: float, time_limit: float) -> Tuple[Action, float]:
        """
        Helper function for minimax that performs the recursive search.

        Args:
            asp: The adversarial search problem.
            depth: Current depth in the search tree.
            cutoff_depth: Maximum search depth (0 = start state, 1 = one move ahead).
            stats: Dictionary to track search statistics.
            alpha: The best value that the maximizing player can guarantee at this level or above.
            beta: The best value that the minimizing player can guarantee at this level or above.
        
        Returns:
            The minimax value of the current state.
        """
        if time.time() >= time_limit:
            return None, None   
            
        if asp.is_terminal_state(state):
            return None, 1000*asp.get_result(state)
        
        if depth == cutoff_depth:
            return None, asp.heuristic(state, state.player_to_move())


        if state.player_to_move() == 0:
            return self.max_ab(asp, state, depth, alpha, beta, cutoff_depth, time_limit)
        else:
            return self.min_ab(asp, state, depth, alpha, beta, cutoff_depth, time_limit)

        

    def max_ab(self, asp: HeuristicGoProblem, state: GoState, depth: int, alpha: float, beta: float, cutoff_depth: float, time_limit: float) -> Tuple[Action, float]:
        """
        Helper function for minimax that computes the maximum value for the maximizing player.

        args:
            asp: The adversarial search problem.
            state: The current game state.
            depth: Current depth in the search tree.
            stats: Dictionary to track search statistics.
            alpha: The best value that the maximizing player can guarantee at this level or above.
            beta: The best value that the minimizing player can guarantee at this level or above.
            cutoff_depth: Maximum search depth (0 = start state, 1 = one move ahead).
        """


        v = float('-inf')
        best_action = None

        if time.time() >= time_limit:
            return best_action, v

        for action in asp.get_available_actions(state):
            if time.time() >= time_limit:
                break

            _, value = self.ab_helper(asp, asp.transition(state, action), depth + 1, alpha, beta, cutoff_depth, time_limit)

            if value is None:
                return None, None
            
            if value > v:
                v = value
                best_action = action

            if value >= beta:
                return action, value
            else:
                alpha = max(alpha, v)

        return best_action, v


    def min_ab(self, asp: HeuristicGoProblem, state: GoState, depth: int, alpha: float, beta: float, cutoff_depth: float, time_limit: float) -> Tuple[Action, float]:
        """
        Helper function for minimax that computes the minimum value for the minimizing player.

        args:
            asp: The adversarial search problem.
            state: The current game state.
            depth: Current depth in the search tree.
            stats: Dictionary to track search statistics.
            alpha: The best value that the maximizing player can guarantee at this level or above.
            beta: The best value that the minimizing player can guarantee at this level or above.
            cutoff_depth: Maximum search depth (0 = start state, 1 = one move ahead).
        """
        
        v = float('inf')
        best_action = None

        if time.time() >= time_limit:
            return best_action, v

        for action in asp.get_available_actions(state):
            if time.time() >= time_limit:
                break
            _, value = self.ab_helper(asp, asp.transition(state, action), depth + 1, alpha, beta, cutoff_depth, time_limit)

            if value is None:
                return None, None
            
            if value < v:
                best_action = action
                v = value

            if value <= alpha:
                return action, value
            else:
                beta = min(beta, v)

        return best_action, v



    def __str__(self):
        return f"IterativeDeepening + " + str(self.search_problem)
    


    

class MCTSNode:
    def __init__(self, state, parent=None, children=None, action=None):
        # GameState for Node
        self.state = state

        # Parent (MCTSNode)
        self.parent = parent
        
        # Children List of MCTSNodes
        if children is None:
            children = []
        self.children = children
        
        # Number of times this node has been visited in tree search
        self.visits = 0
        
        # Value of node (number of times simulations from children results in black win)
        self.value = 0
        
        # Action that led to this node
        self.action = action

    def __hash__(self):
        """
        Hash function for MCTSNode is hash of state
        """
        return hash(self.state)
    
class MCTSAgent(GameAgent):
    def __init__(self, c=np.sqrt(2)):
        """
        Args: 
            c (float): exploration constant of UCT algorithm
        """
        super().__init__()
        self.c = c

        # Initialize Search problem
        self.search_problem = GoProblem()

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using MCTS algorithm
        
        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO Part 2: Implement MCTS
        best_action = random.choice(self.search_problem.get_available_actions(game_state))
        node = MCTSNode(game_state)
        time_end = time.time() + time_limit - 1

        while time.time() < time_end:
            leaf = self.select(node, time_end)
            children = self.expand(leaf, time_end)
            results = self.simulate(children, time_end)
            self.backpropagate(results, children, time_end)

        if len(node.children) > 0:
            best_action = node.children[np.argmax([child.visits for child in node.children])].action

        return best_action
    

    

    def select(self, node, time_end):
        if time.time() >= time_end:
            return node
        
        currNode = node
        while len(currNode.children) > 0 and time.time() < time_end:
            if currNode.state.is_terminal_state():
                return currNode
            
            unvisited_children = [child for child in currNode.children if child.visits == 0]
            if unvisited_children:
                return random.choice(unvisited_children)
            else:
                currNode = max(currNode.children, key=lambda n: n.value / n.visits + self.c * np.sqrt(np.log(currNode.visits) / n.visits))

            
        return currNode



    def expand(self, leaf, time_end):
        if time.time() >= time_end:
            return random.choice(leaf.children) if len(leaf.children) > 0 else leaf
        
        if leaf.state.is_terminal_state():
            return [leaf]
        
        children = []
        state = leaf.state
        actions = state.legal_actions()
        for action in actions:
            if time.time() >= time_end:
                break
            newState = self.search_problem.transition(state, action)
            childNode = MCTSNode(newState, parent=leaf, action=action)
            children.append(childNode)
            leaf.children.append(childNode)
        return children
    

    def simulate(self, children, time_end):
        if time.time() >= time_end:
            return []
        
        results = []
        for child in children:
            if time.time() >= time_end:
                break
            state = child.state
            while not state.is_terminal_state() and time.time() < time_end:
                action = random.choice(state.legal_actions())
                state = self.search_problem.transition(state, action)
            results.append(self.search_problem.get_result(state))
        return results
    
    def backpropagate(self, results, children, time_end):
        if time.time() >= time_end:
            return
        for child, result in zip(children, results):
            if time.time() >= time_end:
                break
            currNode = child
            while currNode is not None and time.time() < time_end:
                currNode.visits += 1
                if result == -1 and currNode.state.player_to_move() == 0:
                    currNode.value += 1
                elif result == 1 and currNode.state.player_to_move() == 1:
                    currNode.value += 1
                currNode = currNode.parent


    def __str__(self):
        return "MCTS"
    

###################################################
#
# Part 3: Final Agent
#
###################################################

class HybridAgent(GameAgent):
    def __init__(self, ):
        super().__init__()

    model_path = "value_model.pt"
    feature_size = 78
    model = load_model(model_path, ValueNetwork(feature_size))
    heuristic_search_problem = GoProblemLearnedHeuristic(model)

    learned_agent = GreedyAgent(heuristic_search_problem)
    alphabeta_simple = AlphaBetaAgent(depth_cutoff=2, search_problem=GoProblemSimpleHeuristic())

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        board = game_state.get_board()
        num_pieces = np.sum(board[0]) + np.sum(board[1])
        if num_pieces > 10:
            return self.alphabeta_simple.get_move(game_state, time_limit)
        else:
            return self.learned_agent.get_move(game_state, time_limit)



        

def get_final_agent_5x5():
    """Called to construct agent for final submission for 5x5 board"""
    return HybridAgent()

def get_final_agent_9x9():
    """Called to construct agent for final submission for 9x9 board"""
    return None
