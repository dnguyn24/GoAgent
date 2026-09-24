from typing import Optional, Any, Union
import numpy as np
import torch
from go_search_problem import GoState, HeuristicGoProblem
BLACK = 0
WHITE = 1

class GoProblemSimpleHeuristic(HeuristicGoProblem):
    def __init__(self, size: int = 5, state=None, player_to_move: int = 0):
        super().__init__(size=size, state=state, player_to_move=player_to_move)

    def heuristic(self, state, player_index):
        """
        Very simple heuristic that just compares the number of pieces for each player
        
        Having more pieces than the opponent means that some were captured, capturing is generally good.
        Returns value from BLACK's perspective: positive = good for BLACK, negative = good for WHITE.
        """
        black_stones = len(state.get_pieces_coordinates(BLACK))
        white_stones = len(state.get_pieces_coordinates(WHITE))

        return black_stones - white_stones

    def __str__(self) -> str:
        return "Simple Heuristic"


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

    #legal actions
    legal = np.zeros(26)
    for action in game_state.legal_actions():
        legal[action] = 1

    features = np.concatenate((black_pieces, white_pieces, [player], [stone_diff], legal))


    return features

class GoProblemLearnedHeuristic(HeuristicGoProblem):
    def __init__(self, model=None, size: int = 5, state=None, player_to_move: int = 0):
        super().__init__(size=size, state=state, player_to_move=player_to_move)
        self.model = model

    def encoding(self, state):
        features = get_features(state)
        return features

    def heuristic(self, state, player_index):
        if self.is_terminal_state(state):
            return self.get_result(state)

        value = self.model(torch.tensor(self.encoding(state), dtype=torch.float32)).item()
        return value

    def __str__(self) -> str:
        return "Learned Heuristic"
    

