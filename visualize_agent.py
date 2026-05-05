import matplotlib.pyplot as plt
from agents import MCTSAgent
import numpy as np
from go_search_problem import GoState
import go_utils




def visualize_MCTS(board=5, time_limit=5, c_vals=[0.1, 1, np.sqrt(2), 5, 10]):

    fig, axes = plt.subplots(1, len(c_vals), figsize=(15, 5))


    for i, c in enumerate(c_vals):
        agent = MCTSAgent(c=c)

        initial_state = GoState(go_utils.create_go_game(board))
        best_action = agent.get_move(initial_state, time_limit)

        visit_counts = np.zeros(board * board)

        for child in agent.root.children:
            action = child.action
            if action == board * board:  
                continue
            row = action // board
            col = action % board
            visit_counts[row * board + col] = child.visits

        visit_counts = visit_counts.reshape((board, board))

        
        ax = axes[i] if len(c_vals) > 1 else axes
        im = ax.imshow(visit_counts, cmap='viridis', aspect='auto')
        ax.set_title(f'MCTS Visit Counts (c={c})')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        fig.colorbar(im, ax=ax, label='Visit Count')


    plt.tight_layout()
    plt.savefig('mcts_visit_counts.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    visualize_MCTS()