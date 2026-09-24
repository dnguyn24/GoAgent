[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/pU4J2i_g)
# README

# Task 1.1
The random agent had more wins overall compared to the greedy agent with simple heuristic. It also won more as black than the greedy agent. This was surprising to me because I expected the greedy agent to perform better with its heuristic. However, the random agent's unpredictability may have given it an advantage in certain situations.

# Task 1.2
I found that it was easy to win against the greedy agent with the simple heuristic. The greedy agent's strategy was predictable, and I could easily exploit its weaknesses. This is similar to what I saw in task 1.1.

# Task 1.3
Shuffling the order of actions before evaluating them greatly improved the performance of the greedy agent. This is a stark contrast from the previous version because the shuffle added randomness to the agent's decision making, allowing it to explore different actions and not get stuck in local minima. 

# Task 1.5
Cutoff depth
Minimax: 3
Alphabeta: 6

Overall, this demonstrates that alphabeta pruning allows the agent to search deeper in the game tree compared to minimax, which can lead to better decision making and improved performance.

# Task 2b
2. MCTS performed better than iterative deepening winning all 10 games. Iterative deepening did have more time remaining after the game ended compared to MCTS.

3. For this, I looked at how changing the C value changes the distribution of number of visits. I found that as the C value increases, the distribution of number of visits becomes more uniform across the different actions. This is because a higher C value encourages more exploration, leading to a more even distribution of visits. This evident with c values 0.1 and 1 where the most visited square was the center. As the value increased, other squares were visited more often. With a C value of 10, the distribution of visits was much more uniform across the board. For the visualization, I created 5 heatmaps where each was a 5x5 grid representing the board. Each square in the grid was colored based on the number of visits it received during the MCTS search. The color intensity represented the number of visits, with lighter colors indicating more visits. The graphs can be found in mcts_visit_counts.png.






# Part 3
Sources: 
https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm
https://www.geeksforgeeks.org/deep-learning/implementing-deep-q-learning-using-tensorflow/
https://github.com/google-deepmind/open_spiel/tree/master
https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#reinforcement-learning-tips-and-tricks
https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/rl_environment.py



Process:
My implementation is in the deepq.ipynb notebook. There is a dqn_model.pt file that contains the trained model weights.

I started by copying the structure of the q learning notebook from assignment 9. I also included the get_features function from the supervised learning notebook.

At first, I thought about doing a full deep q learning agent that would be able to pick actions. However, I learned that this would be very difficult and time consuming to train as it could become unstable very easily and require tuning. Instead, I took inspiration from the supervised learning notebook in part 1 of the project and attempted to create a learned heuristic using deep q learning. I followed the overall structure of the q learning and supervised learning notebooks. I copied over the get_features and neural networks from the supervised learning notebook. I later found out that the environment that was provided in the assignment for go already has implementations in place for returning like the state of the game, player-to-move, reward, and legal actions. This made it easier to implement the deep q learning algorithm as I could just use the state and legal actions provided by the environment instead of having to create my own feature extractor.

Through the implementation process, I took inspiration from the sources I provided above as a guide for how deep q learning works. I also looked at the open spiel codebase to understand how the environment works and how to interact with it. I was able to get the model to train and produce a .pt file to be used to create the learned heuristic. However, due to what I mentioned above about extracting the features using the functions from the environment, a lot of the types conflicted with the preexisting code that defined like Gostate, GoProblem, etc. This prevented me from applying the dqn_model.pt file create a heuristic for my agent. If I had more time, I would have tried to resolve these type conflicts and get the learned heuristic working. 

Other improvements that I worked on that can be seen in the Deep Q notebook were I included an opening move at index 12 which is the middle of a 5x5 board. I found this improved the training time of the model because it allowed the model to start with a good move and learn from there instead of having to learn from random moves at the beginning of the game. I also was in the process of implementing a learning curriculum where I would start the training with simpler opponents such as the random agent and then gradually increase the difficulty of the opponents to the greedy agent with simple heuristic and then eventually to the IDS agent. This would allow the model to slowly learn over time and not get overwhelmed by the more difficult opponents at the beginning of training. However, as mentioned before, I ran into type conflicts that prevented me correctly passing in parameters to my agents for the training. 

As for the actual agent running the model, I thought about just using one agent with this new learned heuristic, but as I have found through my own experiments and what the project description mentioned about different agents performing better at different stages of the game, I decided to create a hybrid agent where one would use the learned heuristic for the early game and then switch to a more traditional agent like ids for the late game. This way, I could take advantage of the strengths of both agents and hopefully create a stronger overall agent. I experimented with greedy using learned heuristic and alphabeta pruning with simple heuristic for the late game. I found that the greedy agent with the learned heuristic performed better than the alphabeta pruning agent with simple heuristic. I found that this performed better than each agent on its own, both time wise and performance wise. This makes sense because the learned heuristic should be able to make good decisions in the early game when there are more possible actions and the game is less deterministic, while the alphabeta agent with simple heuristic should be able to make good decisions in the late game when there are fewer possible actions for the agent to explore. 

As a result, my current implementation for the final agent is a hybrid agent that uses Iterative deepening search with the learned heuristic for the early game and then switches to MCTS for the late game. I found that this performed better than using just one of the agents on its own because it allowed for better decision making in both the early and late game. IDS is able to exploit the learned heuristic in the early game when there are little to no pieces on the board, while MCTS takes over in the late game. In later games, there is less branching, so MCTS is able to give better answers in narrow branches instead of being stretched at the beginning of the game. I would have switched the learned heuristic that we did in the supervised learning notebook to the one I trained in the deep q learning notebook if I had more time to resolve the type conflicts and get it working.




Tell us about your implementation!

Answer the conceptual questions!

Hours taken: 50

Collaborators:

Known Bugs:
The learned heuristic for deep q learning does not work correctly due to type conflicts mentioned above between the features extracted from the environment and the features expected by the heuristic.

AI Use Description:
