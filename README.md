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






# Part 3
Sources: 
https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm
https://www.geeksforgeeks.org/deep-learning/implementing-deep-q-learning-using-tensorflow/
https://github.com/google-deepmind/open_spiel/tree/master
https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#reinforcement-learning-tips-and-tricks
https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/rl_environment.py
https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/rl_environment.py



Process:
I started by copying the structure of the q learning notebook from assignment 9. I also included the get_features function from the supervised learning notebook.

At first, I thought about doing a full deep q learning agent that would be able to pick actions. However, I learned that this would be very difficult and time consuming to train as it could become unstable very easily and require tuning. Instead, I took insparation from the supervised learning notebook in part 1 of the project and attempted to create a learned heuristic using deep q learning. I followed the overall structure of the q learning and supervised learning notebooks. I copied over the get_features and neural networks from the supervised learning notebook. I later found out that the environnment that was orovided in the assignemnt for go already has implementaitons in place for returning like the state of the game, player-to-move, reward, and legal actions. This made it easier to implement the deep q learning algorithm as I could just use the state and legal actions provided by the environment instead of having to create my own feature extractor.

Through the implementation process, I took inspiration from the sources I provided above as a guide for how deep q learning works. I also looked at the open spiel codebase to understand how the environment works and how to interact with it. After getting the initial training loop working, I immeidately found that it ran much longer than I expected.I stopped the loop 30 minutes into its trianing. I tried a couple things such as using hardware acceleraiton to train with my gpu. The biggest change that I found to work was training the model with other agents, where it would essentially "play" against agents like random, greedy, and ids. This allowed the model to learn quicker and more effectively. After training, I saved the model as .pt file similar to what we did for supervised learning.

As for the actual agent running the model, I thoguht about just using one agent with this new learned heuristic, but as I have found through my own experiments and what the project description mentioned about different agents perfomring better at different stages of the game, I decided to create a hybrid agent where one would use the learned heuristic for the early game and then switch to a more traditional agent like ids for the late game. This way, I could take advantage of the strengths of both agents and hopefully create a stronger overall agent. I experimented with greedy using learned heuristic and alphabeta prunign with simple heuristic for the late game. I found that the greedy agent with the learned heuristic performed better than the alphabeta pruning agent with simple heuristic. I found that this perfomred better than each agent on its own, both time wise and performance wise. This makes sense because the learned heuristic should be able to make good decisions in the early game when there are more possible actions and the game is less deterministic, while the alphabeta agent with simple heuristic should be able to make good decisions in the late game when there are fewer possible actions for the agent to explore. 




Tell us about your implementation!

Answer the conceptual questions!

Hours taken:

Collaborators:

Known Bugs:

AI Use Description:
