## Reinforcement Learning
Reinforcement learning and evolutionary algorithms both choose the action with the highest expected rewards. The difference comes is in how they improve. Reinforcement learning tries to estimate rewards based on previous experiences. In essence, this algorithm tries to predict the outcome if the agent takes a certain action. After the agent "explores" the world a bit, it can build up a "memory" of environments, actions, and rewards (outcomes). It can then train to better react to the situations it has seen by modifying the parameters (weights) of its neural network through techniques such as [gradient descent](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html). Gradient descent tries to find the configuration of weights that produces the lowest cost, or error. 

![Gradient Descent](https://ml-cheatsheet.readthedocs.io/en/latest/_images/gradient_descent_demystified.png)

[back](./)
