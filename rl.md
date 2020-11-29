## Reinforcement Learning
Reinforcement learning and evolutionary algorithms both choose the action with the highest expected rewards. The difference comes is in how they improve. Reinforcement learning tries to estimate rewards based on previous experiences. In essence, this algorithm tries to predict the outcome if the agent takes a certain action. After the agent "explores" the world a bit, it can build up a "memory" of environments, actions, and rewards (outcomes). 

![Reinforcement Learning](https://miro.medium.com/max/434/1*n1AZU6IkpfjC0l22Md2x0Q.png)

It can then train to better react to the situations it has seen by modifying the parameters (weights) of its neural network through techniques such as [gradient descent](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html). Gradient descent tries to find the configuration of weights that produces the lowest cost, or error. 

![Gradient Descent](https://ml-cheatsheet.readthedocs.io/en/latest/_images/gradient_descent_demystified.png)

However, just looking at the immediate reward would make the algorithm very short-sighted. To fix this issue, this algorithm also takes in account long term effects. This is achieved by including the expected rewards for actions taken at a later time in the current reward. When these later rewards are considered, they are multiplied by a factor, usually called gamma (γ), between 0 and 1 (usually around 0.95) to make the algorithm focus on short term effects but also take into account long term effects. These adjusted rewards are called Q-values. For each state and action, there is a corresponding Q-value.

![Q Learning](https://cdn-media-1.freecodecamp.org/images/s39aVodqNAKMTcwuMFlyPSy76kzAmU5idMzk)

Due to the amount of states an environment can take on, storing all the Q-values is usually impossible. This is where the neural network comes in. The neural network tries to approximate these Q-values by changing its weights, eventually becoming a network that can accurately predict the rewards for each state and action. 

![Deep Q Learning](https://cdn-media-1.freecodecamp.org/images/1*Zplt-1wTWu_7BGmZCBFjbQ.png)

[back](./)
