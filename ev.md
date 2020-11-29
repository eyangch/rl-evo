## Evolutionary Algorithms

Evolutionary algorithms use randomness to become good at a task instead of math. It is very similar to the evolution you learn in science, where the most fit survive. At first, a bunch of agents are initialized with random weights. Then, the few that perform the best (highest reward) are selected to form the "parents" of the next generation. These parents then each produce offspring with slightly modified weights. As time passes, the agents will become better and better at their task, closely resembling evolution in the real world.

<img src="https://eng.uber.com/wp-content/uploads/2017/12/Header-1.png" alt="AI Evolution" width="700"/>

As you can tell, this algorithm is a lot simpler than reinforcement learning, although it requires a lot more simulation due to the number of generations required to pass before the population will significantly improve. However, evolutionary algorithms are also much faster at simulating, as they don't require lots of complex math. 

[back](./)
