# CSB-Q-Learning

Since the new year we have been attempting reinforcement learning on Coders Strike Back (CSB) again. This time we have had a lot more success and reached the number 1 spot on the leaderboard, achieving close to 100% winrate against Pen's excellent AI which stood unchallenged for long. We are thrilled with this achievement and the fact that we have inspired other players like fenrir to pursue the same goal. With the Xmas Rush contest, players brought MCTS and simultaneous-move handling techniques which had been previously unseen on Codingame (CG). And similarly with this work we hope to bring new techniques of reinforcement learning on the platform.

1- Ré-expliquer comment on a fait un runner simple, ça permet de ré-introduire les bases du Q-learning (évaluer chaque action + lister les raffinements utilisés come PER IS, double, etc..)

## Neural Networks

In this article we assume some knowledge of neural networks. The neural network is a machine learning technique among many others (SVM, Random Forest, Gradient Boosting...) which is particularly powerful if you have enough data to avoid overfitting and lead to the era of so-called deep learning. A sufficiently large neural network can learn to mimic any function from inputs to outputs, given enough (input,output) examples. We used the "vanilla" flavor of dense feed-forward neural networks with leaky rectified linear unit activations (leaky relu) and minibatch gradient descent with momentum. To learn about neural networks we recommend online courses, video series, blog posts and spamming @inoryy.

## Q Learning

Q learning is originally a tabular reinforcement learning technique on games with a finite number of states and actions. The value of a state is typically denoted V, this would be the evaluation function a lot of CG players are used to. Q(state,action) is instead the evaluation of playing an action from a state. Q represents the discounted sum of future rewards assuming play is continued according to perfect play:
```
Q(state,action)=reward_1+γ*reward_2+γ^2*reward_3+...
```
with γ<=1 a discount factor. It is allowed for γ to be 1 in finitely long games, but for inifinitely long games γ<1 otherwise Q values are infinite. γ also serves to make the AI collect rewards sooner rather than later as future rewards are discounted.

The goal of Q learning is to learn these Q values, corresponding to perfect play and then play according to these perfect (state,action) values (e.g: in every state greedily choose the action with the highest Q value). Tabular Q learning starts with a randomly initialized grid of size N\_States\*N\_Actions and iteratively converges to the true Q values by playing the game and applying the Bellman equation to every (state,action)->(next_state) transition the AI experiences:
```
Q(state,action)=immediate_reward+γ*maxQ(next_state,action)
```
In 2015 Deepmind published the Deep Q learning [paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) (DQN). Instead of having Q learning restricted to games with a finite number of states and actions they generalise it to infinite games by using a Neural Network (NN) as a function approximator for Q values. Thus instead of looking up Qs in a table, you would feed in a representation of the state action pair and the network would output the Q value. For computational efficiency they restricted themselves to games with finite actions and the network outputs the Q values for all actions of a given state in one forward pass. They call this NN a Deep Q Network (DQN).

If you read the paper carefully you will find that they use several techniques to achieve convergence: 

* Gradient clipping to avoid destroying the network by backpropagating huge gradients on transitions the network really doesn't understand.
* Experience replay memory to store transitions and backpropagate gradients from a mixture of different states. As opposed to only learning from the current transition.
* Freezing a copy of the network to be used as a "target network" for the right side of the Bellman equation. Said copy is updated to the current weights of the network every time a target number of training steps is reached. `DQN(state,action):=immediate_reward+γ*maxDQN_Frozen(next_state,action)`

Deepmind has since published several papers improving upon DQN. For example Deep Double Q learning (DDQN) where, on the right side of the Bellman Equation instead of taking the maximum over actions, the action that the current network would have chosen is taken as target.
```
target_action=argmaxDQN(next_state,action)
DQN(state,action):=immediate_reward+γ*DQN_Frozen(next_state,target_action)
```
See the [paper](https://arxiv.org/pdf/1509.06461.pdf) for a more detailed explanation.

A major improvement we also used is prioritised experience replay where instead of selecting memories uniformily at random from the memory, transitions which are most misunderstood by the network have a higher probability of being selected. When transitions are experienced they are added to memory with a high priority, to encourage that transitions are learned from at least once. And when a transition is selected its Temporal Difference Error (TD Error) is measured and its priority is updated as:
```
prio:=epsilon_prio+TD_Error^prio_alpha
```
where the TD Error is the error on the supposed equality in the Bellman equation. We used the proportional prioritisation variant mentioned in the [paper](https://arxiv.org/pdf/1511.05952.pdf). This used a sum-tree data structure in order to be able to select samples according to priority in logarithmic time.

? - Mentionner tous les hacks techniques pour faire passer un NN sur CG?

## Training a blocker against a fixed opponent

Q-Learning was described above as a value-based technique where an agent learns a value function over all (State, Action) pairs by interacting with its environment.

If the runner is considered as part of the environment, a blocker can be trained can be trained with the same Q-learning framework that was described for the runner alone.

Rewards are defined as follows:
 - -1 when the runner takes a checkpoint
 - 0 otherwise

### Limitations
While this would be the perfect method to specialize against a specific bot on the leaderboard, one can not consider that the blocker will learn a good policy overall.
Indeed, with this approach the blocker learns an optimal policy against a "dumb" runner which never learns to improve.

### Results
This approach was used in 2017 in the following manner:

    (1) Train a runner alone
    (2) Train a blocker against (1)
    (3) Train a runner against  (2)
    (4) Train a blocker against (3)

A bot was submitted on the leaderboard combining the runner (3) and the blocker (4), it reached rank 30 in legend league.

Other agents were trained by repeating more iterations, but those agents performed worse overall on the leaderboard.

    (5) Train a runner against  (4)
    (6) Train a blocker against (5)
Our understanding is that blocker (6) has learned to block a runner which tries to go around the blocker. Blocker (6) was never placed in an environment against runner (1) which goes straight to the checkpoint: hence the blocker's unability to obstruct runner (1)'s path.


## Simultaneous runner/blocker training
### Failed attempt #1
#### Description
As discussed above, the main limitation we were faced with originated from the fact that only one agent learned to improve its policy from the environment while the other agent had a fixed policy.
We created an environment with which two neural networks interacted: one controlled the blocker, the other one controlled the runner.
In a Q-learning framework, each agent predicted the expected future discounted reward for the 6 actions it was allowed to perform.
Our hope was to train both agents simultaneously so that they would converge towards an optimal adversarial strategy.
#### Results
Upon training and submitting the code on CG servers, the results obtained were disappointing.
Our implementation may have been lackluster, as discussions with people familiar in the field have shown that "*it should have worked.*"

### Failed attempt #2
#### Description
Same as above, but with one neural network with 12 output values instead of two neural networks with 6 output values each.
#### Results
Same as above. We made two completely independent implementations of this technique with bad results in both cases.




3- Décrire notre approche avec Nactions^2, MatrixSolver
Link l’article

4- Trois mots expliquant les rangs atteints en depth 0, 1, 2, MCTS



5- Ouverture sur le fait qu’il existe d’autres algos, illustrer avec ce que fait fenrir
Alpha Zero A2C
