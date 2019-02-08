# CSB-Q-Learning

Test Markdown

1- Ré-expliquer comment on a fait un runner simple, ça permet de ré-introduire les bases du Q-learning (évaluer chaque action + lister les raffinements utilisés come PER IS, double, etc..)

? - Mentionner tous les hacks techniques pour faire passer un NN sur CG?

## Training a blocker against a fixed opponent

Q-Learning was described above as a value-based technique where an agent learns a value function above all (State, Action) pairs by interacting with its environment.

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
Same as above.




3- Décrire notre approche avec Nactions^2, MatrixSolver
Link l’article

4- Trois mots expliquant les rangs atteints en depth 0, 1, 2, MCTS



5- Ouverture sur le fait qu’il existe d’autres algos, illustrer avec ce que fait fenrir
Alpha Zero A2C
