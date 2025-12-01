**Tomas Teixeira**

Past Two Weeks:

Created a simplified observation space that relies on preprocessing to lift some of the spatial learning burden from the agent. Allowed the model to break past basic theory and increase its average steps per episode from \~10 to 100+.

The Last Week:

Record my section of the results demonstration. Additionally, write my sections of the paper, including the PPO Agent, and observation space setup.

**Zixuan Wen**

Past two Weeks:

Modify the normal Actor Critic Agent and make it compatible with the latest environment  
Add a tensorboard that used to collect data and demonstrate the training progress

The Last week:

Write the sections of the paper, including the AC Agent & comparison part

**Adrien Gergouil:**

Past two weeks:

Implemented MCAgent into newer environment. Tested different approaches of Epsilon random evaluations and reward calculations. Also training DQN Agent

The Last Week:

Writing about why MC is not a valid approach for this kind of project. Report results from DQN

**Pierson Michalski**

Past 2 weeks:

Working on implementation using each tiles state as an action instead of one big state, using a feature vector to represent every tiles action.

The Last week:

Hopefully completing that implementation tomorrow, then helping with report

**Milo Goodfellow:**

Past Two Weeks:

	I switched out my turn based policy for a score based policy and successfully made progress on training the network. The new model reached parity with the original.  
	After establishing that score works as a reward, I set out to resolve the smearing, variance and sparsity problems that discounted score induces. I landed on using actor-critic.   
The critic model tried to predict the final rewards the policy model would end up getting. The policy network was rewarded with those predicted values and a discount.  
But, the critic model never converged. It was not able to make reliable predictions about how well the policy network would perform. I'm not sure what the problem is. I expect it is almost surely something about my implementation, rather than the environment. 

The Last Week:

	Try a bit more on getting actor critic to work, and to see if another algorithm, like TD, can fix the sparsity/variance/smearing problems.  
	Beyond that, record my section of the final project demo and, having done so, work on the final report.