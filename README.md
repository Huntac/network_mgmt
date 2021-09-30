# Network_mgmt

Repo to learn and try various methods for solving network management

## Network Summary

network management environment represents a multi-echelon supply network with 4 tiers:  

1. Market Nodes (sink node)  
2. Retail Nodes
3. Distributor Nodes  
4. Manufacturer Nodes  
5. Raw Material Nodes (source node)

At each step the following sequence occurs:  
1. Nodes place replenishment orders to their immediate suppliers. These orders are filled according to supplier inventory or capacity and are associated with cost.  
2. Orders scheduled to arrive at each node are recieved
3. Market node demand is realized
4. Backlog or good-will penalties are inurred
5. Holding cost is incurred according to each node's remaining inventory in addition to ordered inventory that has yet to arrive

### Network Details
#### Market Layer
Market nodes are only connected to retail nodes and serve as sink nodes for the network. Each edge between a market node and retail node has the following parameters:  
- a unit price for units sent via the edge
- a backlog cost or good will loss for unfullfilled demand
- a demand distribution

At each step each of the market node's edges will calculate demand based on it's demand distribution. 
This demand is filled occording to the available inventory at the connected retail node.
If backlogs are enabled the unfulfilled sales will be added to the next period's demand and penalized. Otherwise unfulfilled sales are lost with a good-will penalty.  

#### Retail Layer
Retail nodes are connected to market nodes and distributor nodes. Retail nodes order units from distributor nodes and fulfill demand from market nodes. Each retail node has the following parameters:  
- inventory available for purchase by by the down stream layer
- a holding cost  

Each edge between a retail node and a distributor node has the following parameters:  
- lead time
- a unit price for units sent via the edge
- a backlog cost or good will loss for unitsthe distrbutor was unable to provide the retailer

#### Ditribution Layer
Distribution nodes are connected to retail nodes and manufacturer nodes. Distribution nodes order units from manufacturer nodes and fulfill orders from reail nodes.

Distrbutor nodes have the same node attributes as retail nodes, and distribtor to manufacture edges have the same edge attributes as retail to distributor edges.  

#### Manufacturing Layer
Manufacturing nodes are connected to distribution nodes and natural resource nodes. Manufacturing nodes order resources from natural resource nodes and fulfill orders from distribution nodes. 
Manufacturing nodes have the following parameters:
- inventory of raw materials to conver into purchasable units
- a capacity of units created by manufacturing facility in each time step
- a yield rate of raw materials to purchasable units 
- a unit operating cost
- a holding cost

Manufacturing nodes are limited by their capacity in fulfilling orders from distribution nodes. When an order is placed to a manufacturing node,
the manufacturing node will supply the minimum of its capacity and it's yield rate times its raw material inventory.

## DDPG Approach

The DDPG (deep deterministic policy gradient) agent uses the actor-critic framework to learn maps from a given state to a given action via the actor network and maps from a given state and action to a reward via the critic network. DDPG agents interact with an environment in discrete time steps within a given episode of a simulation. Experiences from each time step are stored in a memory bank and a combination of observed states, previous actions, and observed rewards. Over time the agent uses it's memory banks to train itself on how to interact with the environment such that reward form the environment is maximized if the agent is successful.  

### Technical Details
#### Critic Network
The critic network is trained using a loss function comparing a target reward value to predicted reward value which the critic network predicts from the observed state and action that resulted in the observed reward value. The target reward value is calculated by adding the observed reward value to the discounted future reward value of following time steps in an episode unless the observed reward value came from the final time step in an episode. While the discounted future reward value is not observed during simulation, it can be estimated by taking the observed state and using the actor network to generate a set of next actions and then evaluating those next_actions using the critic network. The sum of the observed reward value plus the estimated future reward value is compated to the critic network's predicted reward value from the observed state and action to calculate the critic network's loss. The critic network optimizer then uses the gradient of that loss w.r.t the critic network's parameter 

#### Actor Network
The actor network is trained by feeding an observed state through the actor network which yields predicted actions. Next the observed sate and the predicted actions are feed into the critic network which predicts the expected reward based on the observed state and predicted actions. Finally the actor network optimizer uses the gradient of the negative expected reward w.r.t the actor network's parameters to improve the actor networks output such that the action maximizes the critics expected reward.

#### Local and Target Networks
Both the actor network and the critic network are fit w.r.t the output of the other model, which can cause a very unstable learning environment. Intuitively this is like trying to hit a moving target. One method of dealing with this problem is using a local and target network for the actor and critic. In this framework, you have local actor network, a target actor network, a local critic network and a target critic network where the local and target networks are initialized with the exact same set of weights. The target networks are used in the loss calculation for the local networks and the local network weights are adjusted based on the loss calculation. Finally the local weights are soft copied to the target networks using the parameter tau. For example when the local actor network is trained the local actor's actions are graded by the target critic network. Then after the local actor network's weights have been updated the target actor network is soft updated where it's new weights are equal to (1-tau x target weights) + (tau x local weights). This framework causes the agent to learn in a slow but stable way.
