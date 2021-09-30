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
