# DeepQ-RL_Trading_Agent

## Overview

In financial trading, optimizing investment strategies involves managing risk and maximizing returns. Deep Q-Learning, a reinforcement learning technique, provides a powerful method for developing trading strategies by learning optimal actions from historical data. This approach helps balance the trade-off between risk and return, which is crucial for constructing an effective portfolio.

### Risk-Return Trade-Off

Higher returns typically come with increased risk. Deep Q-Learning enables the development of strategies that aim to optimize this balance, leveraging historical data to make informed decisions and manage risk.

### Portfolio Management

Effective portfolio management involves selecting the right mix of assets to achieve the desired balance between risk and return. By using Deep Q-Learning, we can simulate different trading strategies and assess their impact on portfolio performance. The goal is to create a robust portfolio that maximizes returns while adhering to the risk tolerance set by the investor.

## Dataset

The dataset contains historical data for NIFTY50 stocks from January 2010 to 2019, sourced from [Yahoo Finance](https://finance.yahoo.com/quote/%5ENSEI/history?period1=1262304000&period2=1559347200&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true). The data was cleaned thoroughly, using the daily opening prices as representative values.

## State Representation

The state representation function generates a state based on the dataset, the current day, and a window size (set to 5 as a hyperparameter). This function captures the relative changes in stock prices over the defined window, focusing on recent price movements. It calculates the price differences between consecutive days and normalizes the values using a sigmoid function to ensure they fall between 0 and 1.

## Actions Representation

For this strategy, we are considering three actions: 
- `0`: Hold
- `1`: Buy
- `2`: Sell


## Reward function
Reward function is a simple representation of the profit that we gain due to the actions we take:
$$\ reward=max(sellprice-buyprice,0)$$
(Once again only positive values are only considered in a reward)

## Deep Q Learing in trading
We are using Deep Q learning algorithm to create the RL agent and then train it. The basic objective of a Q-learning/Deep Q-learning  is to maximize the returns $\ G_t$ i.e the net reward inn a long run. In Q_learning we calculate Q_value which is the expected Return at a given state for a given action,
$$\ Q(s,a) = E[G_t|S_t=s,A_t=a] $$ <br>
This Q_value determines what action should be taken based on the maximum value of Q.<br>
In deep Q Learning, deep neural network is used to approximate the values of Q
![photo_2023-07-20_13-17-13](https://github.com/rakshith-2100/Deep-Q-Learning-Trading-Agent/assets/99346822/d609477e-a4a5-4192-8dc3-fc07cac04307) <br>
This image explains the working of the neural network and also tells us about the q_value that we get as an output.
The Q_value from the neural network is then compared with the Q value from the **Bellman equation**<br>
$$\ Q(s,a)=R_t(s,a)+γmax_{a'}(Q'(s',a')) \$$ <br>
The Bellman equation <br>
In the equation $\ s$ is the current state and $\ a\$ is the action taken and $\ s'$ is the next state and $\ R(s,a)$ is the reward that we get after doing action a and $\ γ$ here is the discount factor which is a hyperparameter(*in the code γ=0.95*)that tells how valuable is a reward which we get in a future state and $\ a'$ is the set of actions possible in the next state $\max_a'(Q'(s',a'))$ and  determines the maximum Q_value possible for the next state which is also calculated from the neural network.<br>
The Q_value that we get from the bellmen equation would serve as the target Q_value, so we call it as **Q_target** i.e $\ Q'(s,a)$  so the loss computed would be the mean square difference between q_value and q_target 
$$\ L(θ)=1/N\sum_{i∈N}(Q_θ(S_i,A_i)-Q'_θ(S_i,A_i))^2\$$ <br>
This loss would be used to fine tune the model i.e update weights $\ θ$ using Stocastic Gradient Descent optimizer(SGD)
$$\ θ <- θ-α\frac{∂L}{∂θ}$$

### Training a Deep Q model
Before training we look into the training of the model we have to know two concepts that play a major role in trainng, **epsilon greddy strategy** and **replay memory** .

#### Epsilon Greedy Strategy
This is used to maintain a balance in **exploration** and **exploitation**. Exploration basically means that the agent is trying to explore all the market conditions and try to learn more about the market and exploitation means that the agent is trying to exploit the market conditions thus maximizing the rewards. There should be a balance between exploration and exploitation such that our agent will be prepared for any situations and parallely maximize profits to any situation, hence we use epsilon greedy strategy. Initially the agent  should explore to know  more about the enivironment and learn important details about the market and slowly it should start exploiting to maximize returns to do this, we define **ε**( Epsilon ), the probablity that the agent would explore the environment rather than exploit it. This is initially set to 1 and this epsilon decays by some rate(which is a hyperparameter) as the agent becomes greddy with time. We choose a random number r ranging between 0 and 1 such that if $\ (r>ε)$ it would choose exploitation i.e choose action based on the maximum Q_value that we get from the model and if $\ (r<ε)$ then random actions are taken sure that agent can explore the environment<br>
*( This can be seen in a function called trade from the class called ModelandTrade in the code )*

#### Replay Memory
Before we know about replay memory we should know about **experience**( $\ e_t$). The agents experience is defined as a tuple
$$\ e_t=(s_t,a_t,r_{t+1},s_{t+1})$$
$\ s_t$ is the state at time t $\ a_t$ is the action taken on state $\ s_t$ and $\ r_{t+1}$ is the reward after taking action $\ a_t$ on state $\ s_t$ and $\ s_{t+1}$ is the next state.<br>
We get experiences at each time step from random exploration and exploitation using epsilon greedy strategy and all the agents experience in each timestep is stored in the replay memory.*( in the code it is called as memory )*.


### Training Process

The replay memory is used for training by sampling random experiences. The dataset is divided into batches (batch size of 32). After processing each batch, experiences are recorded, used to calculate Q-target and Q-value, and then the model is optimized. Epsilon decays over time, with 1000 episodes and an epsilon decay rate of 0.01. A boolean flag in the replay memory indicates whether an episode has ended.

## Conclusion

This Deep Q-Learning Trading Agent illustrates the application of advanced reinforcement learning techniques to financial trading. By integrating state representation, reward mechanisms, and neural network approximation, the model aims to develop effective trading strategies while managing risk and optimizing returns. Additionally, the approach to portfolio management using Deep Q-Learning allows for simulating and evaluating different strategies, contributing to a well-balanced portfolio. This project provides a comprehensive framework for leveraging machine learning in dynamic financial markets, highlighting its potential to enhance trading strategies and portfolio management.
