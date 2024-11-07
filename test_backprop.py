

"""

Steps:

1. Run the forward pass for the current CartPoleState
2. Select an action based on the Network output
3. Act on the Network output to get the next state
4. Calculate the target Q-value
5. Calculate the loss and update the network weights to minimize it. 

"""


if __name__ == "__main__":
  