from nn import *
from cartpole import *

NUM_EPISODES = 100
LEARNING_RATE = 0.1

# Checking my backpropogation logic and verifying through this example:
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

if __name__ == "__main__":
  l1 = Layer(2, 2)
  l1_arr = np.array([[1,1],[1,1]])
  l1.set_weights(l1_arr)
  l2 = Layer(3, 2)
  l2_arr = np.array([[1,1],[1,1]])
  l2.set_weights(l2_arr)

  nn = Network([l1, l2])

  nn.forward(network_input=[1,1])
  
  print(nn.backward(target_output=[1,0]))