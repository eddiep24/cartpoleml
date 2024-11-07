from nn import *
import numpy as np

if __name__ == "__main__":
  l1 = Layer(3, 3)
  l1_arr = np.array([[1,1,1],[1,1,1],[1,1,1]])
  l1.set_weights(l1_arr)
  l2 = Layer(3, 2)
  l2_arr = np.array([[1,1],[1,1],[1,1]])
  l2.set_weights(l2_arr)
  print(l1)
  print(l2)


  nn = Network([l1, l2])
  network_input = np.array([1,1,1])
  print("Output of a network of 1s is", nn.forward(network_input))

  
  # l1 = Layer(4, 128)  # CartPole has 4 input features
  # l2 = Layer(128, 256)
  # l3 = Layer(256, 128)
  # l4 = Layer(128, 2)  # Output is [left, right]
  # nn = Network([l1, l2, l3, l4])
