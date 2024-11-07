from nn import *
import numpy as np

if __name__ == "__main__":
  l1 = Layer(3, 3)
  l2 = Layer(3, 2)
  print(l1)
  print(l2)
  nn = Network([l1, l2])
  network_input = np.array([1,1,1])
  print(nn.forward(network_input))
  