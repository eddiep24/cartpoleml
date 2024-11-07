import numpy as np

def relu(x):
    return np.maximum(0, x)

class Layer:
  def __init__(self, d1, d2):  # cols, rows
    self.d1 = d1
    self.d2 = d2
    self.layer = np.random.rand(self.d1, self.d2)

  def __repr__(self):
    return f"Layer(d1={self.d1}, d2={self.d2}, layer=\n{self.layer})"

  def __call__(self, inputs):
    return np.dot(inputs, self.layer)

  def parameters(self):
    print("Layer parameters: ", self.layer.flatten().tolist())
    return self.layer.flatten().tolist()


class Network:
  def __init__(self, layers):
    self.layers = layers

  def forward(self, network_input):
    output = network_input
    for layer in self.layers:
      output = relu(layer(output))
    return output
  
  def parameters(self):
    all_params = []
    for layer in self.layers:
      all_params.extend(layer.parameters())
    return all_params
  
  def __repr__(self):
    return f"Network(layers={self.layers})"