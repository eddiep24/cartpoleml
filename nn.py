import numpy as np



def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class Layer:
  def __init__(self, d1, d2):  # cols, rows
    self.d1 = d1
    self.d2 = d2
    # Without this initialization the weights converge toward infinity
    limit = np.sqrt(2.0 / (d1 + d2))
    self.layer = np.random.uniform(-limit, limit, (d1, d2)) # Xavier Initialization
    self.hidden_layer = None
      
  def __repr__(self):
    return f"Layer(d1={self.d1}, d2={self.d2}, layer=\n{self.layer})"
      
  def __call__(self, inputs):
    """
    Assuming the input has a two-dimensional shape
    """        
    self.inputs = np.array(inputs, dtype=np.float64)  # Convert inputs to NumPy array
    self.z = np.dot(self.inputs, self.layer)  # Linear transformation
    self.z = np.clip(self.z, -10, 10)
    self.hidden_layer = relu(self.z)  # Apply activation function and store it
    return self.hidden_layer

  def set_weights(self, new_layer):
    self.layer = np.array(new_layer, dtype=np.float64)  # Ensure the weights are float64
      
  def parameters(self):
    return self.layer.flatten().tolist()
      
  def backward(self, doutput, learning_rate):
    """
    Compute gradients and update weights using backpropagation.
    Uses MSE for Error and RELU Activation function. The 
    
    Parameters:
    - doutput: previous dE/dW
    
    Returns:
    - dE/dW
    
    Example of an implementation from the exam question 6:
      
    For W2:
  
      ∂L/∂W2 = ∂L/∂h2 * ∂h2/∂z2 * ∂z2/∂W2
      Where:
      - ∂L/∂h2 = (y_pred - y_true)       [MSE derivative]
      - ∂h2/∂z2 = relu'(z2)              [ReLU derivative]
      - ∂z2/∂W2 = h1                    [derivative of matrix multiplication]

      - h2 is the final output (relu(z2))
      - L is Loss function
      - z2 is preactivation of output layer
      - z1 is preactivation of hidden layer
      - h1 is hidden layer output (relu(z1))
      Therefore:
      ∂L/∂W2 = h1.T @ [(y_pred - y_true) * relu'(z2)]

    For W1:
      
      ∂L/∂W1 = ∂L/∂h2 * ∂h2/∂z2 * ∂z2/∂h1 * ∂h1/∂z1 * ∂z1/∂W1 = ∂L/∂W2 * ∂h1/∂z1 * ∂z1/∂W1

      Where:
      - ∂L/∂W2                          [Previous Gradient]
      - ∂h1/∂z1 = relu'(z1)             [ReLU derivative]
      - ∂z1/∂W1 = x                     [derivative of matrix multiplication]

      Therefore:
      ∂L/∂W1 = x.T @ [((y_pred - y_true) * relu'(z2)) @ W2.T * relu'(z1)]

    """
    doutput = np.clip(doutput, -1, 1)
    # Compute gradient of loss with respect to weights (chain rule)
    doutput = doutput * relu_derivative(self.z)  # doutput = ∂h1/∂z1 * ∂L/∂W2
    doutput = np.clip(doutput, -1, 1)
    dweights = np.dot(self.inputs.T, doutput)  # dweights = doutput * ∂z1/∂W1
    dweights = np.clip(dweights, -1, 1)
    
    # Update weights
    # print("Previous layer:\n{}".format(self.layer))
    self.layer -= learning_rate * dweights
    # print("New layer:\n{}".format(self.layer))
    return np.dot(doutput, self.layer.T)  # Return gradient for the previous layer


"""
Usage:
Instatiate layers and their dimensions, ex:
layer1 = Layer(2, 2) creates a 2 x 2 layer with four weights
layer2 = Layer(2, 4) creates a 2 x 4 layer with eight total weights

Then connect them with the network (put layers in an array):
network = Network( [layer1, layer2] )

Run a forward pass:
netowork.forward(input_array)
"""
class Network:
  def __init__(self, layers, learning_rate=1):
    self.layers = layers
    self.learning_rate = learning_rate
      
  def forward(self, network_input):
    self.network_input = network_input

    # Start forward propogation
    output = network_input
    for layer in self.layers:
        output = layer(output)
    return np.clip(output.flatten(), -10, 10)  # Return flattened output for single input
      
  def compute_loss(self, predicted_output, target_output):
    """Calculates Mean Squared Error (MSE) loss."""
    return np.mean((predicted_output - target_output) ** 2)

  def parameters(self):
    all_params = []
    for layer in self.layers:
        all_params.extend(layer.parameters())
    return all_params
      
  def backward(self, target_output):
    """
    Backprop presupposes knowledge of activation function, weights, hidden layers, all of which is known.
    """
    self.target_output = target_output
    # Calculate the error (loss) between the predicted values and target values
    doutput = self.layers[-1].hidden_layer - self.target_output
    
    # Backpropagate
    for layer in reversed(self.layers):
      doutput = layer.backward(doutput, self.learning_rate)

  def __repr__(self):
    return f"Network(layers={self.layers})"