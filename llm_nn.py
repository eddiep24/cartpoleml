import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class Layer:
    def __init__(self, d1, d2):  # cols, rows
        self.d1 = d1
        self.d2 = d2
        # Xavier initialization instead of random * 0.1
        limit = np.sqrt(2.0 / (d1 + d2))
        self.layer = np.random.uniform(-limit, limit, (d1, d2))
        self.hidden_layer = None
        
    def __call__(self, inputs):
        self.inputs = np.array(inputs, dtype=np.float64)
        self.z = np.dot(self.inputs, self.layer)
        # Add gradient clipping to prevent explosion
        self.z = np.clip(self.z, -10, 10)
        self.hidden_layer = relu(self.z)
        return self.hidden_layer

    def backward(self, doutput, learning_rate):
        # Clip incoming gradients
        doutput = np.clip(doutput, -1, 1)
        
        # Compute gradients with clipping
        doutput = doutput * relu_derivative(self.z)
        dweights = np.dot(self.inputs.T, doutput)
        
        # Clip weight updates
        dweights = np.clip(dweights, -1, 1)
        
        # Update weights with smaller learning rate
        self.layer -= learning_rate * dweights
        
        # Clip and return gradients for previous layer
        return np.clip(np.dot(doutput, self.layer.T), -1, 1)

class Network:
    def __init__(self, layers, learning_rate=0.01):  # Reduced default learning rate
        self.layers = layers
        self.learning_rate = learning_rate
    
    def forward(self, network_input):
        output = network_input
        for layer in self.layers:
            output = layer(output)
        # Clip final output
        return np.clip(output.flatten(), -10, 10)
    
    def backward(self, target_output):
        # Clip the initial error
        doutput = np.clip(self.layers[-1].hidden_layer - target_output, -1, 1)
        
        for layer in reversed(self.layers):
            doutput = layer.backward(doutput, self.learning_rate)