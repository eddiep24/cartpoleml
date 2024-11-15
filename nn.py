import numpy as np

LOSS_FUNCTION = "relu"

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

class Layer:
    def __init__(self, d1, d2):  # input_dim, output_dim
        self.d1 = d1
        self.d2 = d2
        limit = np.sqrt(2.0 / (d1 + d2))
        self.layer = np.random.uniform(-limit, limit, (d1, d2))
        self.hidden_layer = None
        
    def __repr__(self):
        return f"Layer(d1={self.d1}, d2={self.d2}, layer=\n{self.layer})"
        
    def __call__(self, inputs):
        """Forward pass through the layer"""
        # Convert inputs to correct shape without forcing batching
        if isinstance(inputs, list):
            inputs = np.array(inputs, dtype=np.float64)
        self.inputs = inputs
        
        # Compute forward pass
        self.z = np.dot(self.inputs, self.layer)
        
        if LOSS_FUNCTION == "sigmoid":
            self.hidden_layer = sigmoid(self.z)
        elif LOSS_FUNCTION == "relu":
            self.hidden_layer = relu(self.z)
        else:
            raise ValueError("Undefined Activation Function")
            
        return self.hidden_layer
    
    def set_weights(self, new_layer):
        """Set weights manually (for testing)"""
        self.layer = np.array(new_layer, dtype=np.float64)
        
    def parameters(self):
        return self.layer.flatten().tolist()
        
    def backward(self, dE_dy, learning_rate):
        """
        Backward pass computing gradients using chain rule
        """
        # Step 1: Calculate dy/dz (derivative of activation function)
        if LOSS_FUNCTION == "sigmoid":
            dy_dz = sigmoid_derivative(self.z)
        elif LOSS_FUNCTION == "relu":
            dy_dz = relu_derivative(self.z)
            
        # Step 2: Calculate dE/dz = dE/dy * dy/dz
        dE_dz = dE_dy * dy_dz
        
        # Step 3: Calculate dE/dw = x * dE/dz
        # Reshape inputs for matrix multiplication if necessary
        if len(self.inputs.shape) == 1:
            inputs_reshaped = self.inputs.reshape(-1, 1)
            dE_dz_reshaped = dE_dz.reshape(1, -1)
            dE_dw = np.dot(inputs_reshaped, dE_dz_reshaped)
        else:
            dE_dw = np.dot(self.inputs.T, dE_dz)
        
        # Step 4: Update weights
        self.layer -= learning_rate * dE_dw
        
        # Step 5: Calculate dE/dx for previous layer
        dE_dx = np.dot(dE_dz, self.layer.T)
        
        return dE_dx

class Network:
    def __init__(self, layers, learning_rate=0.05):
        self.layers = layers
        self.learning_rate = learning_rate
        
    def forward(self, network_input):
        """Forward pass through entire network"""
        self.network_input = network_input
        output = network_input
        
        for layer in self.layers:
            output = layer(output)
            
        return output.flatten()  # Flatten output for compatibility
        
    def compute_loss(self, predicted_output, target_output):
        """Calculate Mean Squared Error (MSE) loss"""
        return np.mean((predicted_output - target_output) ** 2)
        
    def parameters(self):
        all_params = []
        for layer in self.layers:
            all_params.extend(layer.parameters())
        return all_params
        
    def backward(self, target_output):
        """Backward pass through network using chain rule"""
        # Get the predicted output (need to reshape if necessary)
        predicted = self.layers[-1].hidden_layer
        if len(predicted.shape) == 1:
            predicted = predicted.reshape(1, -1)
        target_output = np.array(target_output).reshape(predicted.shape)
        
        # Calculate initial gradient (dE/dy for MSE)
        dE_dy = 2 * (predicted - target_output) / np.prod(target_output.shape)
        
        # Backpropagate through each layer
        for layer in reversed(self.layers):
            dE_dy = layer.backward(dE_dy, self.learning_rate)
            
    def __repr__(self):
        return f"Network(layers={self.layers})"