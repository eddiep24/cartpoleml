import numpy as np

LOSS_FUNCTION = "sigmoid"


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


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
        self.inputs = np.array(inputs, dtype=np.float64)  # Ensure inputs are in float64
        self.z = np.dot(self.inputs, self.layer)

        # Apply activation function
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
        # Derivative of activation function
        if LOSS_FUNCTION == "sigmoid":
            dy_dz = sigmoid_derivative(self.z)
        elif LOSS_FUNCTION == "relu":
            dy_dz = relu_derivative(self.z)
        #print("dE_dy", dE_dy)
        #print("dy_dz",dy_dz)
        dE_dz = dE_dy * dy_dz  # Gradient for the layer output
        #print("dE_dz", dE_dz)
        # Gradient with respect to weights
        dE_dw = np.dot(self.inputs.reshape(-1, 1), dE_dz.reshape(1, -1))
        #print("dE_dw", dE_dw)
        dE_dz_next=np.dot(dE_dz, self.layer.T)
        #print("self.layer", self.layer)
        # Update weights
        self.layer -= learning_rate * dE_dw
        #print("self.layer",self.layer)
        # Gradient for the previous layer (to backpropagate further)
        return dE_dz_next


class Network:
    def __init__(self, layers, learning_rate=0.0002):
        self.layers = layers
        self.learning_rate = learning_rate

    def forward(self, network_input):
        output = network_input
        for layer in self.layers:
            output = layer(output)
        return output.flatten()  # Flatten output for compatibility

    def compute_loss(self, predicted_output, target_output):
        return np.mean((predicted_output - target_output) ** 2)

    def parameters(self):
        all_params = []
        for layer in self.layers:
            all_params.extend(layer.parameters())
        return all_params

    def backward(self, target_output):
        # Initial error gradient for MSE loss
        predicted = self.layers[-1].hidden_layer
        target_output = np.array(target_output).reshape(predicted.shape)

        dE_dy = 2 * (predicted - target_output) / np.prod(target_output.shape)

        # Backpropagate through each layer
        for layer in reversed(self.layers):
            #if dE_dy.shape == (2, 1):
            dE_dy = layer.backward(dE_dy, self.learning_rate)


    def __repr__(self):
        return f"Network(layers={self.layers})"


    def train_policy_gradient(self, state, action, discounted_reward):
        """
        Train the network using policy gradient method.
        
        Args:
            state: Current state input
            action: Action taken
            discounted_reward: Discounted reward for the action
        """
        # Forward pass to get action probabilities
        action_probs = self.forward(state)
        action_probs = softmax(action_probs)

        # Compute policy gradient
        target = np.zeros_like(action_probs)
        target[0, action] = 1.0
        
        # Scale gradients by the discounted reward
        gradient = (target - action_probs) * discounted_reward
        
        # Update network weights using gradient ascent
        self.backward(gradient)

    def get_action(self, state):
        """
        Get an action from the current policy.
        
        Args:
            state: Current state input
            
        Returns:
            action: Selected action
            action_probs: Probability distribution over actions
        """
        action_probs = self.forward(state)
        action_probs = softmax(action_probs)
        action = np.random.choice([0, 1], p=action_probs.ravel())
        return action, action_probs