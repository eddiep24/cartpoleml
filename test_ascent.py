from nn import *
import numpy as np
import unittest

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        # Create a simple network with known weights
        self.l1 = Layer(3, 3)
        # Initialize weights as float64
        self.l1_arr = np.array([[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]], dtype=np.float64)
        self.l1.set_weights(self.l1_arr)
        self.l1.bias = np.zeros((1, 3))  # Set bias to zero for predictable results
        
        self.l2 = Layer(3, 2)
        # Initialize weights as float64
        self.l2_arr = np.array([[1.,1.],[1.,1.],[1.,1.]], dtype=np.float64)
        self.l2.set_weights(self.l2_arr)
        self.l2.bias = np.zeros((1, 2))  # Set bias to zero for predictable results
        
        self.network = Network([self.l1, self.l2], learning_rate=0.1)
        self.test_input = np.array([1.,1.,1.], dtype=np.float64).reshape(1, -1)

    def test_forward_pass(self):
        """Test if forward pass produces expected output with known weights."""
        output = self.network.forward(self.test_input)
        expected = np.array([[.945716, .945716]])  # 3 inputs × 3 weights × 1 = 9 for each output
        print(output)
        np.testing.assert_array_almost_equal(output, expected)

    def test_softmax(self):
        """Test if softmax produces valid probability distribution."""
        test_output = np.array([[9., 9.]])
        probs = softmax(test_output)
        self.assertEqual(probs.shape, (1, 2))
        self.assertAlmostEqual(np.sum(probs), 1.0)
        np.testing.assert_array_almost_equal(probs, np.array([[0.5, 0.5]]))

    def test_policy_gradient_step(self):
        """Test if policy gradient update works as expected."""
        # Initial forward pass
        initial_output = self.network.forward(self.test_input)
        initial_probs = softmax(initial_output)
        
        # Perform a policy gradient update
        action = 0  # Choose first action
        reward = 1.0  # Positive reward
        self.network.train_policy_gradient(self.test_input, action, reward)
        
        # Check if probabilities changed in the expected direction
        new_output = self.network.forward(self.test_input)
        new_probs = softmax(new_output)
        
        # The probability of the chosen action should increase
        self.assertGreater(new_probs[0, action], initial_probs[0, action])

    def test_gradient_direction(self):
        """Test if gradients move in the correct direction for positive and negative rewards."""
        # Test with positive reward
        initial_output = self.network.forward(self.test_input)
        initial_probs = softmax(initial_output)
        
        # Positive reward case
        action = 0
        pos_reward = 1.0
        self.network.train_policy_gradient(self.test_input, action, pos_reward)
        pos_output = self.network.forward(self.test_input)
        pos_probs = softmax(pos_output)
        
        # Probability should increase for positive reward
        self.assertGreater(pos_probs[0, action], initial_probs[0, action])
        
        # Reset network
        self.setUp()
        
        # Negative reward case
        neg_reward = -1.0
        self.network.train_policy_gradient(self.test_input, action, neg_reward)
        neg_output = self.network.forward(self.test_input)
        neg_probs = softmax(neg_output)
        
        # Probability should decrease for negative reward
        self.assertLess(neg_probs[0, action], initial_probs[0, action])

if __name__ == "__main__":
  unittest.main()