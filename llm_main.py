from nn import *
from cartpole import *

NUM_EPISODES = 100
LEARNING_RATE = 0.01  # Reduced learning rate
DISCOUNT_FACTOR = 0.99  # Slightly reduced discount factor

def main():
    # Simplified architecture
    layer1 = Layer(4, 64)    # Reduced from 128
    layer2 = Layer(64, 32)   # Smaller intermediate layers
    layer3 = Layer(32, 2)    # Output layer
    
    network = Network([layer1, layer2, layer3], learning_rate=LEARNING_RATE)
    env = CartPoleEnvironment()
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            state_array = state.to_array().reshape(1, -1)
            action_probabilities = network.forward(state_array)
            action = np.argmax(action_probabilities)
            
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # Scale reward to prevent large updates
            scaled_reward = reward * 0.01
            
            target_output = np.zeros_like(action_probabilities)
            if done:
                target_output[action] = scaled_reward
            else:
                next_state_array = next_state.to_array().reshape(1, -1)
                next_q_values = network.forward(next_state_array)
                target_output[action] = scaled_reward + DISCOUNT_FACTOR * np.max(next_q_values)
            
            network.backward(target_output)
            state = next_state
            
        print(f"Episode {episode+1}/{NUM_EPISODES} - Total Reward: {total_reward}")

if __name__ == "__main__":
    main()