from nn import *
from cartpole import CartPoleEnvironment  # Import your environment module

NUM_EPISODES = 100

if __name__ == "__main__":
  env = CartPoleEnvironment()
  
  
  # In this implementation their layers were 1 -> 128 -> 256 -> 512 -> 256 -> 128 -> 2 
  # https://pythonprogramming.net/openai-cartpole-neural-network-example-machine-learning-tutorial/
  l1 = Layer(4, 128)  # CartPole has 4 input features
  l2 = Layer(128, 256)
  l3 = Layer(256, 128)
  l4 = Layer(128, 2)  # Output is [left, right]
  nn = Network([l1, l2, l3, l4])

  for episode in range(NUM_EPISODES):
    state = env.reset() # Start fresh
    done = False
    total_reward = 0

    while not done:
      state_array = state.to_array()
      action_values = nn.forward(state_array)  # Forward pass
      
      # Select an action based on action_values
      action = np.argmax(action_values)
      
      # Take the chosen action
      state, reward, done = env.step(action)
      total_reward += reward

    print(f"Episode {episode + 1} ended with total reward: {total_reward}")
