"""

Main implmenetation that trains a fully connected neural network using backpropogation and gets feedback via cart pole RL.

"""

from nn import *
from cartpole import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

NUM_EPISODES = 100
LEARNING_RATE = 1
DISCOUNT_FACTOR = 0.99

def main():
  # This architecture seems successful: https://pythonprogramming.net/openai-cartpole-neural-network-example-machine-learning-tutorial/
  # TODO: Could add a dropout rate like them.
  layer1 = Layer(4, 128)
  layer2 = Layer(128, 256)
  layer3 = Layer(256, 512)
  layer4 = Layer(512, 256)
  layer5 = Layer(256, 128)
  layer6 = Layer(128, 2)
  network = Network([layer1, layer2, layer3, layer4, layer5, layer6], learning_rate=LEARNING_RATE)

  # This is similar to how OpenAI implements the gym library
  env = CartPoleEnvironment()

  for episode in range(NUM_EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
      state_array = state.to_array().reshape(1, -1)  # Reshape to (1, 4) for batch processing
      
      action_probabilities = network.forward(state_array)
      action = np.argmax(action_probabilities)
      
      # Go forward one time unit in environment
      next_state, reward, done = env.step(action)
      total_reward += reward
      
      # Prepare target output for training
      target_output = np.zeros_like(action_probabilities)
      
      if done:
        target_output[action] = reward
      else:
        # Convert next_state to array and reshape
        next_state_array = next_state.to_array().reshape(1, -1)
        next_q_values = network.forward(next_state_array)
        target_output[action] = reward + DISCOUNT_FACTOR * np.max(next_q_values)
      
      # print("Current Loss", network.compute_loss(action_probabilities, target_output))
      
      # Train
      network.backward(target_output)
      state = next_state
        
    # Total reward is a decent benchmark for an increase in performance.
    print(f"Episode {episode+1}/{NUM_EPISODES} - Total Reward: {total_reward}")

#   animator = CartPoleAnimator(network, NUM_EPISODES)
#   anim = FuncAnimation(
#     animator.fig,
#     animator.animate,
#     init_func=animator.init_animation,
#     frames=None,
#     interval=20,
#     blit=True,
#     repeat=False
#   )
#   plt.show()



if __name__ == "__main__":
  main()
  