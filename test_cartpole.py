import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from nn import *
from cartpole import CartPoleEnvironment

NUM_EPISODES = 100

def plot_cartpole(state):
    """ Renders the cartpole simulation using matplotlib. """
    cart_position = state.cart_position
    pole_angle = state.pole_angle

    # Create a blank figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-1, 2)

    # Create the cart (a rectangle)
    cart_width = 0.2
    cart_height = 0.1
    cart = plt.Rectangle((cart_position - cart_width / 2, -cart_height / 2), cart_width, cart_height, fc='blue')
    ax.add_patch(cart)

    # Calculate the position of the pole's tip
    pole_length = 0.5  # Length of the pole
    pole_x = cart_position + pole_length * np.sin(pole_angle)
    pole_y = pole_length * np.cos(pole_angle)

    # Create the pole
    ax.plot([cart_position, pole_x], [0, pole_y], lw=3, c='red')

    ax.set_title(f"Cart Position: {cart_position:.2f} | Pole Angle: {pole_angle:.2f} rad")
    plt.show()

if __name__ == "__main__":
  env = CartPoleEnvironment()

  l1 = Layer(4, 128)
  l2 = Layer(128, 256)
  l3 = Layer(256, 128)
  l4 = Layer(128, 2)  # Output is [left, right]
  nn = Network([l1, l2, l3, l4])

  for episode in range(NUM_EPISODES):
    state = env.reset()  # Start fresh
    done = False
    total_reward = 0

    while not done:
      state_array = state.to_array()
      action_values = nn.forward(state_array)  # Forward pass

      action = np.argmax(action_values)
      plot_cartpole(state)
      state, reward, done = env.step(action)
      total_reward += reward

    print(f"Episode {episode + 1} ended with total reward: {total_reward}")
