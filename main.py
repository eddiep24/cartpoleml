"""

Main implmenetation that trains a fully connected neural network using backpropogation and gets feedback via cart pole RL.

"""

from nn import *
from cartpole import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

NUM_EPISODES = 1000
LEARNING_RATE = 0.05
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

  animator = CartPoleAnimator(network, NUM_EPISODES)
  anim = FuncAnimation(
    animator.fig,
    animator.animate,
    init_func=animator.init_animation,
    frames=None,
    interval=20,
    blit=True,
    repeat=False
  )
  plt.show()




if __name__ == "__main__":
  main()
  