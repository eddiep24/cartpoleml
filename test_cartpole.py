from nn import Network, Layer
from cartpole import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

NUM_EPISODES = 100

if __name__ == "__main__":
  layer1 = Layer(4, 64)
  layer2 = Layer(64, 32)
  layer3 = Layer(32, 2) 
  network = Network([layer1, layer2, layer3], learning_rate=0.001)
  
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