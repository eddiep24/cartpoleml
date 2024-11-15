import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from nn import Network, Layer
from cartpole import CartPoleEnvironment

class CartPoleAnimator:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xlim(-2.4, 2.4)
        self.ax.set_ylim(-1, 2)
        
        # Initialize cart and pole objects
        self.cart = plt.Rectangle((-0.1, -0.05), 0.2, 0.1, fc='blue')
        self.ax.add_patch(self.cart)
        self.pole, = self.ax.plot([], [], 'r-', lw=3)
        
        # Text display for episode info
        self.text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
        
        # Initialize environment and neural network
        self.env = CartPoleEnvironment()
        l1 = Layer(4, 128)
        l2 = Layer(128, 256)
        l3 = Layer(256, 128)
        l4 = Layer(128, 2)
        self.nn = Network([l1, l2, l3, l4])
        
        # Animation state
        self.state = self.env.reset()
        self.episode = 0
        self.total_reward = 0
        self.done = False

    def init_animation(self):
        """Initialize the animation"""
        self.cart.set_xy((-0.1, -0.05))
        self.pole.set_data([], [])
        return self.cart, self.pole, self.text

    def animate(self, frame):
        """Update animation frame"""
        if self.done:
            self.episode += 1
            if self.episode >= NUM_EPISODES:
                plt.close()
                return self.cart, self.pole, self.text
            
            print(f"Episode {self.episode + 1} ended with total reward: {self.total_reward}")
            self.state = self.env.reset()
            self.total_reward = 0
            self.done = False

        # Neural network forward pass
        state_array = self.state.to_array()
        action_values = self.nn.forward(state_array)
        action = np.argmax(action_values)
        
        # Update state
        self.state, reward, self.done = self.env.step(action)
        self.total_reward += reward

        # Update cart position
        cart_position = self.state.cart_position
        pole_angle = self.state.pole_angle
        
        # Update cart
        self.cart.set_x(cart_position - 0.1)
        
        # Update pole
        pole_length = 0.5
        pole_x = [cart_position, cart_position + pole_length * np.sin(pole_angle)]
        pole_y = [0, pole_length * np.cos(pole_angle)]
        self.pole.set_data(pole_x, pole_y)
        
        # Update text
        self.text.set_text(f'Episode: {self.episode + 1}/{NUM_EPISODES}\nReward: {self.total_reward:.1f}')
        
        return self.cart, self.pole, self.text

def run_animation():
    animator = CartPoleAnimator()
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
    NUM_EPISODES = 100
    run_animation()