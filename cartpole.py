import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class CartPoleState:
    def __init__(self, cart_position, cart_velocity, pole_angle, pole_velocity):
        self.cart_position = cart_position
        self.cart_velocity = cart_velocity 
        self.pole_angle = pole_angle
        self.pole_velocity = pole_velocity

    def to_array(self):
        return np.array([self.cart_position, self.cart_velocity, self.pole_angle, self.pole_velocity])

    def __repr__(self):
        return (f"CartPoleState(cart_position={self.cart_position}, "
                f"cart_velocity={self.cart_velocity}, "
                f"pole_angle={self.pole_angle}, "
                f"pole_velocity={self.pole_velocity})")


class CartPoleEnvironment:
  def __init__(self):
    self.gravity = 9.8
    self.masscart = 1.0
    self.masspole = 0.1
    self.total_mass = self.masspole + self.masscart
    self.length = 0.5  # Half of the pole's length
    self.force_mag = 10.0
    self.tau = 0.01  # Time step for the simulation

    self.state = None
    self.reset()

  def reset(self):
    # Initialize state with small random values near zero
    self.state = CartPoleState(
        cart_position=np.random.uniform(-0.05, 0.05),
        cart_velocity=0.0,
        pole_angle=np.random.uniform(-0.05, 0.05),
        pole_velocity=0.0
    )
    return self.state

  def step(self, action):
    """
    Simulates a single time step of the CartPole system given an action (0 or 1).
    
    The physics calculations here are based on the dynamics of an inverted pendulum on a cart.
    The goal is to compute the resulting state after applying a force in one direction.
    
    - Inputs:
        action: int
          - 0: Apply a leftward force to the cart
          - 1: Apply a rightward force to the cart

    - Key Calculations:
      - The action determines the applied force on the cart. Based on the direction (left or right), a force magnitude (`self.force_mag`) is set either positively or negatively.
      - The system then uses Newton's laws to calculate both linear and angular accelerations (`x_acc` and `theta_acc`) for the cart and the pole.
      - The system state is then updated by integrating over the time step `tau` using Euler's method.

    - Physics Breakdown:
      1. **Equations of Motion**:
        - The inverted pendulum (pole) applies a torque on the cart based on the gravitational pull acting on the pole's mass.
        - Simultaneously, the applied force affects the cart's motion, creating a combined dynamic system where the cart and pole influence each other.
            
      2. **Force and Angular Accelerations**:
        - `force`: Set based on the action; a positive force moves the cart right, and a negative force moves it left.
        - `cos_theta` and `sin_theta`: The cosine and sine of the pole angle are used to project forces along the pole.
        - `temp`: Represents the force per unit mass on the cart due to the combined effect of the pole and external force.
        - `theta_acc`: Calculated based on gravity, the force on the pole, and the constraints of the pendulum. The pole's angular acceleration reflects both gravitational pull and the applied force's effect on the pole's mass.
        - `x_acc`: Represents the cart's linear acceleration. It's affected by the pole's angular movement (due to `theta_acc`) and the applied force.

      3. **Euler's Method for Integration**:
        - The current velocity and position of both the cart and the pole are incrementally updated over the time step `tau`.
        - Using Euler's method, we compute:
          - New position and velocity of the cart: `x` and `x_dot`.
          - New angle and angular velocity of the pole: `theta` and `theta_dot`.
        
      4. **Episode Termination Conditions**:
        - The episode ends if the cart moves out of bounds (±2.4 units from the center) or if the pole angle exceeds ±12 degrees. If either condition is met, `done` is set to `True`.

      5. **Reward Calculation**:
        - A reward of 1.0 is given for each step the pole remains balanced within bounds. If the episode ends, the reward is set to 0.

  - Returns:
    - The updated `CartPoleState`, `reward` for the action, and `done` flag indicating if the episode is over.
    """

    assert action in [0, 1], "Action must be 0 or 1"

    # Extract state variables for easier handling
    x, x_dot = self.state.cart_position, self.state.cart_velocity
    theta, theta_dot = self.state.pole_angle, self.state.pole_velocity

    # Determine force direction
    force = self.force_mag if action == 1 else -self.force_mag

    # Equations for updating the CartPole state
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    temp = (force + self.masspole * self.length * theta_dot**2 * sin_theta) / self.total_mass
    theta_acc = (self.gravity * sin_theta - cos_theta * temp) / \
                (self.length * (4.0 / 3.0 - self.masspole * cos_theta**2 / self.total_mass))
    x_acc = temp - self.masspole * self.length * theta_acc * cos_theta / self.total_mass

    # Update state using Euler's method
    x += self.tau * x_dot
    x_dot += self.tau * x_acc
    theta += self.tau * theta_dot
    theta_dot += self.tau * theta_acc

    # Update the environment's state
    self.state = CartPoleState(x, x_dot, theta, theta_dot)

    # Calculate reward and check if the episode is done
    done = x < -2.4 or x > 2.4 or theta < -12 * (np.pi / 180) or theta > 12 * (np.pi / 180)
    reward = 1.0 if not done else 0.0

    return self.state, reward, done

  def render(self):
      print(f"Current state: {self.state}")

class CartPoleLearningMetrics:
  def __init__(self, num_episodes):
    self.num_episodes = num_episodes
    self.rewards_history = []
    self.errors_history = []
    self.current_episode = 0
      
  def update_metrics(self, episode_reward, episode_error):
    """Update metrics after each episode"""
    self.rewards_history.append(episode_reward)
    self.errors_history.append(episode_error)
    self.current_episode += 1
      
  def plot_metrics(self):
      """Create and display a plot for rewards"""
      fig, ax = plt.subplots(figsize=(10, 6))
      episodes = range(1, len(self.rewards_history) + 1)
      
      # Plot rewards with rolling average
      ax.plot(episodes, self.rewards_history, 'b-', alpha=0.3, label='Raw')
      window = min(10, len(self.rewards_history))
      if window > 0:
          rolling_mean = np.convolve(self.rewards_history, np.ones(window)/window, mode='valid')
          ax.plot(range(window, len(self.rewards_history) + 1), rolling_mean, 'b-', label='Rolling Average')
      ax.set_xlabel('Episode')
      ax.set_ylabel('Total Reward')
      ax.set_title('Total Reward vs Episode')
      ax.grid(True)
      ax.legend()
      
      plt.show()

    
    # Plot errors with rolling average
    # ax2.plot(episodes, self.errors_history, 'r-', alpha=0.3, label='Raw')
    # if window > 0:
    #     rolling_mean = np.convolve(self.errors_history, np.ones(window)/window, mode='valid')
    #     ax2.plot(range(window, len(self.errors_history) + 1), rolling_mean, 'r-', label='Rolling Average')
    # ax2.set_xlabel('Episode')
    # ax2.set_ylabel('Average Error')
    # ax2.set_title('Training Error vs Episode')
    # ax2.grid(True)
    # ax2.legend()
    
    # plt.tight_layout()


class CartPoleAnimator:
    def __init__(self, neural_network, num_episodes):
        self.NUM_EPISODES = num_episodes
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xlim(-2.4, 2.4)
        self.ax.set_ylim(-1, 2)
        self.ax.grid(True, axis='x', linestyle='-', alpha=0.2)
        
        # Initialize cart and pole objects
        self.cart = plt.Rectangle((-0.1, -0.05), 0.2, 0.1, fc='blue')
        self.ax.add_patch(self.cart)
        self.pole, = self.ax.plot([], [], 'r-', lw=3)
        
        self.text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
        
        self.env = CartPoleEnvironment()
        self.nn = neural_network
        
        # Animation state
        self.state = self.env.reset()
        self.episode = 0
        self.total_reward = 0
        self.done = False
        
        # Learning parameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.replay_buffer = []
        self.batch_size = 32
        self.metrics = CartPoleLearningMetrics(num_episodes)
        self.episode_errors = []
 
    def init_animation(self):
        self.cart.set_xy((-0.1, -0.05))
        self.pole.set_data([], [])
        return self.cart, self.pole, self.text
        
    def select_action(self, state_array):
        if np.random.random() < self.epsilon:
            return np.random.choice([0, 1])
        # Reshape state array to 2D array (batch size of 1)
        state_input = state_array.reshape(1, -1)
        action_values = self.nn.forward(state_input)
        return np.argmax(action_values)
        
    def train_network(self):
      if len(self.replay_buffer) < self.batch_size:
        return
          
      total_error = 0
      batch_indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
      states = []
      targets = []

      for idx in batch_indices:
        state, action, reward, next_state, done = self.replay_buffer[idx]
        
        state_input = state.reshape(1, -1)
        next_state_input = next_state.reshape(1, -1)
        
        current_q = self.nn.forward(state_input)
        target = current_q.copy()
        
        if done:
            target[0][action] = reward
        else:
            next_q = self.nn.forward(next_state_input)
            target[0][action] = reward + self.gamma * np.max(next_q)
        
        # Calculate error for this sample
        error = np.mean((target - current_q) ** 2)
        print("Error = {}".format(error))
        total_error += error
        
        states.append(state)
        targets.append(target[0])
          
      # Store average error for this batch
      self.episode_errors.append(total_error / self.batch_size)
      
      states = np.array(states)
      targets = np.array(targets)
      self.nn.backward(targets)
            
    def animate(self, frame):
        if self.done:
            # Episode finished, update metrics and reset environment
            episode_error = np.mean(self.episode_errors) if self.episode_errors else 0
            self.metrics.update_metrics(self.total_reward, episode_error)
            self.episode_errors = []  # Reset errors for the next episode
            
            print(f"Episode {self.episode + 1} ended with total reward: {self.total_reward}")
            
            # Increment episode counter
            self.episode += 1
            
            if self.episode >= self.NUM_EPISODES:
                # All episodes complete, plot metrics and stop animation
                self.metrics.plot_metrics()
                plt.close()
                return self.cart, self.pole, self.text
            
            # Reset environment and learning parameters for the next episode
            self.state = self.env.reset()
            self.total_reward = 0
            self.done = False
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.train_network()
        
        # Normal step processing
        state_array = self.state.to_array()
        action = self.select_action(state_array)
        next_state, reward, self.done = self.env.step(action)
        next_state_array = next_state.to_array()
        
        # Store transition in replay buffer
        self.replay_buffer.append((state_array, action, reward, next_state_array, self.done))
        if len(self.replay_buffer) > 10000:
            self.replay_buffer.pop(0)
        
        # Update rewards and state
        self.total_reward += reward
        self.state = next_state
        
        # Update visualization
        cart_position = self.state.cart_position
        pole_angle = self.state.pole_angle
        self.cart.set_x(cart_position - 0.1)
        
        pole_length = 0.5
        pole_x = [cart_position, cart_position + pole_length * np.sin(pole_angle)]
        pole_y = [0, pole_length * np.cos(pole_angle)]
        self.pole.set_data(pole_x, pole_y)
        
        self.text.set_text(f'Episode: {self.episode + 1}/{self.NUM_EPISODES}\n'
                          f'Reward: {self.total_reward:.1f}\n'
                          f'Epsilon: {self.epsilon:.2f}')
        
        return self.cart, self.pole, self.text
