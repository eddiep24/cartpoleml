import numpy as np

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
    self.tau = 0.02  # Time step for the simulation

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

    # Update state using Euler’s method
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
    # Visualization logic could go here 
    print(f"Current state: {self.state}")
