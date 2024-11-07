# cartpoleml: Fully Connected Net Reinforcement Learning (Cart pole) w/ BackProp

### Requirements
✓ No framework (PyTorch, Tensorflow, Caffe, etc.) is allowed.
✓ Instead, codes need to be written in MATLAB or Python (with NumPy) to describe VECTOR-
MATRIX-MULTIPLY explicitly.
✓ Weight parameters need to be in conductance unit (0.1~150 μS), which will need noise
component (σ=0.1 μS). This is equivalent to 8-bit precision.
✓ Negative conductance is allowed to make it simple as a class project.


### How It Works:

We're trying to create a fully connected neural network that can solve the cart pole balancing problem. The network should take the cart pole system parameters (cart_position, cart_velocity, pole_angle, pole_velocity) as inputs and predict whether the cart should move right or left. 

In nn.py we define our network constructs. We have a Layer() class where a user can specify a layer's matrix dimensions d1 and d2. The Network() class takes in an array of Layers() as an initializer and implements a forward pass as well as a backpropogation function.