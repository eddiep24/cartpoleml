from nn import *
from cartpole import *
import matplotlib.pyplot as plt
import numpy as np

NUM_EPISODES = 500
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
RUNNING_AVERAGE_WINDOW = 20

def compute_discounted_rewards(rewards):
    """Compute discounted rewards for the entire episode."""
    discounted_rewards = np.zeros_like(rewards, dtype=float)
    running_reward = 0
    for t in reversed(range(len(rewards))):
        running_reward = rewards[t] + DISCOUNT_FACTOR * running_reward
        discounted_rewards[t] = running_reward
    return discounted_rewards

def normalize_rewards(rewards):
    """Normalize rewards for stability."""
    rewards = np.array(rewards)
    return (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

def plot_rewards(episode_rewards):
    """Plot episode rewards and running average."""
    episodes = range(1, len(episode_rewards) + 1)
    
    running_avg = np.convolve(episode_rewards, 
                             np.ones(RUNNING_AVERAGE_WINDOW)/RUNNING_AVERAGE_WINDOW, 
                             mode='valid')
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, episode_rewards, label='Episode Reward', alpha=0.6)
    plt.plot(range(RUNNING_AVERAGE_WINDOW, len(episode_rewards) + 1), 
             running_avg, 
             label=f'{RUNNING_AVERAGE_WINDOW}-Episode Running Average',
             linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('CartPole Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    # Neural network architecture
    layer1 = Layer(4, 128)
    layer2 = Layer(128, 256)
    layer3 = Layer(256, 512)
    layer4 = Layer(512, 256)
    layer5 = Layer(256, 128)
    layer6 = Layer(128, 2)
    network = Network([layer1, layer2, layer3, layer4, layer5, layer6], 
                     learning_rate=LEARNING_RATE)
    env = CartPoleEnvironment()
    
    all_episode_rewards = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        episode_states = []
        episode_actions = []
        episode_rewards = []
        total_reward = 0

        # Collect episode trajectory
        while not done:
            state_array = state.to_array().reshape(1, -1)
            
            # Get action using the network's policy
            action, _ = network.get_action(state_array)
            
            # Take action in environment
            next_state, reward, done = env.step(action)
            
            # Store trajectory information
            episode_states.append(state_array)
            episode_actions.append(action)
            episode_rewards.append(reward)
            total_reward += reward
            
            state = next_state

        # Store total episode reward
        all_episode_rewards.append(total_reward)

        # Compute returns and advantages
        discounted_rewards = compute_discounted_rewards(episode_rewards)
        normalized_rewards = normalize_rewards(discounted_rewards)

        # Update policy for each step in the episode
        for t in range(len(episode_states)):
            network.train_policy_gradient(
                state=episode_states[t],
                action=episode_actions[t],
                discounted_reward=normalized_rewards[t]
            )

        print(f"Episode {episode+1}/{NUM_EPISODES} - Total Reward: {total_reward}")
    
    # Plot the results
    plot_rewards(all_episode_rewards)

if __name__ == "__main__":
    main()