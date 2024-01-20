import gym
import numpy as np
from agent import Agent
from utils import plot_learning

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                   input_dims=[8], lr=0.003)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation, _ = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print(f"Episode {i}: score = {score}, avg_score = {avg_score}, epsilon = {agent.epsilon}")


    x = [i + 1 for i in range(n_games)]
    filename = "lunar_lander.png"
    plot_learning(x, scores, eps_history, filename)

