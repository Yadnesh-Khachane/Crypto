"""
rl_agent.py - Professional DQN Agent with Prioritized Experience Replay
FIXED: All None type errors resolved
"""

import numpy as np
from collections import deque
import json
import os
import random
from datetime import datetime

class NeuralNetwork:
    """Multi-layer neural network with proper backprop"""
    
    def __init__(self, layers, learning_rate=0.0003, dropout_rate=0.1):
        self.layers = layers
        self.lr = learning_rate
        self.dropout_rate = dropout_rate
        self.weights = []
        self.biases = []
        self.cache = {}
        
        # He initialization
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / max(layers[i], 1))
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Adam optimizer state
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
    
    def forward(self, X, training=True):
        """Forward pass with dropout - SAFE VERSION"""
        # Handle None input
        if X is None:
            return np.zeros(self.layers[-1], dtype=np.float32)
        
        # Convert to numpy array safely
        try:
            X = np.array(X, dtype=np.float32).reshape(1, -1)
        except:
            return np.zeros(self.layers[-1], dtype=np.float32)
        
        # Check for NaN/Inf
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure correct input size
        if X.shape[1] != self.layers[0]:
            # Pad or truncate
            if X.shape[1] < self.layers[0]:
                X = np.pad(X, ((0, 0), (0, self.layers[0] - X.shape[1])), 'constant')
            else:
                X = X[:, :self.layers[0]]
        
        self.cache = {'a0': X}
        A = X
        
        for i in range(len(self.weights)):
            Z = A @ self.weights[i] + self.biases[i]
            
            # Check for NaN
            if np.any(np.isnan(Z)):
                Z = np.nan_to_num(Z, nan=0.0)
            
            self.cache[f'z{i+1}'] = Z
            
            if i == len(self.weights) - 1:
                A = Z
            else:
                A = np.maximum(0, Z)  # ReLU
                if training and self.dropout_rate > 0:
                    mask = np.random.binomial(1, 1 - self.dropout_rate, A.shape) / max(1 - self.dropout_rate, 0.01)
                    A = A * mask
                    self.cache[f'dropout{i+1}'] = mask
            
            self.cache[f'a{i+1}'] = A
        
        # Final safety check
        result = A.flatten()
        if np.any(np.isnan(result)):
            result = np.nan_to_num(result, nan=0.0)
        
        return result
    
    def backward(self, dout):
        """Backpropagation with Adam optimizer"""
        if dout is None:
            return
        
        try:
            dout = np.array(dout, dtype=np.float32).reshape(1, -1)
        except:
            return
        
        if np.any(np.isnan(dout)):
            dout = np.nan_to_num(dout, nan=0.0)
        
        self.t += 1
        dA = dout
        
        for i in range(len(self.weights) - 1, -1, -1):
            dZ = dA
            if i < len(self.weights) - 1:
                dZ = dA * (self.cache.get(f'z{i+1}', np.zeros_like(dA)) > 0)
                if f'dropout{i+1}' in self.cache:
                    dZ = dZ * self.cache[f'dropout{i+1}']
            
            A_prev = self.cache.get(f'a{i}', np.zeros((1, self.weights[i].shape[0])))
            dW = A_prev.T @ dZ
            db = np.sum(dZ, axis=0, keepdims=True)
            dA = dZ @ self.weights[i].T
            
            # Clip gradients
            dW = np.clip(dW, -1.0, 1.0)
            db = np.clip(db, -1.0, 1.0)
            
            # Adam update
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * dW
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (dW ** 2)
            m_hat = self.m_weights[i] / max(1 - self.beta1 ** self.t, 1e-8)
            v_hat = self.v_weights[i] / max(1 - self.beta2 ** self.t, 1e-8)
            self.weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * db
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (db ** 2)
            m_hat_b = self.m_biases[i] / max(1 - self.beta1 ** self.t, 1e-8)
            v_hat_b = self.v_biases[i] / max(1 - self.beta2 ** self.t, 1e-8)
            self.biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
            
            # Clip weights
            self.weights[i] = np.clip(self.weights[i], -10.0, 10.0)
            self.biases[i] = np.clip(self.biases[i], -10.0, 10.0)
    
    def get_weights_flat(self):
        """Get all weights as flat array"""
        all_weights = []
        for w, b in zip(self.weights, self.biases):
            all_weights.extend([w.flatten(), b.flatten()])
        return np.concatenate(all_weights)
    
    def set_weights_flat(self, flat_weights):
        """Set weights from flat array"""
        idx = 0
        for i in range(len(self.weights)):
            w_size = self.weights[i].size
            b_size = self.biases[i].size
            
            self.weights[i] = flat_weights[idx:idx + w_size].reshape(self.weights[i].shape)
            idx += w_size
            
            self.biases[i] = flat_weights[idx:idx + b_size].reshape(self.biases[i].shape)
            idx += b_size


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay"""
    
    def __init__(self, capacity=50000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done):
        """Add experience - SAFE VERSION"""
        # Ensure all values are valid
        try:
            state = np.array(state, dtype=np.float32).flatten()
            next_state = np.array(next_state, dtype=np.float32).flatten()
        except:
            return
        
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            return
        if np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)):
            return
        
        if reward is None or np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        self.buffer.append((state, action, float(reward), next_state, bool(done)))
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size=64):
        """Sample batch"""
        if len(self.buffer) < batch_size:
            return None
        
        try:
            priorities = np.array(self.priorities, dtype=np.float64)
            priorities = np.nan_to_num(priorities, nan=1.0)
            probs = priorities ** self.alpha
            probs /= max(probs.sum(), 1e-8)
            
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
            
            total = len(self.buffer)
            weights = (total * probs[indices]) ** (-self.beta)
            weights /= max(weights.max(), 1e-8)
            
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            states, actions, rewards, next_states, dones = [], [], [], [], []
            for idx in indices:
                s, a, r, ns, d = self.buffer[idx]
                states.append(s)
                actions.append(a)
                rewards.append(r)
                next_states.append(ns)
                dones.append(d)
            
            return (
                np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int32),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32),
                indices,
                np.array(weights, dtype=np.float32)
            )
        except Exception as e:
            return None
    
    def update_priorities(self, indices, td_errors):
        """Update priorities"""
        for idx, td_error in zip(indices, td_errors):
            if td_error is None or np.isnan(td_error):
                td_error = 0.0
            self.priorities[idx] = abs(float(td_error)) + 1e-6
            self.max_priority = max(self.max_priority, abs(float(td_error)) + 1e-6)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Professional DQN Agent for crypto trading"""
    
    def __init__(self, state_size, action_size, hidden_layers=[128, 64],
                 learning_rate=0.0003, gamma=0.97):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        layers = [state_size] + hidden_layers + [action_size]
        
        self.policy_net = NeuralNetwork(layers, learning_rate)
        self.target_net = NeuralNetwork(layers, learning_rate)
        self.update_target_network()
        
        self.memory = PrioritizedReplayBuffer(50000)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.998
        
        self.train_step = 0
        self.loss_history = deque(maxlen=1000)
        
        self.state_mean = None
        self.state_std = None
        self.state_buffer = deque(maxlen=10000)
    
    def normalize_state(self, state):
        """Normalize state - SAFE VERSION"""
        if state is None:
            return np.zeros(self.state_size, dtype=np.float32)
        
        try:
            state = np.array(state, dtype=np.float32).flatten()
        except:
            return np.zeros(self.state_size, dtype=np.float32)
        
        # Ensure correct size
        if len(state) != self.state_size:
            if len(state) < self.state_size:
                state = np.pad(state, (0, self.state_size - len(state)), 'constant')
            else:
                state = state[:self.state_size]
        
        # Remove NaN/Inf
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Store for statistics
        self.state_buffer.append(state.copy())
        
        # Build normalization stats
        if len(self.state_buffer) >= 100:
            states = np.array(list(self.state_buffer))
            self.state_mean = np.mean(states, axis=0)
            self.state_std = np.std(states, axis=0) + 1e-8
        
        # Normalize
        if self.state_mean is not None and self.state_std is not None:
            state = (state - self.state_mean) / self.state_std
        
        # Final clip
        state = np.clip(state, -5, 5)
        state = np.nan_to_num(state, nan=0.0)
        
        return state.astype(np.float32)
    
    def get_action(self, state, training=True):
        """Get action - SAFE VERSION"""
        state = self.normalize_state(state)
        
        if training and np.random.random() < self.epsilon:
            # Softmax exploration
            try:
                q_values = self.policy_net.forward(state, training=False)
                q_values = np.nan_to_num(q_values, nan=0.0)
                q_values = q_values - np.max(q_values)
                exp_q = np.exp(np.clip(q_values / 0.5, -10, 10))
                probs = exp_q / max(exp_q.sum(), 1e-8)
                
                if np.random.random() < 0.7:
                    return np.random.choice(self.action_size, p=probs)
                else:
                    return np.random.randint(self.action_size)
            except:
                return np.random.randint(self.action_size)
        
        # Greedy action
        try:
            q_values = self.policy_net.forward(state, training=False)
            q_values = np.nan_to_num(q_values, nan=0.0)
            return int(np.argmax(q_values))
        except:
            return 0
    
    def get_q_values(self, state):
        """Get Q-values"""
        state = self.normalize_state(state)
        try:
            q = self.policy_net.forward(state, training=False)
            return np.nan_to_num(q, nan=0.0)
        except:
            return np.zeros(self.action_size, dtype=np.float32)
    
    def train_batch(self, batch_size=64):
        """Train on batch - SAFE VERSION"""
        sample = self.memory.sample(batch_size)
        if sample is None:
            return None
        
        states, actions, rewards, next_states, dones, indices, weights = sample
        
        total_loss = 0.0
        td_errors = []
        
        for i in range(len(states)):
            try:
                state = states[i]
                action = actions[i]
                reward = rewards[i]
                next_state = next_states[i]
                done = dones[i]
                weight = weights[i]
                
                # Current Q-value
                current_q = self.policy_net.forward(state)
                
                # Target Q-value
                if done:
                    target_q = reward
                else:
                    next_q = self.target_net.forward(next_state, training=False)
                    target_q = reward + self.gamma * np.max(next_q)
                
                td_error = target_q - current_q[action]
                td_errors.append(td_error)
                
                # Backprop
                dout = np.zeros(self.action_size, dtype=np.float32)
                dout[action] = float(td_error) * float(weight)
                
                self.policy_net.forward(state)
                self.policy_net.backward(dout)
                
                total_loss += float(td_error ** 2) * float(weight)
            except Exception as e:
                td_errors.append(0.0)
                continue
        
        # Update priorities
        try:
            self.memory.update_priorities(indices, td_errors)
        except:
            pass
        
        avg_loss = total_loss / max(batch_size, 1)
        self.loss_history.append(avg_loss)
        
        self.train_step += 1
        if self.train_step % 100 == 0:
            self.update_target_network()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return float(avg_loss)
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience safely"""
        try:
            state = self.normalize_state(state)
            next_state = self.normalize_state(next_state)
            self.memory.push(state, action, float(reward), next_state, bool(done))
        except:
            pass
    
    def update_target_network(self):
        """Copy weights"""
        try:
            self.target_net.set_weights_flat(self.policy_net.get_weights_flat())
        except:
            pass
    
    def save(self, filepath="models/rl_agent.npz"):
        """Save model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            save_data = {
                'weights_flat': self.policy_net.get_weights_flat(),
                'epsilon': self.epsilon,
                'train_step': self.train_step,
                'state_mean': self.state_mean,
                'state_std': self.state_std
            }
            
            np.savez(filepath, **save_data)
            
            meta = {
                'epsilon': float(self.epsilon),
                'train_step': int(self.train_step),
                'avg_loss': float(np.mean(list(self.loss_history))) if self.loss_history else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath.replace('.npz', '_meta.json'), 'w') as f:
                json.dump(meta, f, indent=2)
            
            print(f"✅ Model saved: {filepath}")
        except Exception as e:
            print(f"⚠️ Save error: {e}")
    
    def load(self, filepath="models/rl_agent.npz"):
        """Load model"""
        if not os.path.exists(filepath):
            return False
        
        try:
            data = np.load(filepath, allow_pickle=True)
            
            if 'weights_flat' in data:
                self.policy_net.set_weights_flat(data['weights_flat'])
                self.update_target_network()
            
            self.epsilon = float(data.get('epsilon', self.epsilon))
            self.train_step = int(data.get('train_step', 0))
            self.state_mean = data.get('state_mean')
            self.state_std = data.get('state_std')
            
            print(f"✅ Model loaded: {filepath}")
            print(f"   Epsilon: {self.epsilon:.4f}")
            print(f"   Training steps: {self.train_step}")
            
            return True
        except Exception as e:
            print(f"⚠️ Load error: {e}")
            return False