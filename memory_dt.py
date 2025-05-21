import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import gymnasium as gym

from pomdp_envs.velocity_cartpole import VelocityCartPoleEnv
from pomdp_envs.flickering_pendulum import FlickeringPendulumEnv
from pomdp_envs.lidar_mountain_car import LiDARMountainCarEnv

import matplotlib.pyplot as plt


class POMDPDataset(Dataset):
    def __init__(self, data_dir, block_size):
        """
        Dataset for POMDP environments.
        
        Args:
            data_dir: Directory containing trajectory data
            block_size: Context length for the transformer
        """
        self.block_size = block_size
        
        import glob
        import os
        
        files = glob.glob(os.path.join(data_dir, 'train_data_*.npz'))
        files.sort()
        
        self.states = []
        self.actions = []
        self.rtgs = []
        self.timesteps = []
        self.done_idxs = []
        
        current_idx = 0
        
        for file_path in files:
            data = np.load(file_path)
            
            seq_length = len(data['obs'])
            
            self.states.append(data['obs'])
            self.actions.append(data['action'])
            self.rtgs.append(data['rtg'])
            self.timesteps.append(data['timesteps'])
            
            if seq_length > 0:
                self.done_idxs.append(current_idx + seq_length - 1)
                current_idx += seq_length
        
        self.states = np.concatenate(self.states, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.rtgs = np.concatenate(self.rtgs, axis=0)
        self.timesteps = np.concatenate(self.timesteps, axis=0)
        
        # compute vocabulary size (number of possible actions)
        self.vocab_size = int(np.max(self.actions)) + 1
        
        if len(self.states.shape) > 2:
            self.state_dim = np.prod(self.states.shape[1:])
        else:
            self.state_dim = self.states.shape[1] if len(self.states.shape) > 1 else 1
    
    def __len__(self):
        return len(self.states) - self.block_size
    
    def __getitem__(self, idx):
        # endpoint of the current segment
        block_size = self.block_size // 3
        done_idx = idx + block_size
        
        # make sure we don't cross trajectory boundaries
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        
        # adjust start index to maintain block_size
        idx = done_idx - block_size
        
        # get data segments
        states = torch.tensor(self.states[idx:done_idx], dtype=torch.float32)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)
        
        return states, actions, rtgs, timesteps


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MemoryDecisionTransformer(nn.Module):
    def __init__(self, state_dim, n_actions, n_embed=128, n_layer=2, n_head=4, context_length=20, 
                 memory_type='gru', memory_dim=64, dropout=0.1):
        """Simple Decision Transformer with memory for POMDP."""
        super(MemoryDecisionTransformer, self).__init__()
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_embed = n_embed
        self.context_length = context_length
        self.memory_type = memory_type
        self.memory_dim = memory_dim
        self.n_layer = n_layer
        
        # (R,o,a) encoders
        self.state_encoder = nn.Linear(state_dim, n_embed)
        self.action_encoder = nn.Embedding(n_actions, n_embed)
        self.return_encoder = nn.Linear(1, n_embed)
        
        self.pos_encoder = PositionalEncoding(n_embed)
        
        # Memory module (optional)
        if memory_type == 'gru':
            # TODO: Implement GRU memory
            self.memory = nn.GRU(input_size=n_embed, hidden_size=memory_dim, batch_first=True)  # например
            self.memory_proj = nn.Linear(memory_dim, n_embed)
        elif memory_type == 'lstm':
            # TODO: Implement LSTM memory
            self.memory = nn.LSTM(input_size=n_embed, hidden_size=memory_dim, batch_first=True)
            self.memory_proj = nn.Linear(memory_dim, n_embed)
        else:
            self.memory = None
            self.memory_proj = None
        
        # Transformer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=n_embed,
            nhead=n_head,
            dim_feedforward=4*n_embed,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, n_layer)
        
        # action head
        self.action_head = nn.Linear(n_embed, n_actions)
        
        self.hidden_state = None
    
    def reset_memory(self):
        """Reset the memory state."""
        self.hidden_state = None
    
    def forward(self, states, actions, rtgs):
        """Forward pass."""
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        # reshaping as needed
        if len(states.shape) > 3:  # image observations
            states = states.reshape(batch_size, seq_length, -1)
        
        # ensuring safe inputs
        states = torch.nan_to_num(states)
        
        # if action is [batch, seq, 1], squeeze last dim
        if actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        
        # make sure rtgs is [batch, seq, 1]
        if rtgs.dim() == 2:
            rtgs = rtgs.unsqueeze(-1)
        
        # encoding inputs
        state_embeddings = self.state_encoder(states)
        action_embeddings = self.action_encoder(actions)
        return_embeddings = self.return_encoder(rtgs)
        
        device = state_embeddings.device
        dtype = state_embeddings.dtype
        
        # add memory
        if self.memory is not None:
            if self.memory_type == 'gru':
                if self.hidden_state is None:
                    # TODO: Implement GRU memory
                    self.hidden_state = torch.zeros( 1, batch_size, self.memory_dim, device=device, dtype=dtype )
                    
                memory_out, self.hidden_state = self.memory(state_embeddings, self.hidden_state)
            elif self.memory_type == 'lstm':
                if self.hidden_state is None:
                    # TODO: Implement LSTM memory
                    self.hidden_state = ( torch.zeros(1, batch_size, self.memory_dim, device=device, dtype=dtype),
                                          torch.zeros(1, batch_size, self.memory_dim, device=device, dtype=dtype) )
                    
                memory_out, self.hidden_state = self.memory(state_embeddings, self.hidden_state)
            
            # project memory to embedding dimension
            memory_embedding = self.memory_proj(memory_out)
            
            # combine memory with state embeddings
            state_embeddings = state_embeddings + memory_embedding
        
        # prepare sequence for transformer (R_t, o_t, a_t)
        sequence = torch.cat([
            return_embeddings, 
            state_embeddings,
            action_embeddings
        ], dim=1)
        
        # add positional encoding
        sequence = self.pos_encoder(sequence)
        
        # apply transformer
        # create causal attention mask
        seq_len = sequence.size(1)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=sequence.device), 
            diagonal=1
        )
        
        # try different mask parameter names for compatibility across PyTorch versions
        try:
            transformer_outputs = self.transformer(sequence, mask=mask)
        except TypeError:
            try:
                transformer_outputs = self.transformer(sequence, src_mask=mask)
            except TypeError:
                # fall back to no mask if neither works
                transformer_outputs = self.transformer(sequence)
        
        # extract state positions for output
        state_positions = transformer_outputs[:, seq_length:2*seq_length]
        
        # predict actions
        action_preds = self.action_head(state_positions)
        
        return action_preds
    
    def get_action(self, states, actions, rtgs, device=None):
        """Get a single action for inference."""
        if device is None:
            device = next(self.parameters()).device

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rtgs = torch.tensor(rtgs, dtype=torch.float32).to(device)
        
        # handle batch dimension
        if states.dim() == 2:
            states = states.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if rtgs.dim() == 1:
            rtgs = rtgs.unsqueeze(0).unsqueeze(-1)
        elif rtgs.dim() == 2 and rtgs.shape[-1] == 1:
            rtgs = rtgs.unsqueeze(0)
        elif rtgs.dim() == 2 and rtgs.shape[-1] != 1:
            rtgs = rtgs.unsqueeze(-1)
        
        # ensure same sequence length
        seq_len = states.shape[1]
        if actions.shape[1] < seq_len:
            padding = torch.zeros(actions.shape[0], seq_len - actions.shape[1], 
                                 device=device, dtype=actions.dtype)
            actions = torch.cat([actions, padding], dim=1)
        
        if rtgs.shape[1] < seq_len:
            last_val = rtgs[:, -1:, :]
            padding = last_val.repeat(1, seq_len - rtgs.shape[1], 1)
            rtgs = torch.cat([rtgs, padding], dim=1)
        
        # forward pass
        with torch.no_grad():
            action_preds = self.forward(states, actions, rtgs)
            action = torch.argmax(action_preds[0, -1]).item()
        
        return action


def train_memory_dt(
        env_name, dataset_path, n_epochs=10, batch_size=64, context_length=20,
        n_embed=128, n_layer=2, n_head=4, memory_type='gru', memory_dim=64,
        learning_rate=1e-4, weight_decay=1e-4, debug=False
    ):
    """Train a Memory-enabled Decision Transformer."""
    
    dataset = POMDPDataset(dataset_path, block_size=context_length*3)
    
    # split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset stats: total={len(dataset)}, train={train_size}, val={val_size}, state_dim={dataset.state_dim}, actions={dataset.vocab_size}")
    
    model = MemoryDecisionTransformer(
        state_dim=dataset.state_dim,
        n_actions=dataset.vocab_size,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        context_length=context_length,
        memory_type=memory_type,
        memory_dim=memory_dim,
        dropout=0.1,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Mini-inference each epoch
    if env_name == 'velocity_cartpole':
        val_env = VelocityCartPoleEnv()
    elif env_name == 'flickering_pendulum':
        val_env = FlickeringPendulumEnv(flicker_probability=0.3)
    elif env_name == 'lidar_mountain_car':
        val_env = LiDARMountainCarEnv(num_sensors=8)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # linear warmup followed by cosine annealing (as in original DT paper)
    total_steps = len(train_dataloader) * n_epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    train_losses, val_losses, val_returns = [], [], []
    best_val_return = float('-inf')
    best_model_state = None
    patience, patience_counter = 5, 0
    
    for epoch in range(n_epochs):
        # TRAIN PHASE
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")
        
        for batch_idx, (states, actions, rtgs, _) in enumerate(progress_bar):
            states = states.to(device)
            actions = actions.to(device)
            rtgs = rtgs.to(device)
            
            if len(states.shape) > 3:
                states = states.reshape(states.shape[0], states.shape[1], -1)
            
            if actions.shape[-1] == 1:
                actions = actions.squeeze(-1)
            
            # reset memory at the start of each batch
            model.reset_memory()
            
            action_preds = model(states, actions, rtgs)
            
            loss = criterion(
                action_preds.reshape(-1, dataset.vocab_size),
                actions.reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            progress_bar.set_postfix({
                'loss': f"{batch_loss:.4f}",
                'avg_loss': f"{epoch_loss / (batch_idx + 1):.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # VALIDATION ON DATASET
        model.eval()
        val_epoch_loss = 0
        
        with torch.no_grad():
            for states, actions, rtgs, _ in val_dataloader:
                states = states.to(device)
                actions = actions.to(device)
                rtgs = rtgs.to(device)
                
                if len(states.shape) > 3:
                    states = states.reshape(states.shape[0], states.shape[1], -1)
                
                if actions.shape[-1] == 1:
                    actions = actions.squeeze(-1)
                
                model.reset_memory()
                
                action_preds = model(states, actions, rtgs)
                val_loss = criterion(
                    action_preds.reshape(-1, dataset.vocab_size),
                    actions.reshape(-1)
                )
                
                val_epoch_loss += val_loss.item()
        
        avg_val_loss = val_epoch_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{n_epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        # VALIDATION ON ENVIRONMENT - fixed for reliable results
        print("Running environment validation...")
        model.eval()
        
        # pre-train mode fix for evaluation
        model.train(False)  # make absolutely sure we're in eval mode
        
        # keep track of episode returns
        returns = []
        successful_episodes = 0
        num_eval_episodes = 10
        
        for episode in range(num_eval_episodes):
            obs, _ = val_env.reset()
            model.reset_memory()  # reset memory state for each episode (always do this!)
            
            states = []
            actions = []
            
            episode_return = 0
            done = False
            timestep = 0
            max_steps = 500
            
            # setup target return (for RTG) based on environment
            if isinstance(val_env, VelocityCartPoleEnv):
                target_return = 500.0
            elif isinstance(val_env, FlickeringPendulumEnv):
                target_return = -200.0
            elif isinstance(val_env, LiDARMountainCarEnv):
                target_return = -100.0
            else:
                target_return = 500.0
            
            while not done and timestep < max_steps:
                if isinstance(obs, dict):
                    if 'observation' in obs and 'mask' in obs:
                        processed_obs = np.concatenate([
                            obs['observation'].flatten(),
                            np.array([obs['mask']], dtype=np.float32)
                        ])
                    else:
                        processed_obs = np.concatenate([
                            o.flatten() if hasattr(o, 'flatten') else np.array([o], dtype=np.float32)
                            for o in obs.values()
                        ])
                else:
                    processed_obs = obs
                
                states.append(processed_obs)
                
                # get action
                if len(states) <= 1:
                    # first timestep, use default action
                    action = 0
                else:
                    # use model to predict action
                    context_size = min(len(states), context_length)
                    context_states = np.array(states[-context_size:])
                    
                    # prepare action context
                    if len(actions) >= context_size - 1:
                        context_actions = np.array(actions[-(context_size-1):] + [0])
                    else:
                        context_actions = np.array(actions + [0] * (context_size - 1 - len(actions)))
                    
                    # calculate return-to-go
                    rtg = target_return - episode_return
                    context_rtgs = np.full(context_size, rtg)
                    
                    try:
                        action = model.get_action(
                            states=context_states,
                            actions=context_actions,
                            rtgs=context_rtgs.reshape(-1, 1),
                            device=device
                        )
                    except Exception as e:
                        print(f"Error in evaluation: {e}")
                        action = val_env.action_space.sample()
                
                # take step in environment
                next_obs, reward, terminated, truncated, _ = val_env.step(action)
                done = terminated or truncated
                
                actions.append(action)
                episode_return += reward
                
                obs = next_obs
                timestep += 1
            
            # check for success
            if isinstance(val_env, VelocityCartPoleEnv) and episode_return >= 450:
                successful_episodes += 1
            elif isinstance(val_env, FlickeringPendulumEnv) and episode_return >= -250:
                successful_episodes += 1
            elif isinstance(val_env, LiDARMountainCarEnv) and done and timestep < max_steps:
                successful_episodes += 1
            
            returns.append(episode_return)
            print(f"Episode {episode+1}: Return={episode_return:.1f}, Steps={timestep}")
        
        # calculate evaluation metrics
        mean_return = np.mean(returns)
        success_rate = successful_episodes / num_eval_episodes
        val_returns.append(mean_return)
        
        print(f"Validation: Mean Return={mean_return:.2f}, Success Rate={success_rate:.2%}")
        
        # early stopping and model saving
        if mean_return > best_val_return:
            best_val_return = mean_return
            best_model_state = model.state_dict()
            print(f"New best model with return {best_val_return:.2f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # save best model
    os.makedirs('models', exist_ok=True)
    best_model_path = f"models/memory_dt_{env_name}_{memory_type}_best.pt"
    torch.save(best_model_state if best_model_state else model.state_dict(), best_model_path)
    print(f"Best model saved to {best_model_path}")
    
    if hasattr(val_env, 'close'):
        val_env.close()
    
    # plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(val_returns)
    plt.title('Validation Returns')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Return')
    
    plt.tight_layout()
    plt.savefig(f"models/memory_dt_{env_name}_{memory_type}_training.png")
    plt.close()
    
    return model, train_losses, val_returns


def evaluate_memory_dt(model, env, num_episodes=10, render=False, target_return=None, context_length=20, debug=False, return_success_rate=True):
    """
    Evaluate a trained Memory Decision Transformer.
    
    Args:
        model: MemoryDecisionTransformer model
        env: Gymnasium environment
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        target_return: Target return for the agent (None for environment defaults)
        context_length: Context length for the transformer
        debug: Whether to print debug information
        return_success_rate: Whether to return success rate as third value
    
    Returns:
        mean_return: Mean episode return
        returns: List of individual episode returns
        success_rate: Fraction of successful episodes (if return_success_rate=True)
    """
    model.eval()
    model.reset_memory() # always reset memory when evaluating
    
    device = next(model.parameters()).device
    
    if target_return is None:
        if isinstance(env, VelocityCartPoleEnv):
            target_return = 500.0
        elif isinstance(env, FlickeringPendulumEnv):
            target_return = -200.0
        elif isinstance(env, LiDARMountainCarEnv):
            target_return = -100.0
        else:
            target_return = 500.0
    
    returns = []
    successful_episodes = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        model.reset_memory()  # reset memory state for each episode
        
        states = []
        actions = []
        
        episode_return = 0
        done = False
        timestep = 0
        max_steps = 1000
        
        while not done and timestep < max_steps:
            # process observation
            if isinstance(obs, dict):
                if 'observation' in obs and 'mask' in obs:
                    processed_obs = np.concatenate([
                        obs['observation'].flatten(),
                        np.array([obs['mask']], dtype=np.float32)
                    ])
                else:
                    processed_obs = np.concatenate([
                        o.flatten() if hasattr(o, 'flatten') else np.array([o], dtype=np.float32)
                        for o in obs.values()
                    ])
            else:
                processed_obs = obs
            
            states.append(processed_obs)
            
            # Get action
            if len(states) <= 1:
                # first timestep, use default action
                action = 0
            else:
                # use model to predict action
                context_size = min(len(states), context_length)
                context_states = np.array(states[-context_size:])
                
                # prepare action context
                if len(actions) >= context_size - 1:
                    context_actions = np.array(actions[-(context_size-1):] + [0])
                else:
                    context_actions = np.array(actions + [0] * (context_size - 1 - len(actions)))
                
                # calculate return-to-go
                rtg = target_return - episode_return
                context_rtgs = np.full(context_size, rtg)
                
                try:
                    action = model.get_action(
                        states=context_states,
                        actions=context_actions,
                        rtgs=context_rtgs.reshape(-1, 1),
                        device=device
                    )
                except Exception as e:
                    if debug:
                        print(f"Error in evaluation: {e}")
                    action = env.action_space.sample()
            
            # take step in environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            actions.append(action)
            episode_return += reward
            
            obs = next_obs
            timestep += 1
            
            if render and episode == 0:
                env.render()
        
        # check for success
        if isinstance(env, VelocityCartPoleEnv) and episode_return >= 450:
            successful_episodes += 1
        elif isinstance(env, FlickeringPendulumEnv) and episode_return >= -250:
            successful_episodes += 1
        elif isinstance(env, LiDARMountainCarEnv) and done and timestep < max_steps:
            successful_episodes += 1
        
        returns.append(episode_return)
        print(f"Episode {episode+1}: Return={episode_return:.1f}, Steps={timestep}")
    
    mean_return = np.mean(returns)
    success_rate = successful_episodes / num_episodes
    
    print(f"Evaluation Summary:")
    print(f"Mean Return: {mean_return:.2f}")
    print(f"Success Rate: {success_rate:.2%}")
    
    if return_success_rate:
        return mean_return, returns, success_rate
    else:
        return mean_return, returns


def preprocess_obs(obs):
    """Helper to preprocess observations for the model."""
    if isinstance(obs, dict):
        if 'observation' in obs and 'mask' in obs:
            preproc_obs = np.concatenate([
                obs['observation'].flatten(),
                np.array([obs['mask']], dtype=np.float32)
            ])
            
            is_visible = obs['mask'] == 1
            if not is_visible:
                print("Observation HIDDEN (flickering)")
        else:
            preproc_obs = np.concatenate([
                o.flatten() if hasattr(o, 'flatten') else np.array([o], dtype=np.float32)
                for o in obs.values()
            ])
    else:
        preproc_obs = obs

        # for velocity_cartpole - standard observation of 4 elements
        if len(preproc_obs) == 4:
            # 1. position
            preproc_obs[0] = preproc_obs[0] / 3.0
            
            # 2. velocity
            preproc_obs[1] = np.clip(preproc_obs[1] / 5.0, -1.0, 1.0)
            
            # 3. angle
            angle = preproc_obs[2]
            
            # 4. angular velocity
            preproc_obs[3] = np.clip(preproc_obs[3] / 5.0, -1.0, 1.0)
            
            # add sin and cos of the angle for better representation
            sin_theta = np.sin(angle)
            cos_theta = np.cos(angle)
            
            preproc_obs = np.array([
                preproc_obs[0],  # normalized position
                preproc_obs[1],  # normalized velocity
                sin_theta,       # sin of the angle
                cos_theta,       # cos of the angle
                preproc_obs[3]
            ], dtype=np.float32)
    
    # handling NaN and Inf
    if np.isnan(preproc_obs).any() or np.isinf(preproc_obs).any():
        print("WARNING: NaN or Inf in observation, replacing with zeros")
        preproc_obs = np.nan_to_num(preproc_obs, nan=0.0, posinf=0.0, neginf=0.0)
    
    return preproc_obs


def calculate_dataset_stats(dataset_path):
    """
    Calculates statistics of rewards in the dataset.

    Args:
        dataset_path: Path to the dataset directory
    
    Returns:
        dict: Dictionary with statistics (mean reward, min, max, etc.)
    """
    import glob
    import os
    import numpy as np
    from tqdm import tqdm
    
    files = glob.glob(os.path.join(dataset_path, 'train_data_*.npz'))
    files.sort()
    
    total_episodes = 0
    total_steps = 0
    rewards_per_episode = []
    lengths_per_episode = []
    
    print(f"Analyzing {len(files)} trajectories files in {dataset_path}")
    
    for file_path in tqdm(files):
        try:
            data = np.load(file_path)
            
            if 'reward' in data:
                rewards = data['reward']
            elif 'rewards' in data:
                rewards = data['rewards']
            else:
                print(f"WARNING: No rewards found in {file_path}")
                continue
            
            if len(rewards.shape) == 1:
                total_episodes += 1
                rewards_per_episode.append(np.sum(rewards))
                lengths_per_episode.append(len(rewards))
                total_steps += len(rewards)
            else:
                for episode_rewards in rewards:
                    if len(episode_rewards) > 0:
                        total_episodes += 1
                        rewards_per_episode.append(np.sum(episode_rewards))
                        lengths_per_episode.append(len(episode_rewards))
                        total_steps += len(episode_rewards)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # counting statistics
    rewards_per_episode = np.array(rewards_per_episode)
    lengths_per_episode = np.array(lengths_per_episode)
    
    stats = {
        'mean_reward': np.mean(rewards_per_episode),
        'median_reward': np.median(rewards_per_episode),
        'min_reward': np.min(rewards_per_episode),
        'max_reward': np.max(rewards_per_episode),
        'std_reward': np.std(rewards_per_episode),
        'total_episodes': total_episodes,
        'total_steps': total_steps,
        'mean_episode_length': np.mean(lengths_per_episode),
        'min_episode_length': np.min(lengths_per_episode),
        'max_episode_length': np.max(lengths_per_episode),
    }
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        stats[f'reward_p{p}'] = np.percentile(rewards_per_episode, p)
    
    print("\nDataset statistics:")
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Mean reward per episode: {stats['mean_reward']:.2f}")
    print(f"Median reward per episode: {stats['median_reward']:.2f}")
    print(f"Min/Max reward: {stats['min_reward']:.2f}/{stats['max_reward']:.2f}")
    print(f"Reward std: {stats['std_reward']:.2f}")
    print(f"Mean episode length: {stats['mean_episode_length']:.2f}")
    print(f"Reward percentiles: ")
    for p in percentiles:
        print(f"  {p}%: {stats[f'reward_p{p}']:.2f}")
    
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(rewards_per_episode, bins=30, alpha=0.7)
        plt.axvline(stats['mean_reward'], color='r', linestyle='--', label=f"Mean: {stats['mean_reward']:.2f}")
        plt.axvline(stats['median_reward'], color='g', linestyle='--', label=f"Median: {stats['median_reward']:.2f}")
        plt.axvline(stats['reward_p90'], color='orange', linestyle='--', label=f"90th percentile: {stats['reward_p90']:.2f}")
        plt.title('Distribution of Episode Rewards in Dataset')
        plt.xlabel('Total Reward')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        os.makedirs('plots', exist_ok=True)
        dataset_name = os.path.basename(os.path.normpath(dataset_path))
        plt.savefig(f"plots/{dataset_name}_rewards_histogram.png")
        print(f"Saved histogram to plots/{dataset_name}_rewards_histogram.png")
        plt.close()
    except Exception as e:
        print(f"Could not generate histogram: {e}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Execution of dataset operations and model")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--stats_only', action='store_true', help='Only print dataset statistics without training')
    parser.add_argument('--env', type=str, default='velocity_cartpole', 
                      choices=['velocity_cartpole', 'flickering_pendulum', 'lidar_mountain_car'], 
                      help='Environment name')
    parser.add_argument('--memory', type=str, default='gru', choices=['gru', 'lstm', 'none'], 
                      help='Memory type')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    
    args = parser.parse_args()
    
    if args.stats_only:
        # only calculate dataset statistics
        calculate_dataset_stats(args.dataset)
    else:
        # train memory dt
        memory_type = None if args.memory == 'none' else args.memory
        train_memory_dt(
            env_name=args.env,
            dataset_path=args.dataset,
            n_epochs=args.epochs,
            memory_type=memory_type
        )