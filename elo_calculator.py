import random
from itertools import combinations

import numpy as np
from policy_opponents import (
    CriticalFirstPolicy,
    DefensivePolicy,
    CornerEdgePolicy,
    AggressivePolicy,
    PolicyOpponent,
    ValidationSeededPolicy,
    BuildAndExplodePolicy,
    GeminiPolicyV1
)
#import chainreaction_env_policy as cr_env
#from chainreaction_env_policy import ChainReactionEnv
from chainreaction_env_headless import ChainReactionHeadless as ChainReactionEnv
import torch
from model import PPOGridNet

class PPOPolicy(PolicyOpponent):
    """Wrapper for pre-trained PPO model to match the policy interface."""
    def __init__(self, model_path, grid_size=5):
        super().__init__(grid_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model and load state dict
        self.model = PPOGridNet(grid_size=grid_size).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set to evaluation mode

    def get_move(self, grid, player):
        """Convert grid to model input format and get action."""
        # Convert grid to model's input format (2-channel format)
        state = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)
        
        for i in range(self.grid_size):
            for j in range(len(grid[0])):
                cell = grid[i][j]
                if cell is not None:
                    cell_player, atoms = cell
                    if cell_player == player:
                        state[0, i, j] = atoms
                    else:
                        state[1, i, j] = atoms
        
        # Convert to tensor and get model prediction
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, _ = self.model(state_tensor)  # Get raw logits

            valid_moves_mask_bool = np.zeros(self.grid_size * self.grid_size, dtype=bool)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    cell = grid[i][j]
                    if cell is None or cell[0] == player:
                        valid_moves_mask_bool[i * self.grid_size + j] = True

            if not np.any(valid_moves_mask_bool):
                return None

            valid_mask_tensor = torch.tensor(valid_moves_mask_bool, dtype=torch.bool, device=self.device).unsqueeze(0)
            masked_logits = logits.masked_fill(~valid_mask_tensor, -float('inf'))
            action = torch.argmax(masked_logits, dim=1).item()

            row = action // self.grid_size
            col = action % self.grid_size
            return (row, col)
        

def play_single_game(env, policy1, policy2, policy1_starts=True):
    """Play a single game between two policies."""
    env.start_new_game(
        grid_size=env.grid_size,
        max_moves=env.max_moves,
        opponent_first=not policy1_starts,
        opponent_policy=policy2
    )
    
    done = False
    
    while not done:
        # Get current state
        grid = env.grid
        current_player = env.current_player
        
        # Get move from current policy
        move = policy1.get_move(grid, current_player)
        if move is None:
            break
            
        # Convert move to action index
        action_idx = move[0] * env.grid_size + move[1]
        
        # Make move
        next_state, rw = env.step(action_idx)
        done = env.is_done()
    
    winner = env.get_winner()
    if winner == 0:
        return 1.0 
    elif winner == 1:
        return 0.0 
    else:
        return 0.5  # Draw

def update_elo(rating_a, rating_b, actual_score_a, k_factor=32):
    """Update ELO ratings based on game outcome."""
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    change = k_factor * (actual_score_a - expected_a)
    return rating_a + change, rating_b - change

def calculate_elo(model_paths=None, policy_elo_ratings=None, grid_size=5, max_moves=100, games_per_matchup=100, k_factor=32):
    """
    Calculate ELO ratings for policies and optionally for provided models.
    
    Args:
        model_paths: Optional dict mapping model names to their weight file paths
        policy_elo_ratings: Required dict of policy ELO ratings when model_paths is provided
        grid_size: Size of the game grid (default: 5)
        max_moves: Maximum moves per game (default: 100)
        games_per_matchup: Number of games to play per matchup (default: 100)
        k_factor: ELO K-factor for rating adjustments (default: 32)
    
    Returns:
        dict: Mapping of policy/model names to their final ELO ratings
    """
    # Initialize all policies with their names
    policies = [
        ("critical", CriticalFirstPolicy(grid_size)),
        ("defensive", DefensivePolicy(grid_size)),
        ("corner", CornerEdgePolicy(grid_size)),
        ("aggressive", AggressivePolicy(grid_size)),
        ("random", ValidationSeededPolicy(grid_size)),
        ("build", BuildAndExplodePolicy(grid_size)),
        ("gemini", GeminiPolicyV1(grid_size))
    ]
    
    # If we're calculating ELO for models, we need the policy ratings
    if model_paths and not policy_elo_ratings:
        raise ValueError("policy_elo_ratings must be provided when calculating ELO for models")
    
    # Add models if provided
    if model_paths:
        model_policies = []
        for name, path in model_paths.items():
            model_policies.append((name, PPOPolicy(path, grid_size)))
            
        # Initialize ELO ratings for models (starting at 1200)
        elo_ratings = {name: 1200 for name, _ in model_policies}
        # Add the existing policy ratings
        elo_ratings.update(policy_elo_ratings)
        
        # Only play model vs policy and model vs model games
        model_names = set(model_paths.keys())
        matchups = []
        
        # Model vs Policy matchups
        for model_name, model_policy in model_policies:
            for policy_name, policy in policies:
                matchups.append(((model_name, model_policy), (policy_name, policy)))
        
        # Model vs Model matchups
        # for i, (name1, policy1) in enumerate(model_policies):
        #     for name2, policy2 in model_policies[i+1:]:
        #         matchups.append(((name1, policy1), (name2, policy2)))
                
    else:
        # Play all policies against each other
        matchups = list(combinations(policies, 2))
        # Initialize ELO ratings (starting at 1200)
        elo_ratings = {name: 1200 for name, _ in policies}
    
    print("Starting ELO calculation...")
    print(f"Each matchup will play {games_per_matchup} games ({games_per_matchup//2} games per starting position)\n")
    
    # Create a single environment instance to reuse
    env = ChainReactionEnv(grid_size=grid_size, max_moves=max_moves)
    
    # Play all matchups
    for (name1, policy1), (name2, policy2) in matchups:
        print(f"\nMatchup: {name1} vs {name2}")
        total_score1 = 0
        
        # Play half the games with policy1 starting
        games_per_start = games_per_matchup // 2
        for _ in range(games_per_start):
            score = play_single_game(env, policy1, policy2, policy1_starts=True)
            total_score1 += score
            
            # Update ELO after each game
            rating1, rating2 = update_elo(
                elo_ratings[name1],
                elo_ratings[name2],
                score,
                k_factor
            )
            elo_ratings[name1] = rating1
            elo_ratings[name2] = rating2
        
        # Play half the games with policy2 starting
        for _ in range(games_per_start):
            score = play_single_game(env, policy1, policy2, policy1_starts=False)
            total_score1 += score
            
            # Update ELO after each game
            rating1, rating2 = update_elo(
                elo_ratings[name1],
                elo_ratings[name2],
                score,
                k_factor
            )
            elo_ratings[name1] = rating1
            elo_ratings[name2] = rating2
        
        avg_score1 = total_score1 / games_per_matchup
        print(f"Results: {name1} scored {avg_score1:.2f}, {name2} scored {1-avg_score1:.2f}")
    
    # Print final ELO ratings
    print("\nFinal ELO Ratings:")
    sorted_ratings = sorted(
        elo_ratings.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for name, rating in sorted_ratings:
        print(f"{name}: {rating:.1f}")
    
    return elo_ratings

if __name__ == "__main__":
    # Example usage:
    
    # Calculate ELO for all policies
    print("Calculating ELO for all policies:")
    
    policy_elo_ratings = {
        'critical': 1150.2,
        'defensive': 1122.2,
        'corner': 1148.6,
        'aggressive': 1070.1,
        'random': 1106.9,
        'build': 1318.2,
        'gemini': 1500.0
    }
    #policy_elo_ratings = calculate_elo()
    print(policy_elo_ratings)
    
    # Calculate ELO for specific models against policies
    print("\nCalculating ELO for specific models:")
    model_paths = {
        "ModelA": "PPOnet/chain_reaction_A.pth"
    }
    model_elo_ratings = calculate_elo(
        model_paths=model_paths,
        policy_elo_ratings=policy_elo_ratings
    ) 