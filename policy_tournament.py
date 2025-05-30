import random
from itertools import combinations
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
import numpy as np
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

class PolicyTournament:
    def __init__(self, grid_size=5, max_moves=100, games_per_matchup=100, k_factor=32, ppo_model_path=None):
        self.grid_size = grid_size
        self.max_moves = max_moves
        self.games_per_matchup = games_per_matchup
        self.k_factor = k_factor
        
        # Initialize all policies with their names
        self.policies = [
            ("CriticalFirst", CriticalFirstPolicy(grid_size)),
            ("Defensive", DefensivePolicy(grid_size)),
            ("CornerEdge", CornerEdgePolicy(grid_size)),
            ("Aggressive", AggressivePolicy(grid_size)),
            ("ValidationRandom", ValidationSeededPolicy(grid_size)),
            ("BuildAndExplode", BuildAndExplodePolicy(grid_size)),
            ("Gemini", GeminiPolicyV1(grid_size))
        ]
        
        # Add PPO model if path is provided
        if ppo_model_path:
            self.policies.append(("PPO", PPOPolicy(ppo_model_path, grid_size)))
        
        # Initialize ELO ratings (starting at 1200)
        self.elo_ratings = {name: 1200 for name, _ in self.policies}
        
    def expected_score(self, rating_a, rating_b):
        """Calculate expected score using ELO formula."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_elo(self, rating_a, rating_b, actual_score_a):
        """Update ELO ratings based on game outcome."""
        expected_a = self.expected_score(rating_a, rating_b)
        change = self.k_factor * (actual_score_a - expected_a)
        return rating_a + change, rating_b - change
    
    def play_single_game(self, policy1, policy2, policy1_starts=True):
        """Play a single game between two policies."""
        env = ChainReactionEnv(grid_size=self.grid_size, max_moves=self.max_moves)
        env.start_new_game(
            grid_size=self.grid_size,
            max_moves=self.max_moves,
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
            action_idx = move[0] * self.grid_size + move[1]
            
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
    
    def run_tournament(self):
        """Run a complete tournament between all policies."""
        print("Starting tournament...")
        print(f"Each matchup will play {self.games_per_matchup} games ({self.games_per_matchup//2} games per starting position)\n")
        
        # Get all possible pairs of policies
        for (name1, policy1), (name2, policy2) in combinations(self.policies, 2):
            print(f"\nMatchup: {name1} vs {name2}")
            total_score1 = 0
            
            # Play half the games with policy1 starting
            games_per_start = self.games_per_matchup // 2
            for _ in range(games_per_start):
                score = self.play_single_game(policy1, policy2, policy1_starts=True)
                total_score1 += score
                
                # Update ELO after each game
                rating1, rating2 = self.update_elo(
                    self.elo_ratings[name1],
                    self.elo_ratings[name2],
                    score
                )
                self.elo_ratings[name1] = rating1
                self.elo_ratings[name2] = rating2
            
            # Play half the games with policy2 starting
            for _ in range(games_per_start):
                score = self.play_single_game(policy1, policy2, policy1_starts=False)
                total_score1 += score
                
                # Update ELO after each game
                rating1, rating2 = self.update_elo(
                    self.elo_ratings[name1],
                    self.elo_ratings[name2],
                    score
                )
                self.elo_ratings[name1] = rating1
                self.elo_ratings[name2] = rating2
            
            avg_score1 = total_score1 / self.games_per_matchup
            print(f"Results: {name1} scored {avg_score1:.2f}, {name2} scored {1-avg_score1:.2f}")
        
        # Print final ELO ratings
        print("\nFinal ELO Ratings:")
        sorted_ratings = sorted(
            self.elo_ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for name, rating in sorted_ratings:
            print(f"{name}: {rating:.1f}")

if __name__ == "__main__":
    # Run tournament with 100 games per matchup
    tournament = PolicyTournament(
        grid_size=5,
        max_moves=100,
        games_per_matchup=100,
        k_factor=32,
        ppo_model_path="PPOnet/chain_reaction_A.pth"
    )
    tournament.run_tournament() 