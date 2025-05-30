import numpy as np
from policy_opponents import (
    CriticalFirstPolicy,
    DefensivePolicy,
    CornerEdgePolicy,
    AggressivePolicy,
    SeededRandomPolicy,
    BuildAndExplodePolicy,
    ValidationSeededPolicy,
    ModelPolicy,
    GeminiPolicyV1
)

class ChainReactionHeadless:
    """Headless chain reaction environment with policy support."""
    
    def __init__(self, grid_size=5, max_moves=100):
        self.grid_size = grid_size
        self.max_moves = max_moves
        self.my_player_id = 0  # Will be set in reset based on who goes first
        
        # Map policy names to their classes
        self.policy_map = {
            'critical': CriticalFirstPolicy,
            'defensive': DefensivePolicy,
            'corner': CornerEdgePolicy,
            'aggressive': AggressivePolicy,
            'random': SeededRandomPolicy,
            'build': BuildAndExplodePolicy,
            'validation': ValidationSeededPolicy,
            'model': ModelPolicy,  # Add ModelPolicy to the map
            'gemini': GeminiPolicyV1  # Add GeminiPolicy to the map
        }
        
        # Store policy settings for reuse
        self.current_policy_name = 'random'
        self.current_model_path = None
        
        # Default to random policy
        self.opponent_policy = SeededRandomPolicy()
        
        # Initialize environment state
        self.grid = None
        self.current_player = None
        self.game_over = None
        self.winner = None
        self.move_count = 0
        self.reset()
    
    def reset(self):
        """Reset the game state."""
        # Grid: None means empty; otherwise (player, atom_count)
        self.grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.current_player = 0  # Always start with player 0
        self.game_over = False
        self.winner = None
        self.move_count = 0
    
    def get_max_capacity(self, r, c):
        """Get maximum atom capacity for a cell."""
        # Corners hold 2, edges hold 3, center hold 4
        if (r in (0, self.grid_size-1)) and (c in (0, self.grid_size-1)):
            return 2
        if r in (0, self.grid_size-1) or c in (0, self.grid_size-1):
            return 3
        return 4
    
    def get_adjacent(self, r, c):
        """Return list of adjacent cell coordinates."""
        neighbors = []
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                neighbors.append((nr,nc))
        return neighbors
    
    def process_splits(self):
        """Process chain reactions until no cell exceeds its capacity."""
        queue = []
        # Initial scan for cells that need to split
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = self.grid[r][c]
                if cell is not None and cell[1] >= self.get_max_capacity(r,c):
                    queue.append((r,c))
        
        while queue:
            r, c = queue.pop(0)
            cell = self.grid[r][c]
            if not cell: continue
            
            player, atoms = cell
            cap = self.get_max_capacity(r,c)
            if atoms < cap: continue
            
            # Split atoms to adjacent cells
            self.grid[r][c] = None
            for nr, nc in self.get_adjacent(r,c):
                if self.grid[nr][nc] is None:
                    self.grid[nr][nc] = (player, 1)
                else:
                    _, curr_atoms = self.grid[nr][nc]
                    self.grid[nr][nc] = (player, curr_atoms + 1)
                
                # Check if new cell needs to split
                if self.grid[nr][nc][1] >= self.get_max_capacity(nr,nc):
                    queue.append((nr,nc))
            
            # Check for win condition
            atoms_p0 = sum(1 for r in range(self.grid_size) for c in range(self.grid_size) 
                         if self.grid[r][c] and self.grid[r][c][0] == 0)
            atoms_p1 = sum(1 for r in range(self.grid_size) for c in range(self.grid_size) 
                         if self.grid[r][c] and self.grid[r][c][0] == 1)
            
            if atoms_p0 == 0 and atoms_p1 > 0:
                self.game_over = True
                self.winner = 1
                break
            if atoms_p1 == 0 and atoms_p0 > 0:
                self.game_over = True
                self.winner = 0
                break
    
    @staticmethod
    def valid_moves_from_state(state):
        """
        Return valid moves mask from a state array.
        Args:
            state: numpy array of shape (2, grid_size, grid_size) where
                  channel 0 contains current player's atoms
                  channel 1 contains opponent's atoms
        Returns:
            Boolean mask of valid moves (True where move is valid)
        """
        my_atoms = state[0]  # Current player's atoms
        opp_atoms = state[1]  # Opponent's atoms
        # Move is valid if cell is empty (both channels 0) or has my atoms
        return ((my_atoms + opp_atoms == 0) | (my_atoms > 0)).flatten()

    @staticmethod
    def valid_moves_from_states(states):
        """
        Return valid moves mask from batch of state arrays.
        Args:
            states: numpy array of shape (batch_size, 2, grid_size, grid_size) where
                   channel 0 contains current player's atoms
                   channel 1 contains opponent's atoms
        Returns:
            Boolean mask of valid moves for each state in batch
        """
        batch_size, _, G, _ = states.shape
        my_atoms = states[:, 0, :, :]  # shape (N,G,G)
        opp_atoms = states[:, 1, :, :]  # shape (N,G,G)
        # Move is valid if cell is empty (both channels 0) or has my atoms
        return ((my_atoms + opp_atoms == 0) | (my_atoms > 0)).reshape(batch_size, G*G)

    def step(self, action):
        """Take a step in the environment."""
        if self.game_over:
            return self.get_state(), 0
        
        # Convert action to row, col
        row, col = action // self.grid_size, action % self.grid_size
        
        # Validate move
        cell = self.grid[row][col]
        if cell is not None and cell[0] != self.current_player:
            return self.get_state(), -100.0
        
        # Track opponent atoms before move
        opponent = 1 if self.current_player == 0 else 0
        opponent_atoms_before = sum(c[1] for r in self.grid for c in r if c and c[0] == opponent)
        
        # Place atom
        if self.grid[row][col] is None:
            self.grid[row][col] = (self.current_player, 1)
        else:
            _, atoms = self.grid[row][col]
            self.grid[row][col] = (self.current_player, atoms + 1)
        
        # Process chain reactions
        self.process_splits()
        
        # Calculate reward after player's move but before opponent's move
        opponent_atoms_after = sum(c[1] for r in self.grid for c in r if c and c[0] == opponent)
        reward = float(opponent_atoms_before - opponent_atoms_after)
        
        # Adjust reward based on game outcome
        if self.game_over and self.winner is not None:
            if self.winner == 0:  # Player 1 wins
                reward += 1.0
            elif self.winner == 1:  # Player 2 wins
                reward -= 1.0
        
        # Switch player
        self.current_player = 1 if self.current_player == 0 else 0
        self.move_count += 1
        
        # Check for max moves
        if self.move_count >= self.max_moves:
            self.game_over = True
        
        # Make opponent move if game not over
        if not self.game_over and self.opponent_policy:
            move = self.opponent_policy.get_move(self.grid, self.current_player)
            if move:
                opp_row, opp_col = move
                if self.grid[opp_row][opp_col] is None:
                    self.grid[opp_row][opp_col] = (self.current_player, 1)
                else:
                    _, atoms = self.grid[opp_row][opp_col]
                    self.grid[opp_row][opp_col] = (self.current_player, atoms + 1)
                
                self.process_splits()
                self.current_player = 1 if self.current_player == 0 else 0
                self.move_count += 1
        
        return self.get_state(), reward
    
    def get_state(self):
        """Return observation of shape (2, grid_size, grid_size).
        Channel 0 always contains my player's atoms, Channel 1 always contains opponent atoms."""
        obs = np.zeros((2, self.grid_size, self.grid_size), dtype=np.int32)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = self.grid[r][c]
                if cell is not None:
                    player, atoms = cell
                    # If player is my_player_id, put in channel 0, else channel 1
                    channel = 0 if player == self.my_player_id else 1
                    obs[channel, r, c] = atoms
        return obs
    
    def valid_moves_mask(self):
        """Return a boolean mask for valid moves."""
        mask = np.zeros(self.grid_size * self.grid_size, dtype=bool)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = self.grid[r][c]
                if cell is None or cell[0] == self.current_player:
                    idx = r * self.grid_size + c
                    mask[idx] = True
        return mask
    
    def is_done(self):
        """Check if game is finished."""
        return self.game_over
    
    def get_winner(self):
        """Get winner of the game (0 for my player, 1 for opponent, None if ongoing)."""
        if self.winner is None:
            return None
        # Return 0 if my_player_id won, 1 if opponent won
        return 0 if self.winner == self.my_player_id else 1

    def set_opponent_policy(self, policy_name, model_path=None, game_seed=None):
        """
        Set the opponent policy.
        Args:
            policy_name: Name of the policy to use
            model_path: Path to the model weights file (required if policy_name is 'model')
            game_seed: Specific seed to use for validation/random policy (optional)
        """
        if policy_name == 'model':
            if model_path is None:
                if self.current_model_path is None:
                    raise ValueError("model_path must be provided when using 'model' policy")
                # Reuse previous model path if none provided
                model_path = self.current_model_path
            self.opponent_policy = self.policy_map[policy_name](model_path=model_path, grid_size=self.grid_size)
            # Store settings for reuse
            self.current_model_path = model_path
        elif policy_name == 'validation':
            self.opponent_policy = self.policy_map[policy_name](grid_size=self.grid_size)
            if game_seed is not None:  # If a specific seed for this game is provided
                self.opponent_policy.set_game_seed(game_seed)
            else:  # Use the isolated validation seed generator
                from policy_opponents import get_next_validation_seed
                self.opponent_policy.set_game_seed(get_next_validation_seed())
        elif policy_name == 'random':
            self.opponent_policy = self.policy_map[policy_name](grid_size=self.grid_size)
            if game_seed is not None:  # If a specific seed for this game is provided
                self.opponent_policy.set_game_seed(game_seed)
            else:  # Use the isolated training seed generator
                from policy_opponents import get_next_training_seed
                self.opponent_policy.set_game_seed(get_next_training_seed())
        elif policy_name in self.policy_map:
            self.opponent_policy = self.policy_map[policy_name](grid_size=self.grid_size)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")
        
        # Store the policy name for reuse
        self.current_policy_name = policy_name

    def start_new_game(self, grid_size=None, opponent_policy=None, opponent_first=False, max_moves=None, model_path=None, game_seed=None):
        """
        Start a new game with specified settings.
        Args:
            grid_size: Size of the grid (default: None to reuse current)
            opponent_policy: Policy name or instance (default: None to reuse previous)
            opponent_first: Whether opponent moves first (default: False)
            max_moves: Maximum moves before game is drawn (default: None to reuse current)
            model_path: Path to model weights file (required if opponent_policy is 'model')
            game_seed: Specific seed to use for validation/random policy (optional)
        """
        # Update grid_size and max_moves if provided
        if grid_size is not None:
            self.grid_size = grid_size
        if max_moves is not None:
            self.max_moves = max_moves
            
        # Set up the opponent policy only if explicitly provided
        if opponent_policy is not None:
            if isinstance(opponent_policy, str):
                self.set_opponent_policy(opponent_policy, model_path=model_path, game_seed=game_seed)
            else:
                self.opponent_policy = opponent_policy
                self.current_policy_name = 'custom'
                self.current_model_path = None
        elif self.current_policy_name in ['validation', 'random'] and game_seed is not None:
            # Update seed even if reusing existing validation/random policy
            self.opponent_policy.set_game_seed(game_seed)
        
        # Reset the environment
        self.reset()
        self.my_player_id = 1 if opponent_first else 0
        
        # If opponent goes first, make their move
        if opponent_first and self.opponent_policy:
            move = self.opponent_policy.get_move(self.grid, self.current_player)
            if move:
                opp_row, opp_col = move
                # First move is always placing 1 atom in empty cell
                self.grid[opp_row][opp_col] = (self.current_player, 1)
                self.current_player = 1 if self.current_player == 0 else 0
                self.move_count += 1
        
        return self.get_state()

if __name__ == "__main__":
    # Example usage showing independent instances
    print("Instance 1:")
    game1 = ChainReactionHeadless(grid_size=5)
    state = game1.start_new_game(
        opponent_policy='model',
        model_path='path/to/model1.pth'
    )
    print("Game 1 initial state shape:", state.shape)
    
    # Test batch processing
    import numpy as np
    batch_states = np.zeros((32, 2, 5, 5))  # Example batch of 32 states
    valid_moves = game1.valid_moves_from_states(batch_states)
    print("Valid moves shape for batch:", valid_moves.shape)
    
    print("\nInstance 2:")
    game2 = ChainReactionHeadless(grid_size=5)
    state = game2.start_new_game(
        opponent_policy='critical'
    )
    print("Game 2 initial state shape:", state.shape)
    
    # Each instance maintains its own state
    print("\nStarting new games with previous settings:")
    state1 = game1.start_new_game(opponent_first=True)  # Will use model policy
    state2 = game2.start_new_game(opponent_first=True)  # Will use critical policy
    print("Game 1 policy:", game1.current_policy_name)
    print("Game 2 policy:", game2.current_policy_name) 