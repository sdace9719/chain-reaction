import random
import numpy as np
import os
import sys
import torch
from torch.distributions import Categorical # Needed for sampling strategy
import model # Assuming model.py contains PPOGridNet

# Player identifiers
PLAYER1 = 1
PLAYER2 = 2

# Define default strategy weights here for clarity
DEFAULT_STRATEGY_WEIGHTS = {'random': 0.2, 'sample': 0.8, 'argmax': 0.2}
VALID_STRATEGIES = list(DEFAULT_STRATEGY_WEIGHTS.keys())

class ChainReactionEnvMixedOpponent:
    """Chain Reaction environment using a PPOGridNet model for opponent moves with a configurable mixed strategy.
       Accepts partial strategy dictionaries.
    """
    def __init__(self, grid_size=5, opponent_model_path=None, opponent_device=None, strategy_weights=None):
        self.grid_size = grid_size
        self.opponent_model = None
        self.opponent_device = opponent_device
        self._loaded_opponent_path = None
        self.strategy_weights = None # Will be set below

        # --- Opponent Model Path Check & Load --- 
        if opponent_model_path is None: print(f"FATAL ERROR: opponent_model_path must be provided..."); sys.exit(1)
        if not os.path.exists(opponent_model_path): print(f"FATAL ERROR: Opponent weights not found: {opponent_model_path}"); sys.exit(1)
        if self.opponent_device is None: self.opponent_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.opponent_model = model.PPOGridNet(grid_size=self.grid_size, load_weights=opponent_model_path, eval_mode=True).to(self.opponent_device)
            self.opponent_model.eval()
            self._loaded_opponent_path = opponent_model_path
            print(f"Internal opponent model loaded from: {self._loaded_opponent_path}...")
        except Exception as e: print(f"FATAL ERROR loading opponent model: {e}"); sys.exit(1)
        # --- End Model Load --- 

        # --- Strategy Weights Setup (Handles Partial Dicts) --- 
        final_weights = {key: 0.0 for key in VALID_STRATEGIES} # Initialize all to 0

        if strategy_weights is None:
            # Use default if nothing provided
            final_weights.update(DEFAULT_STRATEGY_WEIGHTS)
            print(f"Using default opponent strategy weights: {final_weights}")
        elif isinstance(strategy_weights, dict):
            # Validate provided keys and values
            invalid_keys = set(strategy_weights.keys()) - set(VALID_STRATEGIES)
            if invalid_keys:
                print(f"FATAL ERROR: Invalid keys in strategy_weights: {invalid_keys}. Allowed keys: {VALID_STRATEGIES}."); sys.exit(1)
            if not all(isinstance(v, (int, float)) and v >= 0 for v in strategy_weights.values()):
                 print("FATAL ERROR: strategy_weights values must be non-negative numbers."); sys.exit(1)
            
            # Update final weights with provided values (missing keys remain 0.0)
            final_weights.update(strategy_weights)
            print(f"Using effective opponent strategy weights: {final_weights}")
        else:
            # Invalid type provided
            print("FATAL ERROR: strategy_weights must be a dictionary or None."); sys.exit(1)

        # Ensure weights sum is positive for random.choices
        if sum(final_weights.values()) <= 0:
             print(f"FATAL ERROR: Sum of effective strategy weights must be positive. Got: {final_weights}"); sys.exit(1)

        self.strategy_weights = final_weights # Store the final 3-key dict
        # --- End Strategy Weights Setup ---
        
        self.grid = None; self.current_player = None; self.game_over = None; self.winner = None; self.player1_played = None; self.player2_played = None
        self.reset()

    def reset(self):
        """Reset the game to the initial state and return the initial observation."""
        self.grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.current_player = PLAYER1
        self.game_over = False
        self.winner = None
        self.player1_played = False
        self.player2_played = False
        return self.get_state(perspective_player=PLAYER1)

    def is_done(self):
        """Return True if the game has ended."""
        return self.game_over

    def get_state(self, perspective_player=PLAYER1):
        """Return observation of shape (2, grid_size, grid_size) from perspective player's view."""
        obs = np.zeros((2, self.grid_size, self.grid_size), dtype=np.int32)
        opponent_player = 3 - perspective_player
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = self.grid[r][c]
                if cell is not None:
                    p, a = cell
                    if p == perspective_player: obs[0, r, c] = a
                    elif p == opponent_player: obs[1, r, c] = a
        return obs

    def valid_moves_mask(self, perspective_player=PLAYER1):
        """Return a boolean mask for valid moves of the perspective_player."""
        mask = np.zeros(self.grid_size * self.grid_size, dtype=bool)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = self.grid[r][c]
                if cell is None or cell[0] == perspective_player:
                    mask[r * self.grid_size + c] = True
        return mask

    def get_max_capacity(self, r, c):
        corners = (r == 0 or r == self.grid_size - 1) and (c == 0 or c == self.grid_size - 1)
        edges = (r == 0 or r == self.grid_size - 1) or (c == 0 or c == self.grid_size - 1)
        if corners: return 2
        elif edges: return 3
        else: return 4

    def get_adjacent(self, r, c):
        neighbors = []
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                neighbors.append((nr,nc))
        return neighbors

    def _apply_action(self, action, player):
        row, col = action // self.grid_size, action % self.grid_size
        cell = self.grid[row][col]
        if cell is not None and cell[0] != player:
            print(f"FATAL ERROR: Player {player} attempted invalid move on ({row},{col}) owned by Player {cell[0]}. Action: {action}")
            os._exit(1)
        if cell is None:
            self.grid[row][col] = (player, 1)
        else:
            _, current_atoms = cell
            self.grid[row][col] = (player, current_atoms + 1)
        if player == PLAYER1: self.player1_played = True
        else: self.player2_played = True

    def process_splits(self):
        queue = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = self.grid[r][c]
                if cell is not None and cell[1] >= self.get_max_capacity(r,c):
                    queue.append((r,c))
        while queue and not self.game_over:
            r, c = queue.pop(0)
            cell = self.grid[r][c]
            if cell is None: continue
            p, a = cell
            cap = self.get_max_capacity(r, c)
            if a < cap: continue
            self.grid[r][c] = None
            for nr, nc in self.get_adjacent(r, c):
                neighbor_cell = self.grid[nr][nc]
                if neighbor_cell is None: self.grid[nr][nc] = (p, 1)
                else: self.grid[nr][nc] = (p, neighbor_cell[1] + 1)
                if self.grid[nr][nc][1] >= self.get_max_capacity(nr, nc):
                    if (nr, nc) not in queue: queue.append((nr, nc))
            self.check_win_condition()

    def check_win_condition(self):
        if not self.player1_played or not self.player2_played: return
        player1_atoms = sum(c[1] for r in self.grid for c in r if c and c[0] == PLAYER1)
        player2_atoms = sum(c[1] for r in self.grid for c in r if c and c[0] == PLAYER2)
        if player1_atoms > 0 and player2_atoms == 0: self.game_over = True; self.winner = PLAYER1
        elif player2_atoms > 0 and player1_atoms == 0: self.game_over = True; self.winner = PLAYER2

    def get_random_move(self, player):
        """Return a random valid move (flat index) for the specified player, using tuple selection like baseline."""
        valid_coords = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = self.grid[r][c]
                if cell is None or cell[0] == player:
                    valid_coords.append((r, c)) # Collect valid (r, c) tuples
        
        if not valid_coords:
            print(f"Warning: get_random_move called for Player {player} with no valid moves.")
            return -1 # Indicate failure
            
        # Choose a random tuple
        chosen_r, chosen_c = random.choice(valid_coords) 
        
        # Convert the chosen tuple back to a flat index
        flat_index = chosen_r * self.grid_size + chosen_c
        return flat_index

    def step(self, action):
        """Execute agent (P1) move, then internal opponent (P2) move using the configured mixed strategy.
           Uses baseline random logic when strategy is 'random'.
           Returns state after P1 splits (P1 perspective) and reward for P1's turn."""
        if self.game_over: return self.get_state(perspective_player=PLAYER1), 0
        player = PLAYER1; opponent = PLAYER2
        r_act, c_act = action // self.grid_size, action % self.grid_size
        cell_act = self.grid[r_act][c_act]
        if cell_act is not None and cell_act[0] != player: os._exit(1)
        opponent_atoms_before = sum(c[1] for row in self.grid for c in row if c and c[0] == opponent)
        self._apply_action(action, player)
        self.process_splits()
        opponent_atoms_after = sum(c[1] for row in self.grid for c in row if c and c[0] == opponent)
        reward = float(opponent_atoms_before - opponent_atoms_after)
        state_after_player_move = self.get_state(perspective_player=player)
        done_after_player_move = self.game_over
        # --- End P1 Turn Logic --- 

        # --- Internal Opponent's (Player 2) Mixed Strategy Turn ---
        if not done_after_player_move:
            opp_valid_mask_np = self.valid_moves_mask(perspective_player=opponent)
            if not np.any(opp_valid_mask_np):
                if not self.game_over: self.game_over = True; self.winner = player; print("Warning: Opponent had no moves...")
            else:
                # --- Choose strategy based on stored weights ---
                population = list(self.strategy_weights.keys())
                weights = list(self.strategy_weights.values())
                strategy = random.choices(population=population, weights=weights, k=1)[0]
                # --- End Strategy Choice ---

                opp_action = -1 # Default invalid action

                # --- Determine Opponent Action based on Strategy --- 
                if strategy == "random":
                    # Use baseline random logic (tuple selection)
                    valid_coords = []
                    for r in range(self.grid_size):
                        for c in range(self.grid_size):
                            cell = self.grid[r][c]
                            if cell is None or cell[0] == opponent:
                                valid_coords.append((r, c))
                    if valid_coords:
                        chosen_r, chosen_c = random.choice(valid_coords)
                        opp_action = chosen_r * self.grid_size + chosen_c # Convert to flat index
                    else:
                        # Should have been caught by np.any(opp_valid_mask_np) check above
                        print(f"Warning: Strategy 'random' found no valid_coords despite mask check passing.")

                elif strategy == "sample" or strategy == "argmax":
                    # Use the loaded PPO model
                    opp_obs_np = self.get_state(perspective_player=opponent)
                    opp_obs_tensor = torch.tensor(opp_obs_np, dtype=torch.float32).unsqueeze(0).to(self.opponent_device)
                    opp_valid_mask_torch = torch.tensor(opp_valid_mask_np, dtype=torch.bool, device=self.opponent_device)

                    with torch.no_grad():
                        model_output = self.opponent_model(opp_obs_tensor)
                        if isinstance(model_output, tuple): logits = model_output[0]
                        else: logits = model_output

                        if logits.dim() == 2 and logits.size(0) == 1: logits_squeezed = logits.squeeze(0)
                        elif logits.dim() == 1: logits_squeezed = logits
                        else: logits_squeezed = logits.view(-1)

                        if logits_squeezed.shape[0] != opp_valid_mask_torch.shape[0]:
                            raise ValueError(f"Opponent Logits shape {logits.shape} -> {logits_squeezed.shape} incompatible with mask {opp_valid_mask_torch.shape}")

                        logits_squeezed[~opp_valid_mask_torch] = -float('inf')

                        if strategy == "sample":
                            action_dist = Categorical(logits=logits_squeezed)
                            opp_action = action_dist.sample().item()
                        else: # strategy == "argmax"
                            opp_action = torch.argmax(logits_squeezed).item()
                # --- End Action Determination ---

                # --- Apply the chosen opponent action --- 
                if opp_action != -1:
                     # Apply using the flat index action
                     self._apply_action(opp_action, opponent) 
                     # Process splits resulting from opponent's move
                     self.process_splits() 
                else:
                     print(f"Warning: Opponent strategy '{strategy}' failed to produce a valid action index.")
                     # If no action was determined (e.g., random found no coords unexpectedly),
                     # the game state doesn't change further this turn.
                     pass

        return state_after_player_move, reward

    def get_winner(self):
        """Return 0 if PLAYER1 won, 1 if PLAYER2 won, else None before game end."""
        # Reverted to baseline logic (0/1 index)
        if not self.game_over:
            return None
        # self.winner stores PLAYER1 (1) or PLAYER2 (2)
        return 0 if self.winner == PLAYER1 else 1

# --- Helper Function for Module API --- 
def _get_effective_strategy_weights(provided_weights):
    """Calculates the full 3-key strategy dictionary based on input."""
    final_weights = {key: 0.0 for key in VALID_STRATEGIES}
    if provided_weights is None:
        final_weights.update(DEFAULT_STRATEGY_WEIGHTS)
    elif isinstance(provided_weights, dict):
        # Basic validation (more thorough validation happens in __init__)
        if not all(key in VALID_STRATEGIES for key in provided_weights.keys()):
             # Let __init__ handle the detailed error message and exit
             return None # Indicate potential error
        final_weights.update(provided_weights)
    else:
        # Let __init__ handle the detailed error message and exit
        return None # Indicate potential error
    return final_weights

# --- Module-level API --- 
_env = None

def start_new_game(grid_size=5, opponent_model_path=None, opponent_device=None, opponent_strategy_weights=None):
    """Starts a new game using ChainReactionEnvMixedOpponent, allowing partial strategy override.
       Reuses model/env if path matches and strategy weights are not explicitly changed.
    """
    global _env
    reinitialize = False
    reason = ""

    if _env is None:
        # Must initialize if no environment exists
        print("No existing global mixed-opponent environment found. Initializing.")
        reinitialize = True
        reason = "First initialization"
        if opponent_model_path is None: 
            print("FATAL: opponent_model_path required on first init."); sys.exit(1)
    elif opponent_model_path is not None and opponent_model_path != getattr(_env, '_loaded_opponent_path', None):
        # Reinitialize if a new, different opponent path is given
        print(f"New opponent path specified ('{opponent_model_path}') for mixed env. Re-initializing.")
        reinitialize = True
        reason = "Opponent path changed"
    elif opponent_strategy_weights is not None:
        # Only check/reinitialize for strategy change IF weights are explicitly provided
        effective_input_weights = _get_effective_strategy_weights(opponent_strategy_weights)
        if effective_input_weights is None:
             print("FATAL ERROR: Invalid opponent_strategy_weights format provided to start_new_game."); sys.exit(1)
        
        current_env_weights = getattr(_env, 'strategy_weights', None)
        if effective_input_weights != current_env_weights:
             print(f"New strategy weights explicitly provided ({effective_input_weights}). Re-initializing environment.")
             reinitialize = True
             reason = "Strategy weights changed"
             
    # --- Decision --- 
    if reinitialize:
        print(f"Reason for re-initialization: {reason}")
        # Pass the *original* (potentially partial or None) strategy weights to the constructor
        # If path wasn't provided here but env existed, reuse the path stored in the old env
        path_to_use = opponent_model_path if opponent_model_path is not None else getattr(_env, '_loaded_opponent_path', None)
        if path_to_use is None: # Should not happen if _env existed, but safety check
             print("FATAL ERROR: Cannot determine opponent model path for re-initialization."); sys.exit(1)
             
        _env = ChainReactionEnvMixedOpponent(grid_size,
                                            opponent_model_path=path_to_use,
                                            opponent_device=opponent_device, # Use newly provided device or None
                                            strategy_weights=opponent_strategy_weights) # Pass original weights
        return _env.get_state(perspective_player=PLAYER1)
    else:
        # Reuse existing environment
        #print(f"Reusing existing environment.") 
        return _env.reset()

# --- Standard Module API wrappers --- 
def reset():
    if _env is None: raise RuntimeError("Mixed Opponent Environment not initialized.")
    return _env.reset()
def is_done():
    if _env is None: raise RuntimeError("Mixed Opponent Environment not initialized.")
    return _env.is_done()
def step(action):
    if _env is None: raise RuntimeError("Mixed Opponent Environment not initialized.")
    return _env.step(action)
def valid_moves_mask():
    if _env is None: raise RuntimeError("Mixed Opponent Environment not initialized.")
    return _env.valid_moves_mask(perspective_player=PLAYER1)
def get_winner():
    if _env is None: raise RuntimeError("Mixed Opponent Environment not initialized.")
    return _env.get_winner()
def get_state():
    if _env is None: raise RuntimeError("Mixed Opponent Environment not initialized.")
    return _env.get_state(perspective_player=PLAYER1)

# --- Stateless Helpers --- (Unaffected)
def valid_moves_from_state(state, player=PLAYER1):
    opponent_atoms = state[1]
    mask = (opponent_atoms == 0)
    return mask.flatten()
def valid_moves_from_states(states, player=PLAYER1):
    batch_size, _, G, _ = states.shape
    opponent_atoms = states[:, 1, :, :]
    mask = (opponent_atoms == 0).reshape(batch_size, G*G)
    return mask 