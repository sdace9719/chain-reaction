import random
import numpy as np
import os
import sys
import torch
import model # Assuming model.py contains PPOGridNet

# Player identifiers
PLAYER1 = 1
PLAYER2 = 2

class ChainReactionEnv:
    """Chain Reaction environment using PPOGridNet model for opponent moves."""
    def __init__(self, grid_size=5, opponent_model_path=None, opponent_device=None):
        self.grid_size = grid_size
        self.opponent_model = None
        self.opponent_device = opponent_device
        self._loaded_opponent_path = None # Added to track loaded path

        # --- Mandatory Opponent Model Path Check ---
        if opponent_model_path is None:
            print(f"FATAL ERROR: opponent_model_path must be provided to initialize ChainReactionEnv.")
            sys.exit(1)
        if not os.path.exists(opponent_model_path):
            print(f"FATAL ERROR: Opponent model weights not found at specified path: {opponent_model_path}")
            sys.exit(1)
        # --- End Mandatory Check ---

        # Determine device
        if self.opponent_device is None:
            self.opponent_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Try loading
        try:
            self.opponent_model = model.PPOGridNet(
                grid_size=self.grid_size,
                load_weights=opponent_model_path,
                eval_mode=True
            ).to(self.opponent_device)
            self.opponent_model.eval()
            self._loaded_opponent_path = opponent_model_path # Store the path
            print(f"Internal opponent model loaded from: {self._loaded_opponent_path} onto device: {self.opponent_device}")
        except Exception as e:
            print(f"FATAL ERROR: Failed to load opponent model from {opponent_model_path}. Error: {e}")
            sys.exit(1)

        # Initialize state variables
        self.grid = None
        self.current_player = None
        self.game_over = None
        self.winner = None
        self.player1_played = None
        self.player2_played = None
        self.reset()

    def reset(self):
        """Reset the game to the initial state and return the initial observation."""
        self.grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.current_player = PLAYER1
        self.game_over = False
        self.winner = None
        self.player1_played = False
        self.player2_played = False
        # Return state from Player 1's perspective initially
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
                    if p == perspective_player:
                        obs[0, r, c] = a  # My atoms
                    elif p == opponent_player:
                        obs[1, r, c] = a  # Opponent atoms
        return obs

    def valid_moves_mask(self, perspective_player=PLAYER1):
        """Return a boolean mask for valid moves of the perspective_player."""
        mask = np.zeros(self.grid_size * self.grid_size, dtype=bool)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = self.grid[r][c]
                # Valid if cell is empty or belongs to the perspective player
                if cell is None or cell[0] == perspective_player:
                    idx = r * self.grid_size + c
                    mask[idx] = True
        return mask

    def get_max_capacity(self, r, c):
        """Calculate max atoms a cell can hold based on its position (standard rules)."""
        corners = (r == 0 or r == self.grid_size - 1) and (c == 0 or c == self.grid_size - 1)
        edges = (r == 0 or r == self.grid_size - 1) or (c == 0 or c == self.grid_size - 1)
        if corners: return 2
        elif edges: return 3
        else: return 4

    def get_adjacent(self, r, c):
        """Return list of adjacent cell coordinates."""
        neighbors = []
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                neighbors.append((nr,nc))
        return neighbors

    def _apply_action(self, action, player):
        """Applies a single action (integer) for the given player. Checks for invalid moves."""
        row, col = action // self.grid_size, action % self.grid_size

        cell = self.grid[row][col]
        # Check for invalid move (placing on opponent's cell)
        if cell is not None and cell[0] != player:
            print(f"FATAL ERROR: Player {player} attempted invalid move on ({row},{col}) owned by Player {cell[0]}. Action: {action}")
            os._exit(1) # Use os._exit(1) for error

        # Place/add atom
        if cell is None:
            self.grid[row][col] = (player, 1)
        else:
            _, current_atoms = cell
            self.grid[row][col] = (player, current_atoms + 1)

        # Mark player as having played
        if player == PLAYER1:
            self.player1_played = True
        else:
            self.player2_played = True

    def process_splits(self):
        """Process splits based on self.grid until stable using standard rules."""
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
                if neighbor_cell is None:
                    self.grid[nr][nc] = (p, 1)
                else:
                    op, oa = neighbor_cell
                    self.grid[nr][nc] = (p, oa + 1)

                neighbor_p, neighbor_a = self.grid[nr][nc]
                if neighbor_a >= self.get_max_capacity(nr, nc):
                    if (nr, nc) not in queue:
                       queue.append((nr, nc))
            self.check_win_condition() # Check win after each explosion resolves

    def check_win_condition(self):
        """Checks if the game has ended based on atom counts and player moves."""
        if not self.player1_played or not self.player2_played: return

        player1_atoms = sum(c[1] for r in self.grid for c in r if c and c[0] == PLAYER1)
        player2_atoms = sum(c[1] for r in self.grid for c in r if c and c[0] == PLAYER2)

        if player1_atoms > 0 and player2_atoms == 0:
            self.game_over = True; self.winner = PLAYER1
        elif player2_atoms > 0 and player1_atoms == 0:
            self.game_over = True; self.winner = PLAYER2

    def step(self, action):
        """Execute agent (Player 1) move, calc reward, then perform opponent (Player 2) PPO model move internally using ARGMAX.
           Returns state after agent splits (from P1 perspective) and reward for P1's turn."""
        if self.game_over:
            # Return current state (P1 perspective) and 0 reward if game already finished
            return self.get_state(perspective_player=PLAYER1), 0

        # --- Player 1's Turn ---
        player = PLAYER1 # Explicitly Player 1
        opponent = PLAYER2

        # 0. Check action validity (using PLAYER1 perspective)
        r_act, c_act = action // self.grid_size, action % self.grid_size
        cell_act = self.grid[r_act][c_act]
        if cell_act is not None and cell_act[0] != player:
             print(f"FATAL ERROR: Player {player} (Agent) provided invalid action {action} on cell ({r_act},{c_act}) owned by Player {cell_act[0]}.")
             os._exit(1) # Exit on external invalid move

        # 1. Calculate opponent atoms before player's move
        opponent_atoms_before = sum(c[1] for row in self.grid for c in row if c and c[0] == opponent)

        # 2. Apply Player 1's action
        self._apply_action(action, player)

        # 3. Process splits resulting from Player 1's action
        self.process_splits() # This might set self.game_over

        # 4. Calculate opponent atoms after player's move/splits
        opponent_atoms_after = sum(c[1] for row in self.grid for c in row if c and c[0] == opponent)

        # 5. Calculate immediate reward for Player 1
        reward = float(opponent_atoms_before - opponent_atoms_after)

        # 6. Capture state from Player 1's perspective *after* their move/splits
        state_after_player_move = self.get_state(perspective_player=player)

        # 7. Note if game ended during player's turn
        done_after_player_move = self.game_over

        # --- Internal Opponent's (Player 2) Turn ---
        if not done_after_player_move:
            opp_obs_np = self.get_state(perspective_player=opponent)
            opp_valid_mask_np = self.valid_moves_mask(perspective_player=opponent)

            if not np.any(opp_valid_mask_np):
                # Opponent has no valid moves. Player 1 wins.
                if not self.game_over:
                     self.game_over = True
                     self.winner = player
                     print("Warning: Opponent had no moves, but game wasn't marked over. Setting P1 as winner.")
            else:
                # Opponent has moves, get action by ARGMAX from PPO model
                opp_obs_tensor = torch.tensor(opp_obs_np, dtype=torch.float32).unsqueeze(0).to(self.opponent_device)
                opp_valid_mask_torch = torch.tensor(opp_valid_mask_np, dtype=torch.bool, device=self.opponent_device)

                with torch.no_grad():
                     model_output = self.opponent_model(opp_obs_tensor)
                     if isinstance(model_output, tuple): logits = model_output[0]
                     else: logits = model_output

                     # Handle potential shape mismatch
                     if logits.dim() == 2 and logits.size(0) == 1:
                         logits_squeezed = logits.squeeze(0)
                     elif logits.dim() == 1:
                          logits_squeezed = logits
                     else:
                         logits_squeezed = logits.view(-1)

                     if logits_squeezed.shape[0] != opp_valid_mask_torch.shape[0]:
                         raise ValueError(f"Opponent Logits shape {logits.shape} -> {logits_squeezed.shape} incompatible with mask shape {opp_valid_mask_torch.shape}")

                     # Apply mask: set invalid logits to -inf (or a very small number)
                     logits_squeezed[~opp_valid_mask_torch] = -float('inf')

                     # --- Use argmax instead of sampling ---
                     # action_dist = Categorical(logits=logits_squeezed)
                     # opp_action = action_dist.sample().item()
                     opp_action = torch.argmax(logits_squeezed).item()
                     # --- End argmax change ---

                # Apply opponent's deterministic action (PLAYER2)
                self._apply_action(opp_action, opponent)

                # Process splits from opponent's move
                self.process_splits() # This might set self.game_over and self.winner = opponent

        # --- Return Results ---
        # Return the state captured *after* Player 1's move and the reward for Player 1's turn.
        # The done status is implicitly checked via env.is_done() externally.
        return state_after_player_move, reward


    def get_winner(self):
        """Return PLAYER1 or PLAYER2 if game is over and won, else None."""
        return self.winner # Return player ID directly

# --- Module-level API (Optimized Loading) ---
_env = None

def start_new_game(grid_size=5, opponent_model_path=None, opponent_device=None):
    """Starts a new game, reusing the existing opponent model if the path matches or if no new path is specified."""
    global _env

    # Determine if re-initialization is needed
    reinitialize = False
    if _env is None:
        print("No existing global environment found. Initializing.")
        reinitialize = True
        if opponent_model_path is None:
             # This check is redundant because __init__ handles it, but good for clarity
             print(f"FATAL ERROR: opponent_model_path must be provided for the first call to start_new_game.")
             sys.exit(1)
    elif opponent_model_path is not None and opponent_model_path != getattr(_env, '_loaded_opponent_path', None):
        print(f"New opponent path specified ('{opponent_model_path}'). Re-initializing environment.")
        reinitialize = True
    # Implicit else: if _env exists and (opponent_model_path is None or opponent_model_path matches existing), just reset.

    if reinitialize:
        # Create new instance (loads model, stores path inside __init__)
        _env = ChainReactionEnv(grid_size,
                                opponent_model_path=opponent_model_path,
                                opponent_device=opponent_device)
        # Note: _env.reset() is called inside __init__
        return _env.get_state(perspective_player=PLAYER1) # Return initial state
    else:
        # Reuse existing environment, just reset it
        #print(f"Reusing existing environment with opponent: {getattr(_env, '_loaded_opponent_path', 'Unknown')}")
        return _env.reset() # reset returns initial state

def reset():
    if _env is None: raise RuntimeError("Environment not initialized via start_new_game.")
    return _env.reset()

def is_done():
    if _env is None: raise RuntimeError("Environment not initialized via start_new_game.")
    return _env.is_done()

def step(action):
    # Returns state (P1 perspective), reward (P1's turn)
    if _env is None: raise RuntimeError("Environment not initialized via start_new_game.")
    return _env.step(action)

def valid_moves_mask():
    # Returns mask for Player 1, maintaining original API behavior
    if _env is None: raise RuntimeError("Environment not initialized via start_new_game.")
    return _env.valid_moves_mask(perspective_player=PLAYER1)

def get_winner():
    # Returns PLAYER1 or PLAYER2 or None
    if _env is None: raise RuntimeError("Environment not initialized via start_new_game.")
    return _env.get_winner()

# Keep module get_state for potential external use
def get_state():
    """Return the current 2xGxG state of the environment from Player 1's perspective"""
    if _env is None: raise RuntimeError("Environment not initialized via start_new_game.")
    return _env.get_state(perspective_player=PLAYER1)

# --- Stateless Helpers (Unaffected) ---
def valid_moves_from_state(state, player=PLAYER1):
    """Given a state array (2,G,G) from player's perspective, return boolean mask (G*G)."""
    opponent_atoms = state[1]
    mask = (opponent_atoms == 0)
    return mask.flatten()

def valid_moves_from_states(states, player=PLAYER1):
    """Given batch of states (N,2,G,G) from player's perspective, return mask (N, G*G)."""
    batch_size, _, G, _ = states.shape
    opponent_atoms = states[:, 1, :, :]
    mask = (opponent_atoms == 0).reshape(batch_size, G*G)
    return mask 