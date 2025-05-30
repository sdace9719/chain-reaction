import random
import numpy as np
import os
import sys # Keep sys for os._exit

# Player identifiers (consistent with other env files)
PLAYER1 = 0
PLAYER2 = 1

class ChainReactionEnv:
    """Chain Reaction environment adapted from chain_reaction_computer.py, using a policy-based opponent."""
    def __init__(self, grid_size=5, opponent_policy=None, max_moves=100): # Default grid_size to 5
        self.grid_size = grid_size # Use provided grid_size
        self.opponent_policy = opponent_policy
        self.max_moves = max_moves
        
        # Initialize state variables (mostly from chain_reaction_computer.py's reset_game)
        self.grid = None
        self.current_player = None
        self.game_over = None
        self.winner = None
        # self.splitting is an internal state for process_splits, not needed to be stored in self long-term for env
        self.split_queue = [] # This will be managed locally within process_splits as in reference
        self.player1_played = False # To ensure first move has happened before checking win
        self.player2_played = False # To ensure first move has happened before checking win
        self.move_count = 0
        self.reset()

    def reset(self, opponent_first=False): # Added opponent_first to reset
        """Reset the game to the initial state and return the initial observation."""
        self.grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.current_player = PLAYER1
        self.game_over = False
        self.winner = None
        self.player1_played = False
        self.player2_played = False
        self.move_count = 0

        if opponent_first:
            self._make_policy_opponent_move()
            # Opponent's move counts as one full move if it's the very first action of the game.
            # self.move_count is incremented in step, or here if opponent moves first before any P1 step.
            # However, standard envs typically start with P1. If P2 starts, P1 state is after P2's move.
            # For consistency, we'll let the standard step() handle move_count if P1 plays.
            # If opponent goes absolutely first, we effectively skip P1's part of the first "turn".
            # Let's ensure move_count reflects that a "turn" (P2's action) has occurred.
            # The step function expects P1's action, so this pre-emptive P2 move is outside a normal step.
            # We can consider the first move done by opponent as completing a half-turn.
            # The typical env loop is: P1 acts -> P2 acts -> increment turn.
            # If P2 acts first, P1's next action will be the start of a new turn.
            # For simplicity, if opponent_first, their move sets up the board for P1.
            # The first actual "step" by P1 will then trigger P2's response and increment move_count.
            # So, no direct move_count increment here, it will happen after P1's first action in step().
            # This aligns with the idea that P1 gets the first state from reset().

        return self.get_state(perspective_player=PLAYER1)

    def get_max_capacity(self, row, col):
        """Return the maximum number of atoms a cell can hold before splitting (from reference)"""
        if (row == 0 or row == self.grid_size - 1) and \
           (col == 0 or col == self.grid_size - 1):
            return 2  # Corner cells
        elif row == 0 or row == self.grid_size - 1 or \
             col == 0 or col == self.grid_size - 1:
            return 3  # Edge cells
        else:
            return 4  # Middle cells

    def get_adjacent_cells(self, row, col): # Renamed from get_adjacent for clarity
        """Return coordinates of adjacent cells (from reference)"""
        adjacent = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                adjacent.append((new_row, new_col))
        return adjacent

    def _place_atom(self, row, col, player): # Adapted, player passed as arg
        """Place an atom for the given player. Returns True if successful."""
        # self.splitting is not used in this env structure for place_atom check
        if self.game_over: # Cannot place if game over
             return False
        
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            print(f"FATAL ERROR: Attempted to place atom out of bounds at ({row},{col}) by Player {player}.")
            os._exit(1) # Critical error

        current_cell_content = self.grid[row][col]
        if current_cell_content is None:
            self.grid[row][col] = (player, 1)
            if player == PLAYER1: self.player1_played = True
            else: self.player2_played = True
            return True
        elif current_cell_content[0] == player:
            _, atoms = current_cell_content
            self.grid[row][col] = (player, atoms + 1)
            # player_played flags are already set if they own the cell
            return True
        else: # Trying to place on opponent's cell
            # This is an invalid move from an agent's perspective and should be caught by valid_moves_mask.
            # If it somehow gets here, it's a problem.
            print(f"FATAL ERROR: Player {player} attempted invalid move on ({row},{col}) owned by Player {current_cell_content[0]}.")
            os._exit(1) 
            return False # Should not be reached

    def _check_split_and_queue(self, row, col, queue): # Adapted, takes queue
        """Check if a cell should split and add to queue if needed (from reference)."""
        cell_content = self.grid[row][col]
        if cell_content is None:
            return
        player, atoms = cell_content
        if atoms >= self.get_max_capacity(row, col):
            if (row, col) not in queue: # Avoid duplicates in queue for same cascade
                 queue.append((row, col))

    def process_splits(self):
        """Process all splits until grid is stable (logic from chain_reaction_computer.py)."""
        # self.splitting variable is not used here as it was for Pygame animation delays.
        # The core logic of queuing and processing splits is what matters.
        
        local_split_queue = [] # Use a local queue for this process
        # Initial population of the queue
        for r_idx in range(self.grid_size):
            for c_idx in range(self.grid_size):
                self._check_split_and_queue(r_idx, c_idx, local_split_queue)

        processed_in_cascade = set() # To avoid infinite loops in rare edge cases if check_split_and_queue adds already processed items

        while local_split_queue:
            row, col = local_split_queue.pop(0)

            if (row, col) in processed_in_cascade and self.grid[row][col] is None : # Already processed and cleared
                continue
            processed_in_cascade.add((row,col))


            # It's important to re-fetch cell data, as it might have changed during the cascade
            current_cell_data = self.grid[row][col]
            if current_cell_data is None: # Cell might have been cleared by another split
                continue 
            
            player_causing_split, atoms_in_cell = current_cell_data
            max_cap = self.get_max_capacity(row, col)
            
            if atoms_in_cell >= max_cap:
                self.grid[row][col] = None # Reset the cell that is splitting
                
                for adj_row, adj_col in self.get_adjacent_cells(row, col):
                    neighbor_cell_data = self.grid[adj_row][adj_col]
                    if neighbor_cell_data is None:
                        self.grid[adj_row][adj_col] = (player_causing_split, 1)
                    else:
                        # Capture or reinforce: take ownership, add 1 to existing atoms
                        _original_neighbor_owner, original_neighbor_atoms = neighbor_cell_data
                        self.grid[adj_row][adj_col] = (player_causing_split, original_neighbor_atoms + 1)
                    
                    # After updating neighbor, check if IT needs to split and add to local_split_queue
                    self._check_split_and_queue(adj_row, adj_col, local_split_queue)
                
                # Check win condition *during* splits if a player is eliminated
                # This was part of the reference, good to keep for immediate game end
                if self.check_win_condition(during_split=True):
                    return # Game ended during splits

        # Final win condition check after all splits are stable
        self.check_win_condition()


    def check_win_condition(self, during_split=False): # Add flag
        """Check if the game has been won or drawn. Returns True if game ended."""
        if self.game_over: # If already decided, no need to re-check
            return True

        # Standard win: one player has no atoms, but only after both have played at least once
        if self.player1_played and self.player2_played:
            player1_atoms = sum(c[1] for r_cells in self.grid for c in r_cells if c and c[0] == PLAYER1)
            player2_atoms = sum(c[1] for r_cells in self.grid for c in r_cells if c and c[0] == PLAYER2)

            if player1_atoms > 0 and player2_atoms == 0:
                self.game_over = True
                self.winner = PLAYER1
                return True
            elif player2_atoms > 0 and player1_atoms == 0:
                self.game_over = True
                self.winner = PLAYER2
                return True
        
        # Max moves rule
        if self.move_count >= self.max_moves:
            self.game_over = True
            self.winner = -1 # Draw or max_moves_reached
            return True
        
        return False # Game not over by win/loss/draw

    def _make_policy_opponent_move(self):
        """Internal: Opponent (PLAYER2) makes a move using its policy."""
        if self.game_over: return

        try:
            opp_row, opp_col = self.opponent_policy.get_move(self.grid, PLAYER2)
            # Action for _place_atom is (row, col)
            if self._place_atom(opp_row, opp_col, PLAYER2): # Ensure placement was valid
                self.process_splits() # Process splits after opponent's move
            else:
                # This case (policy suggests invalid move that _place_atom rejects)
                # might mean the policy found no valid moves or an internal error.
                # For robustness, if policy fails to make a valid move, P1 wins by default.
                print(f"Warning: Opponent policy failed to provide a valid move. Grid: {self.grid}")
                self.game_over = True
                self.winner = PLAYER1 
        except ValueError as e: # Policy might raise ValueError if no moves
            print(f"Warning: Opponent policy raised ValueError: {e}. Grid: {self.grid}")
            self.game_over = True # If policy can't move, agent wins
            self.winner = PLAYER1
        
        # Win condition checked within process_splits and after step completion

    def step(self, action):
        """
        Execute agent (PLAYER1) move, then policy opponent (PLAYER2) move.
        Action is an integer: row * grid_size + col.
        Returns: (observation, reward, done, info) - info is empty dict for now.
        """
        if self.game_over:
            return self.get_state(perspective_player=PLAYER1), 0.0, True, {}

        # --- Player 1's Turn (Agent) ---
        row, col = action // self.grid_size, action % self.grid_size
        
        # Validate Player 1's move (must be empty or own cell)
        cell_to_place = self.grid[row][col]
        if not (cell_to_place is None or cell_to_place[0] == PLAYER1):
            print(f"FATAL ERROR: Player 1 (Agent) attempted invalid action {action} on cell ({row},{col}) owned by Player {cell_to_place[0]}.")
            # This should ideally be caught by an agent using valid_moves_mask,
            # but as a safeguard for the environment:
            # Penalize and end game? Or just let P2 win. Let P2 win.
            self.game_over = True
            self.winner = PLAYER2 # Agent made an illegal move not caught by its own checks.
            return self.get_state(perspective_player=PLAYER1), -100.0, True, {} # Heavy penalty

        opponent_atoms_before_p1_move = sum(c[1] for r_cells in self.grid for c in r_cells if c and c[0] == PLAYER2)
        
        self._place_atom(row, col, PLAYER1)
        self.process_splits() # Process splits from Player 1's move

        opponent_atoms_after_p1_move = sum(c[1] for r_cells in self.grid for c in r_cells if c and c[0] == PLAYER2)
        reward = float(opponent_atoms_before_p1_move - opponent_atoms_after_p1_move)
        
        # Check if P1's move ended the game
        if self.check_win_condition():
            # If P1 won, reward might need adjustment based on rules (e.g., +1 for win)
            # If P1 caused a draw by max_moves, reward is as calculated.
            # If P1 lost somehow (e.g. suicide move if possible, though not in this game), reward is as calculated.
            return self.get_state(perspective_player=PLAYER1), reward, True, {}

        # --- Player 2's Turn (Policy Opponent) ---
        if not self.game_over: # Only if P1's move didn't end the game
            self._make_policy_opponent_move()
            # Win condition is checked within _make_policy_opponent_move via process_splits

        # Increment move_count after both P1 and P2 have had a chance to play (or game ended).
        # This means one full turn (P1 move + P2 response) is one "move_count".
        self.move_count += 1
        
        # Final check for game end conditions (including max_moves after incrementing)
        done = self.check_win_condition()
        
        # Adjust reward if game ended after P2's move or by max_moves
        if done:
            if self.winner == PLAYER1:
                reward += 1.0 # Small bonus for winning on this turn
            elif self.winner == PLAYER2:
                reward -= 1.0 # Small penalty for losing on this turn
            # No change for draw (winner == -1)

        return self.get_state(perspective_player=PLAYER1), reward, done, {}

    def get_state(self, perspective_player=PLAYER1):
        """Return observation of shape (2, grid_size, grid_size) from perspective player's view."""
        obs = np.zeros((2, self.grid_size, self.grid_size), dtype=np.int32)
        opponent_player = PLAYER2 if perspective_player == PLAYER1 else PLAYER1

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = self.grid[r][c]
                if cell is not None:
                    p_owner, atoms = cell
                    if p_owner == perspective_player:
                        obs[0, r, c] = atoms  # My atoms
                    elif p_owner == opponent_player:
                        obs[1, r, c] = atoms  # Opponent atoms
        return obs

    def valid_moves_mask(self, player_to_check=PLAYER1): # Parameterized for player
        """Return a boolean mask (grid_size*grid_size) for valid moves of the specified player."""
        mask = np.zeros(self.grid_size * self.grid_size, dtype=bool)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = self.grid[r][c]
                if cell is None or cell[0] == player_to_check:
                    idx = r * self.grid_size + c
                    mask[idx] = True
        return mask

    def is_done(self):
        """Return True if the game has ended."""
        # game_over is set by check_win_condition which includes max_moves
        return self.game_over

    def get_winner(self): # To match API of previous env
        """Return PLAYER1 if P1 won, PLAYER2 if P2 won, -1 if max moves/draw, None if ongoing."""
        if not self.game_over:
            return None
        return self.winner

# Module-level API (similar to other env files)
_env = None

def start_new_game(grid_size=5, opponent_policy=None, max_moves=100, opponent_first=False):
    global _env
    _env = ChainReactionEnv(grid_size, opponent_policy, max_moves)
    # Reset also handles opponent_first logic now
    initial_state = _env.reset(opponent_first=opponent_first) 
    return initial_state

def reset(opponent_first=False): # Pass opponent_first to the instance method
    if _env is None:
        start_new_game(opponent_first=opponent_first) # Ensure env exists
    return _env.reset(opponent_first=opponent_first)

def is_done():
    if _env is None: return True # No game started
    return _env.is_done()

def step(action):
    if _env is None: raise RuntimeError("Game not started. Call start_new_game() first.")
    return _env.step(action)

def valid_moves_mask(player_to_check=PLAYER1):
    if _env is None: 
        # Return all false if game not started, or handle as error
        temp_env = ChainReactionEnv() # Create a temp to get default mask size
        return np.zeros(temp_env.grid_size * temp_env.grid_size, dtype=bool)
    return _env.valid_moves_mask(player_to_check=player_to_check)

def get_winner():
    if _env is None: return None
    return _env.get_winner()

def get_state(perspective_player=PLAYER1):
    if _env is None: raise RuntimeError("Game not started. Call start_new_game() first.")
    return _env.get_state(perspective_player=perspective_player)

if __name__ == '__main__':
    # Example Usage
    print("Starting new game...")
    current_state = start_new_game(grid_size=3, max_moves=10)
    print("Initial state (P1 perspective):\n", current_state)

    done = False
    total_reward = 0
    turn_count = 0

    while not done:
        turn_count +=1
        print(f"\n--- Turn {turn_count} (Player 1) ---")
        
        # Get valid moves for Player 1
        p1_valid_mask_flat = valid_moves_mask(player_to_check=PLAYER1)
        p1_valid_indices = np.where(p1_valid_mask_flat)[0]

        if not p1_valid_indices.size:
            print("Player 1 has no valid moves!")
            break 

        # Player 1 makes a random valid move
        action_idx = np.random.choice(p1_valid_indices)
        action_row, action_col = action_idx // _env.grid_size, action_idx % _env.grid_size
        print(f"Player 1 taking action: place at ({action_row}, {action_col}) (action index: {action_idx})")
        
        current_state, reward, done, info = step(action_idx)
        total_reward += reward
        
        print("Grid after P1 and P2 moves:")
        # For direct grid view, access _env.grid (not usually done by agent)
        for r in range(_env.grid_size):
            row_str = []
            for c in range(_env.grid_size):
                cell = _env.grid[r][c]
                if cell is None: row_str.append(" . ")
                else: row_str.append(f"P{cell[0]}:{cell[1]}")
            print(" | ".join(row_str))

        print(f"State (P1 perspective):\n{current_state}")
        print(f"Reward for P1: {reward}")
        print(f"Is game done? {done}")
        if done:
            print(f"Game Over! Winner: {get_winner()}")
            print(f"Total reward for P1: {total_reward}")
            print(f"Total moves (full turns): {_env.move_count}")

    print("\nExample with opponent going first:")
    current_state = start_new_game(grid_size=3, max_moves=5, opponent_first=True)
    print("Initial state (P1 perspective) after opponent's first move:\n", current_state)
    print("Grid after opponent's first move:")
    for r in range(_env.grid_size):
        row_str = []
        for c in range(_env.grid_size):
            cell = _env.grid[r][c]
            if cell is None: row_str.append(" . ")
            else: row_str.append(f"P{cell[0]}:{cell[1]}")
        print(" | ".join(row_str))
    # ... continue game loop if desired ... 