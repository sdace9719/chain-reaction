import random
import numpy as np
import math # Added for math.inf
from itertools import cycle  # Added for cyclic iterator
import torch
from model import PPOGridNet

# Global validation and training seeds with isolated RNGs
_default_rng = random.Random()  # For any default random operations

# Validation seed management
with open('validation_seeds', 'r') as f:
    content = f.read()
    _validation_seeds = eval(content)  # Safe since we control the file content
_validation_seed_cycler = cycle(_validation_seeds)
_validation_rng = random.Random()  # Dedicated RNG for validation seed generation

# Training seed management
with open('training_seeds_shuffled', 'r') as f:
    content = f.read()
    _training_seeds = eval(content)  # Safe since we control the file content
_training_seed_cycler = cycle(_training_seeds)
_training_rng = random.Random()  # Dedicated RNG for training seed generation

def get_next_validation_seed():
    """Get next validation seed with isolated RNG."""
    seed = next(_validation_seed_cycler)
    _validation_rng.seed(seed)  # Ensure validation RNG state is consistent
    return seed

def get_next_training_seed():
    """Get next training seed with isolated RNG."""
    seed = next(_training_seed_cycler)
    _training_rng.seed(seed)  # Ensure training RNG state is consistent
    return seed

class PolicyOpponent:
    """Base class for policy-based opponents in Chain Reaction."""
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.rng = _default_rng  # Use default RNG unless overridden

    def get_move(self, grid, player, is_new_game=False):
        """
        Get the next move based on the policy.
        Args:
            grid: The current game grid where each cell is either None or (player, atoms)
            player: The player number (1 or 2) for whom to make the move
            is_new_game: Whether this is the first move of a new game
        Returns:
            tuple: (row, col) coordinates for the next move
        """
        raise NotImplementedError("Subclasses must implement get_move")

    def get_max_capacity(self, row, col):
        """Return the maximum number of atoms a cell can hold before splitting"""
        # Corner cells have 2 adjacent cells
        if (row == 0 or row == self.grid_size - 1) and (col == 0 or col == self.grid_size - 1):
            return 2
        # Edge cells have 3 adjacent cells
        elif row == 0 or row == self.grid_size - 1 or col == 0 or col == self.grid_size - 1:
            return 3
        # Middle cells have 4 adjacent cells
        else:
            return 4

class CriticalFirstPolicy(PolicyOpponent):
    """
    Policy that:
    1. If there is a square with critical atoms, always makes it explode
    2. Otherwise, chooses any square randomly that can reach critical mass fastest
    """
    def __init__(self, grid_size):
        super().__init__(grid_size)

    def get_move(self, grid, player, is_new_game=False):
        critical_cells = []
        valid_moves_with_potential = [] # Store (moves_to_crit, r, c)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = grid[r][c]
                max_cap = self.get_max_capacity(r, c)
                
                # Check if the cell is a valid move for the player
                if cell is None or cell[0] == player:
                    current_atoms = 0 if cell is None else cell[1]
                    
                    if current_atoms + 1 >= max_cap: # Placing an atom here causes a split
                        critical_cells.append((r, c))
                    
                    moves_to_crit = max_cap - (current_atoms + 1)
                    valid_moves_with_potential.append((moves_to_crit, r, c))

        if critical_cells: # If there are cells that will explode
            return random.choice(critical_cells)
        
        if not valid_moves_with_potential: # No valid moves at all
            # This should ideally not happen if the game isn't over
            # As a fallback, find any valid cell (empty or own)
            # This fallback is more robust than raising an error immediately
            all_possible_valid_moves = []
            for r_fallback in range(self.grid_size):
                for c_fallback in range(self.grid_size):
                    cell_fallback = grid[r_fallback][c_fallback]
                    if cell_fallback is None or cell_fallback[0] == player:
                        all_possible_valid_moves.append((r_fallback, c_fallback))
            if not all_possible_valid_moves:
                 raise ValueError("No valid moves available for CriticalFirstPolicy.")
            return random.choice(all_possible_valid_moves)

        # Sort by moves_to_crit (ascending), then pick randomly among the best
        valid_moves_with_potential.sort(key=lambda x: x[0])
        min_moves_to_crit = valid_moves_with_potential[0][0]
        fastest_options = [ (r, c) for moves, r, c in valid_moves_with_potential if moves == min_moves_to_crit ]
        
        if not fastest_options:
            # Should be covered by the earlier check, but as a safeguard
            raise ValueError("Error in CriticalFirstPolicy: No fastest options found despite valid moves.")

        return random.choice(fastest_options)

class DefensivePolicy(PolicyOpponent):
    """
    A defensive policy with the following logic:
    1. If board is empty, select a random square.
    2. If not empty, try to find a diagonal move from any own atom to an EMPTY cell. 
       If multiple, pick one randomly. This is the only type of diagonal move considered.
    3. If no diagonal move to an empty cell is possible:
       a. If the player has existing atoms, select one of their OWN cells that is furthest 
          away (Manhattan distance) from the nearest opponent atom and add an atom there.
       b. If the player has no existing atoms (e.g., all wiped out, but board not empty), 
          select an EMPTY cell that is furthest away from the nearest opponent atom.
    """
    def __init__(self, grid_size):
        super().__init__(grid_size)

    def _get_player_valid_moves(self, grid, player):
        """Helper to get a list of (r,c) for all valid moves for the player (empty or own)."""
        valid_moves = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = grid[r][c]
                if cell is None or cell[0] == player:
                    valid_moves.append((r,c))
        return valid_moves

    def get_move(self, grid, player, is_new_game=False):
        own_atoms_coords = []
        opponent_atoms_coords = []
        all_cells_coords = []

        for r_idx in range(self.grid_size):
            for c_idx in range(self.grid_size):
                all_cells_coords.append((r_idx, c_idx))
                cell = grid[r_idx][c_idx]
                if cell is None:
                    pass
                elif cell[0] == player:
                    own_atoms_coords.append((r_idx, c_idx))
                else:
                    opponent_atoms_coords.append((r_idx, c_idx))

        # 1. If board is completely empty
        if not own_atoms_coords and not opponent_atoms_coords:
            if not all_cells_coords:
                raise ValueError("No cells on the board.")
            return random.choice(all_cells_coords)

        # 2. If board is not empty, try diagonal moves to EMPTY cells only.
        if own_atoms_coords: # Only try diagonal if we have atoms to move from
            diagonal_moves_to_empty = []
            
            shuffled_own_atoms = random.sample(own_atoms_coords, len(own_atoms_coords))
            diagonal_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

            for r_own, c_own in shuffled_own_atoms:
                shuffled_diag_dirs = random.sample(diagonal_directions, len(diagonal_directions))
                for dr, dc in shuffled_diag_dirs:
                    nr, nc = r_own + dr, c_own + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        target_cell_content = grid[nr][nc]
                        if target_cell_content is None: # Empty cell
                            diagonal_moves_to_empty.append((nr, nc))
            
            if diagonal_moves_to_empty:
                return random.choice(diagonal_moves_to_empty)
        
        # 3. If no diagonal move to an empty cell was made, apply fallback logic.
        # Determine the set of candidate moves for the "furthest" calculation.
        fallback_candidate_moves = []
        if own_atoms_coords: # Player has existing atoms, so only consider these for reinforcement.
            fallback_candidate_moves = own_atoms_coords
        else: # Player has no atoms left; consider any valid empty cell.
            # _get_player_valid_moves will return only empty cells if own_atoms_coords is empty.
            fallback_candidate_moves = self._get_player_valid_moves(grid, player)

        if not fallback_candidate_moves:
            # This means player has no owned atoms AND no empty cells they can play on (e.g. board full and not theirs)
            raise ValueError(f"DefensivePolicy: No valid moves available for player {player} at fallback stage.")

        # If no opponents on board, pick a random move from the determined candidates.
        if not opponent_atoms_coords:
            return random.choice(fallback_candidate_moves)

        # Calculate furthest move from any opponent using the fallback_candidate_moves.
        best_moves = []
        max_min_dist = -1

        for r_cand, c_cand in fallback_candidate_moves:
            min_dist_to_any_opponent = math.inf
            for r_opp, c_opp in opponent_atoms_coords:
                dist = abs(r_cand - r_opp) + abs(c_cand - c_opp)
                if dist < min_dist_to_any_opponent:
                    min_dist_to_any_opponent = dist
            
            if min_dist_to_any_opponent > max_min_dist:
                max_min_dist = min_dist_to_any_opponent
                best_moves = [(r_cand, c_cand)]
            elif min_dist_to_any_opponent == max_min_dist and min_dist_to_any_opponent != -1:
                best_moves.append((r_cand, c_cand))
        
        if not best_moves:
            # Should not happen if fallback_candidate_moves was populated.
            # As a safety, pick randomly from candidates if somehow best_moves is empty.
            return random.choice(fallback_candidate_moves)

        return random.choice(best_moves)

class CornerEdgePolicy(PolicyOpponent):
    """
    A policy that prioritizes corners and edges in the following order:
    1. If any corner is available (empty or owned), place there.
    2. If no corners available, if any edge is available (empty or owned), place there.
    3. If neither corners nor edges available, make a random valid move.
    """
    def __init__(self, grid_size):
        super().__init__(grid_size)
        # Pre-compute corner and edge coordinates
        self.corners = [
            (0, 0), (0, self.grid_size-1),
            (self.grid_size-1, 0), (self.grid_size-1, self.grid_size-1)
        ]
        self.edges = []
        # Top and bottom edges (excluding corners)
        for c in range(1, self.grid_size-1):
            self.edges.append((0, c))  # Top edge
            self.edges.append((self.grid_size-1, c))  # Bottom edge
        # Left and right edges (excluding corners)
        for r in range(1, self.grid_size-1):
            self.edges.append((r, 0))  # Left edge
            self.edges.append((r, self.grid_size-1))  # Right edge

    def get_move(self, grid, player, is_new_game=False):
        # Get all valid moves (empty or owned cells)
        valid_moves = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = grid[r][c]
                if cell is None or cell[0] == player:
                    valid_moves.append((r, c))

        if not valid_moves:
            raise ValueError(f"CornerEdgePolicy: No valid moves available for player {player}.")

        # 1. Check corners first
        valid_corners = [corner for corner in self.corners if corner in valid_moves]
        if valid_corners:
            return random.choice(valid_corners)

        # 2. Check edges if no corners available
        valid_edges = [edge for edge in self.edges if edge in valid_moves]
        if valid_edges:
            return random.choice(valid_edges)

        # 3. If neither corners nor edges available, make a random move
        return random.choice(valid_moves)

class AggressivePolicy(PolicyOpponent):
    """
    An aggressive policy that actively tries to capture opponent atoms:
    1. If no opponent atoms exist or it's the first move, make a random valid move.
    2. Otherwise:
       a. Find opponent atoms, prioritizing those with lower counts.
       b. For each opponent atom (starting with lowest count), look for adjacent cells
          that we can place in (empty or owned by us).
       c. If we find an adjacent cell that we can build up to trigger a reaction
          (i.e., one that's close to max capacity), prioritize that.
       d. If no cells are close to triggering, pick the adjacent cell with the most
          atoms we own (to build it up towards triggering).
    """
    def __init__(self, grid_size):
        super().__init__(grid_size)
        # Pre-compute adjacent directions (orthogonal only, no diagonals)
        self.adjacent_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def _get_adjacent_cells(self, row, col):
        """Get list of valid adjacent cell coordinates."""
        adjacent = []
        for dr, dc in self.adjacent_dirs:
            new_r, new_c = row + dr, col + dc
            if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
                adjacent.append((new_r, new_c))
        return adjacent

    def get_move(self, grid, player, is_new_game=False):
        # First, collect all valid moves and opponent atoms
        valid_moves = []
        opponent_atoms = []  # Will store (count, row, col) tuples
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = grid[r][c]
                if cell is None or cell[0] == player:
                    valid_moves.append((r, c))
                elif cell[0] != player:  # opponent's cell
                    opponent_atoms.append((cell[1], r, c))  # store count for sorting

        if not valid_moves:
            raise ValueError(f"AggressivePolicy: No valid moves available for player {player}.")

        # If no opponent atoms, make a random move
        if not opponent_atoms:
            return random.choice(valid_moves)

        # Sort opponent atoms by count (ascending) to target weaker positions first
        opponent_atoms.sort()  # Will sort by count since it's first in tuple

        # For each opponent atom (starting with lowest count)
        for opp_count, opp_r, opp_c in opponent_atoms:
            adjacent_cells = self._get_adjacent_cells(opp_r, opp_c)
            best_attack_move = None
            best_attack_score = -1

            for adj_r, adj_c in adjacent_cells:
                if (adj_r, adj_c) not in valid_moves:
                    continue

                cell = grid[adj_r][adj_c]
                current_count = 0 if cell is None else cell[1]
                max_cap = self.get_max_capacity(adj_r, adj_c)

                # Calculate how close this cell is to triggering
                moves_to_trigger = max_cap - (current_count + 1)

                # Score this move based on:
                # 1. How close it is to triggering (lower is better)
                # 2. Current atom count (higher is better, as we're already building there)
                # 3. Opponent's atom count (lower is better, easier to capture)
                attack_score = (
                    1000 * (max_cap - moves_to_trigger)  # Prioritize cells close to triggering
                    + 100 * current_count  # Then consider current build-up
                    - 10 * opp_count  # Slightly prefer targeting smaller opponent groups
                )

                if attack_score > best_attack_score:
                    best_attack_score = attack_score
                    best_attack_move = (adj_r, adj_c)

            # If we found a good attack move against this opponent atom, take it
            if best_attack_move is not None:
                return best_attack_move

        # If no good attack moves found, make a random valid move
        return random.choice(valid_moves)

class BuildAndExplodePolicy(PolicyOpponent):
    """
    A policy that builds up atoms until owning 15, then switches to aggressive explosion strategy:
    Phase 1 (Build Phase, < 15 atoms):
        - Prioritize cells just below critical mass
        - If no such cells, pick any valid cell to build up
    Phase 2 (Explode Phase, >= 15 atoms):
        - Look for any cells that can trigger explosions
        - If none found, continue building near-critical cells
    """
    def __init__(self, grid_size=5):
        super().__init__(grid_size)
        
    def count_owned_atoms(self, grid, player):
        """Count total atoms owned by player."""
        total = 0
        for row in grid:
            for cell in row:
                if cell is not None and cell[0] == player:
                    total += cell[1]
        return total
    
    def get_move(self, grid, player, is_new_game=False):
        owned_atoms = self.count_owned_atoms(grid, player)
        is_build_phase = owned_atoms < 15
        
        # Collect all valid moves with their scores
        moves_with_scores = []  # Will store (score, row, col) tuples
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = grid[r][c]
                if cell is None or cell[0] == player:
                    current_atoms = 0 if cell is None else cell[1]
                    max_cap = self.get_max_capacity(r, c)
                    moves_to_critical = max_cap - (current_atoms + 1)
                    
                    if is_build_phase:
                        # During build phase, prioritize:
                        # 1. Cells one atom away from critical (score: 1000)
                        # 2. Cells with more atoms but not critical (score: 100-999)
                        # 3. Empty cells (score: 0)
                        if moves_to_critical == 1:
                            score = 1000
                        else:
                            score = 100 * current_atoms
                    else:
                        # During explode phase, prioritize:
                        # 1. Cells that will explode (score: 2000)
                        # 2. Cells one away from exploding (score: 1000)
                        # 3. Cells with more atoms (score: 100-999)
                        if moves_to_critical <= 0:  # Will explode
                            score = 2000
                        elif moves_to_critical == 1:
                            score = 1000
                        else:
                            score = 100 * current_atoms
                    
                    moves_with_scores.append((score, r, c))
        
        if not moves_with_scores:
            return None
            
        # Sort by score (descending) and take the highest scoring move
        moves_with_scores.sort(reverse=True)
        best_score = moves_with_scores[0][0]
        best_moves = [(r, c) for score, r, c in moves_with_scores if score == best_score]
        
        return random.choice(best_moves)

class SeededRandomPolicy(PolicyOpponent):
    """
    A policy that makes random moves using training seeds for reproducibility.
    Seeds are managed externally rather than internally cycling to ensure consistency
    across different instances. Uses isolated RNG to prevent seed leakage.
    """
    def __init__(self, grid_size=5):
        super().__init__(grid_size)
        self.rng = random.Random()  # Create a dedicated Random instance
        
    def set_game_seed(self, seed):
        """Set the seed for the current game."""
        self.rng.seed(seed)  # Isolated RNG ensures no state leakage
        
    def get_move(self, grid, player, is_new_game=False):
        """Get the next move based on the policy."""
        valid_moves = []
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                cell = grid[i][j]
                if cell is None or cell[0] == player:
                    valid_moves.append((i, j))
        
        if not valid_moves:
            return None
        
        # Use isolated RNG for move selection
        move = self.rng.choice(valid_moves)
        return move

class ValidationSeededPolicy(PolicyOpponent):
    """
    A policy that makes random moves using validation seeds for reproducibility.
    Seeds are managed externally with an isolated RNG to prevent any seed leakage
    between validation and training seeds.
    """
    def __init__(self, grid_size=5):
        super().__init__(grid_size)
        self.rng = random.Random()  # Create a dedicated Random instance
        
    def set_game_seed(self, seed):
        """Set the seed for the current game."""
        self.rng.seed(seed)  # Isolated RNG ensures no state leakage
        
    def get_move(self, grid, player, is_new_game=False):
        """Get the next move based on the policy."""
        valid_moves = []
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                cell = grid[i][j]
                if cell is None or cell[0] == player:
                    valid_moves.append((i, j))
        
        if not valid_moves:
            return None
        
        # Use isolated RNG for move selection
        move = self.rng.choice(valid_moves)
        return move

class ModelPolicy(PolicyOpponent):
    """
    A policy that uses a trained model to make moves.
    The model should be a PPOGridNet saved as a .pth file.
    Uses argmax for deterministic move selection.
    """
    def __init__(self, model_path, grid_size=5):
        super().__init__(grid_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model and load state dict
        self.model = PPOGridNet(grid_size=grid_size).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set to evaluation mode
        
    def get_move(self, grid, player, is_new_game=False):
        """
        Get the next move using the model's policy.
        Args:
            grid: The current game grid
            player: The player number (1 or 2)
            is_new_game: Whether this is the first move of a new game
        Returns:
            tuple: (row, col) coordinates for the next move
        """
        # Convert grid to model's input format (2-channel format)
        state = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)
        
        # Fill the channels: channel 0 for player's atoms, channel 1 for opponent's atoms
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = grid[i][j]
                if cell is not None:
                    cell_player, atoms = cell
                    if cell_player == player:
                        state[0, i, j] = atoms
                    else:
                        state[1, i, j] = atoms
        
        # Get model prediction
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, _ = self.model(state_tensor)  # Ignore value output
            action_probs = action_probs.cpu().numpy()[0]
            
            # Create valid moves mask
            valid_moves_mask = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    cell = grid[i][j]
                    if cell is None or cell[0] == player:
                        valid_moves_mask[i * self.grid_size + j] = 1
            
            # Apply mask and get argmax
            masked_probs = action_probs * valid_moves_mask
            if not np.any(valid_moves_mask):
                return None
                
            # Get the move with highest probability among valid moves
            action = np.argmax(masked_probs)
            
            # Convert to grid coordinates
            row = action // self.grid_size
            col = action % self.grid_size
            
            return (row, col)

class GeminiPolicyV1(PolicyOpponent):
    """
    Gemini's Heuristic Policy for Chain Reaction.
    Prioritizes:
    1. Immediate winning moves (simplified check: critical move that clears many opponent cells).
    2. Blocking opponent's direct critical setup if possible with own critical move.
    3. Making own critical (exploding) moves, preferring those that capture.
    4. Setting up own critical moves (1 atom away), preferring strategic spots.
    5. Taking strategic empty cells (corners, then edges).
    6. Reinforcing existing cells.
    7. Random valid move.
    """
    def __init__(self, grid_size=5):
        super().__init__(grid_size)

    def get_valid_moves(self, grid_state, player_id_gemini):
        valid = []
        for r_idx in range(self.grid_size):
            for c_idx in range(self.grid_size):
                cell = grid_state[r_idx][c_idx]
                if cell is None or cell[0] == player_id_gemini:
                    valid.append((r_idx, c_idx))
        return valid

    def _get_adjacent_opponent_cells(self, grid_state, r, c, player_id_gemini):
        count = 0
        opponent_id = 1 - player_id_gemini
        # Check cells adjacent to the explosion *site* (r,c)
        # A more complex check would trace the whole chain reaction
        for dr_adj, dc_adj in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr_adj, nc_adj = r + dr_adj, c + dc_adj
            if 0 <= nr_adj < self.grid_size and 0 <= nc_adj < self.grid_size:
                adj_cell = grid_state[nr_adj][nc_adj]
                if adj_cell and adj_cell[0] == opponent_id:
                    count += 1
        return count

    def get_move(self, grid_state, player_id_gemini, is_new_game=False): # Added is_new_game for interface compatibility
        valid_moves = self.get_valid_moves(grid_state, player_id_gemini)
        if not valid_moves:
            return None

        opponent_id = 1 - player_id_gemini

        # --- Tier 1: Offensive Critical Moves (Explosions) ---
        critical_moves_with_scores = [] # (score, (r, c))
        for r, c in valid_moves:
            current_atoms = 0
            cell = grid_state[r][c]
            if cell and cell[0] == player_id_gemini:
                current_atoms = cell[1]
            elif cell is None:
                current_atoms = 0
            else: # Opponent's cell, should not be in valid_moves for placement
                continue
            
            if current_atoms + 1 >= self.get_max_capacity(r, c):
                # Estimate capture potential (simplistic: count adjacent opponent cells)
                capture_potential = self._get_adjacent_opponent_cells(grid_state, r, c, player_id_gemini)
                # Add more weight if the cell itself is an opponent's cell that would be captured
                # (though valid_moves logic for placement usually means it's my cell or empty)
                critical_moves_with_scores.append((-capture_potential, (r, c))) # Negative for max heap behavior if sorted

        if critical_moves_with_scores:
            critical_moves_with_scores.sort() # Sort by potential (most negative capture_potential first)
            return critical_moves_with_scores[0][1]

        # --- Tier 2: Block Opponent's Critical Setup (if I can make a critical move there) ---
        # Find where opponent is 1 away from critical
        opponent_setup_cells = []
        for r_opp in range(self.grid_size):
            for c_opp in range(self.grid_size):
                cell_opp = grid_state[r_opp][c_opp]
                if cell_opp and cell_opp[0] == opponent_id:
                    if cell_opp[1] + 1 >= self.get_max_capacity(r_opp, c_opp): # Opponent is already critical
                        pass # Harder to block directly without full lookahead
                    elif cell_opp[1] + 1 == self.get_max_capacity(r_opp, c_opp) -1: # Opponent is 1 away
                        opponent_setup_cells.append((r_opp, c_opp))
        
        # If opponent has setup cells, see if any of my valid_moves can become critical AND affect them
        # This is a simplified check: can I place critically near their setup?
        if opponent_setup_cells:
            candidate_blocking_critical_moves = []
            for r_my, c_my in valid_moves: # My potential move
                my_current_atoms = 0; cell_my = grid_state[r_my][c_my]
                if cell_my and cell_my[0] == player_id_gemini: my_current_atoms = cell_my[1]
                elif cell_my is None: my_current_atoms = 0
                else: continue

                if my_current_atoms + 1 >= self.get_max_capacity(r_my, c_my): # My move is critical
                    for r_opp_setup, c_opp_setup in opponent_setup_cells:
                        # Check if my critical explosion is adjacent to opponent's setup cell
                        if abs(r_my - r_opp_setup) + abs(c_my - c_opp_setup) == 1:
                             # Simple: if my critical is next to their setup, it might be good
                            candidate_blocking_critical_moves.append((r_my, c_my))
            if candidate_blocking_critical_moves:
                return self.rng.choice(candidate_blocking_critical_moves)


        # --- Tier 3: My Setup Moves (1 away from critical) ---
        setup_moves = []
        for r, c in valid_moves:
            current_atoms = 0; cell = grid_state[r][c]
            if cell and cell[0] == player_id_gemini: current_atoms = cell[1]
            elif cell is None: current_atoms = 0
            else: continue
                
            if current_atoms + 1 == self.get_max_capacity(r, c) - 1:
                setup_moves.append((r,c))
        
        if setup_moves:
            # Prioritize setup moves that are also strategic (corners, then edges)
            corners = [(0,0), (0,self.grid_size-1), (self.grid_size-1,0), (self.grid_size-1,self.grid_size-1)]
            strategic_setup_corners = [m for m in setup_moves if m in corners]
            if strategic_setup_corners: return self.rng.choice(strategic_setup_corners)

            edges = []
            for i_val in range(1, self.grid_size - 1): 
                edges.extend([(0,i_val), (self.grid_size-1,i_val), (i_val,0), (i_val,self.grid_size-1)])
            strategic_setup_edges = [m for m in setup_moves if m in edges]
            if strategic_setup_edges: return self.rng.choice(strategic_setup_edges)
            
            return self.rng.choice(setup_moves)
        
        # --- Tier 4: Strategic Empty Cell Occupation ---
        corners = [(0,0), (0,self.grid_size-1), (self.grid_size-1,0), (self.grid_size-1,self.grid_size-1)]
        empty_valid_corners = [m for m in valid_moves if m in corners and grid_state[m[0]][m[1]] is None]
        if empty_valid_corners: return self.rng.choice(empty_valid_corners)

        edges = []
        for i_val in range(1, self.grid_size - 1): 
            edges.extend([(0,i_val), (self.grid_size-1,i_val), (i_val,0), (i_val,self.grid_size-1)])
        empty_valid_edges = [m for m in valid_moves if m in edges and grid_state[m[0]][m[1]] is None]
        if empty_valid_edges: return self.rng.choice(empty_valid_edges)
        
        empty_cells = [m for m in valid_moves if grid_state[m[0]][m[1]] is None]
        if empty_cells: return self.rng.choice(empty_cells)
        
        # --- Tier 5: Fallback (Reinforce existing cells or any valid move) ---
        return self.rng.choice(valid_moves) 