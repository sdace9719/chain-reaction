import random
import numpy as np
import os  # Added to allow immediate exit on invalid moves

# Player identifiers
PLAYER1 = 1
PLAYER2 = 2

class ChainReactionEnv:
    """Chain Reaction game environment for two players on a square grid."""
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        """Reset the game to the initial state and return the initial observation."""
        # Grid: None means empty; otherwise (player, atom_count)
        self.grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.current_player = PLAYER1
        self.game_over = False
        self.winner = None
        # Track whether each player has played to enforce win condition
        self.player1_played = False
        self.player2_played = False
        return self.get_state()

    def is_done(self):
        """Return True if the game has ended."""
        return self.game_over

    def get_state(self):
        """Return observation of shape (2, grid_size, grid_size)."""
        obs = np.zeros((2, self.grid_size, self.grid_size), dtype=np.int32)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r][c] is not None:
                    p, a = self.grid[r][c]
                    obs[p-1, r, c] = a
        return obs

    def valid_moves_mask(self):
        """Return a boolean mask of shape (grid_size*grid_size,) for valid moves of player 1."""
        mask = np.zeros(self.grid_size * self.grid_size, dtype=bool)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = self.grid[r][c]
                if cell is None or cell[0] == PLAYER1:
                    idx = r * self.grid_size + c
                    mask[idx] = True
        return mask

    def get_max_capacity(self, r, c):
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
        """Process splits until no cell exceeds its capacity."""
        queue = []
        # initial scan
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = self.grid[r][c]
                if cell is not None and cell[1] >= self.get_max_capacity(r,c):
                    queue.append((r,c))
        # track squares before
        p1_before = sum(1 for r in range(self.grid_size) for c in range(self.grid_size) if self.grid[r][c] and self.grid[r][c][0]==PLAYER1)
        p2_before = sum(1 for r in range(self.grid_size) for c in range(self.grid_size) if self.grid[r][c] and self.grid[r][c][0]==PLAYER2)
        while queue:
            r,c = queue.pop(0)
            cell = self.grid[r][c]
            if not cell: continue
            p,a = cell
            cap = self.get_max_capacity(r,c)
            if a < cap: continue
            self.grid[r][c] = None
            for nr,nc in self.get_adjacent(r,c):
                if self.grid[nr][nc] is None:
                    self.grid[nr][nc] = (p,1)
                else:
                    op,oa = self.grid[nr][nc]
                    self.grid[nr][nc] = (p, oa+1)
                if self.grid[nr][nc][1] >= self.get_max_capacity(nr,nc):
                    queue.append((nr,nc))
            # early win check
            atoms1 = sum(self.grid[i][j][1] for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i][j] and self.grid[i][j][0]==PLAYER1)
            atoms2 = sum(self.grid[i][j][1] for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i][j] and self.grid[i][j][0]==PLAYER2)
            if atoms1==0 and atoms2>0:
                self.game_over=True; self.winner=PLAYER2; break
            if atoms2==0 and atoms1>0:
                self.game_over=True; self.winner=PLAYER1; break
        # calculate reward
        p1_after = sum(1 for r in range(self.grid_size) for c in range(self.grid_size) if self.grid[r][c] and self.grid[r][c][0]==PLAYER1)
        gained = p1_after - p1_before
        lost = p1_before - p1_after
        return gained, lost

    def step(self, action):
        """Execute agent move and splits, compute reward, then perform opponent move internally. Returns state after agent splits and reward."""
        # If game already over, no-op
        if self.game_over:
            return self.get_state(), 0
        # map action to grid coords
        r = action // self.grid_size
        c = action % self.grid_size
        # invalid move: cannot place on opponent cell
        if not (self.grid[r][c] is None or self.grid[r][c][0] == PLAYER1):
            raise ValueError(f"Invalid move: {action} is not allowed.")
            os._exit(1)
        # place atom for agent
        if self.grid[r][c] is None:
            self.grid[r][c] = (PLAYER1, 1)
        else:
            p, a = self.grid[r][c]
            self.grid[r][c] = (p, a + 1)
        self.player1_played = True
        # Count opponent atoms before chain reactions
        opp_before = sum(
            self.grid[i][j][1]
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if self.grid[i][j] and self.grid[i][j][0] == PLAYER2
        )
        # process chain reactions from agent's move
        self.process_splits()
        # Count opponent atoms after chain reactions
        opp_after = sum(
            self.grid[i][j][1]
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if self.grid[i][j] and self.grid[i][j][0] == PLAYER2
        )
        # Reward is number of opponent atoms removed (positive) or gained (negative)
        reward = opp_before - opp_after
        # Capture state immediately after agent's splits (before opponent move)
        state_pre = self.get_state()
        # Opponent move (ignored for reward calculation)
        valid = [
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if self.grid[i][j] is None or self.grid[i][j][0] == PLAYER2
        ]
        if valid:
            rr, cc = random.choice(valid)
            if self.grid[rr][cc] is None:
                self.grid[rr][cc] = (PLAYER2, 1)
            else:
                p2, a2 = self.grid[rr][cc]
                self.grid[rr][cc] = (p2, a2 + 1)
            self.player2_played = True
            self.process_splits()
        # Internally updated to opponent's turn; now return only agent snapshot and reward
        return state_pre, reward

    def get_winner(self):
        """Return 0 if PLAYER1 won, 1 if PLAYER2 won, else None before game end."""
        if not self.game_over:
            return None
        return 0 if self.winner == PLAYER1 else 1

# Module-level API
_env = None

def start_new_game(grid_size=5):
    global _env
    _env = ChainReactionEnv(grid_size)
    return _env.get_state()

def reset():
    return _env.reset()

def is_done():
    return _env.is_done()

def step(action):
    return _env.step(action)

def valid_moves_mask():
    return _env.valid_moves_mask()

def get_winner():
    return _env.get_winner()

# Module-level helper to fetch the current state without taking a step
def get_state():
    """Return the current 2xGxG state of the environment for Player1 and Player2"""
    return _env.get_state()

# Module-level helper to compute valid moves from a given state without env
def valid_moves_from_state(state):
    """
    Given a state array of shape (2, G, G), where channel 0 is your atoms and channel 1 is opponent's,
    return a boolean mask of length G*G indicating valid moves (cell is empty or belongs to you).
    """
    # state[1] == 0 means opponent has no atoms => either empty or your atoms
    opp = state[1]
    # Flatten row-major
    return (opp == 0).flatten()

# Module-level helper to compute valid moves from a batch of states
def valid_moves_from_states(states):
    """
    Given a batch of state arrays shape (N,2,G,G), where channel 0 is you and channel 1 is opponent,
    return a boolean mask of shape (N, G*G) indicating valid moves (empty or your cells).
    """
    # states shape: (N,2,G,G)
    # opponent channel = states[:,1,:,:]
    batch_size, _, G, _ = states.shape
    # valid if opponent atoms == 0
    opp = states[:,1,:,:]  # shape (N,G,G)
    mask = (opp == 0).reshape(batch_size, G*G)
    return mask 