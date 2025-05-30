import pygame
import sys
import time
import random

# Initialize pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
BLUE = (50, 50, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
GRID_COLOR = (70, 90, 110)  # Dark blue-gray that looks good on black

# Game settings
GRID_SIZE = 4
CELL_SIZE = 60
MARGIN = 2
TOP_MARGIN = 40  # Space at the top for turn indicator
WINDOW_SIZE = (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE + TOP_MARGIN)
FPS = 60

# Player settings
PLAYER1 = 1  # Human
PLAYER2 = 2  # Computer
PLAYER_COLORS = {
    PLAYER1: RED,
    PLAYER2: BLUE
}

class ChainReactionComputer:
    def __init__(self):
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Chain Reaction vs Computer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 30)
        self.reset_game()
        
    def reset_game(self):
        # Initialize grid: 0 for empty, (player, atoms) for occupied
        self.grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.current_player = PLAYER1
        self.game_over = False
        self.winner = None
        self.splitting = False
        self.split_queue = []
        self.player1_played = False
        self.player2_played = False
        
    def get_max_capacity(self, row, col):
        """Return the maximum number of atoms a cell can hold before splitting"""
        # Corner cells have 2 adjacent cells
        if (row == 0 or row == GRID_SIZE - 1) and (col == 0 or col == GRID_SIZE - 1):
            return 2
        # Edge cells have 3 adjacent cells
        elif row == 0 or row == GRID_SIZE - 1 or col == 0 or col == GRID_SIZE - 1:
            return 3
        # Middle cells have 4 adjacent cells
        else:
            return 4
        
    def get_adjacent_cells(self, row, col):
        """Return coordinates of adjacent cells"""
        adjacent = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE:
                adjacent.append((new_row, new_col))
                
        return adjacent
    
    def place_atom(self, row, col):
        """Place an atom at the specified cell"""
        if self.game_over or self.splitting:
            return False
        
        # Check if the cell is empty or belongs to the current player
        if self.grid[row][col] is None:
            self.grid[row][col] = (self.current_player, 1)
            # Track that this player has played
            if self.current_player == PLAYER1:
                self.player1_played = True
            else:
                self.player2_played = True
            return True
        elif self.grid[row][col][0] == self.current_player:
            player, atoms = self.grid[row][col]
            self.grid[row][col] = (player, atoms + 1)
            return True
        return False
    
    def check_split(self, row, col):
        """Check if a cell should split and add to queue if needed"""
        if self.grid[row][col] is None:
            return
        
        player, atoms = self.grid[row][col]
        if atoms >= self.get_max_capacity(row, col):
            self.split_queue.append((row, col))
            
    def process_splits(self):
        """Process all splits until grid is stable"""
        self.splitting = True
        while self.split_queue:
            row, col = self.split_queue.pop(0)
            if self.grid[row][col] is not None:
                player, atoms = self.grid[row][col]
                max_capacity = self.get_max_capacity(row, col)
                
                if atoms >= max_capacity:
                    # Reset the cell
                    self.grid[row][col] = None
                    
                    # Distribute atoms to adjacent cells
                    adjacent_cells = self.get_adjacent_cells(row, col)
                    for adj_row, adj_col in adjacent_cells:
                        # If cell is empty, create a new atom of current player
                        if self.grid[adj_row][adj_col] is None:
                            self.grid[adj_row][adj_col] = (player, 1)
                        else:
                            # If cell belongs to an opponent, convert all atoms
                            adj_player, adj_atoms = self.grid[adj_row][adj_col]
                            self.grid[adj_row][adj_col] = (player, adj_atoms + 1)
                        
                        # Check if the adjacent cell should split
                        self.check_split(adj_row, adj_col)
                    
                    # Check if one player has no atoms left after this split
                    p1_atoms = 0
                    p2_atoms = 0
                    for r in range(GRID_SIZE):
                        for c in range(GRID_SIZE):
                            if self.grid[r][c] is not None:
                                p, a = self.grid[r][c]
                                if p == PLAYER1:
                                    p1_atoms += a
                                else:
                                    p2_atoms += a
                    if p1_atoms == 0 and p2_atoms > 0:
                        self.game_over = True
                        self.winner = PLAYER2
                        self.splitting = False
                        return
                    elif p2_atoms == 0 and p1_atoms > 0:
                        self.game_over = True
                        self.winner = PLAYER1
                        self.splitting = False
                        return
                    
                    # Render the intermediate state
                    self.render()
                    pygame.display.flip()
                    time.sleep(0.2)  # Add delay to visualize cascade
        
        self.splitting = False
        self.check_win_condition()
        
    def check_win_condition(self):
        """Check if the game is over"""
        if self.splitting:
            return
            
        # Only check win condition if both players have played
        if not (self.player1_played and self.player2_played):
            return
            
        player1_atoms = 0
        player2_atoms = 0
        
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.grid[row][col] is not None:
                    player, atoms = self.grid[row][col]
                    if player == PLAYER1:
                        player1_atoms += atoms
                    else:
                        player2_atoms += atoms
        
        # If a player has no atoms after both have played
        if (player1_atoms == 0 and player2_atoms > 0) or (player2_atoms == 0 and player1_atoms > 0):
            self.game_over = True
            self.winner = PLAYER1 if player2_atoms == 0 else PLAYER2
            
    def switch_player(self):
        """Switch to the other player"""
        if not self.splitting:
            self.current_player = PLAYER2 if self.current_player == PLAYER1 else PLAYER1
            
            # If it's the computer's turn, make a move
            if self.current_player == PLAYER2 and not self.game_over:
                self.render()
                pygame.display.flip()
                time.sleep(0.5)  # Small delay before computer moves
                self.make_computer_move()
        
    def get_valid_moves(self):
        """Get all valid moves for the current player"""
        valid_moves = []
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                # Cell is valid if it's empty or belongs to the current player
                if self.grid[row][col] is None or self.grid[row][col][0] == self.current_player:
                    valid_moves.append((row, col))
        return valid_moves
    
    def make_computer_move(self):
        """Make a random valid move for the computer"""
        valid_moves = self.get_valid_moves()
        if valid_moves:
            row, col = random.choice(valid_moves)
            if self.place_atom(row, col):
                self.check_split(row, col)
                self.process_splits()
                if not self.game_over:
                    self.switch_player()
        
    def render(self):
        """Render the game state"""
        self.screen.fill(BLACK)
        
        # Draw grid with offset for top margin
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                rect = pygame.Rect(
                    col * CELL_SIZE, 
                    row * CELL_SIZE + TOP_MARGIN,  # Add TOP_MARGIN offset
                    CELL_SIZE, 
                    CELL_SIZE
                )
                pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)
                
                # Draw atoms if cell is not empty
                if self.grid[row][col] is not None:
                    player, atoms = self.grid[row][col]
                    color = PLAYER_COLORS[player]
                    
                    # Draw atom count
                    text = self.font.render(str(atoms), True, color)
                    text_rect = text.get_rect(center=(
                        col * CELL_SIZE + CELL_SIZE // 2,
                        row * CELL_SIZE + CELL_SIZE // 2 + TOP_MARGIN  # Add TOP_MARGIN offset
                    ))
                    self.screen.blit(text, text_rect)
        
        # Draw game over message or current player indicator in the top margin
        if self.game_over:
            winner_text = f"{'You' if self.winner == PLAYER1 else 'Computer'} wins!"
            text = self.font.render(winner_text, True, PLAYER_COLORS[self.winner])
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] // 2, TOP_MARGIN // 2))  # Center in top margin
            self.screen.blit(text, text_rect)
        else:
            # Draw current player indicator in top margin
            player_text = "Your turn" if self.current_player == PLAYER1 else "Computer's turn"
            text = self.font.render(player_text, True, PLAYER_COLORS[self.current_player])
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] // 2, TOP_MARGIN // 2))  # Center in top margin
            self.screen.blit(text, text_rect)
                
    def handle_click(self, pos):
        """Handle mouse click events"""
        if self.game_over or self.splitting:
            return
            
        # Only handle clicks if it's the human player's turn
        if self.current_player != PLAYER1:
            return
            
        col = pos[0] // CELL_SIZE
        row = (pos[1] - TOP_MARGIN) // CELL_SIZE  # Adjust for TOP_MARGIN
        
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            # Check if the cell is available for current player
            if self.grid[row][col] is None or self.grid[row][col][0] == self.current_player:
                if self.place_atom(row, col):
                    self.check_split(row, col)
                    self.process_splits()
                    if not self.game_over:
                        self.switch_player()
                    
    def run(self):
        """Run the game loop"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset game with 'R' key
                        self.reset_game()
            
            self.render()
            pygame.display.flip()
            self.clock.tick(FPS)
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = ChainReactionComputer()
    game.run() 