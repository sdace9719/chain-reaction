import pygame
import sys
# import time # No longer needed for opponent move delay here
import numpy as np # For policy opponent
from chainreaction_env_headless import ChainReactionHeadless # Import the environment module

# Initialize pygame
pygame.display.init()
pygame.font.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
BLUE = (50, 50, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
GRID_COLOR = (70, 90, 110)  # Dark blue-gray that looks good on black

# Game settings
# These will be passed to the environment, but also used for UI rendering
GRID_SIZE_UI = 5 # Default for UI, can be overridden if env is started with different
CELL_SIZE = 60
MARGIN = 2
TOP_MARGIN = 40  # Space at the top for turn indicator
BOTTOM_MARGIN = 40  # Space at the bottom for moves counter
RIGHT_MARGIN = 150  # Space on the right for total atoms counter
# WINDOW_SIZE will be set dynamically based on GRID_SIZE_UI from env
FPS = 60
# MAX_MOVES_UI is a default if not specified, env's max_moves is source of truth

# Player settings (PLAYER1 is Human, PLAYER2 is Env's Policy Opponent)
PLAYER1 = 1
PLAYER2 = 2
PLAYER_COLORS = {
    PLAYER1: RED,
    PLAYER2: BLUE
}

# Map policy names to their classes

class ChainReactionPolicyComputer:
    def __init__(self, grid_size=5, opponent_policy='random', opponent_first=False, opponent_model_path=None):
        """Initialize the Chain Reaction game with a policy opponent.
        
        Args:
            grid_size (int): Size of the grid (default: 5)
            opponent_policy (str): Type of policy to use - "critical", "defensive", "corner", 
                                 "aggressive", "random", or "build" (default: "random")
            opponent_first (bool): Whether the opponent should move first (default: False)
        """
        # Start new game with specified policy
        self.grid_size = grid_size
        self.game_env = ChainReactionHeadless()
        self.game_env.start_new_game(grid_size=grid_size, opponent_policy=opponent_policy, opponent_first=opponent_first, model_path=opponent_model_path)
        
        # Set up display
        self.window_width = grid_size * CELL_SIZE + RIGHT_MARGIN
        self.window_height = grid_size * CELL_SIZE + TOP_MARGIN + BOTTOM_MARGIN
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(f"Chain Reaction - vs {opponent_policy.title()} Policy")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 30)

    def reset_game(self, opponent_first=False):
        """Reset the game."""
        self.game_env.start_new_game(grid_size=self.grid_size, opponent_first=opponent_first)

    def handle_click(self, pos):
        """Handle mouse click."""
        if self.game_env.is_done():
            return

        col = pos[0] // CELL_SIZE
        row = (pos[1] - TOP_MARGIN) // CELL_SIZE

        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            action_idx = row * self.grid_size + col
            valid_moves = self.game_env.valid_moves_mask()
            print(self.game_env.get_state())
            print(valid_moves)
            
            if valid_moves[action_idx]:
                state, reward = self.game_env.step(action_idx)
    def render(self):
        """Render the game state."""
        self.screen.fill(BLACK)
        
        # Get current grid from environment's base grid
        grid = self.game_env.grid
        current_player = self.game_env.current_player
        
        # Draw grid
        for r_idx in range(self.grid_size):
            for c_idx in range(self.grid_size):
                rect = pygame.Rect(
                    c_idx * CELL_SIZE + MARGIN,
                    r_idx * CELL_SIZE + TOP_MARGIN + MARGIN,
                    CELL_SIZE - 2 * MARGIN,
                    CELL_SIZE - 2 * MARGIN
                )
                
                cell_data = grid[r_idx][c_idx]
                if cell_data is None:
                    pygame.draw.rect(self.screen, DARK_GRAY, rect)
                else:
                    player, atoms = cell_data
                    color = PLAYER_COLORS.get(player, GRAY)
                    pygame.draw.rect(self.screen, color, rect)
                    
                    text = self.font.render(str(atoms), True, WHITE)
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)
        
        # Draw turn indicator
        if not self.game_env.is_done():
            turn_text = "Your Turn" if current_player == 0 else "Computer's Turn"
            text_surface = self.font.render(turn_text, True, WHITE)
            text_rect = text_surface.get_rect(center=(self.grid_size * CELL_SIZE // 2, TOP_MARGIN // 2))
            self.screen.blit(text_surface, text_rect)

        # Draw total atoms
        total_atoms = sum(c[1] for r in grid for c in r if c is not None)
        atoms_text = f"Total Atoms: {total_atoms}"
        text_surface = self.font.render(atoms_text, True, WHITE)
        text_rect = text_surface.get_rect(center=(self.grid_size * CELL_SIZE + RIGHT_MARGIN // 2, self.window_height // 2))
        self.screen.blit(text_surface, text_rect)
        
        # Draw game over message
        if self.game_env.is_done():
            winner = self.game_env.get_winner()
            if winner == 0:
                message = "You Won!"
            elif winner == 1:
                message = "Computer Won!"
            else:
                message = "Game Over - Draw!"

            text_surface = self.font.render(message, True, WHITE)
            grid_center_x = self.grid_size * CELL_SIZE // 2
            grid_center_y = TOP_MARGIN + (self.grid_size * CELL_SIZE // 2)
            text_rect = text_surface.get_rect(center=(grid_center_x, grid_center_y))
            self.screen.blit(text_surface, text_rect)

    def run(self):
        """Main game loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.game_env.is_done():
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_game(opponent_first=False)
            
            self.render()
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    # Example usage with different policies
    game = ChainReactionPolicyComputer(
        grid_size=5,
        opponent_model_path='PPOnet/chain_reaction_A.pth',
        opponent_policy='critical',  # can be: 'critical', 'defensive', 'corner', 'aggressive', 'random', 'build', 'validation'
        opponent_first=False # set to True to let opponent move first
    )
    game.run() 