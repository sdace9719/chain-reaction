# Chain Reaction Game

A simple implementation of the Chain Reaction game using Python and Pygame.

## Game Rules

- The game is played on a 9x9 grid.
- It is a 2-player turn-based game.
- In each turn, a player selects a square to place an atom.
- When a square is selected, the number of atoms in that square increases by one.
- A player cannot select a square where the opponent's atoms are present.
- If the number of atoms in a square equals or exceeds the number of adjacent squares, the atoms split and move into neighboring squares:
  - Corner cells: Maximum 2 atoms
  - Edge cells: Maximum 3 atoms
  - Middle cells: Maximum 4 atoms
- When atoms split and enter an opponent's square, all atoms in that square are converted to the current player's atoms.
- The splitting continues in a chain reaction until the grid reaches a stable state.
- The game ends when either player has 0 atoms left.

## Installation

1. Make sure you have Python installed (Python 3.6 or higher recommended).
2. Install the required pygame library:

```
pip install pygame
```

## How to Play

1. Run the game:

```
python chain_reaction.py
```

2. Game Controls:
   - Click on a cell to place an atom
   - Press 'R' to reset the game
   - Close the window to exit

## Visual Guide

- Player 1's atoms are shown in red
- Player 2's atoms are shown in blue
- The current player's turn is displayed at the top of the screen
- The number in each cell represents the number of atoms 