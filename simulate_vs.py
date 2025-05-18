import model
import torch
import numpy as np
from torch.distributions import Categorical # Add import for sampling
import os # For path checking
import sys # For exiting

# Import or define the environment class (assuming similar to chainreaction_env_pponet.py)
import chainreaction_env_mixed_opponent as game
import chainreaction_env as randomgame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#game.start_new_game(opponent_model_path="PPOnet/chain_reaction_D.pth",opponent_device=device,opponent_strategy_weights={"random": 1.0})
game = randomgame
game.start_new_game()
games = 0
done = False
ep_steps = 0
net = model.PPOGridNet(grid_size=5,load_weights="PPOnet/chain_reaction_baseline2.pth",eval_mode=True)
#net = model.PPOGridNet_deep_fc(grid_size=5,load_weights="PPOnet/chain_reaction_baseline.pth",eval_mode=True)
wins = 0
while not done:
    game_done = game.is_done()
    if game_done:
        games += 1
        #print(f"Game {games} moves: {ep_steps}")
        ep_steps = 0
        if game.get_winner() == 0:
            wins += 1
        game.start_new_game()  
    obs = game.get_state()
    m = game.valid_moves_mask()
    valid_mask = torch.tensor(m, dtype=torch.bool, device=device)
    with torch.no_grad():
        logits, value = net.forward(torch.tensor(obs,dtype=torch.float32))
        masked_logits = logits.masked_fill(~valid_mask, -1e9)
        action = masked_logits.argmax(dim=-1)
    game.step(action)
    ep_steps += 1
    if games >= 500:
        done = True
print(f"Win rate: {wins/games}")