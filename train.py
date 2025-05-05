import os
import random as r
import sys
import numpy as np
import torch
import model
import time
import math
from torch.utils.tensorboard import SummaryWriter

log_dir = ""
writer = None

# To run TensorBoard with auto-reload:
# tensorboard --logdir=runs --port=8080 --bind_all --reload_multifile=true --reload_interval=5

largest_neg = np.finfo(np.float64).min  

# n_steps = 4096              # Total steps to collect per PPO update
# n_epochs = 5                # Number of times to train on the collected data
# batch_size = 512             # Batch size for PPO update
# gamma = 0.99                # Discount factor
# gae_lambda = 0.95           # GAE smoothing factor
# clip_epsilon = 0.2          # PPO clipping threshold
# entropy_coeff = 0.05

n_steps = 4096              # Total steps to collect per PPO update
n_epochs = 5                # Number of times to train on the collected data
batch_size = 512             # Batch size for PPO update
gamma = 0.99                # Discount factor
gae_lambda = 0.95           # GAE smoothing factor
clip_epsilon = 0.2          # PPO clipping threshold
entropy_coeff = 0.05


net = ''
optimizer = ''

final_lr_fraction = 0.05
T_max = 2000
decay_rate = math.log(1 / final_lr_fraction)

M = 5 # consider last M checkpoints


# Scheduler function
def flattened_exponential_decay(step):
    progress = min(step / T_max, 1.0)
    return final_lr_fraction + (1 - final_lr_fraction) * math.exp(-decay_rate * progress)

save_path = ''

def entropy_decay_schedule(current_step, total_steps=2000, start=0.01, final=0.001):
    decay_rate = (final / start) ** (1 / total_steps)
    return start * (decay_rate ** current_step)


scheduler = ''
#scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=2000)
total_updates = 2000

dynamic_rewards = False

done = False

def split_buffer_batch(buffer):
    obs = []
    action = []
    log_prob = []
    value = []
    reward = []
    done = []
    for batch in buffer:
        obs.append(batch['obs'])
        action.append(batch['action'])
        log_prob.append(batch['log_prob'])
        value.append(batch['value'])
        reward.append(batch['reward'])
        done.append(batch['done'])
    return {
            'obs': obs,
            'actions': action,
            'log_probs': log_prob,
            'value': value,
            'reward': reward,
            'done': done
    }



def make_minibatches(buffer, returns, advantages, batch_size):
    for i in range(0, len(buffer), batch_size):
        yield buffer[i:i+batch_size], returns[i:i+batch_size], advantages[i:i+batch_size]




# Training Loop

def start_training(opp,entropy_decay):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = None
    if opp:
        import chainreaction_env_mixed_opponent as game
        game.start_new_game(opponent_model_path=opp,opponent_device=device)
        import chainreaction_env as randomgame
    else:
        import chainreaction_env as game
        game.start_new_game()
    global entropy_coeff
    model_loadeded_recently = True
    for update in range(total_updates):
        buffer = []
        rw = 0
        ep_reweard = 0
        all_game_rewards = []
        #wins = 0

        # check win rate against random opponent
        validation_games = 0
        randomwins = 0
        if update%50==0:
            imported = False
            try:
                randomgame.start_new_game()
                print("Random game found")
                imported = True
            except NameError:
                print("Random game not found")
                imported = False
            while validation_games < 500 and imported:
                r_obs = randomgame.get_state()
                r_m = randomgame.valid_moves_mask()
                valid_mask = torch.tensor(r_m, dtype=torch.bool, device=device)
                with torch.no_grad():
                    logits, value = net.forward(torch.tensor(r_obs,dtype=torch.float32))
                    masked_logits = logits.masked_fill(~valid_mask, -1e9)
                    action = masked_logits.argmax(dim=-1)
                _ , _ = randomgame.step(action.item())
                done = randomgame.is_done()
                if done and randomgame.get_winner() == 0:
                    randomwins+=1
                if done:
                    validation_games+=1
                    randomgame.start_new_game()
            win_rate = randomwins/validation_games
            writer.add_scalar('Game/Win_Rate', win_rate, update, walltime=time.time())
                
            

        if update % 100 == 0 and update > 0:
            model_loadeded_recently = False
        # collect experience in this loop for n_steps
        for n in range(n_steps):
            obs = game.get_state()
            m = game.valid_moves_mask()
            valid_mask = torch.tensor(m, dtype=torch.bool, device=device)
            with torch.no_grad():
                logits, value = net.forward(torch.tensor(obs,dtype=torch.float32))
                masked_logits = logits.masked_fill(~valid_mask, -1e9)
                dist = torch.distributions.Categorical(logits=masked_logits) #get a dist to sample an action
                action = dist.sample()
                log_prob = dist.log_prob(action)
            next_obs, rw = game.step(action.item())
            done = game.is_done()
            if done and game.get_winner() == 0:
                rw+=200
            if done and game.get_winner() == 1:
                rw-=100
            if n > 900 and dynamic_rewards:
                rw*=5
            buffer.append({
                'obs': obs,
                'action': action.item(),
                'log_prob': log_prob.item(),
                'value': value.item(),
                'reward': rw,
                'done': done
            })
            ep_reweard += rw
            if done:
                all_game_rewards.append(ep_reweard)
                ep_reweard = 0
                # we will sample a model from the last M checkpoints if we decide to use older opponent model, otherwise we will use the latest model
                if not model_loadeded_recently:
                    print(f"Loading model at update {update}th update")
                    select_model = r.choices(population=["best","older"],weights=[0.8,0.2])[0]
                    latest_checkpoint_index = int(update/100)-1
                    if select_model != "best":
                        oldest_checkpoint_index = max(0,latest_checkpoint_index-M)
                        select_checkpoint = r.randint(oldest_checkpoint_index,latest_checkpoint_index)
                        path = f"{save_path}_{select_checkpoint}.pth"
                        game.start_new_game(opponent_model_path=path,opponent_device=device)
                    else:
                        path = f"{save_path}_{latest_checkpoint_index}.pth"
                        game.start_new_game(opponent_model_path=path,opponent_device=device)
                    model_loadeded_recently = True
                else:
                    game.start_new_game()

        avg_game_reward = float(sum(all_game_rewards)) / float(len(all_game_rewards))
        all_game_rewards = []
        #print(f"avg_game_reward: {avg_game_reward}")
        #win_rate = wins / (n_steps / 25)  # Assuming average game length of 25 steps
        
        # Log metrics to TensorBoard
        if n > 900:
            writer.add_scalar('Rewards/Avg_Reward', float(avg_game_reward)/float(5), update, walltime=time.time())
        else:
            writer.add_scalar('Rewards/Avg_Reward', avg_game_reward, update, walltime=time.time())
        #writer.add_scalar('Game/Win_Rate', win_rate, update, walltime=time.time())
        writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], update, walltime=time.time())
        writer.flush()
        # compute return and advantage
        returns = []
        gae = 0
        next_value = 0
        advantages = []
        returns = []
        for t in reversed(range(len(buffer))):
            reward = buffer[t]['reward']
            value = buffer[t]['value']
            done = buffer[t]['done']

            if t < len(buffer) - 1:
                next_value = buffer[t + 1]['value']
            else:
                next_value = 0  # final state

            #reward = np.clip(reward, -10, 10)

            delta = reward + gamma * next_value * (1 - done) - value
            gae = delta + gamma * gae_lambda * (1 - done) * gae

            advantage = gae
            ret = advantage + value

            advantages.insert(0, advantage)
            returns.insert(0, ret)
        
            # Normalize advantages
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = 0
        value_loss = 0

        if entropy_decay:
            entropy_coeff = entropy_decay_schedule(update,2000,start=0.05,final=0.001)

        scheduler.step() # to prevent skipping first lr in optimizer
        for n in range(n_epochs):
            for batch in make_minibatches(buffer, returns, advantages, batch_size):
                buffer_batch = split_buffer_batch(batch[0])
                actions_batch = torch.tensor(buffer_batch['actions'],dtype=torch.int64, device=device)     # (B,)
                old_log_probs = torch.tensor(buffer_batch['log_probs'],dtype=torch.float32, device=device)   # (B,)
                returns_batch = torch.tensor(batch[1],dtype=torch.float32, device=device)     # (B,)
                obs_batch = torch.tensor(buffer_batch['obs'],dtype=torch.float32, device=device)
                adv_batch = torch.tensor(batch[2],dtype=torch.float32, device=device)
                
                # Forward pass
                logits, values = net(obs_batch)
                m = game.valid_moves_from_states(obs_batch)
                valid_mask = torch.tensor(m, dtype=torch.bool, device=device)
                masked_logits = logits.masked_fill(~valid_mask, -1e9)
                dist = torch.distributions.Categorical(logits=masked_logits)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(actions_batch)
                
                # PPO ratio
                ratios = torch.exp(new_log_probs - old_log_probs)                    # (B,)

                # clipped fraction

                clip_fraction = ((ratios > (1 + clip_epsilon)) | (ratios < (1 - clip_epsilon))).float().mean()
                #print(f"clip_fraction: {clip_fraction}")

                # KL divergence
                approx_kl = (old_log_probs - new_log_probs).mean()
                #print(f"Approc KL: {approx_kl}")

                # Clipped surrogate objective
                surr1 = ratios * adv_batch
                surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * adv_batch
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # Value function loss
                value_loss = torch.mean((returns_batch - values.squeeze()) ** 2)

                # Total loss
                loss = policy_loss + 0.7 * value_loss - entropy_coeff * entropy

                # Gradient step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if update % 100 == 0:
            torch.save(net.state_dict(),f"{save_path}.pth") # save as current best
            torch.save(net.state_dict(),f"{save_path}_{int(update/100)}.pth") # save as checkpoint for future
        #print(f"update: {update}/{total_updates}")
        #print(f"policy_loss: {policy_loss}, value_loss: {value_loss}, entropy: {entropy}, avg_game_reward: {avg_game_reward}")

        # Log metrics to TensorBoard
        writer.add_scalar('Losses/Value_Loss', value_loss.item(), update, walltime=time.time())
        writer.add_scalar('Losses/Policy_Loss', policy_loss.item(), update, walltime=time.time())
        writer.add_scalar('Losses/Entropy', entropy.item(), update, walltime=time.time())
        writer.add_scalar('Training/Clip_Fraction', clip_fraction.item(), update, walltime=time.time())
        writer.add_scalar('Training/Approx_KL', approx_kl.item(), update, walltime=time.time())
        writer.flush()

    writer.close()
    torch.save(net.state_dict(),f"{save_path}.pth")
    
def start_new_training(steps,epochs,batch,entropy,name,dynamic_rw,lr,deep,wide,opp,entropy_decay,self):
    global n_steps, n_epochs, batch_size, gamma, gae_lambda, clip_epsilon, entropy_coeff, log_dir, writer, device, scheduler, dynamic_rewards, save_path, net, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = steps              # Total steps to collect per PPO update
    n_epochs = epochs                # Number of times to train on the collected data
    batch_size = batch             # Batch size for PPO update
    gamma = 0.99                # Discount factor
    gae_lambda = 0.95           # GAE smoothing factor
    clip_epsilon = 0.2          # PPO clipping threshold
    dynamic_rewards = dynamic_rw
    entropy_coeff = entropy
    log_dir = f"runs/chain_reaction_{name}"
    save_path = f"PPOnet/chain_reaction_{name}"
    writer = SummaryWriter(log_dir=log_dir, flush_secs=1)

    if deep and wide:
        net = model.PPOGridNet_deep_wide_fc(grid_size=5,load_weights=self)
    elif deep:
        net = model.PPOGridNet_deep_fc(grid_size=5,load_weights=self)
    else:
        net = model.PPOGridNet(grid_size=5,load_weights=self)

    optimizer = torch.optim.Adam(net.parameters(), lr=2e-4)
    scheduler_map = {
    "default": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-5),
    "CosineWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=1000),
    "Linear":  torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.14, total_iters=2000),
    "exp": torch.optim.lr_scheduler.LambdaLR(optimizer,flattened_exponential_decay)
    }

    scheduler = scheduler_map[lr]
    pid = os.getpid()
    print("*****************************************")
    print(f"[Worker PID {pid}] Starting with nsteps={n_steps}, bs={batch_size}, nepochs={n_epochs}, entropy={entropy}, dynamic_rewards={dynamic_rw}, lr_scheduler={type(scheduler).__name__}, deep={deep}, wide={wide}")
    print("*****************************************")
    start_training(opp,entropy_decay)





        

