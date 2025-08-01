import json
import os
import random as r
import numpy as np
import torch
import model
import time
import math

from chainreaction_env_headless import ChainReactionHeadless
from elo_calculator import calculate_elo
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
entropy_start = 0.05


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

    
def load_validation_seeds():
    f = open("validation_seeds","r")
    validation_seeds = json.loads(f.read())
    f.close()
    return validation_seeds

def load_training_seeds():
    f = open("training_seeds_shuffled","r")
    training_seeds = json.loads(f.read())
    f.close()
    return training_seeds


policy_elo_ratings = {'critical': 1156.4470606494483, 
                      'defensive': 1381.1567399615315, 
                      'corner': 1205.3490512553622, 
                      'aggressive': 1079.5573239231255, 
                      'random': 1242.4150647233662, 
                      'build': 1135.074759487168,
                      'gemini': 1500.0}
    

# Training Loop

def start_training(opp,entropy_decay):
    opponent_first = False

    randomgame = ChainReactionHeadless()
    game = ChainReactionHeadless()

    model_selector = r.Random()
    strategy_selector = r.Random()
    policy_selector = r.Random()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game.start_new_game(opponent_policy="random")
    model_loadeded_recently = False
    global_top5_models = {}
    change_update = 0
    for update in range(total_updates):
        buffer = []
        rw = 0
        ep_reweard = 0
        all_game_rewards = []
        #wins = 0
            

        if update == change_update:
            model_loadeded_recently = False
        # collect experience in this loop for n_steps
        training_games = 0
        all_game_rewards = []
        ep_reweard = 0
        for _ in range(n_steps):
            global entropy_coeff
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
            rw = rw*0.1
            done = game.is_done()
            if done and game.get_winner() == 0:
                rw+=100
            if done and game.get_winner() == 1:
                rw-=100
            if done and game.get_winner() == -1:
                rw += 1
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
                training_games += 1
                all_game_rewards.append(ep_reweard)
                ep_reweard = 0
                opponent_first = not opponent_first
                #we will sample a model from the last M checkpoints if we decide to use older opponent model, otherwise we will use the latest model
                if opp and not model_loadeded_recently:
                    print(f"Selecting model at {update}th update")
                    strategy = strategy_selector.choices(population=["random","best"],weights=[0.2,0.8])[0]
                    if strategy == "best":
                        latest_checkpoint_index = int(update/20)
                        oldest_checkpoint_index = max(0,latest_checkpoint_index-M-1)
                        #print(f"oldest_checkpoint_index: {oldest_checkpoint_index}, latest_checkpoint_index: {latest_checkpoint_index}")
                        model_paths = { f"model_{i}" :f"{save_path}_{i}.pth" for i in range(oldest_checkpoint_index,latest_checkpoint_index)}
                        torch.save(net.state_dict(),f"{save_path}_temp.pth")
                        #model_paths["latest"] = f"{save_path}_temp.pth"
                        #print(f"Model paths: {model_paths}")
                        model_elo_ratings = calculate_elo(model_paths=model_paths,policy_elo_ratings=policy_elo_ratings)
                        model_elo_ratings = global_top5_models | model_elo_ratings | policy_elo_ratings
                        writer.add_text("Training/Model_Elo_Ratings", str(model_elo_ratings), update, walltime=time.time())
                        print(f"Model Elo Ratings: {model_elo_ratings}")
                        top5_keys = sorted(model_elo_ratings, key=model_elo_ratings.get, reverse=True)[:5]
                        global_top5_models = {k: model_elo_ratings[k] for k in top5_keys}
                        print(f"Good models: {global_top5_models}")
                        writer.add_text("Training/global_top_5", str(global_top5_models), update, walltime=time.time())
                        if len(global_top5_models) > 0:
                            selected_model = model_selector.choice(list(global_top5_models.keys()))
                            print(f"Selected model: {selected_model} with strategy: {strategy}")
                            writer.add_text("Training/Selected_Model", selected_model, update, walltime=time.time())
                            if selected_model in policy_elo_ratings.keys():
                                game.start_new_game(opponent_policy=selected_model,opponent_first=opponent_first)
                            elif selected_model == "latest":
                                game.start_new_game(model_path=model_paths[selected_model],opponent_first=opponent_first,opponent_policy="model")
                            else:
                                model_index = selected_model.split("_")[1]
                                game.start_new_game(model_path=f"{save_path}_{model_index}.pth",opponent_first=opponent_first,opponent_policy="model")
                        else:
                            print("Error: No good models found")
                            os._exit(0)
                    else:
                        policy = policy_selector.choice(list(policy_elo_ratings.keys()))
                        print(f"Selected policy: {policy} with strategy: {strategy}")
                        game.start_new_game(opponent_policy=policy,opponent_first=opponent_first)
                    change_update = update + 20
                    model_loadeded_recently = True
                    print(f"Next change scheduled at {change_update}th update")
                game.start_new_game(opponent_first=opponent_first)

        avg_game_reward = float(sum(all_game_rewards)) / float(training_games)
        #print(f"avg_game_reward: {avg_game_reward}")
        #win_rate = wins / (n_steps / 25)  # Assuming average game length of 25 steps
        
        # Log metrics to TensorBoard

        writer.add_scalar('Training/Avg_Reward', avg_game_reward, update, walltime=time.time())
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
            entropy_coeff = entropy_decay_schedule(update,total_updates,start=entropy_start,final=0.001)

        for n in range(n_epochs):
            for batch in make_minibatches(buffer, returns, advantages, batch_size):
                buffer_batch = split_buffer_batch(batch[0])
                actions_batch = torch.tensor(buffer_batch['actions'],dtype=torch.int64, device=device)     # (B,)
                old_log_probs = torch.tensor(np.array(buffer_batch['log_probs']),dtype=torch.float32, device=device)   # (B,)
                returns_batch = torch.tensor(batch[1],dtype=torch.float32, device=device)     # (B,)
                obs_batch = torch.tensor(np.array(buffer_batch['obs']),dtype=torch.float32, device=device)
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
                # print(f"Old Log Probs - Min: {old_log_probs.min().item()}, Max: {old_log_probs.max().item()}, Mean: {old_log_probs.mean().item()}")
                # print(f"New Log Probs - Min: {new_log_probs.min().item()}, Max: {new_log_probs.max().item()}, Mean: {new_log_probs.mean().item()}")
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
        if update % 20 == 0:
            torch.save(net.state_dict(),f"{save_path}.pth") # save as current best
            torch.save(net.state_dict(),f"{save_path}_{int(update/20)}.pth") # save as checkpoint for future
        #print(f"update: {update}/{total_updates}")
        #print(f"policy_loss: {policy_loss}, value_loss: {value_loss}, entropy: {entropy}, avg_game_reward: {avg_game_reward}")

        # check win rate against random opponent
        validation_games = 0
        randomwins = 0
        validation_reward = 0
        if update%50==0:
            randomgame.start_new_game(opponent_first=opponent_first,opponent_policy="validation")
            while validation_games < 500:
                r_obs = randomgame.get_state()
                r_m = randomgame.valid_moves_mask()
                valid_mask = torch.tensor(r_m, dtype=torch.bool, device=device)
                with torch.no_grad():
                    logits, value = net.forward(torch.tensor(r_obs,dtype=torch.float32))
                    masked_logits = logits.masked_fill(~valid_mask, -1e9)
                    action = masked_logits.argmax(dim=-1)
                _ , rw = randomgame.step(action.item())
                done = randomgame.is_done()
                rw=rw*0.1
                if done and randomgame.get_winner() == 0:
                    randomwins+=1
                    rw+=100
                if done and randomgame.get_winner() == 1:
                    rw-=100
                if done:
                    validation_games+=1
                    opponent_first = not opponent_first
                    randomgame.start_new_game(opponent_first=opponent_first)
                validation_reward += rw
            win_rate = randomwins/validation_games
            current_model = {"latest": f"{save_path}.pth"}
            validation_elo_ratings = calculate_elo(model_paths=current_model,policy_elo_ratings=policy_elo_ratings)
            print("validation done")
            writer.add_scalar('Game/Elo_Rating', validation_elo_ratings["latest"], update, walltime=time.time())

            writer.add_scalar('Game/Win_Rate', win_rate, update, walltime=time.time())
            writer.add_scalar('Game/Avg_Reward', float(validation_reward)/float(validation_games), update, walltime=time.time())

        # Log metrics to TensorBoard
        writer.add_scalar('Losses/Value_Loss', value_loss.item(), update, walltime=time.time())
        writer.add_scalar('Losses/Policy_Loss', policy_loss.item(), update, walltime=time.time())
        writer.add_scalar('Losses/Entropy', entropy.item(), update, walltime=time.time())
        writer.add_scalar('Training/Clip_Fraction', clip_fraction.item(), update, walltime=time.time())
        writer.add_scalar('Training/Approx_KL', approx_kl.item(), update, walltime=time.time())
        writer.flush()
        scheduler.step() # to prevent skipping first lr in optimizer

    writer.close()
    torch.save(net.state_dict(),f"{save_path}.pth")
    
def start_new_training(steps,epochs,batch,entropy,name,lr,self,deep,wide,opp,entropy_decay,updates):
    global n_steps, n_epochs, batch_size, gamma, gae_lambda, clip_epsilon, entropy_coeff, log_dir, writer, device, scheduler, save_path, net, optimizer, entropy_start, total_updates
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = steps              # Total steps to collect per PPO update
    n_epochs = epochs                # Number of times to train on the collected data
    batch_size = batch             # Batch size for PPO update
    gamma = 0.99                # Discount factor
    gae_lambda = 0.95           # GAE smoothing factor
    clip_epsilon = 0.2          # PPO clipping threshold
    entropy_coeff = entropy
    entropy_start = entropy
    total_updates = updates
    log_dir = f"runs/chain_reaction_{name}"
    save_path = f"PPOnet/chain_reaction_{name}"
    writer = SummaryWriter(log_dir=log_dir, flush_secs=1)

    if deep and wide:
        net = model.PPOGridNet_deep_wide_fc(grid_size=5,load_weights=f"{save_path}.pth")
    elif deep:
        net = model.PPOGridNet_deep_fc(grid_size=5,load_weights=f"{save_path}.pth")
    elif self:
        net = model.PPOGridNet(grid_size=5,load_weights=f"{self}.pth")
    else:
        net = model.PPOGridNet(grid_size=5)

    optimizer = torch.optim.Adam(net.parameters(), lr=9e-5)
    scheduler_map = {
    "default": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=1e-5),
    "CosineWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=1000),
    "Linear":  torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.14, total_iters=2000),
    "exp": torch.optim.lr_scheduler.LambdaLR(optimizer,flattened_exponential_decay)
    }

    scheduler = scheduler_map[lr]
    pid = os.getpid()
    print("*****************************************")
    print(f"[Worker PID {pid}] Starting with nsteps={n_steps}, bs={batch_size}, nepochs={n_epochs}, entropy={entropy}, lr_scheduler={type(scheduler).__name__}, deep={deep}, wide={wide}")
    print("*****************************************")
    start_training(opp,entropy_decay)





        

