import sys
from train import start_new_training
#import threading
import multiprocessing as mp


hyperparameters = [
    {
    "n_steps": 4096,
    "n_epochs": 6,
    "batch_size": 512,
    "entropy": 0.05,
    "name": 'A_against_ai',
    "Dynamic_rewards": False,
    "lr": "default",
    "opp": "PPOnet/chain_reaction_A_against_ai.pth",
    "self": "PPOnet/chain_reaction_A_against_ai.pth",
    "freeze_conv": False,
    "entropy_decay": True,
    "deep": False,
    "wide": False
    }  
    ]

def main():
    # must be before any CUDA or Pool usage
    mp.set_start_method('spawn', force=True)
    ctx = mp.get_context('spawn')

    manager   = mp.Manager()
    processes = []

    for h in hyperparameters:
        p = ctx.Process(target=start_new_training, args=( h["n_steps"],
                                                         h["n_epochs"],
                                                         h["batch_size"],
                                                         h["entropy"],
                                                         h["name"],
                                                         h['Dynamic_rewards'],
                                                         h["lr"],
                                                         h['deep'],
                                                         h['wide'],
                                                         h['opp'],
                                                         h['entropy_decay'],
                                                         h['self']))
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nCtrl+C detected—terminating all child processes…")
        for p in processes:
            if p.is_alive():
                p.terminate()
        sys.exit(1)

if __name__ == '__main__':
    main()