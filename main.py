import json
import sys

import numpy as np
from train import start_new_training
#import threading
import multiprocessing as mp


hyperparameters = [
    {
    "n_steps": 4096,
    "n_epochs": 4,
    "batch_size": 512,
    "entropy": 0.02,
    "name": 'against_policies4',
    "lr": "default",
    "opp": True,
    "self": "PPOnet/chain_reaction_against_policies3_612",
    "freeze_conv": False,
    "entropy_decay": False,
    "deep": False,
    "wide": False,
    "updates": 15000
    } 
    ]

def shuffle_seeds():
    f = open("training_seeds","r")
    training_seeds = json.loads(f.read())
    f.close()
    np.random.shuffle(training_seeds)
    f = open("training_seeds_shuffled","w")
    f.write(json.dumps(training_seeds))
    f.close()

def main():
    # generate shuffled set of training seeds
    shuffle_seeds()
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
                                                         h["lr"],
                                                         h['self'],
                                                         h['deep'],
                                                         h['wide'],
                                                         h['opp'],
                                                         h['entropy_decay'],
                                                         h['updates']))
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