# DDQN_with_PER
simple DDQN and PER implementation 
Look at https://spinningup.openai.com/en/latest/spinningup/keypapers.html for DDQN References.

This repository provides PER and DDQN implementation for PyTorch. 
# brain_ddqn.py
Brain (num_actions, batch_size, gamma, replay_memory, model,tau : float = 0.0, optimizer = None, update_freq : int = 2000,
                 observe_scenarios : int = 0, scheduler = None, log_variance = True, ddqn = True, device = None, use_per = True,
                 eps_annealing = 0)
You can use Vanilla Replay Memory, or PER.

# per.py
PrioritizedMemory(capacity)
Implements Prioritized Memory, see https://github.com/rlcode/per/blob/master/prioritized_memory.py for original code, but its modified in several ways.
1. Duplicate Removal
2. No NAN
3. BugFixes
4. Numba implementation
5. Interpolation, use update(Index, Value, Interpolation ratio(default = 0)

# sumtree.py
Sumtree structure with duplicate removal and interpolation.
