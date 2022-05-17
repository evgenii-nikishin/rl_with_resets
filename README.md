# Deep RL with Resets for Addressing the Primacy Bias

This repository contains a JAX implementation of the resetting mechanism from the paper 

[**The Primacy Bias in Deep Reinforcement Learning**](https://arxiv.org/abs/2205.07802)

by [Evgenii Nikishin](https://evgenii-nikishin.github.io/)\*, [Max Schwarzer](https://scholar.google.com/citations?user=YmWRSvgAAAAJ)\*, [Pierluca D'Oro](https://proceduralia.github.io/)\*, [Pierre-Luc Bacon](https://pierrelucbacon.com/), and [Aaron Courville](https://scholar.google.com/citations?user=km6CP8cAAAAJ).


# Summary

The paper identifies a common flaw of deep RL algorithms called *the primacy bias*, a tendency to overfit initial experiences that damages the rest of the learning process.
An agent impacted by the primacy bias tends to be incapable of leveraging subsequent data because of the accumulated effect of the initial overfitting.
As a remedy, we propose a *resetting* mechanism that allows an agent to forget a part of its knowledge by periodically re-initializing the last few layers of an agent's network while preserving the replay buffer.
Applying the resets to the [SAC](https://arxiv.org/abs/1812.05905), [DrQ](https://arxiv.org/abs/2004.13649), and [SPR](https://arxiv.org/abs/2007.05929) algorithms on [DM Control](https://github.com/deepmind/dm_control) tasks and [Atari 100k](https://arxiv.org/abs/1903.00374) benchmark alleviates the effects of the primacy bias and consistently improves the performance of the agents.

<p align="center">
  <img src="https://user-images.githubusercontent.com/14283069/168692757-29b2e2ba-341c-42e0-b2c0-bbd519c03f37.png" width=800>
</p>

Please cite our work if you find it useful in your research:
```latex
@inproceedings{nikishin2022primacy,
  title={The Primacy Bias in Deep Reinforcement Learning},
  author={Nikishin, Evgenii and Schwarzer, Max and D'Oro, Pierluca and Bacon, Pierre-Luc and Courville, Aaron},
  booktitle={International Conference on Machine Learning},
  year={2022},
  organization={PMLR}
}
```


# Instructions

Discrete and continuous control experiments use two different codebases.

## DeepMind Control Suite

Install the necessary dependencies for SAC and DrQ algorithms using `continuous_control_requirements.txt`.
To train a continuous control agent with resets on a DMC task, use one of following example commands:

```bash
python train_dense.py --env_name quadruped-run --max_steps 2_000_000 --config.updates_per_step 3 --resets --reset_interval 200_000
MUJOCO_GL=egl python train_pixels.py --env_name quadruped-run --max_steps 2_000_000 --resets --reset_interval 100_000  # due to action repeats, the interval will be higher
```

Note that lines 107-111 in `train_dense.py` and 141-175 in `train_pixels.py` are the only modifications to code needed to equip the SAC and DrQ agents with resets.

## Atari 100k

To set up discrete control experiments, first create a Python 3.9 environment and run the following command to install the dependencies:
``` 
# Install from jax releases
 pip install --no-cache-dir -f https://storage.googleapis.com/jax-releases/jax_releases.html -r ./discrete_control_requirements.txt
```

To train an SPR agent without resets, run:
```bash
python -m discrete_control.train --base_dir ./test_dir/\
 --gin_files discrete_control/configs/SPR.gin \
 --gin_bindings='atari_lib.create_atari_environment.game_name = "Pong"' \
 --run_number 1
```

To train an SPR agent with default reset hyperparameters, run:
```bash
python -m discrete_control.train --base_dir ./test_dir/\
 --gin_files discrete_control/configs/SPR_with_resets.gin \
 --gin_bindings='atari_lib.create_atari_environment.game_name = "Pong"' \
 --run_number 1
```

To train an SPR agent with fully customized reset hyperparameters, the following template may be used:
```
python -m discrete_control.train --run_number ${seed} --base_dir ${BASE_DIR}_${seed} "$@" \
     --gin_files discrete_control/configs/SPR_with_resets.gin \
     --gin_bindings='atari_lib.create_atari_environment.game_name = '"\"${map[${f}]}\"" \
     --tag "SPR_resets_${reset_every}_${reset_updates}_${reset_offset}_${resets}_n${nstep}_rr${replay_ratio}" \
     --gin_bindings='JaxSPRAgent.reset_every = '"\"${reset_every}\"" \
     --gin_bindings='JaxSPRAgent.updates_on_reset = '"\"${reset_updates}\"" \
     --gin_bindings='JaxSPRAgent.total_resets = '"\"${resets}\"" \
     --gin_bindings='JaxSPRAgent.reset_offset = '"\"${reset_offset}\"" \
     --gin_bindings='JaxSPRAgent.reset_projection = '"\"${reset_proj}\"" \
     --gin_bindings='JaxSPRAgent.reset_noise = '"\"${reset_noise}\"" \
     --gin_bindings='JaxSPRAgent.reset_encoder = '"\"${reset_encoder}\"" \
     --gin_bindings='JaxDQNAgent.update_horizon = '"${nstep}" \
     --gin_bindings='JaxSPRAgent.replay_ratio = '"${replay_ratio}"  
```

# Results

| Method       | IQM                   | Median                | Mean                  |
| ------------ |:---------------------:|:---------------------:|:---------------------:|
| SAC + resets | **656**   (549,  753) | **617**   (538,  681) | **607**   (547,  667) |
| SAC          |   501     (389,  609) |   475     (407,  563) |   484     (420,  548) |
| DrQ + resets | **762**   (704,  815) | **680**   (625,  731) | **677**   (632,  720) |
| DrQ          |   569     (475,  662) |   521     (470,  600) |   535     (481,  589) |
| SPR + resets | **0.48** (0.46, 0.51) | **0.51** (0.42, 0.57) | **0.91** (0.84, 1.00) |
| SPR          |   0.38   (0.36, 0.39) |   0.43   (0.38, 0.48) |   0.57   (0.56, 0.60) |

Training curves for all agents and environments with and without resets are available in the `curves` folder.


# Acknowledgements

* Our code for continuous control experiments is based on the [JAXRL](https://github.com/ikostrikov/jaxrl) implementation of SAC and DrQ
* The implementation of the SPR algorithm uses [Dopamine](https://github.com/google/dopamine)
* We aggregate scores across tasks using the [rliable](https://github.com/google-research/rliable) recommendations for evaluating RL algorithms
