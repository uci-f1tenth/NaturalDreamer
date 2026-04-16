# Natural Dreamer is a natural.

This is a simple and clean implementation of [DreamerV3 paper](https://arxiv.org/pdf/2301.04104), that makes it maximally easy to study the architecture and understand the training pipeline.

With no fancy (complex) gear, Natural Dreamer is a natural, that's naturally beautiful.

<p align="center">
<img src="additionalMaterials/OthersVsNaturalDreamer.jpg"/>
</p>

Want to learn how it works? I made a [tutorial](https://www.youtube.com/watch?v=viXppDhx4R0) with paper, diagrams and code from this repo.

🚧 Warning 🚧: The repo isn't in it's final form, it's still work in progress. Core algorithm works, but I should benchmark it on many types of environments, while right now, only CarRacing-v3 env (continuous actions, image observations) is solved.


## Performance

Only CarRacing-v3 is solved for now.

<p align="center">
<img src="additionalMaterials/CarRacing-v3.jpg"/>
</p>

The trace shows a moving average of 10 values. This plot is also nicely cropped, since after that point the performance started slowly declining from overtraining.

Environment steps are roughly 10x the gradient steps, so 60k gradient steps is around 600k environments steps. I mostly show gradient steps because I could greatly increase the replay ratio to minimize the environment steps. I could probably slightly increase the learning rate to learn faster as well, but oh well, I have no idea what axis to show.

## How to use it?

DreamerV3 usage examples have been shown in my [tutorial](https://www.youtube.com/watch?v=viXppDhx4R0).

### Setup

```bash
git clone <this repo>
cd NaturalDreamer
pip install -r requirements.txt
```

### CarRacing-v3 (image observations)

```bash
python main.py --config car-racing-v3.yml
```

### Rustoracer — F1tenth simulator (vector observations)

NaturalDreamer supports Rustoracer, our F1tenth racing simulator. It uses LIDAR + velocity observations instead of images, so training is much faster and memory-efficient (~104 MB buffer vs ~4.6 GB for images).

**Prerequisites:** Install the `rustoracerpy` package (already listed in `requirements.txt`). Maps are included in the `maps/` folder.

**Step 1 — Smoke test on Mac (CPU, ~30–40 min)**

Run this first to verify nothing is broken before committing to a long GPU run. Checks for NaN, reward trend, entropy decay.

```bash
python main.py --config rustoracer_smoke.yml
```

**Step 2 — Full training on a CUDA machine**

```bash
python main.py --config rustoracer.yml
```

**Available maps** (see `maps/` folder):

| Map | Description |
|-----|-------------|
| `my_map` | Custom track (default) |
| `berlin` | Berlin racetrack |
| `levine` | Levine Hall indoor map |
| `vegas` | Vegas circuit |
| `skirk` | Skirk map |
| `stata_basement` | Stata Center basement |

To switch maps, edit `mapYaml` in the config file, e.g. `maps/berlin.yaml`.

**Resuming training**

Set `resume: True` and `checkpointToLoad: "50k"` in the config, then re-run the same command.

For the list of available arguments check `main.py`.

Note: The code is being developed on Linux/Mac. Windows is untested and unsupported.

## TODO

- TwoHot loss isn't implemented yet with rewardModel and critic. I made an attempt and wrote neat code for it, but it didn't work and I'm out of ideas on how to proceed, so that's yet to be finished. For now, rewardModel and critic use normal distribution.
- Discrete actions. That will be easy, just a few lines of code when I'll start solving more environments.
- Add vector observations encoder and decoder.
- Reconstruction loss is huge (11k), messing up graphs' axis scale. But graphs are only for debugging, so it's not a priority.
- Soft Critic cannot be neatly implemented with the current setup, without unnecessarily many net passes. I'm ignoring it for now, but maybe it will be needed one day.
- Continue prediction is untested, so there is no guarantee that it works.
- Remake the buffer. Buffer is currently taken from SimpleDreamer repo, but I should remake it to make it clear and clean like the rest. We don't even need to buffer the nextObservation.


## Acknowledgements

This implementation would never came to be if not for:

[SimpleDreamer](https://github.com/kc-ml2/SimpleDreamer) - which is the cleanest implementation of DreamerV1 I could find, which helped tremendously.

[SheepRL DreamerV3](https://github.com/Eclectic-Sheep/sheeprl) - which was the performant and complex DreamerV3 version, that I studied a lot for all these little performance tricks. It's complex and hard and fairly messy, but still miles ahead of original DreamerV3 code, so I could study it successfully.

[Cameron Redovian](https://github.com/naivoder) as a person, who was working on his own DreamerV3 at the same time as I was, and his attempts somehow helped me understand the training process, especially prior and posterior nets cooperation.
