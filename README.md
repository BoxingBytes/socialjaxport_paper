# socialjaxport_paper

Experiments & figures for the paper. Porting socialjax environments using pufferlib.

Requires python interpreter which has PufferLib installed editable with the
`common_harvest` CPU backend built into `pufferlib._C`:

## Schelling diagram

```sh
python schelling.py                      # sweep, ~1 min -> out/schelling.json, out/schelling.log
python plotting/schelling_diagram.py     # -> out/schelling_diagram.png, prints the SSD conditions
```

Group size, seed count and the cooperator threshold live in `config.ini` under `[schelling]`.

## Watching the policies

Replays one episode of the sweep on screen, writing nothing.

```sh
python schelling.py --render --composition 7 --seed 1   # all cooperate
python schelling.py --render --composition 0 --seed 1   # all defect
```

`--composition j` is the number of cooperators (0..num_agents); `(j, seed)` reproduces exactly the
episode behind a data point. 
