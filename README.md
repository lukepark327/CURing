# CURing

CURing Large Models: Compression via CUR Decomposition


---


# Prerequirements

### lm-evaluation-harness

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness

cd lm-evaluation-harness
pip install -e .
```

### MoRA

```bash
git clone https://github.com/kongds/MoRA.git

cd MoRA
pip install -e ./peft-mora
```

Change project name to `peft_mora`:
- In `pyproject.toml`, update `known-first-party = ["peft_mora"]`.
- In `setup.py`, update `name="peft_mora",`.


# Run

### CURing Decomposition

```bash
$ curing.sh
```

### Healing

```bash
$ healing.sh
```

### Tensorboard

```bash
$ tensorboard --logdir=runs --host 0.0.0.0 --port 6006 --samples_per_plugin scalars=1000000
```

http://localhost:6006/?darkMode=true#scalars


---


# Acknowledgement

Our code is based on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [MoRA](https://github.com/kongds/MoRA).
