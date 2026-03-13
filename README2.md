# submit

`submit` is a lightweight job submission tool for running Python scripts either
locally or on a SLURM cluster. It uses Jinja2 templates and YAML configuration
to define execution modes, runtimes, script arguments, and batching behavior.

The goal is to keep experiment submission simple while still supporting:

- local execution on a laptop or workstation
- remote execution on a VM or cloud shell
- SLURM job submission on a cluster
- multiple runtimes such as `.venv`, Conda, or Singularity
- sequential batching of multiple commands into one job
- grouped execution with `iter: true`

## Core concepts

`submit` separates two concerns:

- **mode**: where the job runs  
  Examples: `local`, `cloud_local`, `slurm`
- **runtime**: how Python is provided  
  Examples: `.venv`, Conda, Singularity, plain Python

This keeps the configuration flexible without hardcoding environment logic into
the submission script.

## Features

- Jinja2-based command and SLURM script templating
- YAML-driven script registry
- CLI overrides for script arguments
- Cartesian product expansion across parameter values
- grouped execution with `iter: true`
- optional batching with `slurm_batch_size`
- support for Docker-based SLURM emulation during testing
- optional project-specific extensions such as prior discovery

## Quick example

```yaml
mode:
  local:
    template: "./submit/templates/local_job_cmd.j2"
    runtime: "venv"
    shell_executable: "bash"

  cloud_local:
    template: "./submit/templates/cloud_local_job_cmd.j2"
    runtime: "conda"
    shell_executable: "bash"

  slurm:
    template: "./submit/templates/slurm_job.sh.j2"
    runtime: "singularity"

runtime:
  venv:
    python_cmd: "./.venv/bin/python"

  conda:
    python_cmd: "python"
    pre_command: |
      eval "$(conda shell.bash hook)"
      conda activate myenv

  singularity:
    python_cmd: "python"
    command_wrapper: "singularity exec --bind /mnt:/mnt --nv python.sif bash -lc"

scripts:
  my_script:
    path: "path/to/script.py"
    default_args:
      seed: [0, 1, 2]
      lr: [1e-3, 1e-4]
````

Run a script locally:

```bash
python submit/submit.py --mode local --runtime venv --script my_script
```

Run a SLURM job:

```bash
python submit/submit.py --mode slurm --runtime singularity --script my_script --partition gpu
```

## Configuration overview

A `run.yaml` file contains:

* `mode:` entries describing execution contexts
* `runtime:` entries describing environment activation / Python invocation
* `scripts:` entries describing available scripts and default arguments

Example script config:

```yaml
scripts:
  train_model:
    path: "scripts/train.py"
    default_args:
      dataset: ["a", "b"]
      seed: [0, 1, 2]
```

## Grouped execution with `iter: true`

If an argument should vary **within** one logical job rather than creating a new
job by itself, mark it with `iter: true`:

```yaml
scripts:
  my_script:
    default_args:
      dataset:
        values: ["a", "b", "c"]
        iter: true
      seed: [0, 1, 2]
```

This creates three logical jobs, one per `seed`, and each job runs the three
dataset variants sequentially.

If multiple arguments use `iter: true`, their product is executed inside the
same logical job.

## Batched execution

To reduce queue pressure, logical jobs can be packed into larger physical SLURM
submissions using `slurm_batch_size`:

```yaml
scripts:
  my_script:
    default_args:
      dataset:
        values: ["a", "b", "c"]
        iter: true
      seed: [0, 1, 2]
      slurm_batch_size: 2
```

In that case, batching happens **after** logical jobs are formed.

## Documentation

* See [`SETUP.md`](./SETUP.md) for installation and setup
* See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for tests, Docker, and development
* See [`OSP.md`](./OSP.md) for project-specific workflow notes

## Requirements

* Python 3.11 recommended
* Jinja2
* PyYAML

## Status

This repository is designed to stay lightweight and adaptable. The core should
remain small, while project-specific behavior can be layered on top through
configuration and narrowly scoped extensions.

