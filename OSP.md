# OSP

This document contains project-specific workflow notes for this repository.

It is intentionally separate from the general `submit` documentation. The goal
is to capture how this project uses `submit` in practice without turning the
main README into a project logbook.

## Current intended workflow

This project works across three execution contexts:

- **local**: development on a MacBook or Linux workstation
- **cloud_local**: direct execution on a remote VM or SSH session
- **slurm**: queued execution on the cluster

The preferred runtimes are:

- **local** → `.venv`
- **cloud_local** → Conda or plain Python
- **slurm** → Singularity or Conda, depending on the setup

## Local development

For local work on macOS, prefer `.venv` over Conda.

Recommended config pattern:

```yaml
mode:
  local:
    template: "./submit/templates/local_job_cmd.j2"
    runtime: "venv"
    shell_executable: "bash"

runtime:
  venv:
    python_cmd: "./.venv/bin/python"
````

This keeps local execution simple and avoids activation problems.

## Cloud usage

For remote interactive work without queue submission, use `cloud_local`.

Typical runtime choice:

```yaml
runtime:
  conda:
    python_cmd: "python"
    pre_command: |
      eval "$(conda shell.bash hook)"
      conda activate nosplace-py311
```

This is useful when the environment already exists on the machine and container
startup is unnecessary.

## SLURM usage

For queued jobs, prefer Singularity when reproducibility matters and the cluster
workflow is container-based.

Typical runtime choice:

```yaml
runtime:
  singularity:
    python_cmd: "python"
    command_wrapper: "singularity exec --bind /mnt:/mnt --nv python.sif bash -lc"
```

## Batching semantics used in this project

This project relies on grouped execution to reduce queue pressure.

Desired rule:

* arguments marked with `iter: true` vary **within** one logical job
* arguments without `iter: true` define separate jobs
* `slurm_batch_size` packs logical jobs into larger physical submissions

Example:

```yaml
default_args:
  data_name:
    values: ["d1", "d2", "d3", "d4", "d5"]
    iter: true

  num_samples:
    values: [100, 1000]
    iter: true

  seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

Intended behavior:

* 10 logical jobs
* each logical job runs 10 commands sequentially

This behavior should be protected by tests.

## Project-specific extensions

This repository currently includes FSP-related extensions such as prior
auto-discovery and prior validation.

Example:

```yaml
default_args:
  data_name: ["pos_2", "base_2"]
  prior_name: ["auto"]
```

In that case, `submit` should resolve priors per dataset and only create valid
`(data_name, prior_name)` combinations.

## Testing strategy

Development should follow this order:

1. pure unit tests for planning logic
2. fake-`sbatch` integration tests in Docker
3. real SLURM-in-Docker tests for end-to-end validation

The fake-SLURM Docker path is the default test route during development.

## Notes for maintainers

* keep the core generic
* keep project-specific logic isolated
* do not let config shortcuts drift away from documented schema
* prefer runtime abstraction over mode proliferation
* improve tests before adding more submission behavior