## `SETUP.md`
# Setup

This document explains how to set up `submit` for local development, cloud
usage, and SLURM workflows.

## 1. Repository layout

Typical structure:

```text
your-project/
├── submit/
│   ├── submit.py
│   ├── init.py
│   ├── templates/
│   └── examples/
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── pyproject.toml
└── src/
````

## 2. Quick setup

Clone `submit` into your project:

```bash
git clone <submit-repo-url> submit/
```

## 3. Automated setup on the cloud

Start an interactive SLURM session, for example:

```bash
srun --partition=2080-galvani --gres=gpu:1 --pty bash
```

Then run:

```bash
python3 -m submit.init
```

Useful options:

* `--non-interactive`
* `--force`
* `--run-yaml-only`
* `--singularity-only`
* `--verbose`

The setup script can:

* discover Python scripts
* generate `run.yaml`
* generate `Singularity.def`
* generate `build_container.sh`

## 4. Local setup on macOS or Linux

For local development, the recommended runtime is usually a `.venv`.

Example local mode:

```yaml
mode:
  local:
    template: "./submit/templates/local_job_cmd.j2"
    runtime: "venv"
    shell_executable: "bash"

runtime:
  venv:
    python_cmd: "./.venv/bin/python"
```

This avoids activation issues and is usually more robust than shell-based
activation.

## 5. Conda setup

If you want to use Conda:

```yaml
runtime:
  conda:
    python_cmd: "python"
    pre_command: |
      eval "$(conda shell.bash hook)"
      conda activate myenv
```

This is useful on cloud VMs or shared systems where Conda environments already
exist.

## 6. Singularity setup

For ML Cloud or cluster workflows, Singularity is often the preferred runtime.

Example:

```yaml
runtime:
  singularity:
    python_cmd: "python"
    command_wrapper: "singularity exec --bind /mnt:/mnt --nv python.sif bash -lc"
```

To build a container manually:

```bash
export SINGULARITY_CACHEDIR="/scratch_local/$USER-$SLURM_JOBID"
export SINGULARITY_TMPDIR="/scratch_local/$USER-$SLURM_JOBID"

singularity build --fakeroot --force --bind /mnt:/mnt --nv python.sif submit/Singularity.def
```

## 7. Example `run.yaml`

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
  train_model:
    path: "scripts/train.py"
    default_args:
      seed: [0, 1, 2]
```

## 8. Running jobs

Basic usage:

```bash
python submit/submit.py --mode <mode> --runtime <runtime> --script <script_name>
```

Examples:

```bash
python submit/submit.py --mode local --runtime venv --script train_model
python submit/submit.py --mode cloud_local --runtime conda --script train_model
python submit/submit.py --mode slurm --runtime singularity --script train_model --partition gpu
```

## 9. Notes

* `--runtime` can be optional if the mode already specifies a default runtime
* local execution is ideal for development and debugging
* SLURM execution should be reserved for queueable cluster runs
* cloud-local mode is useful when you SSH into a remote machine but still want
  to run directly without `sbatch`