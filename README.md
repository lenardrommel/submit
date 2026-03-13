# `submit`

A lightweight job submission tool that supports both local execution and SLURM cluster submissions. It uses Jinja2 templates and YAML configuration to manage job parameters and execution.

## Quick Setup (Recommended)

For automated setup in your Python repository (on the MLCloud):

1. **Clone submit into your project:**
```bash
git clone <submit-repo-url> submit/
```

2. **Run automated setup:**
Start an interactive session with

```bash
srun --partition=2080-galvani --gres=gpu:1 --pty bash
```

and then run

```bash
python3 -m submit.init
```

Consider the following additional arguments:
- `--non-interactive`: setup in yolo mode.
- `--force`: overwrite existing configuration files.
- `--run-yaml-only`: only rebuild `run.yaml` file.
- `--singularity-only`: only rebuild `Singularity.def` and `build_container.sh` files.
- `--verbose`: enable verbose logging for debugging script discovery issue 

3. **Build container (if prompted or manually):**
```bash
./build_container.sh  # Run in SLURM interactive session for cloud usage
```

### Expected Folder Structure

**Before setup:**
```
your-project/
├── submit/                 # Cloned submit repository
│   ├── submit.py
│   ├── init.py
│   ├── templates/
│   └── examples/          # Example configuration files
│       ├── run.yaml
│       └── Singularity.def
├── scripts/                # Your Python scripts (optional)
│   ├── train.py
│   └── evaluate.py
├── pyproject.toml         # Or setup.py, requirements.txt
└── src/                   # Your source code
```

**After setup:**
```
your-project/
├── submit/
│   ├── submit.py
│   ├── init.py
│   ├── run.yaml           # Generated configuration
│   ├── templates/
│   └── examples/          # Example files (unchanged)
├── scripts/               # Your scripts (discovered automatically)
├── Singularity.def        # Generated container definition
├── build_container.sh     # Generated build script
├── python.sif            # Built container (after running build script)
├── logs/                  # Generated logs directory
└── pyproject.toml
```

The automated setup will:
- Discover Python scripts in `**/scripts/*.py` directories
- Generate container definition based on your `pyproject.toml`/`setup.py`
- Create run configuration with found scripts
- Optionally build the Singularity container

## Manual Cloud Setup Tutorial

My current workflow is as follows:
1. Start by cloning `submit` in your working repository (e.g. `$WORK/repos/my-repo/submit`) 
2. Next copy `submit/examples/*` to `submit/*`, e.g. via
```bash
cp -rf ./submit/examples/* ./submit/examples/
```
3. **Container/Environment setup:**
   - **Option A (Singularity):** I prefer running jobs on the ML Cloud using a singularity container (cf. https://portal.mlcloud.uni-tuebingen.de/user-guide/tutorials/singularity/). To do so, copy and modify the `submit/Singularity.def` file in your working repository. Then, start an interactive session (using e.g. `srun ...`) and build the singularity container with the following command:
```bash
# Set cache and tmp directories for the Singularity build - (not optimal)
export SINGULARITY_CACHEDIR="/scratch_local/$USER-$SLURM_JOBID"
export SINGULARITY_TMPDIR="/scratch_local/$USER-$SLURM_JOBID"

# Build the singularity containers
singularity build --fakeroot --force --bind /mnt:/mnt --nv python.sif submit/Singularity.def
```
   - **Option B (Conda):** Alternatively, you can run jobs using a Conda environment. You don't need to build a container. Instead, define a `runtime` entry with `python_cmd: "python"` and a `pre_command` that activates your Conda environment before the script runs.

4. Next, open and modify the `submit/run.yaml` file in the main `submit` directory. It is important to change the entries below scripts, `regression...` to the scripts you want to run and have in the repo. (Note: `default_args` is optional, so in most cases this section can be removed).
5. From the working repository, run jobs with the following command structure:
```bash
python3 submit/submit.py --mode [local|cloud_local|slurm] --script <script_name> [--slurm_args <slurm_args>] [--script_args <script_args>]
```

You can find some more details below.

Remarks:
- If you want to use jax, keep in mind to install it with cuda dependencies (e.g. `pip install -U "jax[cuda12]"`)
- All examples/* files are excluded when placed in the main `submit` directory to not mess with this repository. 

## Requirements
- Python 3.6+
- Jinja2
- PyYAML

> This tool is designed to run on the ML Cloud without the need for a python environment or package installations.

## Configuration

The tool uses a YAML configuration file (`run.yaml`) to define:
- Execution modes (`local`, `cloud_local`, `slurm`)
- Runtime definitions (`venv`, Conda, Singularity, or plain Python)
- Template paths
- Script configurations
- Default arguments

Example configuration structure:
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
    slurm_defaults:
      nodes: 1
      ntasks: 1
      slurm_log_dir: "./logs"
    partition_defaults:
      2080-galvani:
        gres: "gpu:1"
        cpus_per_task: 12
        mem_per_cpu: "12G"
        time_limit: "3-00:00:00"
    partition_aliases:
      2080: "2080-galvani"

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
      param1: [1.0, 2.0]
      param2: ["value1", "value2"]
```

Default is to create such a `run.yaml` file in the main `submit` directory.

## More details on usage

Basic usage:
```bash
python submit/submit.py --mode [local|cloud_local|slurm] --runtime <runtime_name> --script <script_name> [--slurm_args <slurm_args>] [--script_args <script_args>]
```

### Command Line Arguments

Required arguments:
- `--script`: Name of the script configuration from run.yaml
- `--config_file`: Path to YAML config file (default: ./submit/run.yaml)

Optional arguments:
- `--mode`: Execution mode (local or slurm, default: local)
- `--runtime`: Runtime name from the YAML `runtime:` section. If omitted, `mode.<name>.runtime` is used.

SLURM-specific arguments:
- `--partition`: SLURM partition
- `--nodes`: Number of nodes
- `--cpus-per-task`: CPUs per task
- `--mem-per-cpu`: Memory per CPU (e.g. 4G)
- `--gres`: Generic resources (e.g. gpu:1)
- `--time`: Time limit (e.g. 3-00:00:00)

Additional arguments:
- Any `--key value1 value2 ...` pairs will be passed to the script. If multiple values are provided, the script will be run with all combinations of the script values.

### Examples

Run a local job with default parameters:
```bash
python submit/submit.py --mode local --script my_script
```

Run a SLURM job with custom parameters:
```bash
python submit/submit.py --mode slurm --script my_script --partition gpu --cpus-per-task 4 --mem-per-cpu 4G
```

If your config defines `mode.slurm.partition_defaults`, then choosing a
partition can auto-fill the other SLURM resources:

```bash
python submit/submit.py --mode slurm --partition 2080-galvani --script my_script
python submit/submit.py --mode slurm --partition 2080 --script my_script
```

The second form uses `mode.slurm.partition_aliases`.

### Batched Execution

To group runs sequentially and reduce the number of individual jobs queued on the cluster, the `submit.py` script supports multi-dimensional batching. You can pool multiple execution configurations into a single SLURM job submission.

To prevent collision with any `--batch_size` arguments parsed by your underlying Python scripts, use the `slurm_batch_size` argument in your configuration:

```yaml
scripts:
  my_script:
    path: "path/to/script.py"
    default_args:
      param1: ["value1", "value2", "value3"]
      batch_size: 32  # This is cleanly passed to python script args

      # Group parameter combos sequentially inside a single #SBATCH job
      slurm_batch_size: 3   # submit 3 sequential evaluations per 1 SLURM sbatch command
```

If you want specific parameter dimensions to vary within one logical job, mark them with `iter: true`:

```yaml
scripts:
  my_script:
    default_args:
      param1:
        values: ["value1", "value2", "value3"]
        iter: true
      seed: [0, 1, 2]
```

With that configuration, `submit` creates three logical jobs, one per `seed`, and
each logical job runs the three `param1` values sequentially. If
`slurm_batch_size` is also set, it batches whole logical jobs on top of that.

You can also use a top-level shorthand if you prefer to keep the parameter
values as plain lists:

```yaml
scripts:
  my_script:
    default_args:
      param1: ["value1", "value2", "value3"]
      seed: [0, 1, 2]
      iter: [param1]
```

Both YAML list styles work there, so this is equivalent:

```yaml
iter:
  - param1
```

### Testing with Docker

For fast SLURM emulation, this repository ships a fake-`sbatch` test path that
runs on Linux in Docker without starting a real cluster. It covers:

- batch-script rendering
- `sbatch` invocation
- CLI-level SLURM planning with `iter: true`

Run it with:

```bash
docker compose -f docker-compose.test.yml build
docker compose -f docker-compose.test.yml run --rm submit-tests
```

The Docker test runner targets only the submit-focused tests:

- `tests/test_submit_framework.py`
- `tests/test_slurm_executor.py`
- `tests/test_submit_cli_slurm.py`

For a real multi-container SLURM environment, a vendored cluster setup is
available under `submit/slurm-docker-cluster/`. That is the next step when the
fake-`sbatch` path is green and you want to validate `sbatch`, `squeue`, and
cluster behavior more end-to-end.

### Auto-Discovering FSP Priors

When submitting FSP jobs, you can let `submit` read calibrated prior hashes from
`models/gp/d=<data_name>_gp_prior/prior_registry.json` and the on-disk prior
folders automatically:

```yaml
scripts:
  train_fsp:
    default_args:
      data_name: ["pos_2", "base_2"]
      prior_name: ["auto"]
```

`submit` resolves `["auto"]` separately for each `data_name`, only creates valid
`(data_name, prior_name)` combinations, and passes hashes to the Python script
in quoted form such as `--prior_name 'a74704d4'`.

### Contributing

If you think features are missing or issues occur, please reach out.
