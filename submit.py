"""Submit jobs with various arguments to SLURM or to run locally."""

import argparse
import datetime
import subprocess
import sys
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Union

import yaml
from jinja2 import Template

from luno_experiments.fsp.prior_discovery import (
    format_prior_table,
    list_available_priors,
    validate_prior,
)


class ExecutionMode(Enum):
    """Enumeration of supported job execution modes."""

    SLURM = "slurm"
    LOCAL = "local"
    CLOUD_LOCAL = "cloud_local"

    @classmethod
    def from_str(cls, value: str) -> "ExecutionMode":
        """Convert a string to an ExecutionMode enum value.

        Args:
            value: String representation of the execution mode

        Returns:
            Corresponding ExecutionMode enum value

        Raises:
            ValueError: If the string doesn't match any enum value
        """
        try:
            return cls(value)
        except ValueError as err:
            msg = (
                f"Invalid execution mode: {value}. "
                f"Must be one of {[m.value for m in cls]}"
            )
            raise ValueError(msg) from err


class LocalJob:
    """Class for managing and executing jobs locally."""

    def __init__(
        self,
        cmd_template: Union[str, Path],
        job_name: str,
        template_vars: Union[dict, None] = None,
        log_path: Path = Path("logs"),
    ) -> None:
        """Initialize a local job.

        Args:
            cmd_template: Command template string or path to template file
            job_name: Name of the job
            template_vars: Variables to substitute in the template
            log_path: Directory to store log files
        """
        self._cmd_template = cmd_template
        self._template_vars = template_vars or {}
        self._job_name = job_name
        self._log_path = log_path

    def _render_cmd(self):
        """Render the command template with the provided variables."""
        if isinstance(self._cmd_template, Path):
            template_str = self._cmd_template.read_text()
        else:
            template_str = self._cmd_template

        template = Template(template_str)
        return template.render(**self._template_vars)

    def submit(self) -> None:
        """Execute the job locally and log its output."""
        # Render the final command
        cmd_str = self._render_cmd()

        # Setup log file
        self._log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self._log_path / f"{timestamp}_{self._job_name}.out"

        print(f"Running job '{self._job_name}' locally")
        print(f"Command: {cmd_str}")
        print(f"Logging to: {log_file}")

        with log_file.open("w") as f:
            f.write(f"Job Name: {self._job_name}\n")
            f.write(f"Command: {cmd_str}\n")
            f.write("-" * 80 + "\n\n")
            process = subprocess.Popen(
                cmd_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
            while True:
                line = process.stdout.readline()
                # in text mode, readline() returns "" on EOF
                if line == "" and process.poll() is not None:
                    break
                if line:
                    line = line.rstrip()
                    print(line)
                    f.write(line + "\n")
                    f.flush()
            return_code = process.poll()
            if return_code != 0:
                print(f"Job failed with return code {return_code}", file=sys.stderr)
                sys.exit(return_code)


class SlurmJob:
    """Class for managing and submitting jobs to SLURM scheduler."""

    def __init__(
        self,
        cmd_template: Union[str, Path],
        job_name: str,
        template_vars: dict,
        log_path: Path = Path("logs"),
    ) -> None:
        """Initialize a SLURM job.

        Args:
            cmd_template: Command template string or path to template file
            job_name: Name of the job
            template_vars: Variables to substitute in the template
            log_path: Directory to store log files
        """
        self._template = cmd_template
        self._vars = template_vars
        self._job_name = job_name
        self._log_path = log_path

    def _render(self) -> str:
        """Render the SLURM script template with the provided variables."""
        tpl = (
            Path(self._template).read_text()
            if isinstance(self._template, Path)
            else self._template
        )
        return Template(tpl).render(job_name=self._job_name, **self._vars)

    def submit(self) -> None:
        """Submit the job to the SLURM scheduler."""
        # ensure log dir exists (for SBATCH --output=...)
        self._log_path.mkdir(parents=True, exist_ok=True)

        script_fp = Path(f"{self._job_name}.slurm.sh")
        script_fp.write_text(self._render())
        script_fp.chmod(0o700)

        # submit and clean up
        subprocess.run(["sbatch", str(script_fp)], check=True)
        script_fp.unlink()


JOB_OPTIONS = {
    ExecutionMode.LOCAL: LocalJob,
    ExecutionMode.SLURM: SlurmJob,
    ExecutionMode.CLOUD_LOCAL: LocalJob,
}


def arg_to_string(val):
    """Turn argument to string without backslash"""
    return str(val).replace("/", "-")


def _handle_list_priors(data_names: list[str]) -> None:
    """Print available priors for each dataset and exit."""
    for data_name in data_names:
        priors = list_available_priors(data_name)
        print(f"\nAvailable priors for '{data_name}':")
        print(format_prior_table(priors))
    print()


def _resolve_auto_priors(
    keys: list[str],
    all_values: list[list],
    extra_args: dict,
) -> tuple[list[str], list[list]]:
    """Replace ``prior_name=["auto"]`` with discovered hashes per data_name."""
    if "prior_name" not in extra_args:
        return keys, all_values
    pn_vals = extra_args["prior_name"]
    if not (len(pn_vals) == 1 and str(pn_vals[0]).strip("'\"") == "auto"):
        return keys, all_values

    data_names = extra_args["data_name"] if "data_name" in extra_args else []
    all_hashes: set[str] = set()
    for dn in data_names:
        priors = list_available_priors(str(dn))
        for p in priors:
            all_hashes.add(p["hash"])

    if not all_hashes:
        print("WARNING: prior_name='auto' but no calibrated priors found on disk.")
        return keys, all_values

    sorted_hashes = sorted(all_hashes)
    print(f"Auto-discovered {len(sorted_hashes)} prior(s): {', '.join(sorted_hashes)}")
    pn_idx = keys.index("prior_name")
    all_values[pn_idx] = [f"'{h}'" for h in sorted_hashes]
    return keys, all_values


def _validate_prior_combinations(
    combinations: list[tuple],
    keys: list[str],
) -> list[tuple]:
    """Drop combinations whose (data_name, prior_name) pair is invalid."""
    if "prior_name" not in keys or "data_name" not in keys:
        return combinations

    pn_idx = keys.index("prior_name")
    dn_idx = keys.index("data_name")

    valid = []
    warned: set[tuple[str, str]] = set()
    for combo in combinations:
        dn = str(combo[dn_idx])
        pn = str(combo[pn_idx]).strip("'\"")
        is_ok, _, err = validate_prior(dn, pn)
        if is_ok:
            valid.append(combo)
        else:
            pair = (dn, pn)
            if pair not in warned:
                warned.add(pair)
                priors = list_available_priors(dn)
                print(f"WARNING: {err}")
                print(f"  Available priors for '{dn}':")
                print(format_prior_table(priors))
                print()
    return valid


def main() -> None:
    """Main entry point for job submission.

    Parses command line arguments and submits jobs according to the specified mode
    (local or SLURM) and configuration.
    """
    parser = argparse.ArgumentParser(description="Submit jobs.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[m.value for m in ExecutionMode],
        default=ExecutionMode.LOCAL.value,
        help="Execution mode (e.g. slurm or local)",
    )

    parser.add_argument(
        "--script",
        type=str,
        default=None,
        help="Which entry under `scripts:` in the YAML to run.",
    )

    parser.add_argument(
        "--config_file",
        type=Path,
        default=Path("./submit/run.yaml"),
        help="YAML config file, containing all run variables.",
    )

    # Special slurm arguments
    parser.add_argument("--partition", type=str, help="SLURM partition")
    parser.add_argument("--nodes", type=int, help="Number of nodes")
    parser.add_argument("--cpus-per-task", type=int, help="CPUs per task")
    parser.add_argument("--mem-per-cpu", type=str, help="Memory per CPU (e.g. 4G)")
    parser.add_argument("--gres", type=str, help="Generic resources (e.g. gpu:1)")
    parser.add_argument("--time", type=str, help="Time limit (e.g. 3-00:00:00)")
    parser.add_argument(
        "--slurm_log_dir",
        type=str,
        default="./logs",
        help="Log directory for slurm job.",
    )
    parser.add_argument(
        "--list-priors",
        nargs="+",
        metavar="DATA_NAME",
        default=None,
        help="List available priors for given dataset(s) and exit.",
    )
    parser.add_argument(
        "--skip-prior-validation",
        action="store_true",
        default=False,
        help="Skip prior_name validation against disk when submitting FSP jobs.",
    )

    # Split off any --key value1 value2 ... into `unknown`
    args, unknown = parser.parse_known_args()

    if args.list_priors is not None:
        _handle_list_priors(args.list_priors)
        return

    if args.script is None:
        parser.error("--script is required when not using --list-priors")

    # Convert mode string to enum
    args.mode = ExecutionMode.from_str(args.mode)

    # If SlurmJob then get slurm args
    if args.mode == ExecutionMode.SLURM:
        mode_specific_overrides = {
            "partition": args.partition,
            "nodes": args.nodes,
            "cpus_per_task": args.cpus_per_task,
            "mem_per_cpu": args.mem_per_cpu,
            "gres": args.gres,
            "time_limit": args.time,
            "slurm_log_dir": args.slurm_log_dir,
        }
    else:
        mode_specific_overrides = {}

    # Load YAML config
    with args.config_file.open("r") as f:
        config = yaml.safe_load(f)

    # Grab the right mode-block
    mode_cfg = config["mode"][args.mode.value]
    pykernel = mode_cfg.get("pykernel", "")
    pre_command = mode_cfg.get("pre_command", "")
    template_fp = Path(mode_cfg["template"])

    # Grab the selected script-block
    script_cfg = config["scripts"][args.script]
    script_path = script_cfg["path"]
    default_args = script_cfg.get("default_args", {})

    # Combine variables to create job arrays
    extra_args = {}
    iter_multiplier = 1
    
    for k, v in default_args.items():
        if isinstance(v, dict) and "values" in v:
            vals = v["values"]
            vals_list = vals if isinstance(vals, list) else [vals]
            extra_args[k] = vals_list
            if v.get("iter") is True:
                iter_multiplier *= len(vals_list)
        else:
            extra_args[k] = v if isinstance(v, list) else [v]
            
    # Extract slurm_batch_size if specified and remove from script arguments
    slurm_batch_size = extra_args.pop("slurm_batch_size", [1])[0]
    if not isinstance(slurm_batch_size, int) or slurm_batch_size < 1:
        slurm_batch_size = 1
        
    slurm_batch_size *= iter_multiplier
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if not tok.startswith("--"):
            parser.error(f"Unexpected token {tok!r}")
        key = tok.lstrip("--")
        i += 1
        vals = []
        # consume until next --foo or end
        while i < len(unknown) and not unknown[i].startswith("--"):
            vals.append(unknown[i])
            i += 1
        if not vals:
            # store_true argument (boolean flag with no value)
            extra_args[key] = [True]
        else:
            extra_args[key] = vals

    keys = list(extra_args.keys())
    all_values = [extra_args[k] for k in keys]

    keys, all_values = _resolve_auto_priors(keys, all_values, extra_args)

    # Show job creation details
    total_commands = len(list(product(*all_values)))
    total_jobs = (total_commands + slurm_batch_size - 1) // slurm_batch_size
    print(f"Creating {total_jobs} job(s) for {total_commands} commands with the following parameters (batch size: {slurm_batch_size}):")
    if keys:
        for key, values in extra_args.items():
            values_str = ", ".join(str(v) for v in values)
            print(f"  {key}: [{values_str}]")
        print()
    else:
        print("  (no parameters specified)")
        print()

    # Create sequential command arrays for batch execution
    all_combinations = list(product(*all_values))

    if not args.skip_prior_validation and "prior_name" in keys:
        pre_count = len(all_combinations)
        all_combinations = _validate_prior_combinations(all_combinations, keys)
        skipped = pre_count - len(all_combinations)
        if skipped:
            print(f"Skipped {skipped} combination(s) with invalid priors.\n")
        if not all_combinations:
            print("ERROR: No valid (data_name, prior_name) combinations remain.")
            sys.exit(1)
        total_commands = len(all_combinations)
        total_jobs = (total_commands + slurm_batch_size - 1) // slurm_batch_size
    
    for batch_idx in range(total_jobs):
        start_idx = batch_idx * slurm_batch_size
        batch_combos = all_combinations[start_idx:start_idx + slurm_batch_size]
        
        command_suffixes = []
        for combo in batch_combos:
            combo_dict = dict(zip(keys, combo))
            suffix = " ".join(
                f"--{k}" if v is True else f"--{k} {v}" 
                for k, v in combo_dict.items()
            )
            command_suffixes.append(suffix)
            
        # Determine job name
        if slurm_batch_size == 1:
            first_combo_dict = dict(zip(keys, batch_combos[0]))
            suffix_desc = "_".join(
                f"{arg_to_string(k)}={arg_to_string(v)}" for k, v in first_combo_dict.items()
            )
            name = args.script if not suffix_desc else f"{args.script}_{suffix_desc}"
        else:
            name = f"{args.script}_batch_{batch_idx}"

        # Prepare the vars that go into Jinja
        template_vars = {
            "pykernel": pykernel,
            "pre_command": pre_command,
            "script_path": str(script_path),
            "command_suffixes": command_suffixes,
            "script_args": {}, # Backwards compatibility if needed
        }

        # Add mode specific arguments
        template_vars.update(
            {k: v for k, v in mode_specific_overrides.items() if v is not None}
        )

        job = JOB_OPTIONS[args.mode](
            cmd_template=template_fp, job_name=name, template_vars=template_vars
        )
        job.submit()

    print(f"Submitted {total_jobs} job(s)")


if __name__ == "__main__":
    main()
