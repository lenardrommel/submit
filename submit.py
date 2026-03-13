"""Submit parameterized jobs either locally or through SLURM.

This module reads a YAML configuration file describing execution modes,
runtimes, and scripts, expands parameter grids into concrete command
combinations, and then either executes those commands locally or submits them
to SLURM.

Features include:

- local, cloud-local, and SLURM execution contexts
- runtime selection for venv, Conda, Singularity, or plain Python
- parameter-grid expansion from YAML and CLI overrides
- logical job grouping via ``iter: true``
- optional batching of multiple logical jobs into one submission
- automatic prior discovery for FSP experiments
- validation of ``(data_name, prior_name)`` combinations before submission
"""

from __future__ import annotations

import argparse
import ast
import datetime
import difflib
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Union

import yaml
from jinja2 import Template

from luno_experiments.fsp.prior_discovery import (
    format_prior_table,
    list_available_priors,
    validate_prior,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_ITER_WARNING = (
    "Top-level `iter` in `default_args` is deprecated and ignored. "
    "Use `param_name: {values: [...], iter: true}` instead."
)
DEFAULT_SLURM_RESOURCES: dict[str, Any] = {
    "partition": "2080-galvani",
    "nodes": 1,
    "ntasks": 1,
    "cpus_per_task": 12,
    "mem_per_cpu": "4G",
    "gres": "gpu:1",
    "time_limit": "3-00:00:00",
    "slurm_log_dir": "./logs",
}
SLURM_RESOURCE_LABELS = {
    "partition": "--partition",
    "nodes": "--nodes",
    "ntasks": "--ntasks",
    "cpus_per_task": "--cpus-per-task",
    "mem_per_cpu": "--mem-per-cpu",
    "gres": "--gres",
    "time_limit": "--time",
    "slurm_log_dir": "--slurm_log_dir",
}
SUBMIT_INTERNAL_ARGUMENTS = {"slurm_batch_size"}


class ExecutionMode(Enum):
    """Supported job execution backends."""

    SLURM = "slurm"
    LOCAL = "local"
    CLOUD_LOCAL = "cloud_local"

    @classmethod
    def from_str(cls, value: str) -> "ExecutionMode":
        """Return the enum member matching ``value``."""
        try:
            return cls(value)
        except ValueError as err:
            msg = (
                f"Invalid execution mode: {value}. "
                f"Must be one of {[m.value for m in cls]}"
            )
            raise ValueError(msg) from err


@dataclass(frozen=True)
class ExecutionSettings:
    """Resolved execution settings for a single submission."""

    template_path: Path
    shell_executable: str
    python_cmd: str
    pre_command: str
    command_wrapper: str
    runtime_name: str | None
    legacy_pykernel: str


class LocalJob:
    """Run a rendered command locally and stream its output to a log file."""

    def __init__(
        self,
        cmd_template: Union[str, Path],
        job_name: str,
        template_vars: dict[str, Any] | None = None,
        log_path: Path = Path("logs"),
        shell_executable: str = "bash",
    ) -> None:
        self._cmd_template = cmd_template
        self._template_vars = template_vars or {}
        self._job_name = job_name
        self._log_path = log_path
        self._shell_executable = shell_executable

    def _render_cmd(self) -> str:
        """Render the command template into a concrete shell command string."""
        if isinstance(self._cmd_template, Path):
            template_str = self._cmd_template.read_text()
        else:
            template_str = self._cmd_template

        template = Template(template_str)
        return template.render(**self._template_vars)

    def submit(self) -> None:
        """Execute the rendered command locally."""
        cmd_str = self._render_cmd()

        self._log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self._log_path / f"{timestamp}_{self._job_name}.out"

        print(f"Running job '{self._job_name}' locally")
        print(f"Shell: {self._shell_executable} -lc")
        print(f"Command: {cmd_str}")
        print(f"Logging to: {log_file}")

        with log_file.open("w") as f:
            f.write(f"Job Name: {self._job_name}\n")
            f.write(f"Shell: {self._shell_executable} -lc\n")
            f.write(f"Command: {cmd_str}\n")
            f.write("-" * 80 + "\n\n")

            process = subprocess.Popen(
                [self._shell_executable, "-lc", cmd_str],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip()
                print(line)
                f.write(line + "\n")
                f.flush()

            return_code = process.wait()
            if return_code != 0:
                print(f"Job failed with return code {return_code}", file=sys.stderr)
                sys.exit(return_code)


class SlurmJob:
    """Render a SLURM submission script and submit it with ``sbatch``."""

    def __init__(
        self,
        cmd_template: Union[str, Path],
        job_name: str,
        template_vars: dict[str, Any],
        log_path: Path = Path("logs"),
    ) -> None:
        self._template = cmd_template
        self._vars = template_vars
        self._job_name = job_name
        self._log_path = log_path

    def _render(self) -> str:
        """Render the SLURM script template into a concrete batch script."""
        tpl = (
            Path(self._template).read_text()
            if isinstance(self._template, Path)
            else self._template
        )
        render_vars = dict(self._vars)
        render_vars.setdefault("job_name", self._job_name)
        return Template(tpl).render(**render_vars)

    def submit(self) -> None:
        """Submit the rendered script to SLURM via ``sbatch``."""
        self._log_path.mkdir(parents=True, exist_ok=True)

        script_fp = Path(f"{self._job_name}.slurm.sh")
        script_fp.write_text(self._render())
        script_fp.chmod(0o700)

        result = subprocess.run(
            ["sbatch", str(script_fp)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            error_output = (result.stderr or result.stdout).strip()
            resources = self._vars.get("slurm_resources_summary", "(unknown)")
            partition = self._vars.get("partition")
            profile_name = self._vars.get("slurm_partition_profile")
            profile_hint = ""
            if partition and not profile_name:
                profile_hint = (
                    "\nNo partition defaults were configured for this partition. "
                    "You can either pass --gres/--cpus-per-task/--mem-per-cpu/"
                    "--time explicitly, or add "
                    f"`mode.slurm.partition_defaults.{partition}` to the config."
                )
            raise SystemExit(
                f"sbatch failed for job '{self._job_name}': {error_output}\n"
                f"Effective SLURM resources: {resources}\n"
                f"Generated script kept at: {script_fp}\n"
                "Check the requested SLURM resources such as --partition, "
                "--gres, --cpus-per-task, --mem-per-cpu, and --time."
                f"{profile_hint}"
            )
        if result.stdout.strip():
            print(result.stdout.strip())
            status_dir = self._vars.get("status_dir")
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if status_dir and match is not None:
                job_id = match.group(1)
                status_path = f"{status_dir}/{self._job_name}_{job_id}.status"
                print(
                    "Progress status file: "
                    f"{status_path}"
                )
                print(
                    "Check progress with: "
                    f"python submit/submit.py --show-status {status_path}"
                )
        script_fp.unlink()


JOB_OPTIONS = {
    ExecutionMode.LOCAL: LocalJob,
    ExecutionMode.SLURM: SlurmJob,
    ExecutionMode.CLOUD_LOCAL: LocalJob,
}


def _warn(message: str) -> None:
    """Print a warning message."""
    print(f"WARNING: {message}")


def _format_yaml_load_error(config_file: Path, err: yaml.YAMLError) -> str:
    """Return a user-facing YAML parse error with submit-specific hints."""
    parts = [f"Invalid YAML in '{config_file}'."]

    problem_mark = getattr(err, "problem_mark", None)
    if problem_mark is not None:
        parts.append(
            f"Location: line {problem_mark.line + 1}, column {problem_mark.column + 1}."
        )

    problem = getattr(err, "problem", None)
    if problem:
        parts.append(f"YAML parser message: {problem}.")
    else:
        parts.append(str(err))

    parts.append(
        "If this is related to `iter`, use one of these supported forms:"
    )
    parts.append("1. `param_name: {values: [...], iter: true}`")
    parts.append("2. `param_name: [...]` together with `iter: [param_name]`")
    parts.append(
        "Do not place `iter:` beside a dashed YAML list under the same parameter."
    )

    return " ".join(parts)


def _format_slurm_resources_summary(resources: dict[str, Any]) -> str:
    """Return a compact string of effective SLURM resources."""
    ordered_keys = [
        "partition",
        "nodes",
        "ntasks",
        "cpus_per_task",
        "mem_per_cpu",
        "gres",
        "time_limit",
    ]
    return ", ".join(f"{key}={resources[key]}" for key in ordered_keys)


def _resolve_slurm_resource_settings(
    mode_cfg: dict[str, Any],
    cli_overrides: dict[str, Any],
) -> tuple[dict[str, Any], str | None, tuple[str, str] | None]:
    """Resolve effective SLURM resources from defaults, partition profiles, and CLI."""
    resources = dict(DEFAULT_SLURM_RESOURCES)

    configured_defaults = mode_cfg.get("slurm_defaults", {})
    if configured_defaults:
        resources.update({key: value for key, value in configured_defaults.items() if value is not None})

    requested_partition = cli_overrides.get("partition") or resources.get("partition")
    partition_aliases = {
        str(key): str(value)
        for key, value in mode_cfg.get("partition_aliases", {}).items()
    }
    partition_alias = None
    resolved_partition = requested_partition
    if requested_partition in partition_aliases:
        resolved_partition = partition_aliases[requested_partition]
        partition_alias = (requested_partition, resolved_partition)

    partition_defaults = {
        str(key): value for key, value in mode_cfg.get("partition_defaults", {}).items()
    }
    matched_profile = None
    if resolved_partition is not None:
        partition_profile = partition_defaults.get(resolved_partition)
        if partition_profile is not None:
            resources.update({key: value for key, value in partition_profile.items() if value is not None})
            matched_profile = resolved_partition
        resources["partition"] = resolved_partition

    resources.update({key: value for key, value in cli_overrides.items() if value is not None})
    if partition_alias is not None:
        resources["partition"] = partition_alias[1]
    return resources, matched_profile, partition_alias


def _resolve_status_dir(path_value: str) -> str:
    """Resolve a status/log directory to an absolute path."""
    raw_path = Path(path_value).expanduser()
    if raw_path.is_absolute():
        return str(raw_path)
    return str((Path.cwd() / raw_path).resolve())


def _parse_status_file(status_file: Path) -> dict[str, str]:
    """Parse a simple ``key=value`` status file."""
    status: dict[str, str] = {}
    for line in status_file.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        status[key.strip()] = value.strip()
    return status


def _iter_status_files(targets: list[str]) -> list[Path]:
    """Expand file and directory targets into concrete ``.status`` files."""
    status_files: list[Path] = []
    for target in targets:
        path = Path(target).expanduser()
        if path.is_dir():
            status_files.extend(sorted(path.glob("*.status")))
        else:
            status_files.append(path)
    return status_files


def _show_status(status_targets: list[str], parser: argparse.ArgumentParser) -> None:
    """Print progress summaries for one or more status files and exit."""
    status_files = _iter_status_files(status_targets)
    if not status_files:
        parser.error(
            "No status files found. Pass one or more .status files or a directory "
            "containing them."
        )

    missing_files = [status_file for status_file in status_files if not status_file.exists()]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        parser.error(f"Status file(s) not found: {missing}")

    for idx, status_file in enumerate(status_files):
        status = _parse_status_file(status_file)
        state = status.get("state", "unknown")
        current = status.get("current", "?")
        total = status.get("total", "?")
        updated_at = status.get("updated_at", "unknown")
        job_name = status.get("job_name", status_file.stem)
        job_id = status.get("job_id", "unknown")
        current_command = status.get("current_command", "")

        print(
            f"{status_file.resolve()}: "
            f"job={job_name} job_id={job_id} state={state} "
            f"progress={current}/{total} updated_at={updated_at}"
        )
        if current_command:
            print(f"  current_command: {current_command}")
        if idx != len(status_files) - 1:
            print()


def _extract_script_argument_names(script_path: Path) -> set[str]:
    """Collect supported long-form CLI argument names from ``argparse`` calls."""
    try:
        tree = ast.parse(script_path.read_text(), filename=str(script_path))
    except SyntaxError as err:
        raise ValueError(
            f"Could not parse script '{script_path}' for argument validation: {err.msg}"
        ) from err

    argument_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            ):
                argument_names.add(arg.value[2:])

    return argument_names


def _extract_script_argument_defaults(script_path: Path) -> dict[str, Any]:
    """Collect simple ``argparse`` defaults keyed by argument destination name."""
    try:
        tree = ast.parse(script_path.read_text(), filename=str(script_path))
    except SyntaxError as err:
        raise ValueError(
            f"Could not parse script '{script_path}' for argument validation: {err.msg}"
        ) from err

    defaults: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue

        long_flags = [
            arg.value[2:]
            for arg in node.args
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            )
        ]
        if not long_flags:
            continue

        dest = None
        default_value = None
        has_default = False
        for keyword in node.keywords:
            if keyword.arg == "dest":
                try:
                    dest = ast.literal_eval(keyword.value)
                except (ValueError, SyntaxError):
                    dest = None
            if keyword.arg == "default":
                try:
                    default_value = ast.literal_eval(keyword.value)
                    has_default = True
                except (ValueError, SyntaxError):
                    has_default = False

        if not has_default:
            continue

        arg_name = dest or long_flags[0]
        defaults[arg_name] = default_value

    return defaults


def _format_unknown_script_argument_error(
    invalid_args: list[str],
    valid_args: set[str],
    script_path: Path,
) -> str:
    """Build a user-facing validation error for unsupported script args."""
    lines = [
        f"Script '{script_path}' does not define these argument(s): "
        f"{', '.join(invalid_args)}."
    ]

    suggestion_lines = []
    for invalid_arg in invalid_args:
        matches = difflib.get_close_matches(invalid_arg, sorted(valid_args), n=3, cutoff=0.5)
        if matches:
            suggestion_lines.append(
                f"{invalid_arg} -> {', '.join(matches)}"
            )
    if suggestion_lines:
        lines.append("Closest matches:")
        lines.extend(f"  {line}" for line in suggestion_lines)

    preview_args = ", ".join(sorted(valid_args)[:12])
    if len(valid_args) > 12:
        preview_args += ", ..."
    lines.append(f"Known script arguments include: {preview_args}")
    lines.append(
        "If you intended a submit-only setting, keep it out of `default_args`."
    )
    return "\n".join(lines)


def _validate_script_argument_names(
    script_path: Path,
    arg_specs: dict[str, dict[str, Any]],
    parser: argparse.ArgumentParser,
) -> None:
    """Abort before submission if config/CLI args are not defined by the script."""
    valid_args = _extract_script_argument_names(script_path)
    if not valid_args:
        _warn(
            f"Could not discover argparse arguments in '{script_path}'. "
            "Skipping submit-side argument-name validation."
        )
        return

    provided_args = [
        key for key in arg_specs if key not in SUBMIT_INTERNAL_ARGUMENTS
    ]
    invalid_args = sorted(key for key in provided_args if key not in valid_args)
    if invalid_args:
        parser.error(
            _format_unknown_script_argument_error(
                invalid_args,
                valid_args,
                script_path,
            )
        )


def arg_to_string(val: Any) -> str:
    """Return a job-name-safe string representation of an argument value."""
    return str(val).strip("'\"").replace("/", "-").replace(" ", "")


def _handle_list_priors(data_names: list[str]) -> None:
    """Print all registered priors for the requested datasets and exit."""
    for data_name in data_names:
        priors = list_available_priors(data_name)
        print(f"\nAvailable priors for '{data_name}':")
        print(format_prior_table(priors))
    print()


def _as_value_list(value: Any) -> list[Any]:
    """Normalize scalars and lists to a list."""
    return value if isinstance(value, list) else [value]


def _normalize_arg_specs(default_args: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize YAML argument definitions to ``{values, iter}`` records."""
    arg_specs: dict[str, dict[str, Any]] = {}
    iter_arg_names: set[str] = set()

    if "iter" in default_args:
        iter_value = default_args["iter"]
        if isinstance(iter_value, str):
            iter_arg_names.add(iter_value)
        elif isinstance(iter_value, list):
            if not all(isinstance(item, str) for item in iter_value):
                raise ValueError(
                    "Top-level `iter` must contain only argument names when "
                    "provided as a list."
                )
            iter_arg_names.update(iter_value)
        elif not (isinstance(iter_value, dict) and "values" in iter_value):
            _warn(LEGACY_ITER_WARNING)

    for key, value in default_args.items():
        if key == "iter" and not (isinstance(value, dict) and "values" in value):
            continue

        if isinstance(value, dict):
            if "values" not in value:
                raise ValueError(
                    f"Argument '{key}' uses mapping syntax but is missing a "
                    "`values` field."
                )
            unknown_keys = set(value) - {"values", "iter"}
            if unknown_keys:
                unknown = ", ".join(sorted(unknown_keys))
                raise ValueError(
                    f"Argument '{key}' contains unsupported keys: {unknown}"
                )
            arg_specs[key] = {
                "values": _as_value_list(value["values"]),
                "iter": bool(value.get("iter", False)),
            }
            continue

        arg_specs[key] = {"values": _as_value_list(value), "iter": False}

    for iter_arg_name in iter_arg_names:
        if iter_arg_name not in arg_specs:
            raise ValueError(
                f"Top-level `iter` references unknown argument '{iter_arg_name}'."
            )
        arg_specs[iter_arg_name]["iter"] = True

    return arg_specs


def _apply_cli_overrides(
    arg_specs: dict[str, dict[str, Any]],
    unknown: list[str],
    parser: argparse.ArgumentParser,
) -> None:
    """Apply ``--key value1 value2 ...`` overrides from the CLI."""
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if not token.startswith("--"):
            parser.error(f"Unexpected token {token!r}")

        key = token.lstrip("--")
        i += 1
        values: list[Any] = []
        while i < len(unknown) and not unknown[i].startswith("--"):
            values.append(unknown[i])
            i += 1

        resolved_values = values or [True]
        if key in arg_specs:
            arg_specs[key]["values"] = resolved_values
        else:
            arg_specs[key] = {"values": resolved_values, "iter": False}


def _pop_submission_batch_size(arg_specs: dict[str, dict[str, Any]]) -> int:
    """Read and remove the submission batch size from normalized arg specs."""
    batch_spec = arg_specs.pop("slurm_batch_size", None)
    if batch_spec is None:
        return 1

    if batch_spec.get("iter"):
        _warn("`slurm_batch_size` ignores `iter: true` and uses the first value.")

    values = batch_spec["values"]
    if not values:
        return 1
    if len(values) > 1:
        _warn("`slurm_batch_size` only supports one value. Using the first one.")

    try:
        batch_size = int(values[0])
    except (TypeError, ValueError):
        _warn(f"Invalid slurm_batch_size {values[0]!r}. Falling back to 1.")
        return 1

    if batch_size < 1:
        _warn(f"Invalid slurm_batch_size {batch_size!r}. Falling back to 1.")
        return 1

    return batch_size


def _is_auto_prior_request(arg_values: dict[str, list[Any]]) -> bool:
    """Return whether ``prior_name`` was set to the special value ``auto``."""
    if "prior_name" not in arg_values:
        return False

    pn_vals = arg_values["prior_name"]
    if len(pn_vals) != 1:
        return False

    return str(pn_vals[0]).strip("'\"").lower() == "auto"


def _quoted_prior_hashes(data_name: str) -> list[str]:
    """Return all discovered prior hashes for ``data_name`` as quoted strings."""
    hashes = sorted({prior["hash"] for prior in list_available_priors(str(data_name))})
    return [f"'{prior_hash}'" for prior_hash in hashes]


def _build_argument_combinations(
    arg_specs: dict[str, dict[str, Any]],
) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Expand argument values into concrete command combinations."""
    keys = list(arg_specs.keys())
    arg_values = {key: spec["values"] for key, spec in arg_specs.items()}

    if not _is_auto_prior_request(arg_values):
        all_values = [arg_values[key] for key in keys]
        return keys, list(product(*all_values))

    if "data_name" not in arg_values:
        _warn("prior_name='auto' requires data_name to be set.")
        return keys, []

    other_keys = [key for key in keys if key not in {"data_name", "prior_name"}]
    other_values = [arg_values[key] for key in other_keys]
    other_combinations = list(product(*other_values)) if other_values else [()]

    combinations: list[tuple[Any, ...]] = []
    missing_data_names: list[str] = []

    for data_name in arg_values["data_name"]:
        quoted_hashes = _quoted_prior_hashes(str(data_name))
        if not quoted_hashes:
            missing_data_names.append(str(data_name))
            continue

        print(
            f"Auto-discovered {len(quoted_hashes)} prior(s) for '{data_name}': "
            f"{', '.join(quoted_hashes)}"
        )

        for other_combo in other_combinations:
            other_args = dict(zip(other_keys, other_combo))
            for prior_name in quoted_hashes:
                combo = []
                for key in keys:
                    if key == "data_name":
                        combo.append(data_name)
                    elif key == "prior_name":
                        combo.append(prior_name)
                    else:
                        combo.append(other_args[key])
                combinations.append(tuple(combo))

    for data_name in missing_data_names:
        _warn(
            "prior_name='auto' but no calibrated priors found for "
            f"'{data_name}'."
        )

    return keys, combinations


def _validate_prior_combinations(
    combinations: list[tuple[Any, ...]],
    keys: list[str],
    script_arg_defaults: dict[str, Any] | None = None,
) -> list[tuple[Any, ...]]:
    """Filter out invalid ``(data_name, prior_name)`` combinations."""
    if "prior_name" not in keys or "data_name" not in keys:
        return combinations

    pn_idx = keys.index("prior_name")
    dn_idx = keys.index("data_name")
    mean_fn_idx = keys.index("mean_fn_type") if "mean_fn_type" in keys else None
    default_mean_fn = None
    if script_arg_defaults is not None:
        default_mean_fn = script_arg_defaults.get("mean_fn_type")

    valid = []
    warned: set[tuple[str, str, str | None]] = set()
    for combo in combinations:
        data_name = str(combo[dn_idx])
        prior_name = str(combo[pn_idx]).strip("'\"")
        mean_fn = default_mean_fn
        if mean_fn_idx is not None:
            mean_fn = str(combo[mean_fn_idx])
        is_ok, _, err = validate_prior(data_name, prior_name, mean_fn=mean_fn)
        if is_ok:
            valid.append(combo)
            continue

        pair = (data_name, prior_name, mean_fn)
        if pair in warned:
            continue

        warned.add(pair)
        priors = list_available_priors(data_name)
        _warn(str(err))
        print(f"  Available priors for '{data_name}':")
        print(format_prior_table(priors))
        print()

    return valid


def _group_commands_by_iter_flags(
    combinations: list[tuple[Any, ...]],
    keys: list[str],
    arg_specs: dict[str, dict[str, Any]],
) -> list[list[tuple[Any, ...]]]:
    """Group concrete command dictionaries into logical jobs.

    Arguments marked with ``iter: true`` vary within a job, while all other
    arguments define separate logical jobs.

    Example:
        If ``data_name`` has 5 values and ``num_samples`` has 2 values, both
        marked with ``iter: true``, and ``seed`` has 10 values without
        ``iter``, then this returns 10 logical jobs, each containing 10
        commands.
    """
    iter_keys = {key for key, spec in arg_specs.items() if spec["iter"]}
    if not iter_keys:
        return [[combo] for combo in combinations]

    group_key_indices = [idx for idx, key in enumerate(keys) if key not in iter_keys]
    grouped: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}

    for combo in combinations:
        group_key = tuple(combo[idx] for idx in group_key_indices)
        grouped.setdefault(group_key, []).append(combo)

    return list(grouped.values())


def _batch_logical_jobs(
    logical_jobs: list[list[tuple[Any, ...]]],
    batch_size: int,
) -> list[list[list[tuple[Any, ...]]]]:
    """Pack logical jobs into physical submissions."""
    return [
        logical_jobs[idx : idx + batch_size]
        for idx in range(0, len(logical_jobs), batch_size)
    ]


def _flatten_logical_job_batch(
    logical_job_batch: list[list[tuple[Any, ...]]],
) -> list[tuple[Any, ...]]:
    """Flatten one physical submission back to a command list."""
    return [combo for logical_job in logical_job_batch for combo in logical_job]


def _format_command_suffix(combo_dict: dict[str, Any]) -> str:
    """Format one concrete command suffix from a resolved argument mapping."""
    return " ".join(
        f"--{key}" if value is True else f"--{key} {value}"
        for key, value in combo_dict.items()
    )


def _logical_job_name_suffix(
    logical_job: list[tuple[Any, ...]],
    keys: list[str],
    arg_specs: dict[str, dict[str, Any]],
) -> str:
    """Build a job-name suffix from the non-iter arguments of one logical job."""
    non_iter_keys = [key for key in keys if not arg_specs[key]["iter"]]
    if not non_iter_keys:
        return ""

    combo_dict = dict(zip(keys, logical_job[0]))
    return "_".join(
        f"{arg_to_string(key)}={arg_to_string(combo_dict[key])}"
        for key in non_iter_keys
    )


def _resolve_config_path(
    path_value: Union[str, Path],
    config_file: Path,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve config-relative, repo-relative, or cwd-relative paths."""
    raw_path = Path(path_value).expanduser()
    if raw_path.is_absolute():
        normalized = raw_path
        if must_exist and not normalized.exists():
            raise FileNotFoundError(f"Resolved path does not exist: {normalized}")
        return normalized

    candidates = []
    for base in (Path.cwd(), PROJECT_ROOT, config_file.parent):
        candidate = (base / raw_path).resolve()
        if candidate not in candidates:
            candidates.append(candidate)
            if candidate.exists():
                return candidate

    if must_exist:
        candidate_str = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            f"Could not resolve '{path_value}' from config '{config_file}'. "
            f"Tried:\n{candidate_str}"
        )

    return candidates[0]


def _maybe_resolve_python_cmd(python_cmd: str, config_file: Path) -> str:
    """Resolve path-like Python commands without dereferencing venv symlinks."""
    stripped = python_cmd.strip()
    if not stripped:
        return stripped

    if "/" not in stripped and not stripped.startswith(("~", ".")):
        return stripped

    raw_path = Path(stripped).expanduser()
    if raw_path.is_absolute():
        return str(raw_path)

    for base in (config_file.parent, Path.cwd(), PROJECT_ROOT):
        candidate = base / raw_path
        if candidate.exists():
            return str(candidate)

    return str(PROJECT_ROOT / raw_path)


def _resolve_execution_settings(
    config: dict[str, Any],
    config_file: Path,
    mode: ExecutionMode,
    runtime_name: str | None,
) -> ExecutionSettings:
    """Resolve template and runtime settings for one execution mode."""
    mode_cfg = config["mode"][mode.value]
    template_path = _resolve_config_path(mode_cfg["template"], config_file)
    shell_executable = mode_cfg.get("shell_executable", "bash")

    runtime_cfgs = config.get("runtime")
    if runtime_cfgs:
        selected_runtime = (
            runtime_name
            or mode_cfg.get("runtime")
            or mode_cfg.get("default_runtime")
            or config.get("default_runtime")
        )
        if selected_runtime is None:
            if len(runtime_cfgs) == 1:
                selected_runtime = next(iter(runtime_cfgs))
            else:
                raise ValueError(
                    f"Config '{config_file}' defines multiple runtimes. "
                    "Pass --runtime or set mode.<name>.runtime."
                )
        if selected_runtime not in runtime_cfgs:
            raise ValueError(
                f"Runtime '{selected_runtime}' is not defined in '{config_file}'."
            )

        runtime_cfg = runtime_cfgs[selected_runtime] or {}
        python_cmd = _maybe_resolve_python_cmd(
            str(runtime_cfg.get("python_cmd", "python")),
            config_file,
        )
        pre_command = str(runtime_cfg.get("pre_command", ""))
        command_wrapper = str(runtime_cfg.get("command_wrapper", ""))
        legacy_pykernel = python_cmd if mode == ExecutionMode.LOCAL else command_wrapper

        return ExecutionSettings(
            template_path=template_path,
            shell_executable=shell_executable,
            python_cmd=python_cmd or "python",
            pre_command=pre_command,
            command_wrapper=command_wrapper,
            runtime_name=selected_runtime,
            legacy_pykernel=legacy_pykernel,
        )

    if runtime_name is not None:
        raise ValueError(
            f"Config '{config_file}' does not define a runtime section, "
            f"so --runtime {runtime_name!r} cannot be used."
        )

    legacy_pykernel = str(mode_cfg.get("pykernel", "python"))
    pre_command = str(mode_cfg.get("pre_command", ""))
    if mode == ExecutionMode.LOCAL:
        python_cmd = _maybe_resolve_python_cmd(legacy_pykernel or "python", config_file)
        command_wrapper = ""
    else:
        python_cmd = "python"
        command_wrapper = legacy_pykernel

    return ExecutionSettings(
        template_path=template_path,
        shell_executable=shell_executable,
        python_cmd=python_cmd or "python",
        pre_command=pre_command,
        command_wrapper=command_wrapper,
        runtime_name=None,
        legacy_pykernel=legacy_pykernel,
    )


def main() -> None:
    """Parse CLI arguments, expand commands, and submit jobs."""
    parser = argparse.ArgumentParser(description="Submit jobs.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[m.value for m in ExecutionMode],
        default=ExecutionMode.LOCAL.value,
        help="Execution mode (local, cloud_local, or slurm).",
    )
    parser.add_argument(
        "--runtime",
        type=str,
        default=None,
        help="Runtime configuration name defined under `runtime:` in the YAML.",
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
        help="YAML config file containing execution modes, runtimes, and scripts.",
    )
    parser.add_argument("--partition", type=str, help="SLURM partition")
    parser.add_argument("--nodes", type=int, help="Number of nodes")
    parser.add_argument("--ntasks", type=int, help="Number of tasks")
    parser.add_argument("--cpus-per-task", type=int, help="CPUs per task")
    parser.add_argument("--mem-per-cpu", type=str, help="Memory per CPU (e.g. 4G)")
    parser.add_argument("--gres", type=str, help="Generic resources (e.g. gpu:1)")
    parser.add_argument("--time", type=str, help="Time limit (e.g. 3-00:00:00)")
    parser.add_argument(
        "--slurm_log_dir",
        type=str,
        default=None,
        help="Log directory for slurm job output.",
    )
    parser.add_argument(
        "--list-priors",
        nargs="+",
        metavar="DATA_NAME",
        default=None,
        help="List available priors for given dataset(s) and exit.",
    )
    parser.add_argument(
        "--show-status",
        nargs="+",
        metavar="STATUS_PATH",
        default=None,
        help="Show progress from one or more .status files or directories and exit.",
    )
    parser.add_argument(
        "--skip-prior-validation",
        action="store_true",
        default=False,
        help="Skip prior_name validation against disk when submitting FSP jobs.",
    )

    args, unknown = parser.parse_known_args()

    if args.list_priors is not None:
        _handle_list_priors(args.list_priors)
        return

    if args.show_status is not None:
        _show_status(args.show_status, parser)
        return

    if args.script is None:
        parser.error("--script is required when not using --list-priors or --show-status")

    args.mode = ExecutionMode.from_str(args.mode)
    config_file = args.config_file.resolve()

    try:
        with config_file.open("r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as err:
        parser.error(_format_yaml_load_error(config_file, err))

    try:
        execution_settings = _resolve_execution_settings(
            config,
            config_file,
            args.mode,
            args.runtime,
        )
    except (FileNotFoundError, ValueError) as err:
        parser.error(str(err))

    if args.mode == ExecutionMode.SLURM:
        mode_cfg = config["mode"][args.mode.value]
        cli_slurm_overrides = {
            "partition": args.partition,
            "nodes": args.nodes,
            "ntasks": args.ntasks,
            "cpus_per_task": args.cpus_per_task,
            "mem_per_cpu": args.mem_per_cpu,
            "gres": args.gres,
            "time_limit": args.time,
            "slurm_log_dir": args.slurm_log_dir,
        }
        (
            mode_specific_overrides,
            partition_profile,
            partition_alias,
        ) = _resolve_slurm_resource_settings(
            mode_cfg,
            cli_slurm_overrides,
        )
    else:
        mode_specific_overrides = {}
        partition_profile = None
        partition_alias = None

    script_cfg = config["scripts"][args.script]
    try:
        script_path = _resolve_config_path(script_cfg["path"], config_file)
    except FileNotFoundError as err:
        parser.error(str(err))
    try:
        script_arg_defaults = _extract_script_argument_defaults(script_path)
    except ValueError as err:
        parser.error(str(err))

    default_args = script_cfg.get("default_args", {})
    try:
        arg_specs = _normalize_arg_specs(default_args)
    except ValueError as err:
        parser.error(str(err))

    _apply_cli_overrides(arg_specs, unknown, parser)
    _validate_script_argument_names(script_path, arg_specs, parser)
    submission_batch_size = _pop_submission_batch_size(arg_specs)

    keys, all_combinations = _build_argument_combinations(arg_specs)

    if not args.skip_prior_validation and "prior_name" in keys:
        pre_count = len(all_combinations)
        all_combinations = _validate_prior_combinations(
            all_combinations,
            keys,
            script_arg_defaults=script_arg_defaults,
        )
        skipped = pre_count - len(all_combinations)
        if skipped:
            print(f"Skipped {skipped} combination(s) with invalid priors.\n")
        if not all_combinations:
            print("ERROR: No valid (data_name, prior_name) combinations remain.")
            sys.exit(1)

    logical_jobs = _group_commands_by_iter_flags(all_combinations, keys, arg_specs)
    physical_jobs = _batch_logical_jobs(logical_jobs, submission_batch_size)
    total_commands = len(all_combinations)
    total_logical_jobs = len(logical_jobs)
    total_submissions = len(physical_jobs)

    print(
        "Creating "
        f"{total_submissions} submission(s) from "
        f"{total_logical_jobs} logical job(s) and {total_commands} command(s) "
        f"(batch size: {submission_batch_size}) with the following parameters:"
    )
    if keys:
        for key in keys:
            values = arg_specs[key]["values"]
            iter_suffix = " (iter)" if arg_specs[key]["iter"] else ""
            values_str = ", ".join(str(value) for value in values)
            print(f"  {key}{iter_suffix}: [{values_str}]")
        print()
    else:
        print("  (no parameters specified)\n")

    if args.mode == ExecutionMode.SLURM:
        print(
            "Resolved SLURM resources: "
            f"{_format_slurm_resources_summary(mode_specific_overrides)}"
        )
        if partition_alias is not None:
            print(
                "Resolved partition alias: "
                f"{partition_alias[0]} -> {partition_alias[1]}"
            )
        if partition_profile is not None:
            print(f"Using partition defaults for: {partition_profile}")
        print()
        status_dir = _resolve_status_dir(str(mode_specific_overrides["slurm_log_dir"]))
    else:
        status_dir = _resolve_status_dir("logs")

    for batch_idx, logical_job_batch in enumerate(physical_jobs):
        batch_combos = _flatten_logical_job_batch(logical_job_batch)
        command_suffixes = []
        for combo in batch_combos:
            combo_dict = dict(zip(keys, combo))
            command_suffixes.append(_format_command_suffix(combo_dict))

        if len(logical_job_batch) == 1:
            suffix_desc = _logical_job_name_suffix(
                logical_job_batch[0],
                keys,
                arg_specs,
            )
            if suffix_desc:
                name = f"{args.script}_{suffix_desc}"
            elif total_submissions == 1:
                name = args.script
            else:
                name = f"{args.script}_iter_group_{batch_idx}"
        else:
            name = f"{args.script}_batch_{batch_idx}"

        template_vars = {
            "python_cmd": execution_settings.python_cmd,
            "pre_command": execution_settings.pre_command,
            "command_wrapper": execution_settings.command_wrapper,
            "script_path": str(script_path),
            "working_directory": str(Path.cwd()),
            "job_name": name,
            "status_dir": status_dir,
            "command_suffixes": command_suffixes,
            "runtime_name": execution_settings.runtime_name,
            "slurm_partition_profile": partition_profile,
            "slurm_partition_alias": partition_alias,
            "slurm_resources_summary": (
                _format_slurm_resources_summary(mode_specific_overrides)
                if args.mode == ExecutionMode.SLURM
                else ""
            ),
            "script_args": {},  # backward compatibility for existing templates
            "pykernel": execution_settings.legacy_pykernel,
        }
        template_vars.update(
            {key: value for key, value in mode_specific_overrides.items() if value is not None}
        )

        job_kwargs = {
            "cmd_template": execution_settings.template_path,
            "job_name": name,
            "template_vars": template_vars,
        }

        if args.mode in {ExecutionMode.LOCAL, ExecutionMode.CLOUD_LOCAL}:
            job_kwargs["shell_executable"] = execution_settings.shell_executable

        job = JOB_OPTIONS[args.mode](**job_kwargs)
        job.submit()

    print(f"Submitted {total_submissions} job(s)")


if __name__ == "__main__":
    main()
