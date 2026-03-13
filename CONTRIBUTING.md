# Contributing

Thank you for contributing to `submit`.

The project should stay lightweight, explicit, and easy to reason about. New
features should improve the submission workflow without making the core logic
opaque.

## Principles

- keep the core small
- prefer explicit configuration over hidden behavior
- separate planning logic from execution logic
- make new behavior testable before expanding it
- keep project-specific extensions isolated where possible

## Development priorities

Contributions are especially welcome in these areas:

- clearer config schema
- better test coverage
- improved template consistency
- cleaner runtime abstraction
- batching and `iter` semantics
- better docs

## Recommended architecture

When extending the code, prefer separating concerns into modules such as:

- config loading and normalization
- planning / combination expansion
- template rendering
- executors (`LocalJob`, `SlurmJob`)
- CLI wiring

This makes testing much easier than pushing all logic through `main()`.

## Tests

### Fast local tests

Use pytest for pure logic tests such as:

- argument normalization
- Cartesian product expansion
- `iter: true` grouping
- template rendering
- prior validation and auto-discovery

### Fake-SLURM tests with Docker

The repository supports a fake-`sbatch` test path for Linux-based Docker
testing. This should be the default integration test path for development.

Run:

```bash
docker compose -f docker-compose.test.yml build
docker compose -f docker-compose.test.yml run --rm submit-tests
````

These tests should cover:

* batch script rendering
* `sbatch` invocation
* CLI-level planning
* current batching behavior

### Real SLURM tests

For more realistic end-to-end tests, use a Dockerized SLURM cluster. This
should be reserved for higher-level validation and not for every small change.

## Config conventions

Use the nested form for `iter`:

```yaml
param1:
  values: ["a", "b", "c"]
  iter: true
```

Avoid top-level `iter: true` flags that are detached from a specific argument.

If shorthand forms are supported for backward compatibility, document them
clearly and test them explicitly.

## Runtime conventions

Keep runtime concerns separate from execution mode.

Examples of runtimes:

* `.venv`
* Conda
* Singularity
* plain Python

Examples of modes:

* `local`
* `cloud_local`
* `slurm`

## Documentation

When updating docs:

* keep `README.md` general
* put setup instructions into `SETUP.md`
* keep this file focused on contribution and development workflow
* place project-specific guidance into `OSP.md`

## Pull requests

A good pull request should:

* explain the motivation
* describe the behavioral change
* include or update tests
* update docs if config or usage changed

## Style

* prefer clear names over clever names
* avoid mixing environment logic into unrelated code paths
* keep templates consistent across modes
* use docstrings for functions that encode important planning behavior