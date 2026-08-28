# Worst-Shinka


                ██╗   ██╗    ██╗  ██████╗  ██████╗   ███████╗ ████████╗
                ██║   ██║    ██║ ██╔═══██╗ ██╔══██╗  ██╔════╝ ╚══██╔══╝
                ██║   ██║    ██║ ██║   ██║ ██████╔╝  ███████╗    ██║
                ██║   ██║    ██║ ██║   ██║ ██╔══██╗  ╚════██║    ██║
                ╚██████╔██████╔╝ ╚██████╔╝ ██║  ██║  ███████║    ██║
                 ╚═════╝ ╚════╝   ╚═════╝  ╚═╝  ╚═╝  ╚══════╝    ╚═╝
                                    S H I N K A



Worst-Shinka is a customized system for evolving reinforcement-learning agents. It combines an evolution-loop orchestrator, language-model selection through OpenRouter, and an Atari Tennis training and evaluation pipeline.

The system starts from a standardized initial model, creates and evaluates candidate generations, records their results, and exposes the resulting run data for visualization, gameplay, and reset operations. Model selection is filtered by OpenRouter metadata, context size, and a configured cost limit before a run starts.


## How the system works

1. `worst-shinka run` loads the initial model and validates the configured OpenRouter models.
2. OpenRouter is queried for API-key information and the user's model catalog.
3. Models are filtered by price and context profile (`low`, `medium`, or `high`), then a pool of five models is selected.
4. The initial generation is prepared in the run directory.
5. For each requested generation, the integration layer selects parents, asks the evolution adapter for proposals, trains candidates, runs an Atari Tennis tournament, aggregates metrics, and judges the candidates.
6. Run metadata, model selections, metrics, solutions, lineage, tournament data, and score history are saved under the run directory.
7. `visualize`, `play`, and `reset` operate on a selected run after it has been created.

The reinforcement-learning integration currently reuses the scripts in `RL_training/`. The command-line package supplies the orchestration and persistent run structure around that training code.

## Repository structure

```text
worst-shinka/
├── initial_model/                  Standardized zero-generation model
│   ├── algorithm.py
│   └── config.yaml
├── RL_training/                   Reinforcement-learning pipeline
│   ├── config.yaml                RL model configuration
│   ├── model.py                   DQN model definition and training step
│   ├── train.py                   Model training loop
│   ├── run_training.py            Training entry point for one generation
│   ├── run_tournament.py          Model-vs-model tournament and Elo updates
│   ├── run_aggregate_data.py      Collect generation and run metrics
│   ├── run_play.py                Standalone Atari Tennis gameplay
│   ├── run_reset.py               Standalone training reset utility
│   ├── utils.py                   Shared paths, model I/O, scores, and logging
│   ├── loop_test.py               End-to-end RL pipeline smoke test
│   └── notes.txt                  Development notes
├── worst_shinka/
│   ├── cli/
│   │   ├── cli.py                 Command-line parser and dispatch
│   │   ├── config.py              Run configuration and model profiles
│   │   ├── orchestrator.py         Evolution-loop orchestration
│   │   ├── integrations.py        Adapter for RL_training and run artifacts
│   │   ├── play.py                Game process management
│   │   ├── visualization.py       Evolution-tree visualization
│   │   ├── settings.py            OpenRouter credential management
│   │   └── default_models.json    Default OpenRouter model pool
│   ├── brainstorming_system/      LLM-based proposal generation
│   │   ├── brainstorm.py          Generate algorithm proposals from parents
│   │   ├── evaluation_adapter.py  Connect proposals to candidate evaluation
│   │   └── judge_adapter.py       Connect evaluated candidates to the judge
│   ├── judge/                     Candidate comparison and winner history
│   │   ├── judge.py               Evaluate and select winning candidates
│   │   └── history/               Persistent judge history
│   │       └── winners.jsonl      One JSON record per selected winner
│   ├── parent_selector/           Parent selection for new generations
│   │   └── parents_selector.py    Select parents from previous candidates
│   └── llm/
│       ├── connect.py             OpenRouter connection and model validation
│       ├── client.py               LLM request client
│       ├── pool_reader.py          Read and normalize the model pool
│       └── selector.py             Select models for brainstorming
├── global_config.yaml              Shared training and run settings
├── pyproject.toml                 Package metadata and dependencies
└── README.md                      Project documentation
```

### Reinforcement-learning pipeline

The files in `RL_training/` implement the executable Atari Tennis training layer. The pipeline uses PettingZoo's RAM-based Tennis environment and PyTorch models.

- `config.yaml` contains the neural-network configuration used by the RL model.
- `model.py` defines `DQLModel`, including the network, optimizer, prediction, training step, and model serialization.
- `train.py` runs episodes, selects actions through the algorithm module, trains the model, logs rewards, and records score history.
- `run_training.py` prepares a generation directory, copies its configuration and algorithm, and starts training for that generation.
- `run_tournament.py` evaluates the current model against every previous generation, writes the tournament matrix, and updates Elo scores.
- `run_aggregate_data.py` reads the generation source files, training logs, tournament table, and score history into one aggregate result.
- `run_play.py` launches a standalone Atari Tennis match between configured generation models, or between a model and random actions.
- `run_reset.py` removes generated generations and training artifacts while preserving the generation-zero source files.
- `utils.py` centralizes result paths, model loading and saving, score-history handling, Atari constants, and progress logging.
- `loop_test.py` runs training, tournament, and aggregation across several generations as an end-to-end pipeline check.
- `notes.txt` contains development notes for the RL implementation.

### Brainstorming and evolution components

The `feature/brainstorming` branch adds the proposal-to-judgement workflow around the RL pipeline:

- `brainstorming_system/brainstorm.py` asks selected language models to propose changes to an existing algorithm and produces candidate source files.
- `brainstorming_system/evaluation_adapter.py` converts brainstorm proposals into the standardized inputs expected by RL training.
- `brainstorming_system/judge_adapter.py` converts evaluation results into the input format consumed by the candidate judge.
- `parent_selector/parents_selector.py` chooses parent candidates from the available run history for the next evolution generation.
- `judge/judge.py` compares evaluated candidates, applies the acceptance criteria, and records selected winners.
- `judge/history/winners.jsonl` stores the judge's winner history in append-only JSON Lines format.

### LLM integration

The `llm/` package isolates communication with OpenRouter and model-pool management:

- `connect.py` authenticates with OpenRouter, reads model metadata, validates pricing and context limits, and selects compatible models by mode.
- `client.py` provides the request-level LLM client used by higher-level brainstorming components.
- `pool_reader.py` loads and normalizes model-pool definitions before selection.
- `selector.py` chooses the models used for a brainstorming or evolution step.

## Setup

The project requires Python 3.10 or newer. Create and activate a virtual environment, then install the project in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
AutoROM --accept-license
```

Editable installation means that changes to the local Python package are immediately available without reinstalling it. `AutoROM --accept-license` downloads and installs the Atari ROMs required by the PettingZoo Atari environment.

Configure an OpenRouter API key before running an evolution:

```bash
worst-shinka config login
```

The key is stored in the user configuration directory, normally `~/.config/worst-shinka/.env`. An existing `OPENROUTER_API_KEY` environment variable takes precedence over the saved key.

Check the current credential status with:

```bash
worst-shinka config status
```

## Command reference

### `run`

Run the evolution loop.

```bash
worst-shinka run [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--models MODEL [MODEL ...]` | System model list | OpenRouter model identifiers to validate and use. At least two are required when provided; a run ultimately selects five valid models. |
| `--mode {low,medium,high}` | `medium` | Context-size profile used when selecting models. `low` accepts 8,192-262,144 tokens, `medium` accepts 131,072-400,000 tokens, and `high` accepts at least 131,072 tokens with no upper limit. |
| `--results-dir PATH` | `results` | Root directory where run directories are created. |
| `--name NAME` | Timestamp | Explicit run directory name. It must be a directory name, not a path. |
| `--initial-model PATH` | `initial_model` | Directory containing the standardized initial model files. |
| `--generations N` | `5` | Number of generations to execute. Must be greater than zero. |
| `--workers N` | `1` | Maximum number of workers for candidate evaluation. |
| `--parents N` | `4` | Number of parent candidates requested for each generation. |

Examples:

```bash
worst-shinka run
```
```bash
worst-shinka run --name experiment-01 --generations 10 --workers 4
```
```bash
worst-shinka run --mode high --models anthropic/claude-sonnet-4 google/gemini-2.5-pro qwen/qwen3-coder moonshotai/kimi-k2.6 amazon/nova-pro-v1
```

Each run is stored in `results/<run-name>/`. Existing run directories are continued from their latest generation when the same `--name` is used.

### `config`

Manage the OpenRouter API key and display configuration status.

```bash
worst-shinka config login
worst-shinka config status
worst-shinka config logout
```

- `login` securely prompts for and saves an API key.
- `status` shows whether a key is available, its masked value, and its source.
- `logout` removes the locally saved credentials. It does not unset `OPENROUTER_API_KEY` from the current shell.

### `play`

Launch Atari Tennis using a trained model.

```bash
worst-shinka play --model-path PATH [--opponent-path PATH]
```

`--model-path` is required and must point to a model file such as `results/run-name/gen_3/model.pt`. Without `--opponent-path`, a human plays against the selected model. With it, two models from the same run play against each other. Press `Q` to stop the game.

### `visualize`

Generate an HTML evolution-tree visualization for a run.

```bash
worst-shinka visualize --run-path PATH [--save-to PATH] [--no-open]
```

- `--run-path` is required and points to a run directory.
- `--save-to` saves a persistent HTML file at the selected path. Without it, a temporary file is generated.
- `--no-open` generates the HTML without opening a browser.

### `reset`

Reset training progress for a selected run.

```bash
worst-shinka reset --run-path PATH
```

The command delegates to the RL reset pipeline and removes the run manifest and generation-zero metadata that are recreated during the next run.

## Run artifacts

A typical run directory contains:

```text
results/<run-name>/
├── run.json                       Run configuration and lifecycle status
├── models.json                    Selected OpenRouter model pool
├── model-validation.json          OpenRouter validation results
├── lineage.json                   Complete evolution lineage
├── tournament_table.csv           Tournament results
├── model_score_history.csv        Aggregated scores and Elo values
└── gen_N/
	├── config.yaml                Training configuration
	├── algorithm.py               Candidate algorithm
	├── model.pt                   Trained model
	├── metrics.json               Generation metrics
	├── solutions.json             Candidate and acceptance data
	└── lineage.json               Lineage snapshot for the generation
```

## Help

Run the installed CLI from the repository for help.

```bash
python -m worst_shinka.cli --help
worst-shinka --help
```

The package uses `setuptools` via `pyproject.toml`.