# When Is an NBA Game Decided?

Outcome uncertainty inside NBA games, estimated from play-by-play sequences.

Submission to the MIT Sloan Sports Analytics Conference Research Paper Competition.

## What this does

We model the NBA play-by-play stream as a sequence: given the preceding elements and the
state of the game, we estimate the distribution of the next element and of the time until
it occurs. The models are chained into a simulator that generates whole games while
enforcing collective NBA rules. From the simulated games we estimate the probability of
each final outcome (home win, away win, overtime) conditioned on remaining time, score
difference and ball possession, and read those estimates off real games of the following
season.

A game's **decision point** is the first moment at which one outcome reaches 0.95 and stays
there to the end of the fourth quarter.

The model does not distinguish the teams on court: its probabilities describe the league's
average matchup, home advantage aside.

## Data

Play-by-play records of 7,098 NBA games from the 2018/19 through 2023/24 seasons,
3,321,118 events, collected from the NBA's public stats endpoints.

The raw and intermediate files are too large for this repository (the tokenized
play-by-play is ~176 MB; each season of simulated games is ~1 GB). What is included here:

| file | what it is |
|---|---|
| `data/tokens_info.txt` | the event-token vocabulary with frequencies |
| `data/integer_to_token.pkl` | token index used by the models |
| `data/juncoes_bpe.csv` | the merges produced by the adapted Byte-Pair Encoding |
| `data/prop_end_period_by_remaining_time.pkl` | empirical probability that a period has already ended, by period and remaining time |
| `data/resultados_simulacoes.csv` | simulator evaluation metrics, by configuration and season |
| `data/games_nba.csv` | game index: teams, dates, seasons, final scores |

The collection and preprocessing notebooks regenerate the large files from the public
endpoints.

## Repository layout

- `notebooks/00_*` to `01_*` — collection, preprocessing, covariate construction
- `notebooks/01_*` to `05_*train*` — model training
- `notebooks/05_*` to `09_*simulator*` — game simulation
- `notebooks/11_*` to `18_*` — predictive evaluation and consolidation
- `notebooks/19_*` — final simulations
- `notebooks/21b_*` — outcome-probability matrices and decision points
- `notebooks/22_*` — figures
- `figures/` — the two figures submitted with the abstract

## Requirements

Python 3.9+, with `numpy`, `pandas`, `scipy`, `matplotlib`, `pyarrow` and `torch`.

## Status

This repository accompanies a master's dissertation in progress at the Department of
Computer Science, Universidade Federal de Minas Gerais. Code is released as is; the data
are from the NBA's public endpoints and are subject to the NBA's terms of use.
