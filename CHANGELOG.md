# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2](https://github.com/sunsided/game-tree-search/compare/v0.2.1...v0.2.2) - 2026-11-23

### Changed

- Say what a search asks at a noisy horizon ([#49](https://github.com/sunsided/game-tree-search/pull/49))
- Pack shared table entries into atomic words for lock-free Lazy SMP ([#48](https://github.com/sunsided/game-tree-search/pull/48))
- Say what a search asks at a noisy horizon
- Let Narrow fill the search's move buffer and forward noisy moves
- Model-check the packed table's races with loom and run Miri on the tables
- Say where else a search evaluates a position
- Pack shared table entries into atomic words for lock-free Lazy SMP
- Keep the OnceLock mention unlinked for no_std docs
- Show how a position caches costly derived state
- Skew the helpers' quiescence moves as well
- Stop every Lazy SMP thread on an answer and ignore helpers cut at once
- Make Lazy SMP helpers diverge and let the furthest thread answer
- Give every seed of the tabled bench its own table and searcher
- Benchmark the searches and property-test them on generated trees

## [0.2.1](https://github.com/sunsided/game-tree-search/compare/v0.2.0...v0.2.1) - 2026-11-22

### Added

- Lazy SMP: `deepen_shared` runs several deepening threads over one `SharedTable`, a
  transposition table in locked shards (feature `parallel`)
- Batched leaf evaluation for MCTS: `Mcts::search_batched` with a `BatchEvaluator`, virtual
  loss for waiting leaves, and `Unbatched` to use any `Evaluator` that way
- Self-play support: `MctsResult::policy` and `MctsResult::value` as training targets,
  `MctsResult::sample` to choose a move by temperature, and Dirichlet root noise with
  `Mcts::with_root_noise`

## [0.2.0](https://github.com/sunsided/game-tree-search/compare/v0.1.0...v0.2.0) - 2026-11-22

### Added

- Root-parallel Monte Carlo tree search: `Mcts::search_parallel`, behind the new `parallel`
  feature, the only feature that needs `std`
- PUCT selection (`Puct`) with move priors from `Evaluator::priors`, and
  `Selection::unvisited_first` for rules that score unvisited children
- Tree reuse between moves: `Mcts::advance`; a kept tree is continued only for a position
  with the same side to move and the same moves
- Documentation and tests for running many searches at once, one searcher per thread or agent

### Changed

- Principal variation search searches the first move with the full window and scouts only
  later moves, so it searches fewer positions
- MTD(f) stops as soon as a bound reaches the end of the score range, instead of running a
  full-window search, which helps games that score wins and losses at the integer extremes

## [0.1.0](https://github.com/sunsided/game-tree-search/releases/tag/v0.1.0) - 2026-11-22

### Added

- Add dormant release automation with release-plz
- Add CONTRIBUTING.md
- Add git hooks and Copilot review instructions
- Add README badges, a usage example and package excludes
- Add REUSE licensing metadata
- Add supply-chain checks
- Add Codecov configuration
- Add Taskfile and CI workflows

### Changed

- Record the hand-set first version in the release notes
- Prepare the 0.1.0 release
- Value a newly reached finished game by its reward in MCTS
- Let MCTS draw from any rand_core generator behind a feature
- Average weighted scores without overflow
- Grow game trees by simulation with Monte Carlo tree search
- Warn that only expectimax reads chance outcomes
- Value chance positions by their expectation with expectimax
- Report a depth-0 root's noisy moves and correct the ordering docs
- Search noisy moves past the horizon with quiescence search
- Point hidden-card sampling at constraint-search in the README ([#16](https://github.com/sunsided/game-tree-search/pull/16))
- Aspirate deepening iterations through the Deepening hook ([#14](https://github.com/sunsided/game-tree-search/pull/14))
- Point hidden-card sampling at constraint-search in the README
- Qualify the exactness claims for games that satisfice
- Point puzzles at constraint-search in the README
- Lead the crate description with computer opponents
- Aspirate deepening iterations through the Deepening hook
- Close in on the root score with MTD(f)
- Scout later moves with principal variation search
- Run the featureless clippy pass in task lint:clippy too
- Offer a hash-map transposition table behind a hashbrown feature
- Remember searched positions in a transposition table
- Check the budget before asking the deepening hook
- Let a hook stop deepening and report the deepest ply
- Name search-core in the README and test that the previous line helps
- Deepen alpha-beta iteratively within a search-core budget
- Cover Searcher::default and leaf expansion in tests
- Score every principal-variation step and reuse search buffers
- Require Clone on Game::Move and refine the satisfices docs
- Prune game trees with fail-soft alpha-beta
- Negate integer scores by bitwise NOT
- Search game trees with minimax and negamax
- Test the Game trait through a minimal Nim implementation
- Tighten lints and forbid unsafe code
- Render mermaid diagrams on docs.rs and doctest the README
- Reserve the crate name with a 0.0.0 placeholder

### Dependencies

- Bump taiki-e/install-action to v2.90.8

### Fixed

- Repair the MCTS test file after the merge

### Removed

- Drop the now unneeded expansion check in MCTS selection
- Drop the unused Clone bound on Game::Move
