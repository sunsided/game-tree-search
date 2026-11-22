# game-tree-search

Adversarial game-tree search in Rust: choosing a move when an opponent chooses the next one.

## Scope

- Minimax and negamax
- Alpha-beta pruning with move ordering, principal variation search (PVS/NegaScout) and MTD(f)
- Iterative deepening with transposition tables, aspiration windows and quiescence search
- Expectimax for games with chance nodes
- Monte Carlo tree search with UCT

Single-agent search, including iterative deepening (IDDFS, IDA*), lives in [heuristic-search](https://github.com/sunsided/heuristic-search).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/game-tree-search/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/game-tree-search/blob/main/LICENSE-MIT))

at your option.
