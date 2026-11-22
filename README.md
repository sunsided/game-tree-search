# game-tree-search

[![crates.io](https://img.shields.io/crates/v/game-tree-search.svg)](https://crates.io/crates/game-tree-search)
[![docs.rs](https://img.shields.io/docsrs/game-tree-search)](https://docs.rs/game-tree-search)
[![CI](https://github.com/sunsided/game-tree-search/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/game-tree-search/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/game-tree-search/graph/badge.svg)](https://codecov.io/gh/sunsided/game-tree-search)
[![MSRV](https://img.shields.io/crates/msrv/game-tree-search)](https://crates.io/crates/game-tree-search)
[![License](https://img.shields.io/crates/l/game-tree-search)](https://github.com/sunsided/game-tree-search#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Adversarial game-tree search in Rust: choosing a move when an opponent chooses the next one.

## Scope

- Minimax and negamax
- Alpha-beta pruning with move ordering, principal variation search (PVS/NegaScout) and MTD(f)
- Iterative deepening with transposition tables, aspiration windows and quiescence search
- Expectimax for games with chance nodes
- Monte Carlo tree search with UCT

Single-agent search, including iterative deepening (IDDFS, IDA*), lives in [heuristic-search](https://github.com/sunsided/heuristic-search).

## Usage

```sh
cargo add game-tree-search
```

A game describes its positions, the moves available in them, which side is to move and how
good a position is for the maximizing side. The side to move belongs to the position, so one
side may move several times in a row. Here, one-pile Nim: players take one or two stones,
and whoever cannot move has lost.

```rust
use game_tree_search::{Game, Side, minimax};

/// One-pile Nim; a position is the number of stones left and the side to move.
struct Nim;

impl Game for Nim {
    type Position = (u32, Side);
    type Move = u32;
    type Score = i32;

    fn side_to_move(&self, &(_, side): &Self::Position) -> Side {
        side
    }

    fn moves(&self, &(stones, _): &Self::Position, moves: &mut Vec<u32>) {
        moves.extend((1..=2).filter(|&take| take <= stones));
    }

    fn play(&self, &(stones, side): &Self::Position, take: &u32) -> Self::Position {
        let next = if side == Side::Max { Side::Min } else { Side::Max };
        (stones - take, next)
    }

    fn evaluate(&self, &(stones, side): &Self::Position) -> i32 {
        // With no stones left, the side to move has lost.
        match (stones, side) {
            (0, Side::Max) => -1,
            (0, Side::Min) => 1,
            _ => 0,
        }
    }
}

// Four stones: taking one leaves three, a lost pile for the opponent.
let result = minimax(&Nim, &(4, Side::Max), 10);
assert_eq!(result.score, 1);
assert_eq!(result.best(), Some(&1));
```

The crate is `no_std` and needs `alloc`. It has no dependencies.

## Minimum supported Rust version

Rust 1.85 (edition 2024). The MSRV applies to the library; tests may need a newer
toolchain. It may be raised in a minor release when a dependency or a language feature
calls for it.

## Contributing

See [CONTRIBUTING.md](https://github.com/sunsided/game-tree-search/blob/main/CONTRIBUTING.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/game-tree-search/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/game-tree-search/blob/main/LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.
