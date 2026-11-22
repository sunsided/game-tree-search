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

A game describes its positions and moves, when a position ends the game, and how good a
position is for the player to move. Here, one-pile Nim: players take stones in turn, and
whoever cannot move has lost.

```rust
use game_tree_search::Game;

/// One-pile Nim; a position is the number of stones left.
struct Nim;

impl Game for Nim {
    type Position = u32;
    type Move = u32;

    fn is_terminal(&self, stones: &u32) -> bool {
        *stones == 0
    }

    fn evaluate(&self, stones: &u32) -> i32 {
        // With no stones left, the player to move has lost.
        if *stones == 0 { -1 } else { 0 }
    }
}

assert!(Nim.is_terminal(&0));
assert_eq!(Nim.evaluate(&0), -1);
```

The crate is `no_std` and has no dependencies.

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
