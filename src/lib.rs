//! # Game-tree search
//!
//! Adversarial search over game trees: choosing a move when an opponent chooses the next one.
//!
//! ## Scope
//!
//! - Minimax and negamax
//! - Alpha-beta pruning with move ordering, principal variation search (PVS/NegaScout) and MTD(f)
//! - Iterative deepening with transposition tables, aspiration windows and quiescence search
//! - Expectimax for games with chance nodes
//! - Monte Carlo tree search with UCT
//!
//! ## Usage
//!
//! A [`Game`] describes positions, their moves, the [`Side`] to move and a [`Score`] from
//! [`Side::Max`]'s point of view. The side to move belongs to the position, so one side may
//! move several times in a row.
//!
//! [`minimax()`] and [`negamax()`] search every line to a fixed depth and serve as the
//! reference. [`alpha_beta()`] returns the same score while skipping moves that cannot change
//! the decision, helped by [`Game::order_moves`] and, where a game allows it, by
//! [`Game::satisfices`]. Each returns a [`SearchResult`]: the score and its [`Bound`], the
//! principal variation and every root move searched.
//!
//! Repository: <https://github.com/sunsided/game-tree-search>

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

mod alpha_beta;
mod bound;
mod game;
mod minimax;
mod negamax;
mod result;
mod score;

pub use alpha_beta::alpha_beta;
pub use bound::Bound;
pub use game::{Game, Side};
pub use minimax::minimax;
pub use negamax::negamax;
pub use result::{RootMove, SearchResult, Stats};
pub use score::{Score, TotalF32, TotalF64};

/// Doctests for the code examples in README.md (not part of the API).
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}
