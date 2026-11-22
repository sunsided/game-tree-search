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
//! principal variation with a score and bound per step, and every root move searched. A
//! [`Searcher`] keeps the working buffers between searches, so repeated searches stop
//! allocating once warm.
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
mod searcher;

pub use alpha_beta::alpha_beta;
pub use bound::Bound;
pub use game::{Game, Side};
pub use minimax::minimax;
pub use negamax::negamax;
pub use result::{ScoredMove, SearchResult, Stats};
pub use score::{Score, TotalF32, TotalF64};
pub use searcher::Searcher;

/// Doctests for the code examples in README.md (not part of the API).
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}
