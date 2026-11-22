//! # Game-tree search
//!
//! Adversarial game-tree search for computer opponents in board, card and turn-based games:
//! choosing a move when an opponent chooses the next one.
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
//! reference. [`alpha_beta()`], [`pvs()`] and [`mtdf()`] return the same score while skipping
//! moves that cannot change the decision, helped by [`Game::order_moves`] and, where a game
//! allows it, by [`Game::satisfices`], which trades that exactness for speed. Each returns a [`SearchResult`]: the score and its [`Bound`], the
//! principal variation with a score and bound per step, and every root move searched.
//! [`deepen()`] runs alpha-beta one ply deeper at a time until a [`Budget`] of positions or
//! time runs out, trying each iteration's principal variation first in the next; a
//! [`Deepening`] hook can stop it after any iteration or give it aspiration windows.
//! [`Mcts`] grows a tree by random simulations instead, for games too wide or too hard to
//! evaluate for a full search; its selection rule, leaf evaluation and random source are
//! replaceable.
//!
//! [`Game::noisy_moves`] turns on quiescence search: past the horizon, every search goes on
//! through moves such as captures until the position is quiet.
//!
//! A [`Hashed`] game can give its [`Searcher`] a transposition [`Table`], such as
//! [`SlotTable`], with [`Searcher::with_table`]: positions reached by different move orders
//! are then answered from the table instead of being searched again. A
//! [`Searcher`] keeps the working buffers between searches, so repeated searches stop
//! allocating once warm.
//!
//! Repository: <https://github.com/sunsided/game-tree-search>

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

mod alpha_beta;
mod bound;
mod deepening;
mod game;
mod mcts;
mod minimax;
mod mtdf;
mod negamax;
mod pvs;
mod result;
mod score;
mod searcher;
mod table;

pub use alpha_beta::{Window, alpha_beta};
pub use bound::Bound;
pub use deepening::{Deepening, deepen};
pub use game::{Game, Side};
#[cfg(feature = "rand_core")]
pub use mcts::RandCore;
pub use mcts::{
    Evaluation, Evaluator, Mcts, MctsResult, MoveStats, NodeStats, Random, Rewarded, Rollout,
    Selection, SplitMix64, StaticValue, TreeStats, Uct,
};
pub use minimax::minimax;
pub use mtdf::mtdf;
pub use negamax::negamax;
pub use pvs::pvs;
pub use result::{ScoredMove, SearchResult, Stats};
pub use score::{Score, TotalF32, TotalF64};
pub use search_core::Budget;
pub use searcher::{Algorithm, Searcher};
#[cfg(feature = "hashbrown")]
pub use table::MapTable;
pub use table::{Entry, Hashed, Memory, NoTable, SlotTable, Table, TableKey, Transpositions};

/// Doctests for the code examples in README.md (not part of the API).
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}
