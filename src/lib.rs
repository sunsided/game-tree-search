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
//! Repository: <https://github.com/sunsided/game-tree-search>

#![no_std]
#![forbid(unsafe_code)]

/// A two-player, zero-sum game as seen by the searches.
///
/// Implementations describe positions, the moves available in them and how to score
/// positions from the point of view of the player to move.
pub trait Game {
    /// A game position.
    type Position;
    /// A move that leads from one position to the next.
    type Move;

    /// Whether `position` ends the game.
    fn is_terminal(&self, position: &Self::Position) -> bool;

    /// The score of `position` for the player to move; higher is better for that player.
    fn evaluate(&self, position: &Self::Position) -> i32;
}

/// Doctests for the code examples in README.md (not part of the API).
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}
