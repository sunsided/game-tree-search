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

/// Unit tests for the `Game` trait, through a minimal implementation.
#[cfg(test)]
mod tests {
    use super::Game;

    /// One-pile Nim: players take one or two stones in turn, and whoever cannot move has lost.
    struct Nim;

    impl Game for Nim {
        type Position = u32;
        type Move = u32;

        fn is_terminal(&self, stones: &u32) -> bool {
            *stones == 0
        }

        fn evaluate(&self, stones: &u32) -> i32 {
            if *stones == 0 { -1 } else { 0 }
        }
    }

    /// Scores `position` the way a search sees it: through the trait bound only.
    fn score<G: Game>(game: &G, position: &G::Position) -> Option<i32> {
        game.is_terminal(position).then(|| game.evaluate(position))
    }

    /// A terminal position is scored from the point of view of the player to move, who has lost.
    #[test]
    fn terminal_position_is_lost_for_the_player_to_move() {
        assert_eq!(score(&Nim, &0), Some(-1));
    }

    /// A position with stones left is not terminal, so a search has to expand it.
    #[test]
    fn position_with_stones_left_is_not_terminal() {
        assert_eq!(score(&Nim, &3), None);
    }
}
