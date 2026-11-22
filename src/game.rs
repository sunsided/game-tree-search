//! The [`Game`] trait the searches run on, and the [`Side`] that moves in a position.

use alloc::vec::Vec;

use crate::Score;

/// The player who chooses the move in a position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    /// Picks the move with the highest [`Score`].
    Max,
    /// Picks the move with the lowest [`Score`].
    Min,
}

/// A two-player, zero-sum game as seen by the searches.
///
/// The side to move is a property of the position, not of the move count: one side may move
/// several times in a row, as in games where combatants act by speed. Every score is from
/// [`Side::Max`]'s point of view, whoever is to move.
///
/// A position without moves ends the game; [`Game::evaluate`] then gives its final score.
///
/// # Examples
///
/// ```
/// use game_tree_search::{Game, Side, minimax};
///
/// /// One-pile Nim: take one or two stones; whoever cannot move has lost.
/// struct Nim;
///
/// impl Game for Nim {
///     /// The stones left and the side to move.
///     type Position = (u32, Side);
///     type Move = u32;
///     type Score = i32;
///
///     fn side_to_move(&self, &(_, side): &Self::Position) -> Side {
///         side
///     }
///
///     fn moves(&self, &(stones, _): &Self::Position, moves: &mut Vec<u32>) {
///         moves.extend((1..=2).filter(|&take| take <= stones));
///     }
///
///     fn play(&self, &(stones, side): &Self::Position, take: &u32) -> Self::Position {
///         let next = if side == Side::Max { Side::Min } else { Side::Max };
///         (stones - take, next)
///     }
///
///     fn evaluate(&self, &(stones, side): &Self::Position) -> i32 {
///         // With no stones left, the side to move has lost.
///         match (stones, side) {
///             (0, Side::Max) => -1,
///             (0, Side::Min) => 1,
///             _ => 0,
///         }
///     }
/// }
///
/// // Four stones: taking one leaves a multiple of three, a lost pile for the opponent.
/// let result = minimax(&Nim, &(4, Side::Max), 10);
/// assert_eq!(result.score, 1);
/// assert_eq!(result.best(), Some(&1));
/// ```
pub trait Game {
    /// A game position, including whatever decides the side to move.
    type Position;
    /// A move that leads from one position to the next.
    type Move: Clone;
    /// The value of a position, from [`Side::Max`]'s point of view.
    type Score: Score;

    /// The side that chooses the move in `position`.
    ///
    /// Only asked for positions that have moves.
    fn side_to_move(&self, position: &Self::Position) -> Side;

    /// Appends the legal moves of `position` to `moves`, which arrives empty.
    ///
    /// Appending none ends the game at `position`. The order is the order the searches try
    /// the moves in, and the first of equally good moves wins.
    fn moves(&self, position: &Self::Position, moves: &mut Vec<Self::Move>);

    /// The position reached by playing `mv`, one of the [`Game::moves`] of `position`.
    fn play(&self, position: &Self::Position, mv: &Self::Move) -> Self::Position;

    /// The score of `position` from [`Side::Max`]'s point of view.
    ///
    /// Asked for positions that end the game, where it is the final score, and for positions
    /// where the search runs out of depth, where it is a heuristic estimate.
    fn evaluate(&self, position: &Self::Position) -> Self::Score;
}

/// The moves a search expands at `position` with `depth` plies left: none at the horizon
/// (`depth == 0`) or where the game ends, else the [`Game::moves`].
pub(crate) fn expanded_moves<G: Game>(
    game: &G,
    position: &G::Position,
    depth: u32,
) -> Vec<G::Move> {
    let mut moves = Vec::new();
    if depth > 0 {
        game.moves(position, &mut moves);
    }

    moves
}

/// `score`, given from [`Side::Max`]'s point of view, from `side`'s point of view.
pub(crate) fn oriented<S: Score>(score: S, side: Side) -> S {
    match side {
        Side::Max => score,
        Side::Min => score.negate(),
    }
}
