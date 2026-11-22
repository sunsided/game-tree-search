//! The [`Game`] trait the searches run on, and the [`Side`] that moves in a position.

use alloc::vec::Vec;

use crate::{Score, Stats};

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
    ///
    /// Cloned where a search keeps a move apart from its principal variation, such as in the
    /// root move report, and compared to find a remembered move, such as the previous
    /// iteration's principal variation, among the legal ones. Moves are usually small values
    /// that are cheap to copy and compare.
    type Move: Clone + PartialEq;
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

    /// Reorders the [`Game::moves`] of `position`, most promising first.
    ///
    /// Pruning searches such as [`alpha_beta()`](crate::alpha_beta()) cut more of the tree
    /// when the best move comes first; the reference searches [`minimax()`](crate::minimax())
    /// and [`negamax()`](crate::negamax()) do not call it. The default keeps the order.
    fn order_moves(&self, position: &Self::Position, moves: &mut [Self::Move]) {
        let _ = (position, moves);
    }

    /// Whether `side`, having found a move worth `score`, stops looking for a better one.
    ///
    /// Lets a game prune by its own knowledge, such as "any line that wins is good enough".
    /// Pruning searches such as [`alpha_beta()`](crate::alpha_beta()) ask it before each move
    /// after the first, with the best score found so far at that position; on `true` they skip
    /// the remaining moves. A position whose moves were all searched is never affected. The
    /// reference searches [`minimax()`](crate::minimax()) and [`negamax()`](crate::negamax())
    /// never ask.
    ///
    /// The default returns `false`, which keeps every search exact: with a full window, the
    /// root score is the minimax value and its [`Bound`](crate::Bound) is
    /// [`Exact`](crate::Bound::Exact).
    ///
    /// Returning `true` anywhere weakens that guarantee:
    ///
    /// - A satisfied position is bounded: its value is at least its score when [`Side::Max`]
    ///   moves, at most when [`Side::Min`] moves. If the move that set the score is itself only
    ///   bounded the other way, the position is [`Bound::Approximate`](crate::Bound::Approximate)
    ///   instead, which says nothing about the value.
    /// - These bounds propagate up: wherever a lower and an upper bound meet on one line, the
    ///   result is [`Bound::Approximate`](crate::Bound::Approximate). Satisficing on both sides
    ///   makes that common.
    /// - The root score can differ from the minimax value, and the chosen move is the best by
    ///   the reported scores, not necessarily by minimax.
    ///
    /// What still holds: every reported score admits the minimax value under its bound
    /// ([`Bound::admits`](crate::Bound::admits)), and the principal variation replays to the
    /// root score (unless it ends early at a transposition table entry).
    fn satisfices(&self, side: Side, score: Self::Score) -> bool {
        let _ = (side, score);
        false
    }
}

/// The moves a search expands at `position` with `depth` plies left: none at the horizon
/// (`depth == 0`), counted in `stats`, or where the game ends, else the [`Game::moves`].
pub(crate) fn expanded_moves<G: Game>(
    game: &G,
    position: &G::Position,
    depth: u32,
    stats: &mut Stats,
) -> Vec<G::Move> {
    let mut moves = Vec::new();
    if depth == 0 {
        stats.horizon += 1;
    } else {
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
