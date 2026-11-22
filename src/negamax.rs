//! Negamax: minimax written for one side, negating scores where the side to move changes.

use alloc::vec::Vec;

use crate::game::{expanded_moves, oriented};
use crate::{Game, Score, SearchResult, Side, Stats};

/// Searches every line from `position` up to `depth` plies, in negamax form.
///
/// Each position is scored from the point of view of the side that moved into it, and every
/// position picks its highest child. Unlike textbook negamax, a child's score is negated
/// only where the side to move changes, not at every ply, so games where one side moves
/// several times in a row are searched correctly. Returns the same score and principal
/// variation as [`minimax()`](crate::minimax()).
///
/// # Examples
///
/// ```
/// # use game_tree_search::{Game, Side};
/// # struct Nim;
/// # impl Game for Nim {
/// #     type Position = (u32, Side);
/// #     type Move = u32;
/// #     type Score = i32;
/// #     fn side_to_move(&self, &(_, side): &Self::Position) -> Side { side }
/// #     fn moves(&self, &(stones, _): &Self::Position, moves: &mut Vec<u32>) {
/// #         moves.extend((1..=2).filter(|&take| take <= stones));
/// #     }
/// #     fn play(&self, &(stones, side): &Self::Position, take: &u32) -> Self::Position {
/// #         (stones - take, if side == Side::Max { Side::Min } else { Side::Max })
/// #     }
/// #     fn evaluate(&self, &(stones, side): &Self::Position) -> i32 {
/// #         match (stones, side) { (0, Side::Max) => -1, (0, Side::Min) => 1, _ => 0 }
/// #     }
/// # }
/// use game_tree_search::{minimax, negamax};
///
/// let root = (7, Side::Max);
/// assert_eq!(negamax(&Nim, &root, 10), minimax(&Nim, &root, 10));
/// ```
#[must_use]
pub fn negamax<G: Game>(
    game: &G,
    position: &G::Position,
    depth: u32,
) -> SearchResult<G::Move, G::Score> {
    let mut stats = Stats::default();
    let (score, mut pv) = search(game, position, depth, Side::Max, &mut stats);
    pv.reverse();

    SearchResult { score, pv, stats }
}

/// The score of `position` with `depth` plies left from `viewer`'s point of view, and its
/// principal variation in reverse (last move first).
fn search<G: Game>(
    game: &G,
    position: &G::Position,
    depth: u32,
    viewer: Side,
    stats: &mut Stats,
) -> (G::Score, Vec<G::Move>) {
    stats.nodes += 1;
    let moves = expanded_moves(game, position, depth);
    if moves.is_empty() {
        return (oriented(game.evaluate(position), viewer), Vec::new());
    }

    let to_move = game.side_to_move(position);
    let mut best: Option<(G::Score, Vec<G::Move>)> = None;
    for mv in moves {
        let (score, mut line) = search(game, &game.play(position, &mv), depth - 1, to_move, stats);
        if best.as_ref().is_none_or(|&(best, _)| score > best) {
            line.push(mv);
            best = Some((score, line));
        }
    }

    let (score, line) = best.expect("a position with moves has a best child");
    let score = if to_move == viewer {
        score
    } else {
        score.negate()
    };

    (score, line)
}
