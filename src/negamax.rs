//! Negamax: minimax written for one side, negating scores where the side to move changes.

use alloc::vec::Vec;

use crate::game::{expanded_moves, oriented};
use crate::minimax::exact_root_move;
use crate::{Bound, Game, RootMove, Score, SearchResult, Side, Stats};

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
) -> SearchResult<G::Move, G::Score>
where
    G::Move: Clone,
{
    let mut stats = Stats::default();
    let mut root_moves = Vec::new();
    let (score, mut pv) = search(
        game,
        position,
        depth,
        Side::Max,
        Some(&mut root_moves),
        &mut stats,
    );
    pv.reverse();

    SearchResult {
        score,
        bound: Bound::Exact,
        pv,
        root_moves,
        unexamined: 0,
        stats,
    }
}

/// The score of `position` with `depth` plies left from `viewer`'s point of view, and its
/// principal variation in reverse (last move first).
///
/// Records every child of the root in `root_moves`, given only at the root, with scores
/// from [`Side::Max`]'s point of view.
fn search<G: Game>(
    game: &G,
    position: &G::Position,
    depth: u32,
    viewer: Side,
    mut root_moves: Option<&mut Vec<RootMove<G::Move, G::Score>>>,
    stats: &mut Stats,
) -> (G::Score, Vec<G::Move>)
where
    G::Move: Clone,
{
    stats.nodes += 1;
    let moves = expanded_moves(game, position, depth);
    if moves.is_empty() {
        return (oriented(game.evaluate(position), viewer), Vec::new());
    }

    let to_move = game.side_to_move(position);
    let mut best: Option<(G::Score, Vec<G::Move>)> = None;
    for mv in moves {
        let child = game.play(position, &mv);
        let (score, mut line) = search(game, &child, depth - 1, to_move, None, stats);
        if let Some(root_moves) = root_moves.as_deref_mut() {
            root_moves.push(exact_root_move(&mv, oriented(score, to_move)));
        }

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

/// Unit tests for negamax at the edges of the score range.
#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use crate::{Game, Side, minimax, negamax};

    /// One [`Side::Min`] move to either of two leaves scored `i32::MIN` and `i32::MIN + 1`.
    struct Extremes;

    impl Game for Extremes {
        /// `None` at the root, else the chosen leaf.
        type Position = Option<u8>;
        /// The leaf to move to.
        type Move = u8;
        /// Leaves at the bottom of the `i32` range.
        type Score = i32;

        fn side_to_move(&self, _: &Option<u8>) -> Side {
            Side::Min
        }

        fn moves(&self, position: &Option<u8>, moves: &mut Vec<u8>) {
            if position.is_none() {
                moves.extend([1, 0]);
            }
        }

        fn play(&self, _: &Option<u8>, &leaf: &u8) -> Option<u8> {
            Some(leaf)
        }

        fn evaluate(&self, position: &Option<u8>) -> i32 {
            i32::MIN + i32::from(position.unwrap_or(0))
        }
    }

    /// Negation keeps `i32::MIN` apart from `i32::MIN + 1`, so negamax still agrees with
    /// minimax there.
    #[test]
    fn agrees_with_minimax_at_the_integer_minimum() {
        let result = negamax(&Extremes, &None, 1);
        assert_eq!(result.score, i32::MIN);
        assert_eq!(result, minimax(&Extremes, &None, 1));
    }
}
