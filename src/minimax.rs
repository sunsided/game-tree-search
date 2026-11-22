//! Plain minimax: the reference every faster search must agree with.

use alloc::vec::Vec;

use crate::game::expanded_moves;
use crate::result::Line;
use crate::{Bound, Game, Score, ScoredMove, SearchResult, Side, Stats};

/// Searches every line from `position` up to `depth` plies and returns the minimax score.
///
/// [`Side::Max`] picks the highest child score, [`Side::Min`] the lowest; the side to move
/// comes from [`Game::side_to_move`], so one side may move several times in a row. Visits
/// every position of the tree, which grows exponentially with `depth`; it serves as the
/// reference the faster searches are tested against. Ties go to the earliest move.
///
/// # Examples
///
/// See [`Game`] for a complete game.
#[must_use]
pub fn minimax<G: Game>(
    game: &G,
    position: &G::Position,
    depth: u32,
) -> SearchResult<G::Move, G::Score> {
    let mut stats = Stats::default();
    let mut root_moves = Vec::new();
    let (score, mut pv) = search(game, position, depth, Some(&mut root_moves), &mut stats);
    pv.reverse();

    SearchResult {
        score,
        bound: Bound::Exact,
        pv,
        root_moves,
        unexamined: 0,
        depth,
        proven: stats.horizon == 0,
        stats,
    }
}

/// The minimax score of `position` with `depth` plies left, and its principal variation
/// in reverse (last move first), so each level appends its move in constant time.
///
/// Records every child of the root in `root_moves`, given only at the root.
fn search<G: Game>(
    game: &G,
    position: &G::Position,
    depth: u32,
    mut root_moves: Option<&mut Vec<ScoredMove<G::Move, G::Score>>>,
    stats: &mut Stats,
) -> (G::Score, Line<G>) {
    stats.nodes += 1;
    let moves = expanded_moves(game, position, depth, stats);
    if moves.is_empty() {
        return (game.evaluate(position), Vec::new());
    }

    let side = game.side_to_move(position);
    let mut best: Option<(G::Score, Line<G>)> = None;
    for mv in moves {
        let (score, mut line) = search(game, &game.play(position, &mv), depth - 1, None, stats);
        if let Some(root_moves) = root_moves.as_deref_mut() {
            root_moves.push(ScoredMove::exact(mv.clone(), score));
        }

        if best
            .as_ref()
            .is_none_or(|&(best, _)| improves(side, score, best))
        {
            line.push(ScoredMove::exact(mv, score));
            best = Some((score, line));
        }
    }

    best.expect("a position with moves has a best child")
}

/// Whether `score` is strictly better for `side` than `best`.
pub(crate) fn improves<S: Score>(side: Side, score: S, best: S) -> bool {
    match side {
        Side::Max => score > best,
        Side::Min => score < best,
    }
}
