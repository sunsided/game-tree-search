//! Principal variation search (PVS), also known as negascout.

use crate::{Algorithm, Game, SearchResult, Searcher};

/// Searches `position` up to `depth` plies with principal variation search.
///
/// Alpha-beta that assumes the first move at each position is the best: every later move is
/// first scouted with a null window, which only tells whether it beats the best so far, and
/// is searched again with the full window only if it does. With good move ordering, most
/// scouts fail, and their null windows prune more than the full window would; with poor
/// ordering, the re-searches cost more than they save, so measure on the game at hand.
///
/// With the default [`Game::satisfices`], returns the same score and bound as
/// [`alpha_beta()`](crate::alpha_beta()); a game that satisfices gets a result that still
/// admits the minimax value, but not necessarily alpha-beta's. Counts re-searches in
/// [`Stats::researches`](crate::Stats::researches). For a searcher that reuses its buffers,
/// or deepens with PVS, use [`Searcher::with_algorithm`] with [`Algorithm::Pvs`].
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
/// use game_tree_search::{alpha_beta, pvs};
///
/// let root = (10, Side::Max);
/// assert_eq!(pvs(&Nim, &root, 10).score, alpha_beta(&Nim, &root, 10).score);
/// ```
#[must_use]
pub fn pvs<G: Game>(
    game: &G,
    position: &G::Position,
    depth: u32,
) -> SearchResult<G::Move, G::Score> {
    Searcher::new()
        .with_algorithm(Algorithm::Pvs)
        .search(game, position, depth)
}
