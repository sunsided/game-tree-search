//! The algorithms every cross-check runs, and a one-call search with each.

use game_tree_search::{Algorithm, Budget, Game, SearchResult, Searcher, Stats};

/// Every algorithm a [`Searcher`] can run; the shared checks loop over them.
pub const ALGORITHMS: [Algorithm; 2] = [Algorithm::AlphaBeta, Algorithm::Pvs];

/// Searches `position` to `depth` with `algorithm` and a fresh searcher.
pub fn search<G: Game>(
    algorithm: Algorithm,
    game: &G,
    position: &G::Position,
    depth: u32,
) -> SearchResult<G::Move, G::Score> {
    Searcher::new()
        .with_algorithm(algorithm)
        .search(game, position, depth)
}

/// Deepens from `position` up to `max_depth` with `algorithm` and a fresh searcher.
pub fn deepen_with_algorithm<G: Game>(
    algorithm: Algorithm,
    game: &G,
    position: &G::Position,
    max_depth: u32,
    budget: impl Budget<Stats>,
) -> SearchResult<G::Move, G::Score> {
    Searcher::new()
        .with_algorithm(algorithm)
        .deepen(game, position, max_depth, budget)
}
