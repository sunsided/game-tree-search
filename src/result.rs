//! What a search returns: [`SearchResult`] with the score, the principal variation and
//! [`Stats`].

use alloc::vec::Vec;

/// Counters collected during one search run.
///
/// Non-exhaustive: later algorithms may add counters. Start from [`Stats::default`] where a
/// value is needed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Stats {
    /// Positions visited, the root included: each one was either expanded or evaluated.
    pub nodes: u64,
}

/// The result of searching a position.
///
/// Non-exhaustive: later algorithms may report more.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SearchResult<M, S> {
    /// The score of the root from [`Side::Max`](crate::Side::Max)'s point of view, assuming
    /// both sides play the principal variation.
    pub score: S,
    /// The principal variation: the best line of play from the root, one move per ply.
    ///
    /// Ends where the game ends or the depth runs out; empty if the root is evaluated
    /// directly. Playing it from the root reaches a position whose
    /// [`evaluate`](crate::Game::evaluate) is [`SearchResult::score`].
    pub pv: Vec<M>,
    /// What the search did.
    pub stats: Stats,
}

impl<M, S> SearchResult<M, S> {
    /// The move to play at the root: the first move of the principal variation.
    ///
    /// `None` if the root ends the game or was searched to depth `0`.
    #[must_use]
    pub fn best(&self) -> Option<&M> {
        self.pv.first()
    }
}
