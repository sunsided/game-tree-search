//! What a search returns: [`SearchResult`] with the score, its [`Bound`], the principal
//! variation, the [`RootMove`]s tried and [`Stats`].

use alloc::vec::Vec;

use crate::Bound;

/// Counters collected during one search run.
///
/// Non-exhaustive: later algorithms may add counters. Start from [`Stats::default`] where a
/// value is needed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Stats {
    /// Positions visited, the root included: each one was either expanded or evaluated.
    pub nodes: u64,
    /// Positions whose remaining moves were skipped, by a window cutoff or because
    /// [`Game::satisfices`](crate::Game::satisfices) said so. Always `0` for the reference
    /// searches.
    pub cuts: u64,
}

/// A move tried at the root, with what the search learned about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootMove<M, S> {
    /// The move.
    pub mv: M,
    /// The score of the position the move leads to, from
    /// [`Side::Max`](crate::Side::Max)'s point of view.
    pub score: S,
    /// How [`RootMove::score`] relates to that position's minimax value. Moves other than the
    /// best are often only bounded, since the search stops once they are proven worse.
    pub bound: Bound,
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
    /// How [`SearchResult::score`] relates to the root's minimax value; always
    /// [`Bound::Exact`] for the reference searches.
    pub bound: Bound,
    /// The principal variation: the best line of play from the root, one move per ply.
    ///
    /// Ends where the game ends or the depth runs out; empty if the root is evaluated
    /// directly. Playing it from the root reaches a position whose
    /// [`evaluate`](crate::Game::evaluate) is [`SearchResult::score`].
    pub pv: Vec<M>,
    /// Every root move searched, in the order searched; the best is among them.
    pub root_moves: Vec<RootMove<M, S>>,
    /// The legal root moves never searched because the search stopped early.
    pub unexamined: usize,
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
