//! What a search returns: [`SearchResult`] with the score, its [`Bound`], the principal
//! variation and the root moves tried as [`ScoredMove`]s, and [`Stats`].

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
    /// Positions scored by [`Game::evaluate`](crate::Game::evaluate) because the depth ran out,
    /// whether or not the game goes on there.
    pub horizon: u64,
    /// The deepest ply visited: the most moves played from the root to any position the
    /// search reached.
    pub max_ply: u32,
    /// Positions answered by a transposition table entry instead of being searched.
    pub table_hits: u64,
    /// Children searched a second time because a principal variation search scout found
    /// them better than expected. Always `0` for the other algorithms.
    pub researches: u64,
    /// Root searches MTD(f) ran, its final and fallback searches included. Always `0` for
    /// the other algorithms.
    pub passes: u64,
    /// Iterations searched again with a wider window because the score fell outside an
    /// aspiration window (see [`Deepening::window`](crate::Deepening::window)).
    pub widenings: u64,
}

impl Stats {
    /// All counters at zero.
    pub(crate) const ZERO: Self = Self {
        nodes: 0,
        cuts: 0,
        horizon: 0,
        max_ply: 0,
        table_hits: 0,
        researches: 0,
        passes: 0,
        widenings: 0,
    };

    /// Counts a visit to a position `ply` moves below the root.
    pub(crate) fn visit(&mut self, ply: u32) {
        self.nodes += 1;
        self.max_ply = self.max_ply.max(ply);
    }
}

/// A move with the score of the position it leads to: a step of the principal variation,
/// or a move tried at the root.
///
/// Non-exhaustive: later searches may report more per move.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ScoredMove<M, S> {
    /// The move.
    pub mv: M,
    /// The score of the position the move leads to, from
    /// [`Side::Max`](crate::Side::Max)'s point of view.
    pub score: S,
    /// How [`ScoredMove::score`] relates to that position's minimax value. Root moves other
    /// than the best are often only bounded, since the search stops once they are proven
    /// worse.
    pub bound: Bound,
}

/// A line of play in `G`, one scored move per ply.
pub(crate) type Line<G> = Vec<ScoredMove<<G as crate::Game>::Move, <G as crate::Game>::Score>>;

impl<M, S> ScoredMove<M, S> {
    /// `mv`, leading to a position whose minimax value is `score`.
    pub(crate) const fn exact(mv: M, score: S) -> Self {
        Self {
            mv,
            score,
            bound: Bound::Exact,
        }
    }
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
    /// The principal variation: the best line of play from the root, one move per ply, each
    /// with the score and bound of the position it leads to.
    ///
    /// Ends where the game ends or the depth runs out; empty if the root is evaluated
    /// directly. Playing it from the root reaches a position whose
    /// [`evaluate`](crate::Game::evaluate) is [`SearchResult::score`], unless the line ends
    /// early at a position a transposition table answered, which keeps no line below it.
    pub pv: Vec<ScoredMove<M, S>>,
    /// Every root move searched, in the order searched; the best is among them.
    pub root_moves: Vec<ScoredMove<M, S>>,
    /// The legal root moves never searched because the search stopped early.
    pub unexamined: usize,
    /// The depth searched, in plies. For [`deepen()`](crate::deepen()), the deepest iteration
    /// that finished.
    pub depth: u32,
    /// Whether no line ran into the depth limit, so every line was searched to the end of the
    /// game: a deeper search would return the same result. A line whose game ends exactly at
    /// the depth limit still counts as running into it. Only searched lines count, so a search
    /// that prunes more, for example thanks to a better move order, is proven sooner.
    pub proven: bool,
    /// What the search did. For [`deepen()`](crate::deepen()), summed over every iteration,
    /// including one the budget stopped.
    pub stats: Stats,
}

impl<M, S> SearchResult<M, S> {
    /// The move to play at the root: the first move of the principal variation.
    ///
    /// `None` if the root ends the game or was searched to depth `0`.
    #[must_use]
    pub fn best(&self) -> Option<&M> {
        self.pv.first().map(|step| &step.mv)
    }
}
