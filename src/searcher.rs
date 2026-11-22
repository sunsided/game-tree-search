//! [`Searcher`]: the reusable working memory of the searches.

use alloc::vec::Vec;
use core::mem;

use crate::{Game, ScoredMove, Stats};

/// Working memory for repeated searches: move lists and principal variations per ply.
///
/// Keep one per search thread or game agent and reuse it. Its buffers grow to the deepest
/// search run so far and are kept between searches, so a warm searcher allocates only the
/// results it returns. The free functions such as [`alpha_beta()`](crate::alpha_beta())
/// create a fresh one per call.
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
/// use game_tree_search::Searcher;
///
/// let mut searcher = Searcher::new();
/// for stones in 1..=9 {
///     let result = searcher.alpha_beta(&Nim, &(stones, Side::Max), 9);
///     assert_eq!(result.score, if stones % 3 == 0 { -1 } else { 1 });
/// }
/// ```
#[derive(Debug)]
pub struct Searcher<G: Game> {
    /// Move lists by ply, empty between searches; kept for their capacity.
    moves: Vec<Vec<G::Move>>,
    /// The principal variation found below each ply in the current search.
    lines: Vec<Vec<ScoredMove<G::Move, G::Score>>>,
    /// The previous iteration's principal variation, tried first while a search follows it.
    pub(crate) hint: Vec<G::Move>,
    /// Whether the budget stopped the current search; its result is then discarded.
    pub(crate) aborted: bool,
    /// The counters of the current or last search.
    pub(crate) stats: Stats,
}

impl<G: Game> Searcher<G> {
    /// A searcher with empty buffers; they grow on first use.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            moves: Vec::new(),
            lines: Vec::new(),
            hint: Vec::new(),
            aborted: false,
            stats: Stats::ZERO,
        }
    }

    /// Prepares the buffers of `ply` for a new position: clears its line and hands out its
    /// move list, empty. Give the list back with [`Searcher::leave`].
    pub(crate) fn enter(&mut self, ply: usize) -> Vec<G::Move> {
        while self.lines.len() <= ply {
            self.lines.push(Vec::new());
            self.moves.push(Vec::new());
        }

        self.lines[ply].clear();
        mem::take(&mut self.moves[ply])
    }

    /// Returns the move list of `ply`, emptied, so its capacity serves the next position.
    pub(crate) fn leave(&mut self, ply: usize, mut moves: Vec<G::Move>) {
        moves.clear();
        self.moves[ply] = moves;
    }

    /// Makes `step` followed by the line found below it, at `ply + 1`, the line of `ply`.
    pub(crate) fn adopt(&mut self, ply: usize, step: ScoredMove<G::Move, G::Score>) {
        let (above, below) = self.lines.split_at_mut(ply + 1);
        let line = &mut above[ply];
        line.clear();
        line.push(step);
        line.extend_from_slice(&below[0]);
    }

    /// The principal variation found below `ply`.
    pub(crate) fn line(&self, ply: usize) -> &[ScoredMove<G::Move, G::Score>] {
        &self.lines[ply]
    }
}

impl<G: Game> Default for Searcher<G> {
    fn default() -> Self {
        Self::new()
    }
}
