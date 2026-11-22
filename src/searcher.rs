//! [`Searcher`]: the reusable working memory of the searches.

use alloc::vec::Vec;
use core::mem;

use crate::table::{Memory, Transpositions};
use crate::{Game, Hashed, NoTable, ScoredMove, Stats, Table};

/// Working memory for repeated searches: move lists and principal variations per ply, and
/// optionally a transposition table `T`.
///
/// Keep one per search thread or game agent and reuse it. Its buffers grow to the deepest
/// search run so far and are kept between searches, so a warm searcher allocates only
/// results: the one it returns, plus one per iteration while deepening. The free functions such as [`alpha_beta()`](crate::alpha_beta())
/// create a fresh one per call.
///
/// Built with [`Searcher::with_table`], it also remembers positions in a [`Table`]
/// across searches, which needs a [`Hashed`] game. Entries are only valid for
/// the game configuration that produced them: when the game passed to the searcher changes
/// its rules, clear the table, for example with
/// [`SlotTable::clear`](crate::SlotTable::clear) through [`Searcher::table_mut`].
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
///     let result = searcher.search(&Nim, &(stones, Side::Max), 9);
///     assert_eq!(result.score, if stones % 3 == 0 { -1 } else { 1 });
/// }
/// ```
#[derive(Debug)]
pub struct Searcher<G: Game, T = NoTable> {
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
    /// The transposition table, kept between searches.
    pub(crate) table: T,
    /// The search run at each depth.
    pub(crate) algorithm: Algorithm,
    /// The score MTD(f) starts from: the previous iteration's while deepening.
    pub(crate) guess: Option<G::Score>,
    /// The moves searched at the root in the last root search.
    pub(crate) root_moves: Vec<ScoredMove<G::Move, G::Score>>,
    /// Plies deepening starts ahead of depth 1; a Lazy SMP helper thread runs ahead so its
    /// entries are ready when the main thread gets there.
    pub(crate) lead: u32,
}

/// The search a [`Searcher`] runs at each depth, for one search and for every iteration of
/// deepening.
///
/// With the default [`Game::satisfices`], every algorithm returns the same score; they differ
/// in how many positions they visit. Under satisficing, each returns a sound result, not
/// necessarily the same one.
/// Non-exhaustive: more algorithms may follow.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Algorithm {
    /// Fail-soft alpha-beta: see [`alpha_beta()`](crate::alpha_beta()).
    #[default]
    AlphaBeta,
    /// Principal variation search: see [`pvs()`](crate::pvs()).
    Pvs,
    /// MTD(f), null-window searches that close in on the score: see [`mtdf()`](crate::mtdf()).
    Mtdf,
}

impl<G: Game> Searcher<G> {
    /// A searcher with empty buffers and no transposition table; the buffers grow on first
    /// use.
    #[must_use]
    pub const fn new() -> Self {
        Self::with_memory(NoTable)
    }
}

impl<G: Hashed, T: Table<G::Key, G::Move, G::Score>> Searcher<G, Transpositions<T>> {
    /// A searcher with empty buffers that remembers positions in `table` across searches.
    #[must_use]
    pub const fn with_table(table: T) -> Self {
        Self::with_memory(Transpositions(table))
    }

    /// The transposition table.
    #[must_use]
    pub const fn table(&self) -> &T {
        &self.table.0
    }

    /// The transposition table, for example to clear it.
    pub const fn table_mut(&mut self) -> &mut T {
        &mut self.table.0
    }
}

impl<G: Game, T: Memory<G>> Searcher<G, T> {
    /// A searcher with empty buffers and the given memory.
    const fn with_memory(table: T) -> Self {
        Self {
            moves: Vec::new(),
            lines: Vec::new(),
            hint: Vec::new(),
            aborted: false,
            stats: Stats::ZERO,
            table,
            algorithm: Algorithm::AlphaBeta,
            guess: None,
            root_moves: Vec::new(),
            lead: 0,
        }
    }

    /// This searcher, running `algorithm` from now on.
    #[must_use]
    pub const fn with_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// The algorithm this searcher runs.
    #[must_use]
    pub const fn algorithm(&self) -> Algorithm {
        self.algorithm
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
