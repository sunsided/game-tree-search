//! Iterative deepening: alpha-beta searches one ply deeper at a time, within a budget.

use search_core::Budget;

use crate::alpha_beta::Run;
use crate::{Game, SearchResult, Searcher, Stats};

/// Decides after each finished iteration whether deepening goes on.
///
/// [`Searcher::deepen_with`] asks it before each deeper iteration, with the result of the
/// iteration that just finished; it is not asked when deepening stops anyway, at the maximum
/// depth, on a proven result or when the budget runs out. Implemented for every `FnMut(&SearchResult<M, S>) -> bool`;
/// [`deepen()`] uses a hook that always goes on, leaving the stop to the maximum depth, the
/// budget and a proven result.
///
/// # Examples
///
/// Stop as soon as an iteration finds a forced win, as a game AI that only needs a winning
/// move would:
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
/// use game_tree_search::{SearchResult, Searcher, Stats};
///
/// let mut searcher = Searcher::new();
/// let unlimited = |_: &Stats| false;
/// let until_won = |result: &SearchResult<u32, i32>| result.score < 1;
/// let early = searcher.deepen_with(&Nim, &(2, Side::Max), 40, unlimited, until_won);
/// assert_eq!((early.score, early.depth), (1, 1));
///
/// // Without the hook, deepening goes on until the whole game is searched.
/// let full = searcher.deepen(&Nim, &(2, Side::Max), 40, unlimited);
/// assert_eq!((full.score, full.depth), (1, 3));
/// ```
pub trait Deepening<M, S> {
    /// Whether to search one ply deeper after the iteration that returned `result`.
    fn proceed(&mut self, result: &SearchResult<M, S>) -> bool;
}

impl<M, S, F: FnMut(&SearchResult<M, S>) -> bool> Deepening<M, S> for F {
    fn proceed(&mut self, result: &SearchResult<M, S>) -> bool {
        self(result)
    }
}

/// Searches `position` with alpha-beta at depth 1, 2, … up to `max_depth`, until `budget`
/// runs out.
///
/// Returns the result of the deepest iteration that finished. An iteration the budget stops
/// is discarded, so the result is always a complete search to [`SearchResult::depth`]. The
/// first iteration ignores the budget, so a move is always found. Deepening stops early once
/// an iteration is [`proven`](SearchResult::proven), since searching deeper cannot change it.
///
/// Each iteration tries the previous one's principal variation first, which makes the
/// deeper search prune more. The budget is asked before every position visited, with the
/// [`Stats`] summed over all iterations so far; [`Budget`] is implemented for any
/// `FnMut(&Stats) -> bool`, so a node limit or a deadline is one closure.
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
/// use game_tree_search::{Stats, deepen};
///
/// // Unlimited: deepening stops once the whole game is searched. The longest game takes 7
/// // plies; a game that ends exactly at the depth limit still counts as cut off there, so
/// // the search is proven at depth 8.
/// let solved = deepen(&Nim, &(7, Side::Max), 50, |_: &Stats| false);
/// assert!(solved.proven);
/// assert_eq!((solved.score, solved.depth), (1, 8));
///
/// // At most 30 positions: the result is the deepest search that fit.
/// let limited = deepen(&Nim, &(20, Side::Max), 50, |stats: &Stats| stats.nodes >= 30);
/// assert!(!limited.proven && limited.depth < 20);
/// ```
#[must_use]
pub fn deepen<G: Game>(
    game: &G,
    position: &G::Position,
    max_depth: u32,
    budget: impl Budget<Stats>,
) -> SearchResult<G::Move, G::Score> {
    Searcher::new().deepen(game, position, max_depth, budget)
}

impl<G: Game> Searcher<G> {
    /// Searches `position` with iterative deepening, reusing this searcher's buffers.
    ///
    /// Returns what [`deepen()`](crate::deepen()) returns.
    #[must_use]
    pub fn deepen(
        &mut self,
        game: &G,
        position: &G::Position,
        max_depth: u32,
        budget: impl Budget<Stats>,
    ) -> SearchResult<G::Move, G::Score> {
        let always = |_: &SearchResult<G::Move, G::Score>| true;
        self.deepen_with(game, position, max_depth, budget, always)
    }

    /// Searches `position` with iterative deepening, asking `deepening` after each finished
    /// iteration whether to go on.
    ///
    /// Works like [`Searcher::deepen`], and also stops once `deepening` declines. The result
    /// is that of the iteration it declined after.
    #[must_use]
    pub fn deepen_with(
        &mut self,
        game: &G,
        position: &G::Position,
        max_depth: u32,
        mut budget: impl Budget<Stats>,
        mut deepening: impl Deepening<G::Move, G::Score>,
    ) -> SearchResult<G::Move, G::Score> {
        self.stats = Stats::ZERO;
        self.hint.clear();
        let mut unlimited = |_: &Stats| false;
        let mut first = Run {
            game,
            budget: &mut unlimited,
        };
        let mut best = self.iterate(&mut first, position, max_depth.min(1));

        let mut run = Run {
            game,
            budget: &mut budget,
        };
        for depth in 2..=max_depth {
            if best.proven || !deepening.proceed(&best) {
                break;
            }

            self.hint.clear();
            self.hint.extend(best.pv.iter().map(|step| step.mv.clone()));
            let result = self.iterate(&mut run, position, depth);
            if self.aborted {
                break;
            }

            best = result;
        }

        best.stats = self.stats;
        best
    }
}
