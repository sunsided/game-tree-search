//! Lazy SMP: several threads deepen the same root, sharing one transposition table
//! (feature `parallel`).

use alloc::vec::Vec;
use core::cell::Cell;
use core::sync::atomic::{AtomicBool, Ordering};
use std::panic;
use std::thread;

use search_core::Budget;

use crate::{Algorithm, Deepening, Hashed, SearchResult, Searcher, Stats, Table, Window};

/// Deepens `position` with `threads` threads that share `table` (Lazy SMP) and returns the
/// result of the thread that got furthest.
///
/// Every thread runs iterative deepening with `algorithm`; every other helper starts one
/// ply ahead. Each helper keeps the move the game orders first at every position and
/// rotates the others by its own index, so the threads try the game's best guess first,
/// as the main thread does, but search the alternatives in different orders instead of in
/// lockstep. They do not split the work explicitly: entries one thread stores answer or
/// order the others' searches. The search ends when the main thread is done, spending
/// `budget` as [`Searcher::deepen`] would, or when any thread reaches `max_depth` or proves
/// the score: every thread then stops, the main thread included. The result is a finished
/// thread's if there is one, the deepest thread's otherwise, ties going to the main thread;
/// a helper stopped before it finished its first iteration has no result.
///
/// The gain depends on what a node costs. In the `parallel` benchmarks, with an evaluation
/// of about a microsecond, eight threads finish in about 60% of one thread's time; with
/// nodes of a few dozen nanoseconds, the shared table's synchronisation eats the gain, so
/// measure it on the game at hand. With the default
/// [`Game::satisfices`](crate::Game::satisfices), the score equals a single-threaded
/// search's at the same depth, but the line, the root moves' bounds and the statistics,
/// which are the chosen thread's, vary from run to run.
///
/// `table` is a shared reference to a table for several threads: a
/// [`SharedTable`](crate::SharedTable), which takes any move and score, or a
/// [`PackedTable`](crate::PackedTable), lock-free for moves and scores that pack into a
/// few bits. It keeps its entries for later searches; clear it before searching a
/// different game. A `threads` of `0` runs one thread.
///
/// Available with the `parallel` feature, which needs `std`.
///
/// # Panics
///
/// Panics if a thread panics, for example in the game's own code; the first such panic is
/// passed on.
///
/// # Examples
///
/// ```
/// # use game_tree_search::{Game, Hashed, Side};
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
/// # impl Hashed for Nim {
/// #     type Key = u64;
/// #     fn key(&self, &(stones, side): &Self::Position) -> u64 {
/// #         u64::from(stones) * 2 + u64::from(side == Side::Max)
/// #     }
/// # }
/// use game_tree_search::{Algorithm, SharedTable, Stats, deepen_shared};
///
/// let table = SharedTable::new(1 << 12, 8);
/// let unlimited = |_: &Stats| false;
/// let root = (20, Side::Max);
/// let result = deepen_shared(&Nim, &root, 21, (4, Algorithm::AlphaBeta), &table, unlimited);
/// assert_eq!(result.best(), Some(&2));
/// ```
pub fn deepen_shared<G, B>(
    game: &G,
    position: &G::Position,
    max_depth: u32,
    setup: (usize, Algorithm),
    table: impl Table<G::Key, G::Move, G::Score> + Copy + Send + Sync,
    budget: B,
) -> SearchResult<G::Move, G::Score>
where
    G: Hashed + Sync,
    G::Position: Sync,
    G::Key: Send,
    G::Move: Send,
    G::Score: Send,
    B: Budget<Stats> + Send,
{
    let always = |_: &SearchResult<G::Move, G::Score>| true;
    deepen_shared_with(game, position, max_depth, setup, table, budget, always)
}

/// Deepens `position` like [`deepen_shared`], with the main thread asking `deepening` after
/// each finished iteration whether to go on, and for each iteration's aspiration window, as
/// [`Searcher::deepen_with`] does.
///
/// Only the main thread asks `deepening`; the helpers deepen on until the search ends. When
/// `deepening` declines, the main thread's result answers the search like a finished one,
/// so the search ends there unless a helper had already reached `max_depth` or proved the
/// score, whose result is then returned.
///
/// Available with the `parallel` feature, which needs `std`.
///
/// # Panics
///
/// Panics if a thread panics, for example in the game's own code; the first such panic is
/// passed on.
pub fn deepen_shared_with<G, B>(
    game: &G,
    position: &G::Position,
    max_depth: u32,
    (threads, algorithm): (usize, Algorithm),
    table: impl Table<G::Key, G::Move, G::Score> + Copy + Send + Sync,
    budget: B,
    deepening: impl Deepening<G::Move, G::Score>,
) -> SearchResult<G::Move, G::Score>
where
    G: Hashed + Sync,
    G::Position: Sync,
    G::Key: Send,
    G::Move: Send,
    G::Score: Send,
    B: Budget<Stats> + Send,
{
    let done = &AtomicBool::new(false);
    let declined = Cell::new(false);
    let searcher = || Searcher::with_table(table).with_algorithm(algorithm);
    thread::scope(|scope| {
        // Set on every way out, a failed spawn included, so no helper outlives the call.
        let stop = Stop(done);
        let helpers: Vec<_> = (1..threads.max(1))
            .map(|helper| {
                let stopped = |_: &Stats| done.load(Ordering::Relaxed);
                scope.spawn(move || {
                    let mut helper_searcher = searcher();
                    helper_searcher.lead = u32::from(helper % 2 == 1);
                    helper_searcher.budget_first = true;
                    helper_searcher.skew = helper;
                    let result = helper_searcher.deepen(game, position, max_depth, stopped);
                    // A helper stopped in its first iteration has no answer at all; one that
                    // reached the full depth, or proved the score, ends the search.
                    let answer = helper_searcher.answered.then_some(result);
                    if answer
                        .as_ref()
                        .is_some_and(|result| finished(result, max_depth))
                    {
                        done.store(true, Ordering::Relaxed);
                    }

                    answer
                })
            })
            .collect();

        let mut budget = budget;
        let main_budget = |stats: &Stats| done.load(Ordering::Relaxed) || budget.exhausted(stats);
        let watched = Watched {
            deepening,
            declined: &declined,
        };
        let main = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            searcher().deepen_with(game, position, max_depth, main_budget, watched)
        }));
        drop(stop);
        let main = rethrow(main);
        let main_finished = declined.get();
        let rank = |result: &SearchResult<G::Move, G::Score>, settled: bool| {
            (settled || finished(result, max_depth), result.depth)
        };
        let mut best_rank = rank(&main, main_finished);
        helpers
            .into_iter()
            .filter_map(|helper| rethrow(helper.join()))
            .fold(main, |best, helper| {
                let helper_rank = rank(&helper, false);
                if helper_rank > best_rank {
                    best_rank = helper_rank;
                    helper
                } else {
                    best
                }
            })
    })
}

/// Whether `result` answers the search: it reached `max_depth` or proved its score.
const fn finished<M, S>(result: &SearchResult<M, S>, max_depth: u32) -> bool {
    result.depth == max_depth || result.proven
}

/// The main thread's [`Deepening`] hook, noting when it declines to go on.
struct Watched<'a, D> {
    /// The caller's hook.
    deepening: D,
    /// Set once `deepening` declined.
    declined: &'a Cell<bool>,
}

impl<M, S, D: Deepening<M, S>> Deepening<M, S> for Watched<'_, D> {
    fn proceed(&mut self, result: &SearchResult<M, S>) -> bool {
        let proceed = self.deepening.proceed(result);
        if !proceed {
            self.declined.set(true);
        }

        proceed
    }

    fn window(&mut self, previous: &SearchResult<M, S>, attempt: u32) -> Window<S> {
        self.deepening.window(previous, attempt)
    }
}

/// Tells the helper threads to stop when dropped.
struct Stop<'a>(&'a AtomicBool);

impl Drop for Stop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// The value of a thread that finished, or its panic passed on to the caller.
pub(crate) fn rethrow<T>(outcome: thread::Result<T>) -> T {
    outcome.unwrap_or_else(|panic| panic::resume_unwind(panic))
}
