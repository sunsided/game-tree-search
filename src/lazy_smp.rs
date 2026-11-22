//! Lazy SMP: several threads deepen the same root, sharing one transposition table
//! (feature `parallel`).

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use std::panic;
use std::thread;

use search_core::Budget;

use crate::{Algorithm, Hashed, SearchResult, Searcher, SharedTable, Stats};

/// Deepens `position` with `threads` threads that share `table` (Lazy SMP) and returns the
/// main thread's result.
///
/// Every thread runs iterative deepening with `algorithm`; every other helper starts one
/// ply ahead. They do not split the work explicitly: entries one thread stores answer or
/// order the others' searches, and a helper running ahead has the main thread's next
/// iteration partly in the table before it starts. On the random test trees, eight threads
/// cut the main thread's positions by about 60%, but on searches of a few milliseconds the
/// thread start-up and lock contention cost more time than that saves: Lazy SMP pays off on
/// long searches, so measure it on the game at hand. The main thread spends
/// `budget` and decides the result, as [`Searcher::deepen`] would; the helpers stop when
/// it is done. With the default [`Game::satisfices`](crate::Game::satisfices), the score
/// equals a single-threaded search's at the same depth, but the line, the root moves'
/// bounds and the statistics, which are the main thread's, vary from run to run.
///
/// `table` keeps its entries for later searches; clear it with [`SharedTable::clear`]
/// before searching a different game. A `threads` of `0` runs one thread.
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
    (threads, algorithm): (usize, Algorithm),
    table: &SharedTable<G::Key, G::Move, G::Score>,
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
    let done = AtomicBool::new(false);
    let searcher = || Searcher::with_table(table).with_algorithm(algorithm);
    thread::scope(|scope| {
        let helpers: Vec<_> = (1..threads.max(1))
            .map(|helper| {
                let stopped = |_: &Stats| done.load(Ordering::Relaxed);
                scope.spawn(move || {
                    let mut helper_searcher = searcher();
                    helper_searcher.lead = u32::from(helper % 2 == 1);
                    helper_searcher.deepen(game, position, max_depth, stopped)
                })
            })
            .collect();

        let main = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            searcher().deepen(game, position, max_depth, budget)
        }));
        done.store(true, Ordering::Relaxed);
        for helper in helpers {
            helper
                .join()
                .unwrap_or_else(|panic| panic::resume_unwind(panic));
        }

        main.unwrap_or_else(|panic| panic::resume_unwind(panic))
    })
}
