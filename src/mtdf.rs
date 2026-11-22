//! MTD(f): null-window searches of the root that close in on its score.

use search_core::Budget;

use crate::alpha_beta::{Node, Run, Window};
use crate::table::Memory;
use crate::{Algorithm, Bound, Game, Score, SearchResult, Searcher, Stats};

/// The most null-window searches per depth before MTD(f) falls back to one full-window
/// search.
const MAX_PASSES: u32 = 64;

/// Searches `position` up to `depth` plies with MTD(f).
///
/// Each pass searches the root with a null window, which only proves the score lies above or
/// below one value; the bounds close in until they meet. A final search with a window just
/// around the score then returns it exactly, with its principal variation and root moves.
/// Passes revisit the same positions, so MTD(f) pays off with a transposition table (see
/// [`Searcher::with_table`]) and a good first guess. While deepening, each iteration starts
/// from the previous one's score; a single search starts from the root's
/// [`evaluate`](Game::evaluate).
///
/// With the default [`Game::satisfices`], returns the same score and bound as
/// [`alpha_beta()`](crate::alpha_beta()); a game that satisfices gets a result that still
/// admits the minimax value, but not necessarily alpha-beta's. If the bounds
/// stop closing in, which [`Game::satisfices`] can cause, or [`MAX_PASSES`] passes do not
/// suffice, one full-window search gives the result instead. Passes are counted in
/// [`Stats::passes`].
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
/// use game_tree_search::{Algorithm, Bound, Searcher, SlotTable, alpha_beta, mtdf};
///
/// let root = (10, Side::Max);
/// let result = mtdf(&Nim, &root, 10);
/// assert_eq!((result.score, result.bound), (alpha_beta(&Nim, &root, 10).score, Bound::Exact));
///
/// # impl Hashed for Nim {
/// #     type Key = u64;
/// #     fn key(&self, &(stones, side): &Self::Position) -> u64 {
/// #         u64::from(stones) * 2 + u64::from(side == Side::Max)
/// #     }
/// # }
/// // With a table and deepening, each pass reuses what the passes before it learned.
/// let mut searcher = Searcher::with_table(SlotTable::new(1024)).with_algorithm(Algorithm::Mtdf);
/// let deep = searcher.deepen(&Nim, &(30, Side::Max), 30, |_: &_| false);
/// assert_eq!(deep.score, alpha_beta(&Nim, &(30, Side::Max), deep.depth).score);
/// ```
#[must_use]
pub fn mtdf<G: Game>(
    game: &G,
    position: &G::Position,
    depth: u32,
) -> SearchResult<G::Move, G::Score> {
    Searcher::new()
        .with_algorithm(Algorithm::Mtdf)
        .search(game, position, depth)
}

/// What the passes so far proved about the root score: a lower and an upper bound, `None`
/// where still open.
#[derive(Debug, Clone, Copy)]
struct Interval<S> {
    /// The root score is at least this.
    lower: Option<S>,
    /// The root score is at most this.
    upper: Option<S>,
}

impl<S: Score> Interval<S> {
    /// The interval after a pass returned `score` with `bound`, or `None` if the pass proved
    /// nothing new.
    fn narrowed(self, score: S, bound: Bound) -> Option<Self> {
        let raise = |lower: Option<S>| Some(lower.map_or(score, |lower| lower.max(score)));
        let drop = |upper: Option<S>| Some(upper.map_or(score, |upper| upper.min(score)));
        let narrowed = match bound {
            Bound::Exact => Self {
                lower: Some(score),
                upper: Some(score),
            },
            Bound::Lower => Self {
                lower: raise(self.lower),
                ..self
            },
            Bound::Upper => Self {
                upper: drop(self.upper),
                ..self
            },
            Bound::Approximate => return None,
        };

        (narrowed.lower != self.lower || narrowed.upper != self.upper).then_some(narrowed)
    }

    /// The root score, once both bounds meet, or once one bound sits at the end of the score
    /// range, where nothing lies beyond it.
    fn value(self) -> Option<S> {
        let top = self.lower.filter(|&lower| lower.next_up() == lower);
        let bottom = self.upper.filter(|&upper| upper.next_down() == upper);
        let met = self
            .lower
            .zip(self.upper)
            .and_then(|(lower, upper)| (lower == upper).then_some(lower));

        met.or(top).or(bottom)
    }

    /// The null window of the next pass around the guess `g`: it asks whether the score
    /// reaches `g`, or the next score above `g` if `g` is already proven as a lower bound.
    /// `None` at the bottom of the score range, where no null window fits.
    fn probe(self, g: S) -> Option<Window<S>> {
        let beta = if self.lower == Some(g) {
            g.next_up()
        } else {
            g
        };
        let alpha = beta.next_down();

        (alpha < beta).then_some(Window {
            alpha: Some(alpha),
            beta: Some(beta),
        })
    }
}

/// The window just around `value`, so a search returns it exactly.
fn around<S: Score>(value: S) -> Window<S> {
    let below = value.next_down();
    let above = value.next_up();

    Window {
        alpha: (below < value).then_some(below),
        beta: (above > value).then_some(above),
    }
}

impl<G: Game, T: Memory<G>> Searcher<G, T> {
    /// MTD(f) at the root `position` to `depth`, starting from the searcher's guess or the
    /// root's evaluation.
    pub(crate) fn mtdf<B: Budget<Stats>>(
        &mut self,
        run: &mut Run<'_, G, B>,
        position: &G::Position,
        depth: u32,
    ) -> Node<G::Score> {
        let mut g = self.guess.unwrap_or_else(|| run.game.evaluate(position));
        let mut interval = Interval {
            lower: None,
            upper: None,
        };
        let mut window = Window::FULL;
        for _ in 0..MAX_PASSES {
            let Some(probe) = interval.probe(g) else {
                break;
            };

            let pass = self.pass(run, position, depth, probe);
            if self.aborted {
                return pass;
            }

            let Some(narrowed) = interval.narrowed(pass.score, pass.bound) else {
                break;
            };

            (interval, g) = (narrowed, pass.score);
            if let Some(value) = interval.value() {
                window = around(value);
                break;
            }
        }

        self.finish_mtdf(run, position, depth, window)
    }

    /// The last search of MTD(f): within `window`, falling back to the full window if that
    /// leaves the score inexact.
    fn finish_mtdf<B: Budget<Stats>>(
        &mut self,
        run: &mut Run<'_, G, B>,
        position: &G::Position,
        depth: u32,
        window: Window<G::Score>,
    ) -> Node<G::Score> {
        let node = self.pass(run, position, depth, window);
        if window.is_full() || self.aborted || node.bound == Bound::Exact {
            return node;
        }

        self.pass(run, position, depth, Window::FULL)
    }

    /// One counted root search within `window`.
    fn pass<B: Budget<Stats>>(
        &mut self,
        run: &mut Run<'_, G, B>,
        position: &G::Position,
        depth: u32,
        window: Window<G::Score>,
    ) -> Node<G::Score> {
        self.stats.passes += 1;
        self.root(run, position, depth, window)
    }
}

/// Unit tests for the pass bookkeeping.
#[cfg(test)]
mod tests {
    use super::{Interval, around};
    use crate::Bound;

    /// An interval with the given bounds.
    const fn interval(lower: Option<i8>, upper: Option<i8>) -> Interval<i8> {
        Interval { lower, upper }
    }

    /// Each bound narrows its side; an exact score closes both; nothing new is no progress.
    #[test]
    fn passes_narrow_the_interval() {
        let open = interval(None, None);
        let lower = open.narrowed(2, Bound::Lower).expect("progress");
        assert_eq!((lower.lower, lower.upper), (Some(2), None));
        let both = lower.narrowed(5, Bound::Upper).expect("progress");
        assert_eq!((both.lower, both.upper), (Some(2), Some(5)));
        assert!(both.narrowed(1, Bound::Lower).is_none());
        assert!(both.narrowed(3, Bound::Approximate).is_none());
        assert_eq!(
            both.narrowed(3, Bound::Exact).and_then(Interval::value),
            Some(3)
        );
        assert_eq!(both.value(), None);
    }

    /// A lower bound at the top of the range, or an upper bound at its bottom, is the score.
    #[test]
    fn bounds_at_the_ends_of_the_range_are_the_score() {
        assert_eq!(interval(Some(i8::MAX), None).value(), Some(i8::MAX));
        assert_eq!(interval(None, Some(i8::MIN)).value(), Some(i8::MIN));
        assert_eq!(interval(Some(i8::MAX - 1), None).value(), None);
    }

    /// The probe asks about the guess, or just above it once the guess is a proven lower
    /// bound; no probe fits at the bottom of the range.
    #[test]
    fn probes_step_past_proven_lower_bounds() {
        let probe = interval(None, None).probe(4).expect("fits");
        assert_eq!((probe.alpha, probe.beta), (Some(3), Some(4)));
        let probe = interval(Some(4), None).probe(4).expect("fits");
        assert_eq!((probe.alpha, probe.beta), (Some(4), Some(5)));
        assert!(interval(None, None).probe(i8::MIN).is_none());
    }

    /// The window around a value stays open on a side where the range ends.
    #[test]
    fn windows_around_extremes_stay_open() {
        let window = around(i8::MAX);
        assert_eq!((window.alpha, window.beta), (Some(i8::MAX - 1), None));
        let window = around(i8::MIN);
        assert_eq!((window.alpha, window.beta), (None, Some(i8::MIN + 1)));
    }
}
