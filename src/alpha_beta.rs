//! Alpha-beta pruning: minimax that skips moves which cannot change the decision.

use alloc::vec::Vec;

use core::mem;

use search_core::Budget;

use crate::minimax::improves;
use crate::table::{Entry, Memory};
use crate::{Algorithm, Bound, Game, Score, ScoredMove, SearchResult, Searcher, Side, Stats};

/// Searches `position` up to `depth` plies with fail-soft alpha-beta pruning.
///
/// Returns the [`minimax()`](crate::minimax()) score while visiting far fewer positions,
/// most so when [`Game::order_moves`] puts good moves first. Like minimax, the side to move
/// comes from [`Game::side_to_move`], so one side may move several times in a row.
///
/// With the default [`Game::satisfices`], the root score is the minimax value and its bound
/// is [`Bound::Exact`]; the best root move is exact too, while the other
/// [`SearchResult::root_moves`] are often only bounded. A game that overrides
/// [`Game::satisfices`] trades that exactness for speed, as documented there.
///
/// Allocates its working buffers on every call; [`Searcher::search`] reuses them across
/// searches.
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
/// use game_tree_search::{Bound, alpha_beta, minimax};
///
/// let root = (10, Side::Max);
/// let pruned = alpha_beta(&Nim, &root, 10);
/// let full = minimax(&Nim, &root, 10);
/// assert_eq!((pruned.score, pruned.bound), (full.score, Bound::Exact));
/// assert!(pruned.stats.nodes < full.stats.nodes);
/// ```
#[must_use]
pub fn alpha_beta<G: Game>(
    game: &G,
    position: &G::Position,
    depth: u32,
) -> SearchResult<G::Move, G::Score> {
    Searcher::new().search(game, position, depth)
}

impl<G: Game, T: Memory<G>> Searcher<G, T> {
    /// Searches `position` up to `depth` plies with this searcher's [`Algorithm`], reusing its
    /// buffers and table.
    ///
    /// Returns what [`alpha_beta()`](crate::alpha_beta()), [`pvs()`](crate::pvs()) or
    /// [`mtdf()`](crate::mtdf()) return. Once the buffers have grown to the search depth, a
    /// search allocates only the [`SearchResult`] it returns.
    #[must_use]
    pub fn search(
        &mut self,
        game: &G,
        position: &G::Position,
        depth: u32,
    ) -> SearchResult<G::Move, G::Score> {
        self.stats = Stats::ZERO;
        self.hint.clear();
        self.guess = None;
        let mut unlimited = |_: &Stats| false;
        let mut run = Run {
            game,
            budget: &mut unlimited,
        };

        let node = self.iterate(&mut run, position, depth, Window::FULL);
        self.result(&node, depth)
    }

    /// One search of `position` to `depth` with the searcher's algorithm within `window`,
    /// trying the hint's moves first; MTD(f) ignores `window` and picks its own.
    ///
    /// Adds to the searcher's counters instead of resetting them. If the budget stopped the
    /// search, the searcher's abort flag is set and the node must be discarded.
    pub(crate) fn iterate<B: Budget<Stats>>(
        &mut self,
        run: &mut Run<'_, G, B>,
        position: &G::Position,
        depth: u32,
        window: Window<G::Score>,
    ) -> Node<G::Score> {
        self.aborted = false;
        if self.algorithm == Algorithm::Mtdf {
            return self.mtdf(run, position, depth);
        }

        self.root(run, position, depth, window)
    }

    /// One search of the root `position` to `depth` within `window`, leaving its line and
    /// root moves in the searcher's buffers.
    pub(crate) fn root<B: Budget<Stats>>(
        &mut self,
        run: &mut Run<'_, G, B>,
        position: &G::Position,
        depth: u32,
        window: Window<G::Score>,
    ) -> Node<G::Score> {
        let mut root_moves = mem::take(&mut self.root_moves);
        root_moves.clear();
        let frame = Frame {
            ply: 0,
            depth,
            window,
            on_hint: true,
        };
        let node = self.visit(run, position, frame, Some(&mut root_moves));
        self.root_moves = root_moves;

        node
    }

    /// The result of the last root search, which returned `node` at `depth`.
    pub(crate) fn result(
        &self,
        node: &Node<G::Score>,
        depth: u32,
    ) -> SearchResult<G::Move, G::Score> {
        SearchResult {
            score: node.score,
            bound: node.bound,
            pv: self.line(0).to_vec(),
            root_moves: self.root_moves.clone(),
            unexamined: node.skipped,
            depth,
            proven: node.proven,
            stats: self.stats,
        }
    }

    /// Searches the position at `frame`, recording every child in `root_moves` (given only at
    /// the root).
    ///
    /// Leaves the position's principal variation in the line buffer of its ply. Once the
    /// budget is exhausted, it sets the searcher's abort flag and returns at once.
    fn visit<B: Budget<Stats>>(
        &mut self,
        run: &mut Run<'_, G, B>,
        position: &G::Position,
        frame: Frame<G::Score>,
        root_moves: Option<&mut Vec<ScoredMove<G::Move, G::Score>>>,
    ) -> Node<G::Score> {
        let game = run.game;
        if run.budget.exhausted(&self.stats) {
            self.aborted = true;
            return Node::horizon(game.evaluate(position));
        }

        self.stats
            .visit(u32::try_from(frame.ply).expect("a ply never exceeds the u32 search depth"));
        let mut moves = self.enter(frame.ply);
        if frame.depth == 0 {
            return self.quiesce(run, position, frame, moves, root_moves);
        }

        game.moves(position, &mut moves);
        if moves.is_empty() {
            self.leave(frame.ply, moves);
            return Node::leaf(game.evaluate(position));
        }

        game.order_moves(position, &mut moves);
        let skew = self.skew % moves.len();
        moves.rotate_left(skew);

        let key = self.table.key(game, position);
        let stored = self.table.recall(&key);
        if let Some(node) = stored.as_ref().and_then(|entry| answer(entry, frame)) {
            self.stats.table_hits += 1;
            self.leave(frame.ply, moves);
            return node;
        }

        if let Some(best) = stored.and_then(|entry| entry.best) {
            promote(&mut moves, &best);
        }

        let hint = self.promote_hint(&mut moves, frame);
        let expansion = Expansion {
            moves,
            hint,
            stand: None,
        };
        let node = self.expand(run, position, frame, expansion, root_moves);
        if !self.aborted {
            let entry = self.entry(&node, frame);
            self.table.remember(key, entry);
        }

        node
    }

    /// The table entry for the position at `frame`, just searched to `node`.
    fn entry(&self, node: &Node<G::Score>, frame: Frame<G::Score>) -> Entry<G::Move, G::Score> {
        Entry {
            depth: frame.depth,
            score: node.score,
            bound: node.bound,
            best: self.line(frame.ply).first().map(|step| step.mv.clone()),
            proven: node.proven,
            reach: node.reach,
        }
    }

    /// Quiescence at the horizon position at `frame`: its evaluation, unless one of its
    /// [`Game::noisy_moves`] beats it, searched the same way below. `moves` is the ply's empty
    /// move list; `root_moves`, given only at a depth-0 root, records the noisy moves tried.
    fn quiesce<B: Budget<Stats>>(
        &mut self,
        run: &mut Run<'_, G, B>,
        position: &G::Position,
        frame: Frame<G::Score>,
        mut moves: Vec<G::Move>,
        root_moves: Option<&mut Vec<ScoredMove<G::Move, G::Score>>>,
    ) -> Node<G::Score> {
        let game = run.game;
        self.stats.horizon += 1;
        let stand = Node::horizon(game.evaluate(position));
        game.noisy_moves(position, &mut moves);
        if moves.is_empty() {
            self.leave(frame.ply, moves);
            return stand;
        }

        game.order_moves(position, &mut moves);
        let hint = self.promote_hint(&mut moves, frame);
        let expansion = Expansion {
            moves,
            hint,
            stand: Some(stand),
        };
        self.expand(run, position, frame, expansion, root_moves)
    }

    /// Moves the hint's move for this ply to the front of `moves`, if the line to the
    /// position follows the hint and the move is legal here; returns it.
    fn promote_hint(&self, moves: &mut [G::Move], frame: Frame<G::Score>) -> Option<G::Move> {
        let hint = self.hint.get(frame.ply).filter(|_| frame.on_hint)?;

        promote(moves, hint).then(|| hint.clone())
    }

    /// Searches the `moves` of the position at `frame`, `hint` among them, until one of them
    /// stops the search there; in quiescence, `stand` is the score to beat.
    fn expand<B: Budget<Stats>>(
        &mut self,
        run: &mut Run<'_, G, B>,
        position: &G::Position,
        frame: Frame<G::Score>,
        expansion: Expansion<G::Move, G::Score>,
        mut root_moves: Option<&mut Vec<ScoredMove<G::Move, G::Score>>>,
    ) -> Node<G::Score> {
        let Expansion {
            mut moves,
            hint,
            stand,
        } = expansion;
        let game = run.game;
        let side = game.side_to_move(position);
        let total = moves.len();
        let mut window = frame.window;
        if let Some(stand) = stand {
            window.tighten(side, stand.score);
        }

        let mut best = stand;
        let (mut widened, mut proven, mut searched, mut reach) = (false, stand.is_none(), 0, 0);
        for mv in moves.drain(..) {
            if best.is_some_and(|best| stops(game, side, best.score, &window)) {
                break;
            }

            searched += 1;
            let child_frame = frame.child(window, hint.as_ref() == Some(&mv));
            let child = self.child(
                run,
                &game.play(position, &mv),
                child_frame,
                (side, best.is_some()),
            );
            if self.aborted {
                break;
            }

            let step = ScoredMove {
                mv,
                score: child.score,
                bound: child.bound,
            };
            if let Some(root_moves) = root_moves.as_deref_mut() {
                root_moves.push(step.clone());
            }

            widened |= child.bound.widens(side);
            proven &= child.proven;
            reach = reach.max(child.reach + 1);
            window.tighten(side, child.score);
            if best.is_none_or(|best| improves(side, child.score, best.score)) {
                best = Some(child);
                self.adopt(frame.ply, step);
            }
        }

        self.leave(frame.ply, moves);
        let Some(best) = best else {
            return Node::horizon(game.evaluate(position));
        };

        let node = Node {
            proven,
            reach,
            ..best
        };
        self.finish(side, node, total - searched, widened)
    }

    /// Searches the child at `frame`, a move of `side`'s.
    ///
    /// With [`Algorithm::Pvs`], a child after the first (`later`) is scouted first with a
    /// null window at the edge of `frame`'s window, which only tells whether it beats that
    /// edge; only a child that does is searched again with the full window. The first child
    /// is searched with the full window right away, as it is expected to be the best.
    fn child<B: Budget<Stats>>(
        &mut self,
        run: &mut Run<'_, G, B>,
        position: &G::Position,
        frame: Frame<G::Score>,
        (side, later): (Side, bool),
    ) -> Node<G::Score> {
        let scout = frame
            .window
            .null(side)
            .filter(|_| later && self.algorithm == Algorithm::Pvs);
        if let Some(window) = scout {
            let node = self.visit(run, position, Frame { window, ..frame }, None);
            if self.aborted || !frame.window.needs_research(&node) {
                return node;
            }

            self.stats.researches += 1;
        }

        self.visit(run, position, frame, None)
    }

    /// The position's result from its best child, the moves `skipped` and whether a searched
    /// child `widened` the value.
    fn finish(
        &mut self,
        side: Side,
        best: Node<G::Score>,
        skipped: usize,
        widened: bool,
    ) -> Node<G::Score> {
        let cut = skipped > 0;
        if cut && !self.aborted {
            self.stats.cuts += 1;
        }

        Node {
            bound: Bound::combine(side, cut, best.bound, widened),
            skipped,
            ..best
        }
    }
}

/// Moves `mv` to the front of `moves`, keeping the others in order; whether it was there.
fn promote<M: PartialEq>(moves: &mut [M], mv: &M) -> bool {
    let Some(index) = moves.iter().position(|candidate| candidate == mv) else {
        return false;
    };

    moves[..=index].rotate_right(1);
    true
}

/// The node `entry` answers at `frame` without a search; `None` if the position must be
/// searched.
///
/// An exact entry answers, and so does a bound already outside `frame`'s window. A bound
/// never narrows the window: a search inside a narrowed window could return only a bound
/// where the full search is exact. The root is always searched, so it can report every move
/// it tries.
fn answer<M, S: Ord + Copy>(entry: &Entry<M, S>, frame: Frame<S>) -> Option<Node<S>> {
    if frame.ply == 0 || !entry.serves(frame.depth) {
        return None;
    }

    let answers = match entry.bound {
        Bound::Exact => true,
        Bound::Lower => frame.window.closes(Side::Max, entry.score),
        Bound::Upper => frame.window.closes(Side::Min, entry.score),
        Bound::Approximate => false,
    };

    answers.then(|| Node::stored(entry))
}

/// Whether `side`, holding `score`, skips its remaining moves: a window cutoff, or the game
/// is satisfied.
fn stops<G: Game>(game: &G, side: Side, score: G::Score, window: &Window<G::Score>) -> bool {
    window.closes(side, score) || game.satisfices(side, score)
}

/// The moves a position searches: all of them, or in quiescence the noisy ones.
struct Expansion<M, S> {
    /// The moves to search, in the order to search them.
    moves: Vec<M>,
    /// The hint's move among them, if the line follows the hint.
    hint: Option<M>,
    /// In quiescence, the position's own evaluation, kept unless a move beats it.
    stand: Option<Node<S>>,
}

/// What one search run works with besides the searcher: the game and the budget.
pub(crate) struct Run<'a, G, B> {
    /// The game searched.
    pub(crate) game: &'a G,
    /// Stops the run early; asked before every position visited.
    pub(crate) budget: &'a mut B,
}

/// Where a position sits in the search: its ply, the plies left, its window and whether the
/// line to it follows the searcher's hint.
#[derive(Debug, Clone, Copy)]
struct Frame<S> {
    /// Moves played from the root to the position.
    ply: usize,
    /// Plies left to search below the position.
    depth: u32,
    /// The scores still worth telling apart.
    window: Window<S>,
    /// Whether every move to the position is the hint's move at its ply.
    on_hint: bool,
}

impl<S> Frame<S> {
    /// The frame of a child within `window`, reached by the hint's move if `hinted`.
    const fn child(&self, window: Window<S>, hinted: bool) -> Self {
        Self {
            ply: self.ply + 1,
            depth: self.depth.saturating_sub(1),
            window,
            on_hint: self.on_hint && hinted,
        }
    }
}

/// The scores still worth telling apart at a position, from [`Side::Max`]'s point of view.
///
/// With the default [`Game::satisfices`], a search within a window returns the exact score
/// if it lies strictly inside, and only a bound otherwise; a game that satisfices can get a
/// bound either way. `None` is unbounded on that side. Used to give deepening an aspiration
/// window, see [`Deepening::window`](crate::Deepening::window).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window<S> {
    /// The score [`Side::Max`] is already sure of elsewhere: scores at or below it need not
    /// be told apart.
    pub alpha: Option<S>,
    /// The score [`Side::Min`] is already sure of elsewhere: scores at or above it need not
    /// be told apart.
    pub beta: Option<S>,
}

impl<S> Window<S> {
    /// The window that tells every score apart.
    pub const FULL: Self = Self {
        alpha: None,
        beta: None,
    };
}

impl<S: Ord + Copy> Window<S> {
    /// Whether the window is unbounded on both sides.
    pub(crate) const fn is_full(&self) -> bool {
        self.alpha.is_none() && self.beta.is_none()
    }

    /// Whether `side`, holding `score`, can no longer affect the decision above.
    fn closes(&self, side: Side, score: S) -> bool {
        match side {
            Side::Max => self.beta.is_some_and(|beta| score >= beta),
            Side::Min => self.alpha.is_some_and(|alpha| score <= alpha),
        }
    }

    /// The null window at the edge `side` has to beat: just above alpha for [`Side::Max`],
    /// just below beta for [`Side::Min`]; `None` if that edge is open or at the end of the
    /// score range.
    fn null(&self, side: Side) -> Option<Self>
    where
        S: Score,
    {
        let (alpha, beta) = match side {
            Side::Max => {
                let alpha = self.alpha?;
                (alpha, alpha.next_up())
            }
            Side::Min => {
                let beta = self.beta?;
                (beta.next_down(), beta)
            }
        };

        (alpha < beta).then_some(Self {
            alpha: Some(alpha),
            beta: Some(beta),
        })
    }

    /// Whether a scout result `node` needs a search with this full window: it is not exact,
    /// and its score lies strictly inside the window, so the true value may too.
    fn needs_research(&self, node: &Node<S>) -> bool {
        let above = self.alpha.is_none_or(|alpha| node.score > alpha);
        let below = self.beta.is_none_or(|beta| node.score < beta);

        node.bound != Bound::Exact && above && below
    }

    /// Narrows the window once `side` has found a move worth `score`.
    fn tighten(&mut self, side: Side, score: S) {
        match side {
            Side::Max => self.alpha = Some(self.alpha.map_or(score, |alpha| alpha.max(score))),
            Side::Min => self.beta = Some(self.beta.map_or(score, |beta| beta.min(score))),
        }
    }
}

/// What a search learned about one position; its principal variation sits in the
/// searcher's line buffer for its ply.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Node<S> {
    /// The score, from [`Side::Max`]'s point of view.
    pub(crate) score: S,
    /// How [`Node::score`] relates to the position's minimax value.
    pub(crate) bound: Bound,
    /// The moves of the position left unsearched.
    skipped: usize,
    /// Whether no line below the position ran into the depth limit.
    proven: bool,
    /// The fewest plies left at which a search of the position runs into no depth limit.
    reach: u32,
}

impl<S> Node<S> {
    /// A position where the game ends, scored by [`Game::evaluate`].
    const fn leaf(score: S) -> Self {
        Self {
            score,
            bound: Bound::Exact,
            skipped: 0,
            proven: true,
            reach: 1,
        }
    }

    /// A position where the depth runs out, scored by [`Game::evaluate`].
    const fn horizon(score: S) -> Self {
        Self {
            score,
            bound: Bound::Exact,
            skipped: 0,
            proven: false,
            reach: 0,
        }
    }

    /// A position answered by a transposition table entry.
    fn stored<M>(entry: &Entry<M, S>) -> Self
    where
        S: Copy,
    {
        Self {
            score: entry.score,
            bound: entry.bound,
            skipped: 0,
            proven: entry.proven,
            reach: entry.reach,
        }
    }
}

/// Unit tests for the searcher's hint handling.
#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::{Frame, Run, Window, answer};
    use crate::table::Entry;
    use crate::{Bound, Game, Searcher, Side, Stats};

    /// A game whose root offers moves `1` and `2`, each ending the game.
    struct TwoMoves;

    impl Game for TwoMoves {
        /// `true` at the root, `false` once a move was played.
        type Position = bool;
        /// The move played.
        type Move = u8;
        /// Always `0`.
        type Score = i32;

        fn side_to_move(&self, _: &bool) -> Side {
            Side::Max
        }

        fn moves(&self, &root: &bool, moves: &mut Vec<u8>) {
            if root {
                moves.extend([1, 2]);
            }
        }

        fn play(&self, _: &bool, _: &u8) -> bool {
            false
        }

        fn evaluate(&self, _: &bool) -> i32 {
            0
        }
    }

    /// Searches the root of [`TwoMoves`] to depth 2 with `hint` as the previous line.
    fn search_with_hint(hint: Vec<u8>) -> Option<u8> {
        let mut searcher = Searcher::<TwoMoves>::new();
        searcher.hint = hint;
        let mut unlimited = |_: &Stats| false;
        let mut run = Run {
            game: &TwoMoves,
            budget: &mut unlimited,
        };

        let node = searcher.iterate(&mut run, &true, 2, Window::FULL);
        searcher.result(&node, 2).best().copied()
    }

    /// Approximate entries, entries from another depth and bounds inside the window answer
    /// nothing; bounds beyond the window answer.
    #[test]
    fn unusable_entries_answer_nothing() {
        let entry = |bound, depth| Entry::<u8, i32> {
            depth,
            score: 3,
            bound,
            best: None,
            proven: false,
            reach: 0,
        };
        let frame = Frame {
            ply: 1,
            depth: 2,
            window: Window {
                alpha: Some(0),
                beta: Some(5),
            },
            on_hint: false,
        };

        assert!(answer(&entry(Bound::Approximate, 2), frame).is_none());
        assert!(answer(&entry(Bound::Exact, 3), frame).is_none());
        assert!(answer(&entry(Bound::Lower, 2), frame).is_none());
        assert!(answer(&entry(Bound::Upper, 2), frame).is_none());
        let beyond = Frame {
            window: Window {
                alpha: Some(3),
                beta: Some(3),
            },
            ..frame
        };
        assert!(answer(&entry(Bound::Lower, 2), beyond).is_some());
        assert!(answer(&entry(Bound::Upper, 2), beyond).is_some());
    }

    /// Principal variation search scouts a later child with a null window, and re-searches
    /// it when the scout beats the edge, but searches the first child with the full window
    /// right away.
    #[test]
    fn pvs_scouts_only_later_children() {
        let mut searcher = Searcher::<TwoMoves>::new().with_algorithm(crate::Algorithm::Pvs);
        let mut unlimited = |_: &Stats| false;
        let mut run = Run {
            game: &TwoMoves,
            budget: &mut unlimited,
        };
        let frame = Frame {
            ply: 1,
            depth: 1,
            window: Window {
                alpha: Some(-1),
                beta: Some(1),
            },
            on_hint: false,
        };

        let first = searcher.child(&mut run, &true, frame, (Side::Max, false));
        assert_eq!(
            (first.score, first.bound, searcher.stats.researches),
            (0, Bound::Exact, 0)
        );

        let later = searcher.child(&mut run, &true, frame, (Side::Max, true));
        assert_eq!(
            (later.score, later.bound, searcher.stats.researches),
            (0, Bound::Exact, 1)
        );
    }

    /// A searcher that budgets its first iteration stops at once on a spent budget, as a
    /// Lazy SMP helper must when the main thread is done.
    #[test]
    fn a_budgeted_first_iteration_stops_at_once() {
        let mut searcher = Searcher::<TwoMoves>::new();
        searcher.budget_first = true;
        let result = searcher.deepen(&TwoMoves, &true, 5, |_: &Stats| true);
        assert_eq!((result.stats.nodes, result.pv.len()), (0, 0));
    }

    /// A legal hint is searched first and wins the tie; an illegal one is ignored.
    #[test]
    fn only_legal_hints_are_tried_first() {
        assert_eq!(search_with_hint(vec![2]), Some(2));
        assert_eq!(search_with_hint(vec![9]), Some(1));
        assert_eq!(search_with_hint(Vec::new()), Some(1));
    }
}
