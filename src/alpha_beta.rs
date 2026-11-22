//! Alpha-beta pruning: minimax that skips moves which cannot change the decision.

use alloc::vec::Vec;

use search_core::Budget;

use crate::minimax::improves;
use crate::table::{Entry, Memory};
use crate::{Bound, Game, ScoredMove, SearchResult, Searcher, Side, Stats};

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
/// Allocates its working buffers on every call; [`Searcher::alpha_beta`] reuses them across
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
    Searcher::new().alpha_beta(game, position, depth)
}

impl<G: Game, T: Memory<G>> Searcher<G, T> {
    /// Searches `position` up to `depth` plies with fail-soft alpha-beta pruning, reusing
    /// this searcher's buffers.
    ///
    /// Returns what [`alpha_beta()`](crate::alpha_beta()) returns. Once the buffers have grown
    /// to the search depth, a search allocates only the [`SearchResult`] it returns.
    #[must_use]
    pub fn alpha_beta(
        &mut self,
        game: &G,
        position: &G::Position,
        depth: u32,
    ) -> SearchResult<G::Move, G::Score> {
        self.stats = Stats::ZERO;
        self.hint.clear();
        let mut unlimited = |_: &Stats| false;
        let mut run = Run {
            game,
            budget: &mut unlimited,
        };

        self.iterate(&mut run, position, depth)
    }

    /// One alpha-beta search of `position` to `depth`, trying the hint's moves first.
    ///
    /// Adds to the searcher's counters instead of resetting them. If the budget stopped the
    /// search, the searcher's abort flag is set and the result must be discarded.
    pub(crate) fn iterate<B: Budget<Stats>>(
        &mut self,
        run: &mut Run<'_, G, B>,
        position: &G::Position,
        depth: u32,
    ) -> SearchResult<G::Move, G::Score> {
        self.aborted = false;
        let mut root_moves = Vec::new();
        let frame = Frame {
            ply: 0,
            depth,
            window: Window::FULL,
            on_hint: true,
        };
        let node = self.search(run, position, frame, Some(&mut root_moves));

        SearchResult {
            score: node.score,
            bound: node.bound,
            pv: self.line(0).to_vec(),
            root_moves,
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
    fn search<B: Budget<Stats>>(
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
            self.stats.horizon += 1;
            self.leave(frame.ply, moves);
            return Node::horizon(game.evaluate(position));
        }

        game.moves(position, &mut moves);
        if moves.is_empty() {
            self.leave(frame.ply, moves);
            return Node::leaf(game.evaluate(position));
        }

        game.order_moves(position, &mut moves);
        let key = self.table.key(game, position);
        let stored = self.table.recall(&key);
        let mut frame = frame;
        if let Some(node) = stored.as_ref().and_then(|entry| answer(entry, &mut frame)) {
            self.stats.table_hits += 1;
            self.leave(frame.ply, moves);
            return node;
        }

        if let Some(best) = stored.and_then(|entry| entry.best) {
            promote(&mut moves, &best);
        }

        let hint = self.promote_hint(&mut moves, frame);
        let node = self.expand(run, position, frame, (moves, hint), root_moves);
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

    /// Moves the hint's move for this ply to the front of `moves`, if the line to the
    /// position follows the hint and the move is legal here; returns it.
    fn promote_hint(&self, moves: &mut [G::Move], frame: Frame<G::Score>) -> Option<G::Move> {
        let hint = self.hint.get(frame.ply).filter(|_| frame.on_hint)?;

        promote(moves, hint).then(|| hint.clone())
    }

    /// Searches the `moves` of the position at `frame`, `hint` among them, until one of them
    /// stops the search there.
    fn expand<B: Budget<Stats>>(
        &mut self,
        run: &mut Run<'_, G, B>,
        position: &G::Position,
        frame: Frame<G::Score>,
        (mut moves, hint): (Vec<G::Move>, Option<G::Move>),
        mut root_moves: Option<&mut Vec<ScoredMove<G::Move, G::Score>>>,
    ) -> Node<G::Score> {
        let game = run.game;
        let side = game.side_to_move(position);
        let total = moves.len();
        let mut window = frame.window;
        let mut best: Option<Node<G::Score>> = None;
        let (mut widened, mut proven, mut searched, mut reach) = (false, true, 0, 0);
        for mv in moves.drain(..) {
            if best.is_some_and(|best| stops(game, side, best.score, &window)) {
                break;
            }

            searched += 1;
            let child_frame = Frame {
                ply: frame.ply + 1,
                depth: frame.depth - 1,
                window,
                on_hint: frame.on_hint && hint.as_ref() == Some(&mv),
            };
            let child = self.search(run, &game.play(position, &mv), child_frame, None);
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

/// The node `entry` answers at `frame` without a search, narrowing `frame`'s window by the
/// entry's bound otherwise; `None` if the position must be searched.
///
/// The root is always searched, so it can report every move it tries.
fn answer<M, S: Ord + Copy>(entry: &Entry<M, S>, frame: &mut Frame<S>) -> Option<Node<S>> {
    if frame.ply == 0 || !entry.serves(frame.depth) {
        return None;
    }

    match entry.bound {
        Bound::Exact => return Some(Node::stored(entry)),
        Bound::Lower => frame.window.tighten(Side::Max, entry.score),
        Bound::Upper => frame.window.tighten(Side::Min, entry.score),
        Bound::Approximate => {}
    }

    frame.window.is_empty().then(|| Node::stored(entry))
}

/// Whether `side`, holding `score`, skips its remaining moves: a window cutoff, or the game
/// is satisfied.
fn stops<G: Game>(game: &G, side: Side, score: G::Score, window: &Window<G::Score>) -> bool {
    window.closes(side, score) || game.satisfices(side, score)
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

/// The scores still worth telling apart at a position, from [`Side::Max`]'s point of view.
///
/// `None` is unbounded on that side.
#[derive(Debug, Clone, Copy)]
struct Window<S> {
    /// The score [`Side::Max`] is already sure of elsewhere.
    alpha: Option<S>,
    /// The score [`Side::Min`] is already sure of elsewhere.
    beta: Option<S>,
}

impl<S: Ord + Copy> Window<S> {
    /// The window that tells every score apart.
    const FULL: Self = Self {
        alpha: None,
        beta: None,
    };

    /// Whether `side`, holding `score`, can no longer affect the decision above.
    fn closes(&self, side: Side, score: S) -> bool {
        match side {
            Side::Max => self.beta.is_some_and(|beta| score >= beta),
            Side::Min => self.alpha.is_some_and(|alpha| score <= alpha),
        }
    }

    /// Whether no score lies strictly inside the window any more.
    fn is_empty(&self) -> bool {
        self.alpha
            .zip(self.beta)
            .is_some_and(|(alpha, beta)| alpha >= beta)
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
struct Node<S> {
    /// The score, from [`Side::Max`]'s point of view.
    score: S,
    /// How [`Node::score`] relates to the position's minimax value.
    bound: Bound,
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

        searcher.iterate(&mut run, &true, 2).best().copied()
    }

    /// An approximate entry answers nothing and leaves the window alone; an exact one from
    /// another depth is not used.
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
        let mut frame = Frame {
            ply: 1,
            depth: 2,
            window: Window::FULL,
            on_hint: false,
        };

        assert!(answer(&entry(Bound::Approximate, 2), &mut frame).is_none());
        assert!(answer(&entry(Bound::Exact, 3), &mut frame).is_none());
        assert!(frame.window.alpha.is_none() && frame.window.beta.is_none());
    }

    /// A legal hint is searched first and wins the tie; an illegal one is ignored.
    #[test]
    fn only_legal_hints_are_tried_first() {
        assert_eq!(search_with_hint(vec![2]), Some(2));
        assert_eq!(search_with_hint(vec![9]), Some(1));
        assert_eq!(search_with_hint(Vec::new()), Some(1));
    }
}
