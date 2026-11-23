//! Searching with make/unmake: [`Incremental`] games change one position in place along the
//! searched line instead of building a new position per move, and the handles the
//! alpha-beta family reads the current position from.

use search_core::Budget;

use crate::table::Memory;
use crate::{Deepening, Game, SearchResult, Searcher, Stats};

/// A game whose positions can be changed in place and changed back, so the alpha-beta
/// family can search with one position instead of one per move
/// ([`Searcher::search_in_place`], [`Searcher::deepen_in_place`]).
///
/// It pays off where copying a position is a large share of what a node costs. An
/// evaluation that a move changes only a little, such as an NNUE accumulator, does not need
/// it: see the "Incremental evaluation" section of [`Game`]'s documentation.
///
/// `unmake` must restore the position exactly, or later moves search a different
/// position than the game's [`Game::play`] would reach; the searches have no way to tell.
///
/// # Examples
///
/// ```
/// use game_tree_search::{Game, Incremental, Searcher, Side};
///
/// /// One-pile Nim: take one or two stones; whoever cannot move has lost.
/// struct Nim;
///
/// impl Game for Nim {
///     /// The stones left and the side to move.
///     type Position = (u32, Side);
///     type Move = u32;
///     type Score = i32;
///
///     fn side_to_move(&self, &(_, side): &Self::Position) -> Side {
///         side
///     }
///
///     fn moves(&self, &(stones, _): &Self::Position, moves: &mut Vec<u32>) {
///         moves.extend((1..=2).filter(|&take| take <= stones));
///     }
///
///     fn play(&self, position: &Self::Position, take: &u32) -> Self::Position {
///         let mut next = *position;
///         self.make(&mut next, take);
///         next
///     }
///
///     fn evaluate(&self, &(stones, side): &Self::Position) -> i32 {
///         match (stones, side) {
///             (0, Side::Max) => -1,
///             (0, Side::Min) => 1,
///             _ => 0,
///         }
///     }
/// }
///
/// impl Incremental for Nim {
///     /// Nothing: taking stones is undone by putting them back.
///     type Undo = ();
///
///     fn make(&self, (stones, side): &mut Self::Position, take: &u32) {
///         *stones -= take;
///         *side = if *side == Side::Max { Side::Min } else { Side::Max };
///     }
///
///     fn unmake(&self, (stones, side): &mut Self::Position, take: &u32, (): ()) {
///         *stones += take;
///         *side = if *side == Side::Max { Side::Min } else { Side::Max };
///     }
/// }
///
/// let root = (4, Side::Max);
/// let in_place = Searcher::new().search_in_place(&Nim, &root, 10);
/// assert_eq!(in_place, Searcher::new().search(&Nim, &root, 10));
/// assert_eq!(in_place.best(), Some(&1));
/// ```
pub trait Incremental: Game {
    /// What [`Incremental::make`] hands to [`Incremental::unmake`] to undo a move, such as a
    /// captured piece or the previous hash.
    type Undo;

    /// Plays `mv` on `position` in place, returning what undoing it needs.
    fn make(&self, position: &mut Self::Position, mv: &Self::Move) -> Self::Undo;

    /// Takes `mv` back, restoring `position` exactly as it was before
    /// [`Incremental::make`] returned `undo`.
    fn unmake(&self, position: &mut Self::Position, mv: &Self::Move, undo: Self::Undo);
}

/// Where a search reads the position it is at, and how it reaches a child.
pub(crate) trait Place<G: Game> {
    /// The handle of a child position.
    type Child: Place<G>;

    /// The position.
    fn position(&self) -> &G::Position;

    /// Runs `then` at the child `mv` leads to, and comes back.
    fn descend<R>(&mut self, game: &G, mv: &G::Move, then: impl FnOnce(&mut Self::Child) -> R)
    -> R;
}

/// The caller's root position, borrowed; its children are [`Owned`] copies.
pub(crate) struct Borrowed<'a, P>(pub(crate) &'a P);

/// A position the search built with [`Game::play`].
pub(crate) struct Owned<P>(P);

/// The one position an in-place search changes with [`Incremental::make`] and
/// [`Incremental::unmake`]: a clone of the root, so the caller's position stays untouched
/// even if the search panics.
pub(crate) struct InPlace<P>(pub(crate) P);

impl<G: Game> Place<G> for Borrowed<'_, G::Position> {
    type Child = Owned<G::Position>;

    #[inline]
    fn position(&self) -> &G::Position {
        self.0
    }

    #[inline]
    fn descend<R>(
        &mut self,
        game: &G,
        mv: &G::Move,
        then: impl FnOnce(&mut Owned<G::Position>) -> R,
    ) -> R {
        then(&mut Owned(game.play(self.0, mv)))
    }
}

impl<G: Game> Place<G> for Owned<G::Position> {
    type Child = Self;

    #[inline]
    fn position(&self) -> &G::Position {
        &self.0
    }

    #[inline]
    fn descend<R>(&mut self, game: &G, mv: &G::Move, then: impl FnOnce(&mut Self) -> R) -> R {
        then(&mut Self(game.play(&self.0, mv)))
    }
}

impl<G: Incremental> Place<G> for InPlace<G::Position> {
    type Child = Self;

    #[inline]
    fn position(&self) -> &G::Position {
        &self.0
    }

    #[inline]
    fn descend<R>(&mut self, game: &G, mv: &G::Move, then: impl FnOnce(&mut Self) -> R) -> R {
        let undo = game.make(&mut self.0, mv);
        let result = then(self);
        game.unmake(&mut self.0, mv, undo);

        result
    }
}

impl<G: Incremental, T: Memory<G>> Searcher<G, T>
where
    G::Position: Clone,
{
    /// Searches `position` like [`Searcher::search`], but along one clone of it that
    /// [`Incremental::make`] and [`Incremental::unmake`] change in place, instead of a new
    /// position per move.
    ///
    /// Returns what [`Searcher::search`] returns. `position` itself is never changed.
    #[must_use]
    pub fn search_in_place(
        &mut self,
        game: &G,
        position: &G::Position,
        depth: u32,
    ) -> SearchResult<G::Move, G::Score> {
        self.search_from(game, &mut InPlace(position.clone()), depth)
    }

    /// Deepens `position` like [`Searcher::deepen`], along one clone of it changed in
    /// place, as [`Searcher::search_in_place`] does.
    #[must_use]
    pub fn deepen_in_place(
        &mut self,
        game: &G,
        position: &G::Position,
        max_depth: u32,
        budget: impl Budget<Stats>,
    ) -> SearchResult<G::Move, G::Score> {
        let always = |_: &SearchResult<G::Move, G::Score>| true;
        self.deepen_in_place_with(game, position, max_depth, budget, always)
    }

    /// Deepens `position` like [`Searcher::deepen_with`], along one clone of it changed in
    /// place, as [`Searcher::search_in_place`] does.
    #[must_use]
    pub fn deepen_in_place_with(
        &mut self,
        game: &G,
        position: &G::Position,
        max_depth: u32,
        budget: impl Budget<Stats>,
        deepening: impl Deepening<G::Move, G::Score>,
    ) -> SearchResult<G::Move, G::Score> {
        let mut here = InPlace(position.clone());
        self.deepen_from(game, &mut here, max_depth, budget, deepening)
    }
}
