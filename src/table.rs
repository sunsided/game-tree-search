//! Transposition tables: remembering what a search learned about a position, so the same
//! position reached by another move order is not searched again.
//!
//! A game opts in by implementing [`Hashed`]. A [`Searcher`](crate::Searcher) built with
//! [`Searcher::with_table`](crate::Searcher::with_table) then asks its [`Table`] before
//! searching a position and tells it the result afterwards. [`SlotTable`] is the default
//! table; `MapTable` (feature `hashbrown`) or any other type implementing [`Table`] can
//! replace it.

#[cfg(feature = "hashbrown")]
mod map;
#[cfg(feature = "parallel")]
mod packed;
#[cfg(feature = "parallel")]
mod shared;
mod slot;

#[cfg(feature = "hashbrown")]
pub use map::MapTable;
#[cfg(feature = "parallel")]
pub use packed::{Packed, PackedTable};
#[cfg(feature = "parallel")]
pub use shared::SharedTable;
pub use slot::SlotTable;

use crate::{Bound, Game};

/// A game whose positions have a hash key, so a transposition table can recognize them.
///
/// Two positions must get the same key exactly when they play out the same from here on:
/// the same side to move, the same moves and the same scores below. Keys that differ only
/// by chance (hash collisions) make a search reuse a wrong result, so use a well-mixed hash
/// of 64 or 128 bits.
pub trait Hashed: Game {
    /// The key type; `u64`, `u128` or a pair of `u64` halves.
    type Key: TableKey;

    /// The key of `position`.
    fn key(&self, position: &Self::Position) -> Self::Key;
}

/// A position key that a fixed-size table can place in a slot.
pub trait TableKey: Copy + Eq {
    /// 64 well-mixed bits of the key; a table uses the low bits to pick a slot.
    fn slot(&self) -> u64;
}

impl TableKey for u64 {
    fn slot(&self) -> u64 {
        *self
    }
}

impl TableKey for u128 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "folding 128 bits into 64 is the point"
    )]
    fn slot(&self) -> u64 {
        (*self ^ (*self >> 64)) as u64
    }
}

impl TableKey for (u64, u64) {
    fn slot(&self) -> u64 {
        self.0
    }
}

/// What a search learned about a position.
///
/// Non-exhaustive: tables store and return entries, only searches create them.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Entry<M, S> {
    /// The plies that were left to search below the position.
    pub depth: u32,
    /// The score found, from [`Side::Max`](crate::Side::Max)'s point of view.
    pub score: S,
    /// How [`Entry::score`] relates to the position's minimax value at [`Entry::depth`].
    pub bound: Bound,
    /// The best move found, tried first when the position is searched again.
    pub best: Option<M>,
    /// Whether no line below the position ran into the depth limit.
    pub proven: bool,
    /// For a [`proven`](Entry::proven) entry: the fewest plies left at which a search of the
    /// position runs into no depth limit, so the entry holds for any depth from there on.
    pub reach: u32,
}

impl<M, S> Entry<M, S> {
    /// Whether the entry answers a search of its position with `depth` plies left.
    ///
    /// Only an entry from the same depth, or a proven one deep enough, is used: an entry from
    /// a deeper search would give a different score than a search to `depth`.
    pub(crate) const fn serves(&self, depth: u32) -> bool {
        self.depth == depth || (self.proven && depth >= self.reach)
    }
}

/// Storage for [`Entry`] values by position key.
///
/// Implemented by [`SlotTable`] and, with the `hashbrown` feature, `MapTable`. Entries are
/// handed out by value, never by reference, so a table shared between threads can implement
/// this trait on a reference to itself.
pub trait Table<K, M, S> {
    /// The entry stored for `key`, if any.
    fn probe(&mut self, key: &K) -> Option<Entry<M, S>>;

    /// Stores `entry` for `key`; the table may keep or drop it, or evict another entry.
    fn store(&mut self, key: K, entry: Entry<M, S>);
}

/// No transposition table: every position is searched, nothing is remembered.
///
/// The default of [`Searcher`](crate::Searcher); works for every game, [`Hashed`] or not.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoTable;

/// How a [`Searcher`](crate::Searcher) remembers positions of a game `G`.
///
/// Implemented for [`NoTable`], which remembers nothing, and for [`Transpositions`], a
/// [`Table`] of a [`Hashed`] game; there is no need to implement it directly.
pub trait Memory<G: Game> {
    /// What identifies a position: the game's key, or nothing for [`NoTable`].
    type Key;

    /// The key of `position` in `game`.
    fn key(&self, game: &G, position: &G::Position) -> Self::Key;

    /// The entry remembered for `key`, if any.
    fn recall(&mut self, key: &Self::Key) -> Option<Entry<G::Move, G::Score>>;

    /// Remembers `entry` for `key`.
    fn remember(&mut self, key: Self::Key, entry: Entry<G::Move, G::Score>);
}

impl<G: Game> Memory<G> for NoTable {
    type Key = ();

    fn key(&self, _: &G, _: &G::Position) {}

    fn recall(&mut self, (): &()) -> Option<Entry<G::Move, G::Score>> {
        None
    }

    fn remember(&mut self, (): (), _: Entry<G::Move, G::Score>) {}
}

/// A [`Table`] serving as a searcher's [`Memory`]; built by
/// [`Searcher::with_table`](crate::Searcher::with_table).
#[derive(Debug, Clone, Default)]
pub struct Transpositions<T>(pub(crate) T);

impl<G: Hashed, T: Table<G::Key, G::Move, G::Score>> Memory<G> for Transpositions<T> {
    type Key = G::Key;

    fn key(&self, game: &G, position: &G::Position) -> G::Key {
        game.key(position)
    }

    fn recall(&mut self, key: &G::Key) -> Option<Entry<G::Move, G::Score>> {
        self.0.probe(key)
    }

    fn remember(&mut self, key: G::Key, entry: Entry<G::Move, G::Score>) {
        self.0.store(key, entry);
    }
}
