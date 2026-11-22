//! [`SlotTable`]: a fixed-size array of entries, one per slot.

use alloc::vec::Vec;
use core::iter;

use super::{Entry, Table, TableKey};

/// A transposition table with a fixed number of slots, each holding at most one entry.
///
/// A key's slot comes from the low bits of [`TableKey::slot`]. Storing always replaces what
/// the slot held, so memory stays fixed and recent positions win; a probe whose key differs
/// from the slot's misses.
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
/// use game_tree_search::{Hashed, Searcher, SlotTable, alpha_beta};
///
/// impl Hashed for Nim {
///     type Key = u64;
///
///     fn key(&self, &(stones, side): &Self::Position) -> u64 {
///         u64::from(stones) * 2 + u64::from(side == Side::Max)
///     }
/// }
///
/// // Many move orders lead to the same pile, so the table saves most of the work.
/// let mut searcher = Searcher::with_table(SlotTable::new(1024));
/// let remembered = searcher.alpha_beta(&Nim, &(20, Side::Max), 20);
/// let plain = alpha_beta(&Nim, &(20, Side::Max), 20);
/// assert_eq!(remembered.score, plain.score);
/// assert!(remembered.stats.nodes < plain.stats.nodes / 10);
/// ```
#[derive(Debug, Clone)]
pub struct SlotTable<K, M, S> {
    /// The slots; their number is a power of two.
    slots: Vec<Option<(K, Entry<M, S>)>>,
}

impl<K: TableKey, M: Clone, S: Clone> SlotTable<K, M, S> {
    /// A table with `capacity` slots, rounded up to a power of two (at least one).
    ///
    /// # Panics
    ///
    /// Panics if the rounded capacity does not fit in `usize`.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let slots = capacity
            .max(1)
            .checked_next_power_of_two()
            .expect("the capacity fits in usize when rounded up");

        Self {
            slots: iter::repeat_with(|| None).take(slots).collect(),
        }
    }

    /// The number of slots.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Empties every slot, for example before searching a different game.
    pub fn clear(&mut self) {
        self.slots.fill_with(|| None);
    }

    /// The slot of `key`: its low bits, as many as the slot count needs.
    fn index(&self, key: &K) -> usize {
        let mask = u64::try_from(self.slots.len() - 1).expect("a slot count fits in u64");
        usize::try_from(key.slot() & mask).expect("a masked slot is below the slot count")
    }
}

impl<K: TableKey, M: Clone, S: Clone> Table<K, M, S> for SlotTable<K, M, S> {
    fn probe(&mut self, key: &K) -> Option<Entry<M, S>> {
        let index = self.index(key);
        let (stored, entry) = self.slots[index].as_ref()?;

        (stored == key).then(|| entry.clone())
    }

    // shortcut: always replaces; upgrade to depth-preferred buckets if hit rates suffer.
    fn store(&mut self, key: K, entry: Entry<M, S>) {
        let index = self.index(&key);
        self.slots[index] = Some((key, entry));
    }
}

/// Unit tests for slot placement and replacement.
#[cfg(test)]
mod tests {
    use super::{SlotTable, Table};
    use crate::Bound;
    use crate::table::{Entry, TableKey};

    /// An exact entry of `score` at depth 1.
    fn entry(score: i32) -> Entry<u8, i32> {
        Entry {
            depth: 1,
            score,
            bound: Bound::Exact,
            best: None,
            proven: false,
            reach: 0,
        }
    }

    /// Capacities round up to a power of two, and zero becomes one slot.
    #[test]
    fn capacity_rounds_up() {
        assert_eq!(SlotTable::<u64, u8, i32>::new(0).capacity(), 1);
        assert_eq!(SlotTable::<u64, u8, i32>::new(5).capacity(), 8);
    }

    /// Two keys sharing a slot replace each other; the evicted one misses.
    #[test]
    fn keys_sharing_a_slot_replace_each_other() {
        let mut table = SlotTable::new(4);
        table.store(1_u64, entry(10));
        table.store(5_u64, entry(50));
        assert_eq!(table.probe(&1), None);
        assert_eq!(table.probe(&5).map(|entry| entry.score), Some(50));

        table.clear();
        assert_eq!(table.probe(&5), None);
    }

    /// Wide keys fold into 64 slot bits.
    #[test]
    fn wide_keys_fold_into_slot_bits() {
        assert_eq!((1_u128 << 64 | 3).slot(), 2);
        assert_eq!((7_u64, 9_u64).slot(), 7);
        assert_eq!(4_u64.slot(), 4);
    }
}
