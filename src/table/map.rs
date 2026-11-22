//! [`MapTable`]: a hash map of entries with a size cap (feature `hashbrown`).

use core::hash::Hash;

use hashbrown::HashMap;

use super::{Entry, Table};

/// A transposition table backed by a hash map, holding up to a fixed number of entries.
///
/// Unlike [`SlotTable`](crate::SlotTable), keys never share a slot, so an entry stays until
/// [`MapTable::clear`]. Once the map is full, entries for new keys are dropped, while known
/// keys are still updated. Keys need [`Hash`], and memory grows with the entries stored.
///
/// Available with the `hashbrown` feature.
///
/// # Examples
///
/// ```
/// use game_tree_search::{MapTable, Table};
///
/// let mut table = MapTable::<u64, u8, i32>::new(1);
/// assert!(table.is_empty());
/// assert_eq!(table.capacity(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct MapTable<K, M, S> {
    /// The entries by key; never more than [`MapTable::capacity`].
    entries: HashMap<K, Entry<M, S>>,
    /// The most entries kept.
    capacity: usize,
}

impl<K: Eq + Hash, M: Clone, S: Clone> MapTable<K, M, S> {
    /// An empty table that keeps up to `capacity` entries.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity,
        }
    }

    /// The most entries the table keeps.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// The number of entries stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Removes every entry, for example before searching a different game.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl<K: Eq + Hash, M: Clone, S: Clone> Table<K, M, S> for MapTable<K, M, S> {
    fn probe(&mut self, key: &K) -> Option<Entry<M, S>> {
        self.entries.get(key).cloned()
    }

    fn store(&mut self, key: K, entry: Entry<M, S>) {
        if self.entries.len() < self.capacity || self.entries.contains_key(&key) {
            self.entries.insert(key, entry);
        }
    }
}

/// Unit tests for the size cap.
#[cfg(test)]
mod tests {
    use super::{MapTable, Table};
    use crate::Bound;
    use crate::table::Entry;

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

    /// A full table drops new keys but still updates known ones.
    #[test]
    fn a_full_table_keeps_known_keys_only() {
        let mut table = MapTable::new(1);
        table.store(1_u64, entry(10));
        table.store(2_u64, entry(20));
        table.store(1_u64, entry(11));
        assert_eq!(table.len(), 1);
        assert_eq!(table.probe(&1).map(|entry| entry.score), Some(11));
        assert_eq!(table.probe(&2), None);

        assert_eq!(table.capacity(), 1);
        assert!(!table.is_empty());

        table.clear();
        assert!(table.is_empty());
    }
}
