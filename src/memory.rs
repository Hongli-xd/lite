//! Memory backends for zeroclaw-lite.

use crate::traits::{Memory, MemoryCategory, MemoryEntry};
use async_trait::async_trait;

/// No-op memory backend — returns empty results for all operations.
/// Useful for single-shot agents or constrained environments where
/// persistence is not needed.
pub struct NoneMemory;

impl NoneMemory {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoneMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Memory for NoneMemory {
    fn name(&self) -> &str {
        "none"
    }

    async fn store(&self, _key: &str, _content: &str, _category: MemoryCategory, _session_id: Option<&str>) -> anyhow::Result<()> {
        Ok(())
    }

    async fn recall(&self, _query: &str, _limit: usize, _session_id: Option<&str>, _since: Option<&str>, _until: Option<&str>) -> anyhow::Result<Vec<MemoryEntry>> {
        Ok(Vec::new())
    }

    async fn get(&self, _key: &str) -> anyhow::Result<Option<MemoryEntry>> {
        Ok(None)
    }

    async fn health_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn none_memory_is_healthy() {
        let mem = NoneMemory::new();
        assert!(mem.health_check().await);
        assert_eq!(mem.name(), "none");
    }

    #[tokio::test]
    async fn none_memory_store_is_noop() {
        let mem = NoneMemory::new();
        mem.store("key", "value", MemoryCategory::Core, None).await.unwrap();
    }

    #[tokio::test]
    async fn none_memory_recall_is_empty() {
        let mem = NoneMemory::new();
        let results = mem.recall("query", 10, None, None, None).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn none_memory_get_is_none() {
        let mem = NoneMemory::new();
        let result = mem.get("key").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn none_memory_count_is_zero() {
        let mem = NoneMemory::new();
        assert_eq!(mem.count().await.unwrap(), 0);
    }
}
