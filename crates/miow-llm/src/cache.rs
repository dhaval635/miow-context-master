use anyhow::Result;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use tokio::fs;
use tracing::{info, debug};

pub struct LLMCache {
    cache_dir: PathBuf,
}

impl LLMCache {
    pub fn new() -> Self {
        let cache_dir = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(".cache")
            .join("miow-llm");
        
        Self { cache_dir }
    }

    pub async fn init(&self) -> Result<()> {
        if !self.cache_dir.exists() {
            fs::create_dir_all(&self.cache_dir).await?;
        }
        Ok(())
    }

    fn get_cache_key(&self, prompt: &str, model: &str) -> String {
        let mut hasher = DefaultHasher::new();
        prompt.hash(&mut hasher);
        model.hash(&mut hasher);
        let hash = hasher.finish();
        format!("{}_{:x}.json", model, hash)
    }

    pub async fn get(&self, prompt: &str, model: &str) -> Option<String> {
        let key = self.get_cache_key(prompt, model);
        let path = self.cache_dir.join(key);

        if path.exists() {
            debug!("Cache hit for prompt hash");
            if let Ok(content) = fs::read_to_string(path).await {
                return Some(content);
            }
        }
        None
    }

    pub async fn set(&self, prompt: &str, model: &str, response: &str) -> Result<()> {
        let key = self.get_cache_key(prompt, model);
        let path = self.cache_dir.join(key);
        
        // Ensure dir exists (lazy init)
        if !self.cache_dir.exists() {
            fs::create_dir_all(&self.cache_dir).await?;
        }

        fs::write(path, response).await?;
        debug!("Cached response for prompt hash");
        Ok(())
    }
}
