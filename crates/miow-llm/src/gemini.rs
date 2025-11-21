use crate::{LLMConfig, LLMProvider, LLMResponse, Message, Role, LLMCache};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::json;
use tracing::{debug, info, warn, error};
use tokio::time::{sleep, Duration};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub struct GeminiClient {
    api_key: String,
    model: String,
    temperature: f32,
    client: reqwest::Client,
    max_retries: u32,
    base_delay: Duration,
    cache: LLMCache,
}

impl GeminiClient {
    pub fn new(config: LLMConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            anyhow::bail!("Gemini API key is required");
        }

        Ok(Self {
            api_key: config.api_key,
            model: config.model,
            temperature: config.temperature,
            client: reqwest::Client::new(),
            max_retries: 5,
            base_delay: Duration::from_secs(2),
            cache: LLMCache::new(),
        })
    }

    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("GEMINI_API_KEY")
            .context("GEMINI_API_KEY environment variable not set")?;

        Self::new(LLMConfig {
            api_key,
            ..Default::default()
        })
    }

    fn generate_jitter(&self) -> Duration {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        let seed = now.as_nanos() as u64 % 1000;
        Duration::from_millis(seed)
    }

    async fn call_api(&self, messages: Vec<Message>) -> Result<String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );

        debug!("Calling Gemini API with model: {}", self.model);

        let mut contents = Vec::new();
        for message in messages {
            let role = match message.role {
                Role::System => "model",
                Role::User => "user",
                Role::Assistant => "model",
            };

            contents.push(json!({
                "role": role,
                "parts": [{
                    "text": message.content
                }]
            }));
        }

        let request_body = json!({
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "topK": 40,
                "topP": 0.95,
            }
        });

        let mut attempt = 0;

        while attempt <= self.max_retries {
            let start_time = Instant::now();
            let jitter = self.generate_jitter();

            match self.perform_api_call(&url, &request_body).await {
                Ok(response_text) => {
                    info!("Gemini API call successful on attempt {} (took {:?})", attempt + 1, start_time.elapsed());
                    return Ok(response_text);
                }
                Err(e) => {
                    attempt += 1;
                    warn!("Gemini API call failed on attempt {}: {}", attempt, e);

                    if attempt > self.max_retries {
                        error!("All {} retry attempts failed for Gemini API", self.max_retries);
                        return Err(e);
                    }

                    let backoff_delay = self.base_delay * 2u32.pow(attempt - 1);
                    let total_delay = backoff_delay + jitter;

                    warn!("Retrying in {:?} (attempt {}/{}, jitter: {:?})", total_delay, attempt, self.max_retries, jitter);
                    sleep(total_delay).await;
                }
            }
        }

        anyhow::bail!("Unexpected error after retries")
    }

    async fn perform_api_call(&self, url: &str, request_body: &serde_json::Value) -> Result<String> {
        let response = self
            .client
            .post(url)
            .json(request_body)
            .send()
            .await
            .context("Failed to send request to Gemini API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            
            if status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                anyhow::bail!("Gemini API error ({}): {}. This is retryable.", status, error_text);
            } else {
                anyhow::bail!("Gemini API error ({}): {}", status, error_text);
            }
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse Gemini API response")?;

        let text = response_json["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .context("Failed to extract text from Gemini response")?
            .to_string();

        Ok(text)
    }
}

#[async_trait]
impl LLMProvider for GeminiClient {
    async fn generate(&self, prompt: &str) -> Result<LLMResponse> {
        // Check cache first
        if let Some(cached) = self.cache.get(prompt, &self.model).await {
            info!("Returning cached response for prompt");
            return Ok(LLMResponse {
                content: cached,
                finish_reason: None,
                usage: None,
            });
        }

        info!("Generating response with Gemini");

        let messages = vec![Message {
            role: Role::User,
            content: prompt.to_string(),
        }];

        let text = self.call_api(messages).await?;

        // Cache the result
        if let Err(e) = self.cache.set(prompt, &self.model, &text).await {
            warn!("Failed to cache response: {}", e);
        }

        Ok(LLMResponse {
            content: text,
            finish_reason: None,
            usage: None,
        })
    }

    async fn generate_with_context(&self, messages: Vec<Message>) -> Result<LLMResponse> {
        info!("Generating response with Gemini (with context)");

        let text = self.call_api(messages).await?;

        Ok(LLMResponse {
            content: text,
            finish_reason: None,
            usage: None,
        })
    }

    async fn stream_generate(
        &self,
        _prompt: &str,
    ) -> Result<Box<dyn futures::Stream<Item = Result<String>> + Unpin>> {
        // TODO: Implement streaming for Gemini
        unimplemented!("Streaming not yet implemented for Gemini")
    }

    async fn generate_multi_step(&self, steps: Vec<String>, context: &str) -> Result<LLMResponse> {
        let mut final_content = String::new();

        for (i, step_prompt) in steps.iter().enumerate() {
            let full_prompt = format!("Step {}/{}: {}\nContext: {}", i + 1, steps.len(), step_prompt, context);
            let response = self.generate(&full_prompt).await?;
            final_content += &format!("Step {}: {}\n", i + 1, response.content);
        }

        Ok(LLMResponse {
            content: final_content,
            finish_reason: None,
            usage: None,
        })
    }

    async fn generate_with_framework(&self, prompt: &str, framework: &str, lang: &str) -> Result<LLMResponse> {
        let enhanced_prompt = format!(
            "You are an expert {} developer using {} framework.\n\n{}",
            lang, framework, prompt
        );
        self.generate(&enhanced_prompt).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires API key
    async fn test_gemini_client() {
        let client = GeminiClient::from_env().unwrap();
        let response = client.generate("Say hello!").await;
        assert!(response.is_ok());
    }
}
