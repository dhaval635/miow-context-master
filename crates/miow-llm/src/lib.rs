use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

mod gemini;
mod openai;
pub mod question_loop;
pub mod cache;

pub use gemini::GeminiClient;
pub use openai::OpenAIClient;
pub use question_loop::*;
pub use cache::LLMCache;

/// LLM provider trait
#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn generate(&self, prompt: &str) -> Result<LLMResponse>;
    async fn generate_with_context(&self, messages: Vec<Message>) -> Result<LLMResponse>;
    async fn stream_generate(
        &self,
        prompt: &str,
    ) -> Result<Box<dyn futures::Stream<Item = Result<String>> + Unpin>>;
    async fn generate_multi_step(&self, steps: Vec<String>, context: &str) -> Result<LLMResponse>;
    async fn generate_with_framework(&self, prompt: &str, framework: &str, lang: &str) -> Result<LLMResponse>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    pub content: String,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct LLMConfig {
    pub api_key: String,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: usize,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "gemini-2.5-flash".to_string(), // Using Gemini 2.5 Flash
            temperature: 0.7,
            max_tokens: 4096,
        }
    }
}

/// Interactive LLM for context gathering
pub struct InteractiveLLM {
    provider: Box<dyn LLMProvider>,
}

impl InteractiveLLM {
    pub fn new(provider: Box<dyn LLMProvider>) -> Self {
        Self { provider }
    }

    /// Analyze user intent and generate clarifying questions
    pub async fn analyze_intent(&self, user_prompt: &str) -> Result<IntentAnalysis> {
        let system_prompt = r#"You are an expert code analyst. Analyze the user's request and:
1. Identify the main intent (create component, create page, add feature, fix bug, etc.)
2. List what information is needed to complete this task
3. Generate 2-3 clarifying questions to gather missing context

Respond in JSON format:
{
  "intent": "create_component",
  "required_info": ["component type", "props needed", "styling approach"],
  "questions": ["What props should this component accept?", "Should it use existing design tokens?"]
}"#;

        let messages = vec![
            Message {
                role: Role::System,
                content: system_prompt.to_string(),
            },
            Message {
                role: Role::User,
                content: user_prompt.to_string(),
            },
        ];

        let response = self.provider.generate_with_context(messages).await?;
        let analysis: IntentAnalysis = serde_json::from_str(&response.content)?;

        Ok(analysis)
    }

    /// Generate search queries for vector database
    pub async fn generate_search_queries(
        &self,
        user_prompt: &str,
        intent: &str,
    ) -> Result<Vec<String>> {
        let system_prompt = format!(
            r#"Given the user's request and intent, generate 3-5 search queries to find relevant code.
Intent: {}
User request: {}

Generate queries that would find:
- Similar components/functions
- Helper utilities
- Type definitions
- Design tokens

Respond with a JSON array of strings."#,
            intent, user_prompt
        );

        let response = self.provider.generate(&system_prompt).await?;
        let queries: Vec<String> = serde_json::from_str(&response.content)?;

        Ok(queries)
    }

    /// Build comprehensive prompt with gathered context
    pub async fn build_comprehensive_prompt(
        &self,
        user_prompt: &str,
        context: &GatheredContext,
    ) -> Result<String> {
        let system_prompt = r#"You are an expert at creating comprehensive prompts for code generation.
Given the user's request and gathered context, create a detailed prompt that includes:
1. Clear instructions emphasizing reuse of existing code
2. All relevant components, helpers, and utilities
3. Design tokens and styling guidelines
4. Type definitions
5. Step-by-step implementation plan
6. Code examples from similar implementations

Make the prompt extremely detailed so no context is lost."#;

        let context_json = serde_json::to_string_pretty(context)?;
        let user_message = format!(
            "User request: {}\n\nGathered context:\n{}",
            user_prompt, context_json
        );

        let messages = vec![
            Message {
                role: Role::System,
                content: system_prompt.to_string(),
            },
            Message {
                role: Role::User,
                content: user_message,
            },
        ];

        let response = self.provider.generate_with_context(messages).await?;
        Ok(response.content)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentAnalysis {
    pub intent: String,
    pub required_info: Vec<String>,
    pub questions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatheredContext {
    pub components: Vec<ContextItem>,
    pub helpers: Vec<ContextItem>,
    pub types: Vec<ContextItem>,
    pub design_tokens: Vec<ContextItem>,
    pub constants: Vec<ContextItem>,
    pub schemas: Vec<ContextItem>,
    pub similar_implementations: Vec<ContextItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextItem {
    pub name: String,
    pub kind: String,
    pub content: String,
    pub file_path: String,
    pub relevance_score: f32,
    #[serde(default)]
    pub props: Vec<String>,
    #[serde(default)]
    pub references: Vec<String>,
}
