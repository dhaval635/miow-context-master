use anyhow::{Context, Result, anyhow};
use miow_llm::LLMProvider;
use miow_graph::KnowledgeGraph;
use miow_vector::VectorStore;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{info, debug, error};
use async_trait::async_trait;
use crate::tools::{Tool, ToolRegistry, ViewFileTool, ListDirTool, RunCommandTool, WriteFileTool};

/// An autonomous agent that iteratively gathers context and solves tasks using tools
pub struct AutonomousAgent {
    llm: Arc<dyn LLMProvider>,
    tools: ToolRegistry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentContext {
    pub task: String,
    pub gathered_info: Vec<VerifiedInfo>,
    pub history: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedInfo {
    pub content: String,
    pub source: String,
    pub relevance: String,
}

use tokio::sync::mpsc::Sender;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum AgentEvent {
    Step { step: usize, max_steps: usize },
    Thought { content: String },
    ToolCall { tool: String, args: serde_json::Value },
    ToolOutput { output: String },
    Error { error: String },
    Done,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action")]
pub enum AgentAction {
    #[serde(rename = "use_tool")]
    UseTool {
        tool: String,
        args: serde_json::Value,
        reason: String,
    },
    #[serde(rename = "done")]
    Done,
}

impl AutonomousAgent {
    pub fn new(
        llm: Arc<dyn LLMProvider>, 
        graph: Arc<KnowledgeGraph>, 
        vector_store: Option<Arc<VectorStore>>
    ) -> Self {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(ViewFileTool));
        registry.register(Arc::new(ListDirTool));
        registry.register(Arc::new(RunCommandTool));
        registry.register(Arc::new(WriteFileTool));
        registry.register(Arc::new(SearchTool::new(graph, vector_store)));

        Self { llm, tools: registry }
    }

    /// Run the autonomous loop to gather context and solve the task
    pub async fn run(&self, task: &str, event_tx: Option<Sender<AgentEvent>>) -> Result<AgentContext> {
        let mut context = AgentContext {
            task: task.to_string(),
            gathered_info: Vec::new(),
            history: Vec::new(),
        };

        info!("🚀 Starting Autonomous Agent Loop for task: {}", task);

        let max_steps = 15;
        for step in 0..max_steps {
            info!("🔄 Step {}/{}", step + 1, max_steps);
            if let Some(tx) = &event_tx {
                let _ = tx.send(AgentEvent::Step { step: step + 1, max_steps }).await;
            }
            
            // 1. Decide next action
            let action = self.decide_next_step(&context).await?;
            info!("🤖 Decision: {:?}", action);

            match action {
                AgentAction::UseTool { tool, args, reason } => {
                    context.history.push(format!("Action: UseTool {} (Reason: {})", tool, reason));
                    if let Some(tx) = &event_tx {
                        let _ = tx.send(AgentEvent::Thought { content: format!("Decided to use tool '{}' because: {}", tool, reason) }).await;
                        let _ = tx.send(AgentEvent::ToolCall { tool: tool.clone(), args: args.clone() }).await;
                    }
                    
                    if let Some(tool_impl) = self.tools.get(&tool) {
                        info!("   🔨 Executing tool: {}", tool);
                        match tool_impl.execute(args.clone()).await {
                            Ok(output) => {
                                info!("   ✅ Tool success");
                                if let Some(tx) = &event_tx {
                                    let _ = tx.send(AgentEvent::ToolOutput { output: output.chars().take(1000).collect() }).await;
                                }
                                context.history.push(format!("Output: {}", output.chars().take(500).collect::<String>()));
                                
                                // If it was a search or read, add to gathered info
                                if tool == "search" || tool == "view_file" {
                                    context.gathered_info.push(VerifiedInfo {
                                        content: output,
                                        source: format!("Tool: {} Args: {}", tool, args),
                                        relevance: reason,
                                    });
                                }
                            },
                            Err(e) => {
                                error!("   ❌ Tool failed: {}", e);
                                if let Some(tx) = &event_tx {
                                    let _ = tx.send(AgentEvent::Error { error: e.to_string() }).await;
                                }
                                context.history.push(format!("Error: {}", e));
                            }
                        }
                    } else {
                        error!("   ❌ Tool not found: {}", tool);
                        context.history.push(format!("Error: Tool '{}' not found", tool));
                    }
                },
                AgentAction::Done => {
                    info!("✅ Agent decided it is done.");
                    if let Some(tx) = &event_tx {
                        let _ = tx.send(AgentEvent::Done).await;
                    }
                    break;
                }
            }
        }

        Ok(context)
    }

    async fn decide_next_step(&self, context: &AgentContext) -> Result<AgentAction> {
        let tools_schema = serde_json::to_string_pretty(&self.tools.list_tools())?;
        
        let prompt = format!(
            r#"You are an Autonomous Context Engine. Your goal is to build a perfect context for the user's task.

Task: "{}"

Available Tools:
{}

Current Context:
{}

History of Actions:
{}

Decide the next step.
- Use "search" to find relevant symbols.
- Use "list_dir" to explore the file structure.
- Use "view_file" to read file contents.
- Use "run_command" only if necessary (e.g. grep).
- Choose "done" when you have gathered sufficient information.

Respond with JSON ONLY:
{{
  "action": "use_tool",
  "tool": "tool_name",
  "args": {{ ... }},
  "reason": "why this is needed"
}}
OR
{{
  "action": "done"
}}
"#,
            context.task,
            tools_schema,
            self.format_gathered_info(&context.gathered_info),
            context.history.join("\n")
        );

        let response = self.llm.generate(&prompt).await?;
        let clean = self.clean_json(&response.content);
        
        serde_json::from_str(&clean).context("Failed to parse agent decision")
    }

    fn format_gathered_info(&self, info: &[VerifiedInfo]) -> String {
        if info.is_empty() {
            return "No information gathered yet.".to_string();
        }
        info.iter().enumerate()
            .map(|(i, info)| format!("#{}: {} (Source: {})", i, info.content.lines().next().unwrap_or(""), info.source))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn clean_json(&self, input: &str) -> String {
        input.trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
            .to_string()
    }
}

/// Tool to search the knowledge graph and vector store
pub struct SearchTool {
    graph: Arc<KnowledgeGraph>,
    vector_store: Option<Arc<VectorStore>>,
}

impl SearchTool {
    pub fn new(graph: Arc<KnowledgeGraph>, vector_store: Option<Arc<VectorStore>>) -> Self {
        Self { graph, vector_store }
    }
}

#[async_trait]
impl Tool for SearchTool {
    fn name(&self) -> &str { "search" }
    fn description(&self) -> &str { "Search for symbols in the codebase using graph and vector search" }
    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search query" }
            },
            "required": ["query"]
        })
    }
    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let query = args["query"].as_str().ok_or_else(|| anyhow!("Missing 'query' argument"))?;
        
        let mut results = Vec::new();

        // Vector search
        if let Some(vs) = &self.vector_store {
            if let Ok(vec_results) = vs.search_similar(query, 5).await {
                for vr in vec_results {
                    if let Ok(symbols) = self.graph.find_symbols_by_name(&vr.symbol.name) {
                        results.extend(symbols);
                    }
                }
            }
        }

        // Graph search
        if let Ok(graph_results) = self.graph.search_symbols(query) {
            results.extend(graph_results);
        }

        // Deduplicate
        results.sort_by(|a, b| a.file_path.cmp(&b.file_path));
        results.dedup_by(|a, b| a.name == b.name && a.file_path == b.file_path);

        if results.is_empty() {
            return Ok("No results found.".to_string());
        }

        let output = results.iter().take(5)
            .map(|r| format!("Symbol: {} ({})\nFile: {}\nContent:\n{}", r.name, r.kind, r.file_path, r.content))
            .collect::<Vec<_>>()
            .join("\n---\n");
            
        Ok(output)
    }
}
