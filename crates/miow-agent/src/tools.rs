use async_trait::async_trait;
use anyhow::{Result, anyhow, Context};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::process::Command;
use tracing::{info, warn};

/// Trait defining a tool that the agent can use
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn schema(&self) -> serde_json::Value;
    async fn execute(&self, args: serde_json::Value) -> Result<String>;
}

/// Registry to hold available tools
pub struct ToolRegistry {
    tools: std::collections::HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: std::collections::HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn list_tools(&self) -> Vec<serde_json::Value> {
        self.tools.values().map(|t| {
            json!({
                "name": t.name(),
                "description": t.description(),
                "parameters": t.schema()
            })
        }).collect()
    }
}

// --- Tool Implementations ---

/// Tool to view file contents
pub struct ViewFileTool;

#[async_trait]
impl Tool for ViewFileTool {
    fn name(&self) -> &str { "view_file" }
    fn description(&self) -> &str { "Read the contents of a file" }
    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Absolute path to the file" }
            },
            "required": ["path"]
        })
    }
    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let path_str = args["path"].as_str().ok_or_else(|| anyhow!("Missing 'path' argument"))?;
        let path = PathBuf::from(path_str);
        
        if !path.exists() {
            return Err(anyhow!("File not found: {}", path_str));
        }

        let content = tokio::fs::read_to_string(&path).await
            .context(format!("Failed to read file: {}", path_str))?;
            
        Ok(content)
    }
}

/// Tool to list directory contents
pub struct ListDirTool;

#[async_trait]
impl Tool for ListDirTool {
    fn name(&self) -> &str { "list_dir" }
    fn description(&self) -> &str { "List contents of a directory" }
    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Absolute path to the directory" }
            },
            "required": ["path"]
        })
    }
    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let path_str = args["path"].as_str().ok_or_else(|| anyhow!("Missing 'path' argument"))?;
        let path = PathBuf::from(path_str);
        
        if !path.exists() {
            return Err(anyhow!("Directory not found: {}", path_str));
        }

        let mut entries = tokio::fs::read_dir(&path).await
            .context(format!("Failed to read directory: {}", path_str))?;
            
        let mut result = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let file_type = if entry.file_type().await?.is_dir() { "DIR" } else { "FILE" };
            result.push(format!("[{}] {}", file_type, entry.file_name().to_string_lossy()));
        }
        
        Ok(result.join("\n"))
    }
}

/// Tool to run shell commands
pub struct RunCommandTool;

#[async_trait]
impl Tool for RunCommandTool {
    fn name(&self) -> &str { "run_command" }
    fn description(&self) -> &str { "Execute a shell command" }
    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Command to execute" },
                "cwd": { "type": "string", "description": "Working directory" }
            },
            "required": ["command"]
        })
    }
    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let command_str = args["command"].as_str().ok_or_else(|| anyhow!("Missing 'command' argument"))?;
        let cwd = args["cwd"].as_str().unwrap_or(".");
        
        info!("Executing command: '{}' in '{}'", command_str, cwd);

        // Security check (basic)
        if command_str.contains("rm -rf /") || command_str.contains("mkfs") {
            return Err(anyhow!("Command blocked for security reasons"));
        }

        let output = Command::new("sh")
            .arg("-c")
            .arg(command_str)
            .current_dir(cwd)
            .output()
            .await
            .context("Failed to execute command")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        if output.status.success() {
            Ok(stdout.to_string())
        } else {
            Ok(format!("Command failed with code {:?}\nSTDOUT:\n{}\nSTDERR:\n{}", output.status.code(), stdout, stderr))
        }
    }
}

/// Tool to write/overwrite files
pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str { "write_file" }
    fn description(&self) -> &str { "Write content to a file (overwrites)" }
    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Absolute path to the file" },
                "content": { "type": "string", "description": "Content to write" }
            },
            "required": ["path", "content"]
        })
    }
    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let path_str = args["path"].as_str().ok_or_else(|| anyhow!("Missing 'path' argument"))?;
        let content = args["content"].as_str().ok_or_else(|| anyhow!("Missing 'content' argument"))?;
        let path = PathBuf::from(path_str);

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(&path, content).await
            .context(format!("Failed to write file: {}", path_str))?;
            
        Ok(format!("Successfully wrote to {}", path_str))
    }
}
