use serde::{Deserialize, Serialize};

pub mod meta_prompt;
pub mod pruner;
pub mod deduplication;

pub use meta_prompt::*;
pub use pruner::*;
pub use deduplication::*;

/// Prompt generator - creates context-aware prompts for LLMs
pub struct PromptGenerator;

impl PromptGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate a complete prompt with context
    pub fn generate(&self, request: &PromptRequest) -> GeneratedPrompt {
        let system_prompt = self.build_system_prompt(&request.intent);
        let context_block = self.build_context_block(&request.context);
        let user_prompt = self.build_user_prompt(&request.original_prompt, &request.context);
        let implementation_plan = request
            .implementation_plan
            .clone()
            .unwrap_or_else(|| self.build_implementation_plan(&request.context, &request.intent));
        let full_prompt = self.combine_all(
            &system_prompt,
            &context_block,
            &user_prompt,
            &implementation_plan,
        );

        GeneratedPrompt {
            system_prompt,
            context_block,
            user_prompt,
            implementation_plan,
            full_prompt,
        }
    }

    fn build_system_prompt(&self, intent: &str) -> String {
        let base = r#"You are an expert software engineer with deep knowledge of the codebase.
You have been provided with comprehensive context about existing code, components, utilities, and design patterns.

CRITICAL INSTRUCTIONS:
1. ALWAYS use existing components, utilities, and helpers when available
2. ALWAYS follow the existing code style and patterns
3. ALWAYS reuse design tokens (colors, spacing, etc.) from the codebase
4. DO NOT create new components if similar ones exist
5. DO NOT hardcode values that are available as constants or design tokens
"#;

        let intent_specific = match intent {
            "CreateComponent" => "\n6. When creating components, check for similar existing components and reuse their patterns\n7. Use the same prop patterns and naming conventions as existing components",
            "CreateFunction" => "\n6. Check for existing utility functions that solve similar problems\n7. Follow the same function signature patterns",
            "CreatePage" => "\n6. Reuse existing layout components and page structures\n7. Follow the same routing and navigation patterns",
            _ => "",
        };

        format!("{}{}", base, intent_specific)
    }

    fn build_context_block(&self, context: &ContextData) -> String {
        let mut blocks = Vec::new();

        // Add relevant symbols
        if !context.relevant_symbols.is_empty() {
            blocks.push("## Relevant Existing Code\n".to_string());
            for symbol in &context.relevant_symbols {
                blocks.push(format!(
                    "### {} ({})\n**File:** {}\n**Lines:** {}-{}\n```\n{}\n```\n",
                    symbol.name,
                    symbol.kind,
                    symbol.file_path,
                    symbol.start_line,
                    symbol.end_line,
                    symbol.content
                ));
            }
        }

        // Add similar symbols
        if !context.similar_symbols.is_empty() {
            blocks.push("\n## Similar Existing Patterns\n".to_string());
            for symbol in &context.similar_symbols {
                blocks.push(format!(
                    "### {} ({})\n**File:** {}\n```\n{}\n```\n",
                    symbol.name, symbol.kind, symbol.file_path, symbol.content
                ));
            }
        }

        // Add design tokens
        if !context.design_tokens.is_empty() {
            blocks.push("\n## Design Tokens & Styles\n".to_string());
            for token in &context.design_tokens {
                blocks.push(format!(
                    "- **{}**: `{}` ({})\n",
                    token.name, token.value, token.token_type
                ));
            }
        }

        // Add types
        if !context.types.is_empty() {
            blocks.push("\n## Type Definitions\n".to_string());
            for type_info in &context.types {
                blocks.push(format!(
                    "### {} ({})\n```typescript\n{}\n```\n",
                    type_info.name, type_info.kind, type_info.definition
                ));
            }
        }

        // Add constants
        if !context.constants.is_empty() {
            blocks.push("\n## Constants & Configuration\n".to_string());
            for constant in &context.constants {
                blocks.push(format!(
                    "- **{}** ({}) = `{}`\n",
                    constant.name, constant.category, constant.value
                ));
            }
        }

        // Add schemas
        if !context.schemas.is_empty() {
            blocks.push("\n## Validation Schemas\n".to_string());
            for schema in &context.schemas {
                blocks.push(format!(
                    "### {} ({})\n```typescript\n{}\n```\n",
                    schema.name, schema.schema_type, schema.definition
                ));
            }
        }

        // Add imports
        if !context.common_imports.is_empty() {
            blocks.push("\n## Common Imports\n".to_string());
            for import in &context.common_imports {
                blocks.push(format!("- {}\n", import));
            }
        }

        blocks.join("\n")
    }

    fn build_user_prompt(&self, original: &str, context: &ContextData) -> String {
        let mut prompt = format!("## User Request\n{}\n\n", original);

        if !context.relevant_symbols.is_empty() {
            prompt.push_str("## Available Resources\n");
            prompt.push_str("The following components/functions are available for use:\n");
            for symbol in &context.relevant_symbols {
                prompt.push_str(&format!("- `{}` ({})\n", symbol.name, symbol.kind));
            }
            prompt.push('\n');
        }

        prompt
    }

    fn build_implementation_plan(&self, context: &ContextData, intent: &str) -> String {
        let mut plan = String::from("## Suggested Implementation Plan\n\n");

        match intent {
            "CreateComponent" => {
                plan.push_str("1. Review similar existing components for patterns\n");
                plan.push_str("2. Identify reusable sub-components\n");
                plan.push_str("3. Use existing design tokens for styling\n");
                plan.push_str("4. Follow the same prop patterns as similar components\n");
                plan.push_str("5. Add proper TypeScript types\n");
            }
            "CreateFunction" => {
                plan.push_str("1. Check if a similar utility function exists\n");
                plan.push_str("2. Follow the same function signature patterns\n");
                plan.push_str("3. Add proper type annotations\n");
                plan.push_str("4. Include error handling\n");
            }
            "CreatePage" => {
                plan.push_str("1. Reuse existing layout components\n");
                plan.push_str("2. Follow the same page structure pattern\n");
                plan.push_str("3. Use existing components for UI elements\n");
                plan.push_str("4. Apply consistent styling with design tokens\n");
            }
            _ => {
                plan.push_str("1. Review the provided context\n");
                plan.push_str("2. Identify reusable patterns and components\n");
                plan.push_str("3. Follow existing code conventions\n");
            }
        }

        if !context.relevant_symbols.is_empty() {
            plan.push_str("\n### Components/Functions to Reuse:\n");
            for symbol in &context.relevant_symbols {
                plan.push_str(&format!("- `{}`\n", symbol.name));
            }
        }

        plan
    }

    fn combine_all(&self, system: &str, context: &str, user: &str, plan: &str) -> String {
        format!(
            "{}\n\n---\n\n{}\n\n---\n\n{}\n\n---\n\n{}",
            system, context, user, plan
        )
    }
}

impl Default for PromptGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptRequest {
    pub original_prompt: String,
    pub intent: String,
    pub context: ContextData,
    pub implementation_plan: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextData {
    pub relevant_symbols: Vec<SymbolInfo>,
    pub similar_symbols: Vec<SymbolInfo>,
    pub design_tokens: Vec<DesignTokenInfo>,
    pub common_imports: Vec<String>,
    pub types: Vec<TypeInfo>,
    pub constants: Vec<ConstantInfo>,
    pub schemas: Vec<SchemaInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInfo {
    pub name: String,
    pub kind: String,
    pub definition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantInfo {
    pub name: String,
    pub value: String,
    pub category: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaInfo {
    pub name: String,
    pub schema_type: String,
    pub definition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInfo {
    pub name: String,
    pub kind: String,
    pub content: String,
    pub file_path: String,
    pub start_line: i64,
    pub end_line: i64,
    #[serde(default)]
    pub props: Vec<String>,
    #[serde(default)]
    pub references: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignTokenInfo {
    pub name: String,
    pub value: String,
    pub token_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedPrompt {
    pub system_prompt: String,
    pub context_block: String,
    pub user_prompt: String,
    pub implementation_plan: String,
    pub full_prompt: String,
}
