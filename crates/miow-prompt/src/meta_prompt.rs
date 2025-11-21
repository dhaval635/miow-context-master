use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{ConstantInfo, ContextData, SchemaInfo, SymbolInfo, TypeInfo};

/// Simple token counter (rough approximation: 1 token ≈ 4 characters)
struct TokenCounter;

impl TokenCounter {
    fn count(text: &str) -> usize {
        // Rough approximation: 4 characters per token
        (text.len() + 3) / 4
    }

    fn count_context(context: &ContextData) -> usize {
        let mut total = 0;

        for symbol in &context.relevant_symbols {
            total += Self::count(&symbol.content);
            total += Self::count(&symbol.name);
        }

        for type_info in &context.types {
            total += Self::count(&type_info.definition);
            total += Self::count(&type_info.name);
        }

        for constant in &context.constants {
            total += Self::count(&constant.value);
            total += Self::count(&constant.name);
        }

        for schema in &context.schemas {
            total += Self::count(&schema.definition);
            total += Self::count(&schema.name);
        }

        for token in &context.design_tokens {
            total += Self::count(&token.value);
            total += Self::count(&token.name);
        }

        total
    }
}

/// Meta-prompt generator - creates comprehensive, copy-paste ready prompts
pub struct MetaPromptGenerator;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaPromptConfig {
    pub include_full_code: bool,
    pub include_style_guide: bool,
    pub include_implementation_plan: bool,
    pub max_examples_per_type: usize,
    pub token_budget: Option<usize>,
}

impl Default for MetaPromptConfig {
    fn default() -> Self {
        Self {
            include_full_code: true,
            include_style_guide: true,
            include_implementation_plan: true,
            max_examples_per_type: 5,
            token_budget: Some(16000),
        }
    }
}

impl MetaPromptGenerator {
    /// Generate a comprehensive meta-prompt
    pub fn generate(
        user_request: &str,
        context: &ContextData,
        project_info: Option<&str>,
        config: MetaPromptConfig,
    ) -> Result<String> {
        let mut prompt = String::new();

        // ===== HEADER =====
        prompt.push_str(&Self::build_header(user_request, project_info));

        // ===== FILE STRUCTURE =====
        prompt.push_str(&build_file_structure(context));

        // ===== RELEVANT CODEBASE =====
        let context_section = build_relevant_codebase(context, &config);
        prompt.push_str(&context_section);

        // ===== CONSTRAINTS =====
        prompt.push_str(&Self::build_constraints());

        // ===== STYLE GUIDE =====
        if config.include_style_guide {
            prompt.push_str(&build_style_guide(context));
        }
        
        // ===== IMPLEMENTATION PLAN =====
        if config.include_implementation_plan {
            prompt.push_str(&build_implementation_plan(user_request, context));
        }
        
        // ===== EXECUTION INSTRUCTIONS =====
        prompt.push_str(&build_execution_instructions());
        
        Ok(prompt)
    }
    
    fn build_header(user_request: &str, project_info: Option<&str>) -> String {
        let mut header = format!("# TASK: {}\n\n", user_request);

        // Project Context section
        header.push_str("# Project Context\n");
        if let Some(info) = project_info {
            // Extract language and framework from project info
            header.push_str(&Self::format_project_context(info));
        } else {
            header.push_str("Language: Unknown\nFramework: Unknown\n");
        }
        header.push('\n');

        header
    }

    fn format_project_context(project_info: &str) -> String {
        // Try to extract language and framework from the project info
        let language = if project_info.to_lowercase().contains("typescript") {
            "TypeScript"
        } else if project_info.to_lowercase().contains("rust") {
            "Rust"
        } else if project_info.to_lowercase().contains("python") {
            "Python"
        } else {
            "Unknown"
        };

        let framework = if project_info.to_lowercase().contains("next.js") {
            "Next.js"
        } else if project_info.to_lowercase().contains("react") {
            "React"
        } else if project_info.to_lowercase().contains("nest") {
            "NestJS"
        } else if project_info.to_lowercase().contains("axum") {
            "Axum"
        } else if project_info.to_lowercase().contains("django") {
            "Django"
        } else {
            "Unknown"
        };

        format!("Language: {}\nFramework: {}", language, framework)
    }
    
    fn build_constraints() -> String {
        r#"## CONSTRAINTS ⚠️

> **CRITICAL RULES - MUST FOLLOW**
> 
> 1. **REUSE EXISTING CODE**: You MUST use the existing code provided below. Do NOT create new implementations if they already exist.
> 2. **FOLLOW STYLE**: Match the exact coding style, naming conventions, and patterns shown in the existing codebase.
> 3. **USE DESIGN TOKENS**: Use the exact color values, spacing, and design tokens defined below. Do NOT use arbitrary values.
> 4. **IMPORT CORRECTLY**: Use the exact import paths shown for each component/utility.
> 5. **TYPE SAFETY**: Use the existing type definitions. Do NOT create duplicate types.
> 6. **NO HALLUCINATIONS**: Do NOT invent new helper functions or components that are not in the context.
> 7. **CONSISTENCY**: Ensure your code is indistinguishable from the existing codebase.

"#.to_string()
    }
    
    fn build_existing_context(context: &ContextData, config: &MetaPromptConfig) -> String {
        // This method is deprecated, use build_relevant_codebase instead
        build_relevant_codebase(context, config)
    }
    }

    fn build_design_tokens_section(context: &ContextData) -> String {
        let mut section = String::from("### Design Tokens 🎨\n\nUse these EXACT values for styling:\n\n");

        let colors: Vec<_> = context.design_tokens.iter()
            .filter(|t| t.token_type.contains("Color"))
            .take(10)
            .collect();
        let spacing: Vec<_> = context.design_tokens.iter()
            .filter(|t| t.token_type.contains("Spacing") || t.token_type.contains("Size"))
            .take(10)
            .collect();

        if !colors.is_empty() {
            section.push_str("**Colors:**\n");
            for color in colors {
                section.push_str(&format!("- `{}`: `{}`\n", color.name, color.value));
            }
            section.push('\n');
        }

        if !spacing.is_empty() {
            section.push_str("**Spacing/Sizes:**\n");
            for space in spacing {
                section.push_str(&format!("- `{}`: `{}`\n", space.name, space.value));
            }
            section.push('\n');
        }

        section
    }

    fn build_constants_section(context: &ContextData, config: &MetaPromptConfig) -> String {
        let mut section = String::from("### Constants & Configuration\n\n");
        for (i, constant) in context.constants.iter().take(config.max_examples_per_type).enumerate() {
            section.push_str(&format_constant(constant, i + 1));
        }
        section.push('\n');
        section
    }

    fn build_types_section_with_budget(context: &ContextData, config: &MetaPromptConfig, token_budget: usize) -> String {
        let mut section = String::from("### Type Definitions\n\n");
        let mut used_tokens = TokenCounter::count(&section);

        for (i, type_info) in context.types.iter().enumerate() {
            if i >= config.max_examples_per_type {
                break;
            }

            let formatted = format_type(type_info, i + 1);
            let type_tokens = TokenCounter::count(&formatted);

            if used_tokens + type_tokens > token_budget {
                section.push_str(&format!("... ({} more types omitted due to token limit)\n\n", context.types.len() - i));
                break;
            }

            section.push_str(&formatted);
            used_tokens += type_tokens;
        }

        section
    }

    fn build_components_section_with_budget(context: &ContextData, config: &MetaPromptConfig, token_budget: usize) -> String {
        let mut section = String::from("### Components & Functions\n\n");
        let mut used_tokens = TokenCounter::count(&section);

        for (i, symbol) in context.relevant_symbols.iter().enumerate() {
            if i >= config.max_examples_per_type {
                break;
            }

            let formatted = format_symbol(symbol, i + 1);
            let symbol_tokens = TokenCounter::count(&formatted);

            if used_tokens + symbol_tokens > token_budget {
                section.push_str(&format!("... ({} more components omitted due to token limit)\n\n", context.relevant_symbols.len() - i));
                break;
            }

            section.push_str(&formatted);
            used_tokens += symbol_tokens;
        }

        section
    }

    fn build_schemas_section_with_budget(context: &ContextData, config: &MetaPromptConfig, token_budget: usize) -> String {
        let mut section = String::from("### Validation Schemas\n\n");
        let mut used_tokens = TokenCounter::count(&section);

        for (i, schema) in context.schemas.iter().enumerate() {
            if i >= config.max_examples_per_type {
                break;
            }

            let formatted = format_schema(schema, i + 1);
            let schema_tokens = TokenCounter::count(&formatted);

            if used_tokens + schema_tokens > token_budget {
                section.push_str(&format!("... ({} more schemas omitted due to token limit)\n\n", context.schemas.len() - i));
                break;
            }

            section.push_str(&formatted);
            used_tokens += schema_tokens;
        }

        section
    }

    fn summarize_design_tokens(context: &ContextData) -> String {
        let color_count = context.design_tokens.iter()
            .filter(|t| t.token_type.contains("Color"))
            .count();
        let spacing_count = context.design_tokens.iter()
            .filter(|t| t.token_type.contains("Spacing") || t.token_type.contains("Size"))
            .count();

        format!("### Design Tokens 🎨\n\nAvailable: {} colors, {} spacing tokens. Use existing design system patterns.\n\n", color_count, spacing_count)
    }

    fn summarize_constants(context: &ContextData) -> String {
        format!("### Constants & Configuration\n\n{} configuration constants available. Follow existing patterns.\n\n", context.constants.len())
    }

    fn build_file_structure(context: &ContextData) -> String {
        let mut structure = String::from("# File Structure\n");

        // Extract unique directories from file paths
        let mut directories = std::collections::HashSet::new();

        for symbol in &context.relevant_symbols {
            if let Some(dir) = std::path::Path::new(&symbol.file_path).parent() {
                let dir_str = dir.to_string_lossy().to_string();
                if !dir_str.is_empty() {
                    directories.insert(dir_str);
                }
            }
        }

        for _type_info in &context.types {
            // Types don't have file paths in current structure, skip
        }

        for _schema in &context.schemas {
            // Schemas don't have file paths in current structure, skip
        }

        if directories.is_empty() {
            structure.push_str("src/\n");
        } else {
            let mut sorted_dirs: Vec<String> = directories.into_iter().collect();
            sorted_dirs.sort();
            for dir in sorted_dirs {
                structure.push_str(&format!("{}/\n", dir));
            }
        }

        structure.push('\n');
        structure
    }

    fn build_relevant_codebase(context: &ContextData, config: &MetaPromptConfig) -> String {
        let mut codebase = String::from("# Relevant Codebase\n");

        // Build with token budget if specified
        let content = if let Some(token_budget) = config.token_budget {
            let current_tokens = TokenCounter::count(&codebase);
            let remaining_budget = token_budget.saturating_sub(current_tokens);
            build_relevant_codebase_with_budget(context, config, remaining_budget)
        } else {
            build_relevant_codebase_unlimited(context, config)
        };

        codebase.push_str(&content);
        codebase
    }

    fn build_relevant_codebase_unlimited(context: &ContextData, config: &MetaPromptConfig) -> String {
        let mut content = String::new();

        // Components/Symbols - use relevant_symbols
        if !context.relevant_symbols.is_empty() {
            for symbol in context.relevant_symbols.iter().take(config.max_examples_per_type) {
                content.push_str(&format!("## File: {}\n", symbol.file_path));
                content.push_str(&format!("```\n{}\n```\n\n", symbol.content));
            }
        }

        // Types
        if !context.types.is_empty() {
            for type_info in context.types.iter().take(config.max_examples_per_type) {
                // Since types don't have file paths, create a generic one
                content.push_str("## File: types/definitions.ts\n");
                content.push_str(&format!("```\n{}\n```\n\n", type_info.definition));
            }
        }

        // Schemas
        if !context.schemas.is_empty() {
            for schema in context.schemas.iter().take(config.max_examples_per_type) {
                // Since schemas don't have file paths, create a generic one
                content.push_str("## File: schemas/validation.ts\n");
                content.push_str(&format!("```\n{}\n```\n\n", schema.definition));
            }
        }

        content
    }

    fn build_relevant_codebase_with_budget(context: &ContextData, config: &MetaPromptConfig, token_budget: usize) -> String {
        let mut content = String::new();
        let mut used_tokens = 0;

        // Priority: Components first (most important), then Types, then Schemas

        // Components (highest priority)
        if !context.relevant_symbols.is_empty() && used_tokens < token_budget {
            for (i, symbol) in context.relevant_symbols.iter().enumerate() {
                if i >= config.max_examples_per_type {
                    break;
                }

                let formatted = format!("## File: {}\n```\n{}\n```\n\n", symbol.file_path, symbol.content);
                let symbol_tokens = TokenCounter::count(&formatted);

                if used_tokens + symbol_tokens > token_budget {
                    content.push_str(&format!("... ({} more files omitted due to token limit)\n\n", context.relevant_symbols.len() - i));
                    break;
                }

                content.push_str(&formatted);
                used_tokens += symbol_tokens;
            }
        }

        // Types (medium priority)
        if !context.types.is_empty() && used_tokens < token_budget {
            let _remaining_budget = token_budget - used_tokens;
            let mut type_count = 0;

            for type_info in context.types.iter() {
                if type_count >= config.max_examples_per_type {
                    break;
                }

                let formatted = format!("## File: types/definitions.ts\n```\n{}\n```\n\n", type_info.definition);
                let type_tokens = TokenCounter::count(&formatted);

                if used_tokens + type_tokens > token_budget {
                    content.push_str(&format!("... ({} more type definitions omitted due to token limit)\n\n", context.types.len() - type_count));
                    break;
                }

                content.push_str(&formatted);
                used_tokens += type_tokens;
                type_count += 1;
            }
        }

        // Schemas (lowest priority)
        if !context.schemas.is_empty() && used_tokens < token_budget {
            let _remaining_budget = token_budget - used_tokens;
            let mut schema_count = 0;

            for schema in context.schemas.iter() {
                if schema_count >= config.max_examples_per_type {
                    break;
                }

                let formatted = format!("## File: schemas/validation.ts\n```\n{}\n```\n\n", schema.definition);
                let schema_tokens = TokenCounter::count(&formatted);

                if used_tokens + schema_tokens > token_budget {
                    content.push_str(&format!("... ({} more schemas omitted due to token limit)\n\n", context.schemas.len() - schema_count));
                    break;
                }

                content.push_str(&formatted);
                used_tokens += schema_tokens;
                schema_count += 1;
            }
        }

        content
    }
    
    pub(crate) fn format_symbol(symbol: &SymbolInfo, index: usize) -> String {
        let mut info = format!(
            "#### {}. `{}` ({})\n\
            **File**: `{}`\n",
            index, symbol.name, symbol.kind, symbol.file_path
        );

        if !symbol.props.is_empty() {
            info.push_str(&format!("**Props**: {}\n", symbol.props.join(", ")));
        }

        if !symbol.references.is_empty() {
            // Limit references to avoid noise
            let refs: Vec<_> = symbol.references.iter().take(10).collect();
            info.push_str(&format!("**References**: {}\n", refs.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")));
            if symbol.references.len() > 10 {
                info.push_str("... (more)\n");
            }
        }

        info.push_str(&format!("\n```\n{}\n```\n\n", symbol.content));
        info
    }
    
    fn format_type(type_info: &TypeInfo, index: usize) -> String {
        format!(
            "#### {}. `{}`\n\n\
            ```\n{}\n```\n\n",
            index, type_info.name, type_info.definition
        )
    }
    
    fn format_constant(constant: &ConstantInfo, index: usize) -> String {
        format!(
            "{}. `{}` = `{}` (Category: {})\n",
            index, constant.name, constant.value, constant.category
        )
    }
    
    fn format_schema(schema: &SchemaInfo, index: usize) -> String {
        format!(
            "#### {}. `{}` ({})\n\n\
            ```\n{}\n```\n\n",
            index, schema.name, schema.schema_type, schema.definition
        )
    }
    
    fn build_style_guide(context: &ContextData) -> String {
        let mut guide = String::from("## STYLE GUIDE 📐\n\n");
        guide.push_str("Extracted patterns from the existing codebase. Your code MUST match this style:\n\n");
        
        // Extract patterns from symbols
        let mut patterns = vec![];
        
        // Detect common patterns from relevant_symbols
        let has_hooks = context.relevant_symbols.iter().any(|s| s.content.contains("useState") || s.content.contains("useEffect"));
        let has_arrow_functions = context.relevant_symbols.iter().any(|s| s.content.contains("=>"));
        let has_result_type = context.relevant_symbols.iter().any(|s| s.content.contains("Result<"));
        let has_option_type = context.relevant_symbols.iter().any(|s| s.content.contains("Option<"));
        
        if has_hooks {
            patterns.push("- **React Hooks**: Use functional components with hooks (useState, useEffect, etc.)");
        }
        if has_arrow_functions {
            patterns.push("- **Arrow Functions**: Use ES6 arrow function syntax `() => {}`");
        }
        if has_result_type {
            patterns.push("- **Error Handling (Rust)**: Use `Result<T, E>` for error handling");
        }
        if has_option_type {
            patterns.push("- **Null Safety (Rust)**: Use `Option<T>` instead of null");
        }
        
        // Naming convention detection
        let camel_case = context.relevant_symbols.iter().any(|s| {
            s.name.chars().next().map(|c| c.is_lowercase()).unwrap_or(false) && s.name.contains(char::is_uppercase)
        });
        let pascal_case = context.relevant_symbols.iter().any(|s| {
            s.name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
        });
        let snake_case = context.relevant_symbols.iter().any(|s| s.name.contains('_'));
        
        if camel_case {
            patterns.push("- **Naming**: camelCase for variables and functions");
        }
        if pascal_case {
            patterns.push("- **Naming**: PascalCase for components and types");
        }
        if snake_case {
            patterns.push("- **Naming**: snake_case detected (Rust/Python style)");
        }
        
        if patterns.is_empty() {
            guide.push_str("*(No specific style patterns detected)*\n\n");
        } else {
            for pattern in patterns {
                guide.push_str(&format!("{}\n", pattern));
            }
            guide.push('\n');
        }
        
        guide
    }
    
    fn build_implementation_plan(user_request: &str, context: &ContextData) -> String {
        let mut plan = String::from("## IMPLEMENTATION PLAN 📋\n\n");
        plan.push_str("Follow these steps in order:\n\n");
        
        let mut step = 1;
        
        // Step 1: Imports
        if !context.relevant_symbols.is_empty() || !context.types.is_empty() {
            plan.push_str(&format!(
                "{}. **Import existing code**\n   - Import the components, types, and utilities listed above\n   - Use the exact file paths shown\n\n",
                step
            ));
            step += 1;
        }
        
        // Step 2: Types
        if !context.types.is_empty() {
            plan.push_str(&format!(
                "{}. **Use existing types**\n   - Do NOT create duplicate type definitions\n   - Extend existing types if needed\n\n",
                step
            ));
            step += 1;
        }
        
        // Step 3: Main implementation
        plan.push_str(&format!(
            "{}. **Implement {}\n**   - Follow the examples from existing code\n   - Reuse helper functions and utilities\n   - Match the coding style shown above\n\n",
            step, user_request
        ));
        step += 1;
        
        // Step 4: Styling
        if !context.design_tokens.is_empty() {
            plan.push_str(&format!(
                "{}. **Apply styling**\n   - Use design tokens listed above\n   - Do NOT use hardcoded values\n\n",
                step
            ));
            step += 1;
        }
        
        // Step 5: Validation
        if !context.schemas.is_empty() {
            plan.push_str(&format!(
                "{}. **Add validation**\n   - Use existing validation schemas\n   - Follow the same schema patterns\n\n",
                step
            ));
            step += 1;
        }
        
        plan.push_str(&format!(
            "{}. **Test and verify**\n   - Ensure imports work\n   - Check type safety\n   - Verify styling matches design tokens\n\n",
            step
        ));
        
        plan
    }
    
    fn build_execution_instructions() -> String {
        r#"## EXECUTION INSTRUCTIONS 🚀

You are now ready to implement the requested feature. Remember:

1. **START** by importing the existing code shown above
2. **REUSE** components, utilities, and helpers instead of creating new ones
3. **FOLLOW** the exact style and patterns from the existing codebase
4. **USE** the design tokens for all styling (colors, spacing, etc.)
5. **REFER** back to the code examples above when in doubt

Begin your implementation now. Write complete, production-ready code that integrates seamlessly with the existing codebase.

"#.to_string()
    }


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_meta_prompt_generation() {
        let context = ContextData {
            relevant_symbols: vec![],
            similar_symbols: vec![],
            types: vec![],
            constants: vec![],
            design_tokens: vec![],
            schemas: vec![],
            common_imports: vec![],
        };
        
        let config = MetaPromptConfig::default();
        let prompt = MetaPromptGenerator::generate(
            "Create a login page",
            &context,
            Some("TypeScript + React"),
            config,
        ).unwrap();
        
        assert!(prompt.contains("# TASK"));
        assert!(prompt.contains("CONSTRAINTS"));
    }

    #[test]
    fn test_format_symbol_with_metadata() {
        let symbol = SymbolInfo {
            name: "TestComponent".to_string(),
            kind: "component".to_string(),
            content: "function TestComponent() {}".to_string(),
            file_path: "src/components/TestComponent.tsx".to_string(),
            start_line: 1,
            end_line: 1,
            props: vec!["title: string".to_string(), "isActive: boolean".to_string()],
            references: vec!["Button".to_string(), "useState".to_string()],
        };

        let formatted = format_symbol(&symbol, 1);
        
        assert!(formatted.contains("#### 1. `TestComponent` (component)"));
        assert!(formatted.contains("**File**: `src/components/TestComponent.tsx`"));
        assert!(formatted.contains("**Props**: title: string, isActive: boolean"));
        assert!(formatted.contains("**References**: Button, useState"));
        assert!(formatted.contains("```\nfunction TestComponent() {}\n```"));
    }
}
