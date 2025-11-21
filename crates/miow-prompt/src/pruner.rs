use crate::ContextData;
use tracing::{info, debug};

/// Smart context pruner to manage token budget and relevance
pub struct SmartPruner {
    token_budget: usize,
}

impl SmartPruner {
    pub fn new(token_budget: usize) -> Self {
        Self { token_budget }
    }

    /// Prune context to fit within token budget
    pub fn prune(&self, context: &mut ContextData) {
        let current_usage = self.calculate_usage(context);
        
        if current_usage <= self.token_budget {
            debug!("Context usage {} within budget {}", current_usage, self.token_budget);
            return;
        }

        info!("✂️ Pruning context: usage {} > budget {}", current_usage, self.token_budget);

        // Strategy 1: Remove test files and mocks
        self.remove_test_files(context);
        
        if self.calculate_usage(context) <= self.token_budget {
            return;
        }

        // Strategy 2: Limit number of items per category
        self.limit_items(context);
        
        if self.calculate_usage(context) <= self.token_budget {
            return;
        }
        
        // Strategy 3: Truncate large content (keep signatures if possible)
        // For now, just remove lowest priority items
        self.aggressive_prune(context);
    }
    
    fn calculate_usage(&self, context: &ContextData) -> usize {
        let mut chars = 0;
        
        for s in &context.relevant_symbols { chars += s.content.len() + s.name.len(); }
        for s in &context.similar_symbols { chars += s.content.len() + s.name.len(); }
        for t in &context.types { chars += t.definition.len() + t.name.len(); }
        for c in &context.constants { chars += c.value.len() + c.name.len(); }
        for d in &context.design_tokens { chars += d.value.len() + d.name.len(); }
        for s in &context.schemas { chars += s.definition.len() + s.name.len(); }
        
        // Approx 4 chars per token
        chars / 4
    }
    
    fn remove_test_files(&self, context: &mut ContextData) {
        let is_test = |path: &str| {
            path.contains(".test.") || 
            path.contains(".spec.") || 
            path.contains("__tests__") ||
            path.contains("mock")
        };
        
        context.relevant_symbols.retain(|s| !is_test(&s.file_path));
        context.similar_symbols.retain(|s| !is_test(&s.file_path));

        // Constants and tokens usually don't have file paths in the same way or are less likely to be test-only
        // But if they do, filter them too
    }
    
    fn limit_items(&self, context: &mut ContextData) {
        // Keep top N items
        const MAX_ITEMS: usize = 10;
        
        if context.relevant_symbols.len() > MAX_ITEMS {
            context.relevant_symbols.truncate(MAX_ITEMS);
        }
        if context.similar_symbols.len() > MAX_ITEMS {
            context.similar_symbols.truncate(MAX_ITEMS);
        }
        if context.types.len() > MAX_ITEMS {
            context.types.truncate(MAX_ITEMS);
        }
        // ... others
    }
    
    fn aggressive_prune(&self, context: &mut ContextData) {
        // Graduated pruning strategy
        
        // 1. Reduce similar symbols (keep top 5)
        if context.similar_symbols.len() > 5 {
            context.similar_symbols.truncate(5);
            if self.calculate_usage(context) <= self.token_budget { return; }
        }

        // 2. Reduce constants (keep top 5)
        if context.constants.len() > 5 {
            context.constants.truncate(5);
            if self.calculate_usage(context) <= self.token_budget { return; }
        }

        // 3. Reduce design tokens (keep top 10)
        if context.design_tokens.len() > 10 {
            context.design_tokens.truncate(10);
            if self.calculate_usage(context) <= self.token_budget { return; }
        }

        // 4. Clear secondary categories if still over budget
        context.similar_symbols.clear();
        if self.calculate_usage(context) <= self.token_budget { return; }
        
        context.constants.clear();
        if self.calculate_usage(context) <= self.token_budget { return; }
        
        context.design_tokens.clear();
        if self.calculate_usage(context) <= self.token_budget { return; }
        
        // 5. Finally, prune relevant symbols from the end (assuming least relevant are at the end)
        // Note: In a real scenario, we should sort by relevance score first if not already sorted.
        while self.calculate_usage(context) > self.token_budget && !context.relevant_symbols.is_empty() {
            context.relevant_symbols.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContextData, SymbolInfo, ConstantInfo};

    #[test]
    fn test_graduated_pruning() {
        let mut context = ContextData {
            relevant_symbols: vec![],
            similar_symbols: vec![],
            types: vec![],
            constants: vec![],
            design_tokens: vec![],
            schemas: vec![],
            common_imports: vec![],
        };

        // Add 10 constants
        for i in 0..10 {
            context.constants.push(ConstantInfo {
                name: format!("CONST_{}", i),
                value: "value".to_string(),
                category: "test".to_string(),
            });
        }

        // Set a budget that allows ~5 constants but not 10
        // Each constant is roughly: "CONST_X" (7) + "value" (5) + "test" (4) = 16 chars / 4 = 4 tokens
        // 5 constants * 4 tokens = 20 tokens.
        // Let's set budget to 25.
        let pruner = SmartPruner::new(25); 
        pruner.prune(&mut context);

        // Should be reduced to 5, not 0
        assert_eq!(context.constants.len(), 5);
    }
}
