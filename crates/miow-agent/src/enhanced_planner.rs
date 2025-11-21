use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;

/// Generic LLM provider trait
#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn generate(&self, prompt: &str) -> Result<LLMResponse>;
}

#[derive(Debug, Clone)]
pub struct LLMResponse {
    pub content: String,
}

/// Enhanced execution plan with dependencies and fallbacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub goal: String,
    pub steps: Vec<PlanStep>,
    pub estimated_duration: u64, // seconds
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub id: String,
    pub description: String,
    pub tool: String,
    pub arguments: HashMap<String, String>,
    pub expected_output: String,
    pub dependencies: Vec<String>, // step IDs that must complete first
    pub fallback_steps: Vec<PlanStep>,
    pub timeout: u64, // seconds
    pub retries: u32,
}

/// Enhanced planner with dependency management
pub struct EnhancedPlanner {
    llm: Arc<dyn LLMProvider>,
}

impl EnhancedPlanner {
    pub fn new(llm: Arc<dyn LLMProvider>) -> Self {
        Self { llm }
    }
    
    /// Create a detailed execution plan
    pub async fn create_plan(&self, goal: &str, context: &str) -> Result<ExecutionPlan> {
        let prompt = self.build_planning_prompt(goal, context);
        let response = self.llm.generate(&prompt).await?;
        
        self.parse_plan(&response.content, goal)
    }
    
    fn build_planning_prompt(&self, goal: &str, context: &str) -> String {
        format!(
            r#"Create a detailed execution plan for this goal:

Goal: {}

Context:
{}

Create a plan with:
1. Clear, sequential steps
2. Tool to use for each step (search, analyze, generate, execute, verify)
3. Expected output for each step
4. Dependencies between steps (which steps must complete first)
5. Fallback steps if primary step fails
6. Estimated timeout for each step
7. Number of retries allowed

Respond with JSON:
{{
  "goal": "{}",
  "steps": [
    {{
      "id": "step_1",
      "description": "Clear description of what this step does",
      "tool": "search|analyze|generate|execute|verify",
      "arguments": {{"key": "value"}},
      "expected_output": "What we expect to get from this step",
      "dependencies": [],
      "fallback_steps": [
        {{
          "id": "step_1_fallback",
          "description": "Alternative approach if step_1 fails",
          "tool": "...",
          "arguments": {{}},
          "expected_output": "...",
          "dependencies": [],
          "fallback_steps": [],
          "timeout": 30,
          "retries": 1
        }}
      ],
      "timeout": 60,
      "retries": 2
    }}
  ],
  "estimated_duration": 300
}}

Guidelines:
- Keep steps atomic and focused
- Add dependencies to ensure correct order
- Provide meaningful fallbacks
- Set realistic timeouts
- Limit retries to avoid infinite loops

Respond ONLY with valid JSON."#,
            goal, context, goal
        )
    }
    
    fn parse_plan(&self, response: &str, goal: &str) -> Result<ExecutionPlan> {
        // Extract JSON from response
        let json_str = if let Some(start) = response.find('{') {
            if let Some(end) = response.rfind('}') {
                &response[start..=end]
            } else {
                response
            }
        } else {
            response
        };
        
        let mut plan: ExecutionPlan = serde_json::from_str(json_str)
            .unwrap_or_else(|_| ExecutionPlan {
                goal: goal.to_string(),
                steps: vec![],
                estimated_duration: 0,
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            });
        
        plan.created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        Ok(plan)
    }
    
    /// Validate plan for circular dependencies
    pub fn validate_plan(&self, plan: &ExecutionPlan) -> Result<()> {
        let mut visited = HashMap::new();
        
        for step in &plan.steps {
            if self.has_circular_dependency(step, &plan.steps, &mut visited)? {
                anyhow::bail!("Circular dependency detected in plan");
            }
        }
        
        Ok(())
    }
    
    fn has_circular_dependency(
        &self,
        step: &PlanStep,
        all_steps: &[PlanStep],
        visited: &mut HashMap<String, bool>,
    ) -> Result<bool> {
        if let Some(&in_progress) = visited.get(&step.id) {
            return Ok(in_progress);
        }
        
        visited.insert(step.id.clone(), true);
        
        for dep_id in &step.dependencies {
            if let Some(dep_step) = all_steps.iter().find(|s| &s.id == dep_id) {
                if self.has_circular_dependency(dep_step, all_steps, visited)? {
                    return Ok(true);
                }
            }
        }
        
        visited.insert(step.id.clone(), false);
        Ok(false)
    }
    
    /// Get steps that are ready to execute (dependencies met)
    pub fn get_ready_steps<'a>(
        &self,
        plan: &'a ExecutionPlan,
        completed_steps: &[String],
    ) -> Vec<&'a PlanStep> {
        plan.steps
            .iter()
            .filter(|step| {
                // Not already completed
                !completed_steps.contains(&step.id) &&
                // All dependencies completed
                step.dependencies.iter().all(|dep| completed_steps.contains(dep))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_circular_dependency_detection() {
        let planner = EnhancedPlanner {
            llm: Arc::new(MockLLM),
        };
        
        let plan = ExecutionPlan {
            goal: "test".to_string(),
            steps: vec![
                PlanStep {
                    id: "step_1".to_string(),
                    description: "Step 1".to_string(),
                    tool: "search".to_string(),
                    arguments: HashMap::new(),
                    expected_output: "output".to_string(),
                    dependencies: vec!["step_2".to_string()],
                    fallback_steps: vec![],
                    timeout: 60,
                    retries: 2,
                },
                PlanStep {
                    id: "step_2".to_string(),
                    description: "Step 2".to_string(),
                    tool: "analyze".to_string(),
                    arguments: HashMap::new(),
                    expected_output: "output".to_string(),
                    dependencies: vec!["step_1".to_string()],
                    fallback_steps: vec![],
                    timeout: 60,
                    retries: 2,
                },
            ],
            estimated_duration: 120,
            created_at: 0,
        };
        
        assert!(planner.validate_plan(&plan).is_err());
    }
    
    #[test]
    fn test_get_ready_steps() {
        let planner = EnhancedPlanner {
            llm: Arc::new(MockLLM),
        };
        
        let plan = ExecutionPlan {
            goal: "test".to_string(),
            steps: vec![
                PlanStep {
                    id: "step_1".to_string(),
                    description: "Step 1".to_string(),
                    tool: "search".to_string(),
                    arguments: HashMap::new(),
                    expected_output: "output".to_string(),
                    dependencies: vec![],
                    fallback_steps: vec![],
                    timeout: 60,
                    retries: 2,
                },
                PlanStep {
                    id: "step_2".to_string(),
                    description: "Step 2".to_string(),
                    tool: "analyze".to_string(),
                    arguments: HashMap::new(),
                    expected_output: "output".to_string(),
                    dependencies: vec!["step_1".to_string()],
                    fallback_steps: vec![],
                    timeout: 60,
                    retries: 2,
                },
            ],
            estimated_duration: 120,
            created_at: 0,
        };
        
        // Initially, only step_1 is ready
        let ready = planner.get_ready_steps(&plan, &[]);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "step_1");
        
        // After step_1 completes, step_2 is ready
        let ready = planner.get_ready_steps(&plan, &["step_1".to_string()]);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "step_2");
    }
    
    struct MockLLM;
    
    #[async_trait]
    impl LLMProvider for MockLLM {
        async fn generate(&self, _prompt: &str) -> Result<LLMResponse> {
            Ok(LLMResponse {
                content: r#"{"goal": "test", "steps": [], "estimated_duration": 0}"#.to_string(),
            })
        }
    }
}
