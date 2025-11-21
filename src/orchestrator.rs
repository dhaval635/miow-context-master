use anyhow::Result;
use miow_analyzer::ContextAnalyzer;
use miow_agent::{AutonomousAgent, GeminiContextAuditor, GeminiRouterAgent, GeminiWorkerAgent, RouterAgent, SearchPlan, WorkerAgent};
use miow_core::{IntelligentSignatureDetector, ProjectSignature};
use miow_graph::{KnowledgeGraph, RelationshipInferencer};
use miow_llm::{ContextItem, GatheredContext, LLMProvider, LLMResponse, Message, Role};
use miow_prompt::{
    ConstantInfo, ContextData, DesignTokenInfo, PromptGenerator, PromptRequest, SchemaInfo,
    SymbolInfo, TypeInfo,
};
use miow_vector::VectorStore;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::Arc;
use tracing::{info, warn};

/// Orchestrator that ties together all the components with LLM-powered context gathering
#[allow(dead_code)]
pub struct MiowOrchestrator {
    graph: Arc<KnowledgeGraph>,
    analyzer: ContextAnalyzer,
    prompt_generator: PromptGenerator,
    llm: Option<Arc<dyn LLMProvider>>,
    vector_store: Option<Arc<VectorStore>>,
}

#[allow(dead_code)]
impl MiowOrchestrator {
    pub fn new(db_path: &str) -> Result<Self> {
        Ok(Self {
            graph: Arc::new(KnowledgeGraph::new(db_path)?),
            analyzer: ContextAnalyzer::new(),
            prompt_generator: PromptGenerator::new(),
            llm: None,
            vector_store: None,
        })
    }

    /// Create orchestrator with LLM provider
    pub fn with_llm(mut self, llm: Box<dyn LLMProvider>) -> Self {
        self.llm = Some(Arc::from(llm));
        self
    }

    /// Create orchestrator with shared LLM provider
    pub fn with_llm_arc(mut self, llm: Arc<dyn LLMProvider>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Attach a vector store for semantic search
    pub fn with_vector_store(mut self, store: Arc<VectorStore>) -> Self {
        self.vector_store = Some(store);
        self
    }

    /// Generate a context-aware prompt from a user request with advanced LLM-powered analysis
    pub async fn generate_context_prompt(&self, user_prompt: &str) -> Result<String> {
        info!("Generating context-aware prompt for: {}", user_prompt);

        let analyzed = self.analyzer.analyze_prompt(user_prompt);

        // Step 1: Analyze prompt with LLM if available, otherwise use basic analyzer
        let intent_analysis = if let Some(ref llm) = self.llm {
            // Create a new InteractiveLLM with the LLM provider
            // Note: We need to use the LLM directly since InteractiveLLM takes ownership
            let messages = vec![
                Message {
                    role: Role::System,
                    content: r#"You are an expert code analyst. Analyze the user's request and identify the main intent.
Respond with just the intent in one word or short phrase (e.g., "create_component", "create_page", "add_feature", "fix_bug")."#.to_string(),
                },
                Message {
                    role: Role::User,
                    content: user_prompt.to_string(),
                },
            ];

            match llm.generate_with_context(messages).await {
                Ok(response) => {
                    let intent = response.content.trim().to_string();
                    info!("LLM intent analysis: {}", intent);
                    intent
                }
                Err(e) => {
                    warn!(
                        "LLM intent analysis failed: {}, falling back to basic analyzer",
                        e
                    );
                    format!("{:?}", analyzed.intent)
                }
            }
        } else {
            format!("{:?}", analyzed.intent)
        };

        // Step 2: Generate search queries using LLM
        let search_queries = if let Some(ref llm) = self.llm {
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
                intent_analysis, user_prompt
            );

            match llm.generate(&system_prompt).await {
                Ok(response) => {
                    match serde_json::from_str::<Vec<String>>(&response.content) {
                        Ok(queries) => {
                            info!("Generated {} search queries", queries.len());
                            queries
                        }
                        Err(_) => {
                            // Fallback: extract queries from response text
                            let queries: Vec<String> = response
                                .content
                                .lines()
                                .filter_map(|line| {
                                    let line = line.trim();
                                    if line.starts_with('-')
                                        || line
                                            .chars()
                                            .next()
                                            .map(|c| c.is_ascii_digit())
                                            .unwrap_or(false)
                                    {
                                        Some(
                                            line.trim_start_matches(|c: char| {
                                                c.is_ascii_digit()
                                                    || c == '-'
                                                    || c == '.'
                                                    || c.is_whitespace()
                                            })
                                            .to_string(),
                                        )
                                    } else {
                                        None
                                    }
                                })
                                .take(5)
                                .collect();
                            if !queries.is_empty() {
                                queries
                            } else {
                                analyzed.keywords.clone()
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to generate search queries: {}, using keywords", e);
                    analyzed.keywords.clone()
                }
            }
        } else {
            analyzed.keywords.clone()
        };

        let refined_queries = self.refine_keywords(&search_queries);
        let search_queries = if refined_queries.is_empty() {
            if search_queries.is_empty() {
                vec![user_prompt.to_string()]
            } else {
                search_queries
            }
        } else {
            refined_queries
        };

        // Step 3: Analyze intent and keywords
        let _keyword_pool = {
            let mut pool = HashSet::new();
            pool.extend(analyzed.keywords.clone());
            // technical_terms field does not exist in AnalyzedPrompt
            pool
        };

        // Step 3: Gather comprehensive context using multiple search strategies
        let gathered_context = self
            .gather_comprehensive_context(user_prompt, &search_queries, &intent_analysis, None)
            .await?;

        // Step 4: Convert gathered context to prompt context format
        let master_context = self
            .convert_to_context_data(gathered_context, &[], "Master Context")
            .await?;

        // Step 5: Generate multi-step implementation plan using LLM
        let implementation_plan = if let Some(llm) = &self.llm {
            match self
                .generate_implementation_plan(llm.as_ref(), user_prompt, &master_context, &intent_analysis)
                .await
            {
                Ok(plan) => plan,
                Err(err) => {
                    warn!(
                        "LLM implementation plan failed ({}). Falling back to basic plan.",
                        err
                    );
                    self.generate_basic_implementation_plan(&master_context, &intent_analysis)
                }
            }
        } else {
            self.generate_basic_implementation_plan(&master_context, &intent_analysis)
        };

        // Step 6: Generate the final comprehensive prompt
        let request = PromptRequest {
            original_prompt: user_prompt.to_string(),
            intent: intent_analysis,
            context: master_context,
            implementation_plan: Some(implementation_plan),
        };

        let generated = self.prompt_generator.generate(&request);
        Ok(generated.full_prompt)
    }

    /// Enhanced context-aware prompt generation with Universal Knowledge Graph
    /// This method uses ALL new components: Project Signature, Question Loop, Style Analysis, Meta-Prompt
    pub async fn generate_enhanced_prompt(
        &self,
        user_prompt: &str,
        project_root: &std::path::Path,
    ) -> Result<String> {
        info!("🚀 Starting Universal Knowledge Graph workflow (agentic router enabled)...");

        // PHASE 1: Project Signature Detection (with caching)
        info!("📋 Phase 1: Detecting project signature...");
        let project_signature = self.load_or_detect_signature(project_root)?;
        info!("✅ Detected: {}", project_signature.to_description());

        // PHASE 1b: LLM-driven Router Planning (Router Agent)
        let router_plan: Option<SearchPlan> = if let Some(ref llm) = self.llm {
            info!("🧠 Router Agent: planning search strategy with LLM...");
            let router = GeminiRouterAgent::new(llm.clone());
            match router.plan(user_prompt, &project_signature).await {
                Ok(plan) => {
                    info!(
                        "✅ Router plan: intent='{}', {} global queries, {} worker plans, execution order: {:?}",
                        plan.global_intent,
                        plan.search_queries.len(),
                        plan.workers.len(),
                        plan.execution_plan
                    );
                    Some(plan)
                }
                Err(e) => {
                    warn!("Router planning failed, falling back to analyzer keywords: {}", e);
                    None
                }
            }
        } else {
            info!("ℹ️ No LLM available, skipping Router Agent planning");
            None
        };

        let project_language = project_signature.dominant_language().to_string();
        let framework: Option<String> = if !project_signature.framework.is_empty() {
            Some(project_signature.framework.clone())
        } else {
            None
        };

        // PHASE 2a: Execute Workers Sequentially (if router plan exists)
        let worker_results: Vec<miow_agent::WorkerResult> = if let Some(ref plan) = &router_plan {
            if let Some(ref llm) = self.llm {
                info!("🔄 Phase 2a: Executing workers sequentially...");
                self.execute_workers_sequentially(llm.clone(), plan, user_prompt, &project_signature).await
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        if !worker_results.is_empty() {
            info!("✅ Worker execution complete: {} results collected", worker_results.len());
        }

        // PHASE 2: Generate Critical Questions (with detailed logging)
        info!("❓ Phase 3: Generating language-specific critical questions...");
        let critical_questions = if let Some(ref llm) = self.llm {
            info!("💬 [LLM] Calling generate_critical_questions for language: {}, framework: {:?}",
                  project_language, framework);
            let start = std::time::Instant::now();

            // Use LLM to generate questions
            match miow_llm::generate_critical_questions(
                llm.as_ref(),
                user_prompt,
                &project_language,
                framework.as_deref(),
            ).await {
                Ok(questions) => {
                    let duration = start.elapsed();
                    info!("✅ [LLM] Generated {} questions in {:?}", questions.len(), duration);
                    for (i, q) in questions.iter().enumerate() {
                        info!("   Q{}: {} (type: {}, priority: {:?})",
                              i + 1, q.question, q.expected_type, q.priority);
                    }
                    questions
                }
                Err(e) => {
                    let duration = start.elapsed();
                    warn!("❌ [LLM] Failed to generate questions after {:?}: {}, using template", duration, e);
                    // Fallback: Use template questions from project signature
                    project_signature
                        .get_question_templates()
                        .into_iter()
                        .enumerate()
                        .map(|(i, q)| miow_llm::CriticalQuestion {
                            question: q.clone(),
                            search_query: q.split_whitespace().last().unwrap_or("").to_string(),
                            expected_type: if q.contains("component") {
                                "component".to_string()
                            } else if q.contains("type") {
                                "type".to_string()
                            } else {
                                "function".to_string()
                            },
                            priority: if i == 0 {
                                miow_llm::Priority::Critical
                            } else {
                                miow_llm::Priority::High
                            },
                        })
                        .collect()
                }
            }
        } else {
            info!("ℹ️  No LLM available, using template questions");
            // No LLM: use template questions
            project_signature
                .get_question_templates()
                .into_iter()
                .enumerate()
                .map(|(i, q)| miow_llm::CriticalQuestion {
                    question: q.clone(),
                    search_query: q.split_whitespace().last().unwrap_or("").to_string(),
                    expected_type: "unknown".to_string(),
                    priority: if i == 0 {
                        miow_llm::Priority::Critical
                    } else {
                        miow_llm::Priority::Medium
                    },
                })
                .collect()
        };

        info!("✅ Generated {} critical questions", critical_questions.len());

        // PHASE 3: Execute Question Loop with Rollback (with detailed logging)
        info!("🔄 Phase 3: Executing question loop with search-verify-retry...");
        let question_answers = if let Some(ref llm) = self.llm {
            info!("💬 [QUESTION_LOOP] Starting execution of {} questions", critical_questions.len());
            let question_loop = miow_llm::QuestionLoop::new(
                llm.clone(),
                self.vector_store.clone(),
                self.graph.clone(),
            );

            let start = std::time::Instant::now();
            match question_loop.execute_questions(critical_questions.clone()).await {
                Ok(answers) => {
                    let duration = start.elapsed();
                    info!("✅ [QUESTION_LOOP] Completed in {:?} with {} answers", duration, answers.len());
                    for (i, answer) in answers.iter().enumerate() {
                        info!("   Answer {}: {} symbols found, confidence: {:.2}",
                              i + 1, answer.symbols.len(), answer.confidence);
                        for symbol in &answer.symbols {
                            info!("      - {} ({}) from {}", symbol.name, symbol.kind, symbol.file_path);
                        }
                    }
                    answers
                }
                Err(e) => {
                    let duration = start.elapsed();
                    warn!("❌ [QUESTION_LOOP] Failed after {:?}: {}, using basic search", duration, e);
                    Vec::new()
                }
            }
        } else {
            warn!("ℹ️  No LLM available for question loop, skipping");
            Vec::new()
        };

        info!("✅ Question loop completed with {} answers", question_answers.len());

        // PHASE 4: Gather Context (enhanced with router plan + worker results + question answers)
        info!("📚 Phase 4: Gathering comprehensive context...");
        let analyzed = self.analyzer.analyze_prompt(user_prompt);

        // Start from analyzer intent/keywords.
        let mut intent = format!("{:?}", analyzed.intent);
        let mut search_queries = analyzed.keywords.clone();

        // If router produced a plan, let it drive intent + search queries.
        if let Some(plan) = &router_plan {
            if !plan.global_intent.trim().is_empty() {
                intent = plan.global_intent.clone();
            }
            let router_queries = plan.all_query_strings();
            if !router_queries.is_empty() {
                search_queries = router_queries;
            }
        }

        let mut gathered_context = self
            .gather_comprehensive_context(
                user_prompt,
                &search_queries,
                &intent,
                router_plan.as_ref(),
            )
            .await?;

        // Merge worker results into gathered context
        for worker_result in &worker_results {
            for chunk in &worker_result.chunks {
                let item = miow_llm::ContextItem {
                    name: chunk.id.clone(),
                    kind: chunk.kind.clone(),
                    content: chunk.content.clone(),
                    file_path: chunk.file_path.clone(),
                    relevance_score: worker_result.confidence,
                    props: vec![],
                    references: vec![],
                };

                // Categorize based on content type
                if chunk.kind.contains("component") || chunk.kind.contains("function") {
                    gathered_context.components.push(item);
                } else if chunk.kind.contains("type") || chunk.kind.contains("interface") {
                    gathered_context.types.push(item);
                } else if chunk.kind.contains("schema") || chunk.kind.contains("model") {
                    gathered_context.schemas.push(item);
                } else {
                    gathered_context.helpers.push(item);
                }
            }
        }

        // Merge question answers into gathered context
        for answer in question_answers {
            for symbol in answer.symbols {
                let item = miow_llm::ContextItem {
                    name: symbol.name.clone(),
                    kind: symbol.kind.clone(),
                    content: symbol.content.clone(),
                    file_path: symbol.file_path.clone(),
                    relevance_score: answer.confidence,
                    props: vec![],
                    references: vec![],
                };

                // Add to appropriate category
                if symbol.kind.contains("component") {
                    gathered_context.components.push(item);
                } else if symbol.kind.contains("type") {
                    gathered_context.types.push(item);
                } else if symbol.kind.contains("schema") {
                    gathered_context.schemas.push(item);
                } else {
                    gathered_context.helpers.push(item);
                }
            }
        }

        // Optional PHASE 4b: LLM-powered context auditing (Context Auditor Agent)
        if let Some(ref llm) = self.llm {
            info!("🧹 Context Auditor: LLM-driven pruning of gathered context...");
            let auditor = GeminiContextAuditor::new(llm.clone());
            if let Err(e) = auditor.audit(user_prompt, &mut gathered_context).await {
                warn!("Context auditor failed, continuing with unfiltered context: {}", e);
            }
        }

        // PHASE 5: Master Prompt Compilation (aggregate worker results)
        info!("🎯 Phase 5: Compiling master context from worker results...");
        let compiled_context = self.compile_master_context(
            &worker_results,
            &gathered_context,
            user_prompt,
            &project_signature,
        ).await;

        // PHASE 6: Convert to ContextData
        let mut context_data = self
            .convert_to_context_data(compiled_context, &search_queries, user_prompt)
            .await?;

        info!(
            "Compiled: {} relevant symbols, {} types, {} tokens from {} workers",
            context_data.relevant_symbols.len(),
            context_data.types.len(),
            context_data.design_tokens.len(),
            worker_results.len()
        );

        // PHASE 6: Generate Meta-Prompt (copy-paste ready)
        info!("📝 Phase 6: Generating meta-prompt...");
        let config = miow_prompt::MetaPromptConfig {
            include_full_code: true,
            include_style_guide: true,
            include_implementation_plan: true,
            max_examples_per_type: 5,
            token_budget: Some(16000),
        };

        // 5. Deduplicate and Prune Context
        info!("✂️ Optimizing context...");
        miow_prompt::DeduplicationEngine::deduplicate(&mut context_data);

        if let Some(budget) = config.token_budget {
            let pruner = miow_prompt::SmartPruner::new(budget);
            pruner.prune(&mut context_data);
        }

        // 6. Generate Meta-Prompt
        info!("📝 Generating meta-prompt...");
        let project_info = project_signature.to_description(); // Define project_info here
        let prompt = miow_prompt::MetaPromptGenerator::generate(
            user_prompt,
            &context_data,
            Some(&project_info),
            config,
        )?;

        info!("✅ Universal Knowledge Graph workflow complete!");

        Ok(prompt)
    }

    /// Generate a context-aware prompt using the Autonomous Agent Loop
    pub async fn generate_autonomous_prompt(
        &self,
        project_root: &str,
        user_prompt: &str,
        event_tx: Option<tokio::sync::mpsc::Sender<miow_agent::autonomous::AgentEvent>>,
    ) -> Result<String> {
        info!("🤖 Starting Autonomous Context Generation for: {}", project_root);

        // 1. Detect Project Signature (LLM-driven)
        let signature = self.detect_signature_with_llm(std::path::Path::new(project_root)).await?;
        info!("📊 Detected Project Signature: {:?}", signature);

        // 2. Initialize Autonomous Agent
        let llm = self.llm.clone().ok_or_else(|| anyhow::anyhow!("LLM required for autonomous mode"))?;
        let agent = AutonomousAgent::new(
            llm,
            self.graph.clone(),
            self.vector_store.clone(),
        );

        // 3. Run Agent Loop (Gather Context)
        let agent_context = agent.run(user_prompt, event_tx).await?;
        info!("✅ Agent finished gathering context. Items: {}", agent_context.gathered_info.len());

        // 4. Generate Implementation Plan (LLM-driven)
        let plan = self.generate_implementation_plan_with_llm(
            user_prompt,
            &agent_context,
            &signature.to_description()
        ).await?;

        // 5. Prepare Context Data for Meta-Prompt
        let mut context_data = ContextData {
            relevant_symbols: Vec::new(),
            similar_symbols: Vec::new(),
            types: Vec::new(),
            design_tokens: Vec::new(),
            constants: Vec::new(),
            schemas: Vec::new(),
            common_imports: Vec::new(),
        };

        // Add gathered info
        for info in agent_context.gathered_info {
            context_data.relevant_symbols.push(SymbolInfo {
                name: "ContextItem".to_string(),
                kind: "snippet".to_string(),
                file_path: info.source,
                content: info.content,
                start_line: 0,
                end_line: 0,
                props: Vec::new(),
                references: Vec::new(),
            });
        }

        // Add the Plan as a special context item
        context_data.relevant_symbols.push(SymbolInfo {
            name: "ImplementationPlan".to_string(),
            kind: "plan".to_string(),
            file_path: "implementation_plan.md".to_string(),
            content: plan,
            start_line: 0,
            end_line: 0,
            props: Vec::new(),
            references: Vec::new(),
        });

        let config = miow_prompt::MetaPromptConfig::default();
        let prompt = miow_prompt::MetaPromptGenerator::generate(
            user_prompt,
            &context_data,
            Some(&signature.to_description()),
            config,
        )?;

        Ok(prompt)
    }

    async fn detect_signature_with_llm(&self, project_root: &std::path::Path) -> Result<miow_core::ProjectSignature> {
        // List top-level files to give LLM a hint
        let mut file_list = Vec::new();
        if let Ok(entries) = std::fs::read_dir(project_root) {
            for entry in entries.flatten() {
                if let Ok(name) = entry.file_name().into_string() {
                    file_list.push(name);
                }
            }
        }
        let files_str = file_list.join(", ");

        let prompt = format!(
            r#"Analyze the following file list from a project root and determine the technology stack.
            Files: {}

            Respond with JSON ONLY:
            {{
                "language": "Rust/TypeScript/Python/etc",
                "framework": "React/Axum/Django/etc",
                "package_manager": "npm/cargo/pip/etc",
                "ui_library": "Tailwind/MaterialUI/etc (or None)",
                "validation_library": "zod/serde/etc (or None)",
                "auth_library": "auth.js/etc (or None)",
                "description": "Short description of the stack"
            }}
            "#,
            files_str
        );

        let llm = self.llm.as_ref().ok_or_else(|| anyhow::anyhow!("LLM required"))?;
        let response = llm.generate(&prompt).await?;

        // Clean and parse JSON
        let clean = response.content.trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let signature: miow_core::ProjectSignature = serde_json::from_str(clean)
            .unwrap_or_else(|_| miow_core::ProjectSignature::default());

        Ok(signature)
    }

    async fn generate_implementation_plan_with_llm(
        &self,
        task: &str,
        context: &miow_agent::autonomous::AgentContext,
        project_info: &str
    ) -> Result<String> {
        let gathered_summary = context.gathered_info.iter()
            .map(|i| format!("- From {}: {}", i.source, i.relevance))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            r#"Create a comprehensive implementation plan for the following task.

            Task: {}
            Project Context: {}

            Gathered Information:
            {}

            Generate a detailed Markdown plan with:
            1. Goal Description
            2. Proposed Changes (File by File)
            3. Verification Plan
            "#,
            task, project_info, gathered_summary
        );

        let llm = self.llm.as_ref().ok_or_else(|| anyhow::anyhow!("LLM required"))?;
        let response = llm.generate(&prompt).await?;

        Ok(response.content)
    }

    /// Gather comprehensive context from codebase
    /// If a router plan is provided, its target_paths hints are used to filter results by file path.
    async fn gather_comprehensive_context(
        &self,
        _user_prompt: &str,
        search_queries: &[String],
        intent: &str,
        router_plan: Option<&miow_agent::SearchPlan>,
    ) -> Result<GatheredContext> {
        info!("Gathering comprehensive context...");

        // Helper: collect all target path hints for a given plain-text query.
        let get_target_paths = |query: &str| -> Vec<String> {
            let mut paths = Vec::new();
            if let Some(plan) = router_plan {
                let needle = query.trim().to_lowercase();

                for sq in &plan.search_queries {
                    if sq.query.trim().to_lowercase() == needle {
                        paths.extend(sq.target_paths.iter().cloned());
                    }
                }
                for worker in &plan.workers {
                    for sq in &worker.queries {
                        if sq.query.trim().to_lowercase() == needle {
                            paths.extend(sq.target_paths.iter().cloned());
                        }
                    }
                }
            }
            paths
        };

        let mut gathered = GatheredContext {
            components: Vec::new(),
            helpers: Vec::new(),
            types: Vec::new(),
            design_tokens: Vec::new(),
            constants: Vec::new(),
            schemas: Vec::new(),
            similar_implementations: Vec::new(),
        };

        // Explicitly search for common UI primitives
        let ui_primitives = vec!["Button", "Input", "InputBox", "Form", "BaseButton", "DateInput", "PhoneNumberInput"];
        for primitive in &ui_primitives {
            if let Ok(results) = self.graph.search_symbols(primitive) {
                for result in results {
                    let name_lower = result.name.to_lowercase();
                    if name_lower.contains(&primitive.to_lowercase()) {
                        let relevance = if result.name == *primitive || result.name.contains(primitive) {
                            0.95 // High relevance for exact matches
                        } else {
                            0.7
                        };

                        // Parse metadata for props
                        let mut props = Vec::new();
                        if let Some(meta_json) = &result.metadata {
                            if let Ok(meta) = serde_json::from_str::<serde_json::Value>(meta_json) {
                                 if let Some(props_arr) = meta.get("props").and_then(|p| p.as_array()) {
                                     for p in props_arr {
                                         let name = p.get("name").and_then(|s| s.as_str()).unwrap_or("?");
                                         let type_ann = p.get("type_annotation").and_then(|s| s.as_str()).unwrap_or("any");
                                         props.push(format!("{}: {}", name, type_ann));
                                     }
                                 }
                            }
                        }

                        // Get references
                        let references = self.graph.get_symbol_dependencies(result.id).unwrap_or_default();

                        let item = ContextItem {
                            name: result.name.clone(),
                            kind: result.kind.clone(),
                            content: result.content.clone(),
                            file_path: result.file_path.clone(),
                            relevance_score: relevance,
                            props,
                            references,
                        };
                        gathered.components.push(item);
                    }
                }
            }
        }

        // Search for components/helpers using queries, respecting router target_paths when present
        for query in search_queries {
            let target_paths = get_target_paths(query);
            let results = self.graph.search_symbols(query)?;
            for result in results {
                if !target_paths.is_empty()
                    && !target_paths
                        .iter()
                        .any(|p| result.file_path.starts_with(p))
                {
                    continue;
                }

                let kind_lower = result.kind.to_lowercase();
                let name_lower = result.name.to_lowercase();
                let relevance = self.calculate_relevance(&result.name, &result.kind, query, intent);

                // Boost relevance for UI primitives
                let relevance = if ui_primitives.iter().any(|p| name_lower.contains(&p.to_lowercase())) {
                    (relevance + 0.3).min(1.0)
                } else {
                    relevance
                };

                // Parse metadata for props
                let mut props = Vec::new();
                if let Some(meta_json) = &result.metadata {
                    if let Ok(meta) = serde_json::from_str::<serde_json::Value>(meta_json) {
                            if let Some(props_arr) = meta.get("props").and_then(|p| p.as_array()) {
                                for p in props_arr {
                                    let name = p.get("name").and_then(|s| s.as_str()).unwrap_or("?");
                                    let type_ann = p.get("type_annotation").and_then(|s| s.as_str()).unwrap_or("any");
                                    props.push(format!("{}: {}", name, type_ann));
                                }
                            }
                    }
                }

                // Get references
                let references = self.graph.get_symbol_dependencies(result.id).unwrap_or_default();

                let item = ContextItem {
                    name: result.name.clone(),
                    kind: result.kind.clone(),
                    content: result.content.clone(),
                    file_path: result.file_path.clone(),
                    relevance_score: relevance,
                    props,
                    references,
                };

                if kind_lower.contains("component")
                    || (kind_lower.contains("function")
                        && result
                            .name
                            .chars()
                            .next()
                            .map(|c| c.is_uppercase())
                            .unwrap_or(false))
                {
                    gathered.components.push(item);
                } else if kind_lower.contains("type") || kind_lower.contains("interface") {
                    gathered.types.push(item);
                } else if kind_lower.contains("schema") || kind_lower.contains("model") {
                    gathered.schemas.push(item);
                } else if kind_lower.contains("const") {
                    gathered.constants.push(item);
                } else {
                    gathered.helpers.push(item);
                }
            }

            // 2. Vector Search (Semantic)
            if let Some(vs) = &self.vector_store {
                if let Ok(vector_results) = vs.search_similar(query, 5).await {
                    for result in vector_results {
                        // Skip if we have target paths and this file doesn't match
                        if !target_paths.is_empty()
                            && !target_paths
                                .iter()
                                .any(|p| result.symbol.file_path.starts_with(p))
                        {
                            continue;
                        }

                        // Parse metadata for props
                        let mut props = Vec::new();
                        if !result.symbol.metadata.is_empty() {
                            if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&result.symbol.metadata) {
                                    if let Some(props_arr) = meta.get("props").and_then(|p| p.as_array()) {
                                        for p in props_arr {
                                            let name = p.get("name").and_then(|s| s.as_str()).unwrap_or("?");
                                            let type_ann = p.get("type_annotation").and_then(|s| s.as_str()).unwrap_or("any");
                                            props.push(format!("{}: {}", name, type_ann));
                                        }
                                    }
                            }
                        }

                        // Get references
                        let references = if let Ok(id) = result.symbol.id.parse::<i64>() {
                            self.graph.get_symbol_dependencies(id).unwrap_or_default()
                        } else {
                            Vec::new()
                        };

                        let item = ContextItem {
                            name: result.symbol.name.clone(),
                            kind: result.symbol.kind.clone(),
                            content: result.symbol.content.clone(),
                            file_path: result.symbol.file_path.clone(),
                            relevance_score: result.score,
                            props,
                            references,
                        };

                        let kind_lower = result.symbol.kind.to_lowercase();

                        if kind_lower.contains("component") {
                            gathered.components.push(item);
                        } else if kind_lower.contains("type") || kind_lower.contains("interface") {
                            gathered.types.push(item);
                        } else if kind_lower.contains("schema") || kind_lower.contains("model") {
                            gathered.schemas.push(item);
                        } else if kind_lower.contains("const") {
                            gathered.constants.push(item);
                        } else {
                            gathered.helpers.push(item);
                        }
                    }
                }
            }
        }

        // Find similar implementations based on intent
        if intent.contains("Component") || intent.contains("component") {
            let components = self.graph.find_symbols_by_kind("Component")?;
            for comp in components.into_iter().take(5) {
                gathered.similar_implementations.push(ContextItem {
                    name: comp.name,
                    kind: comp.kind,
                    content: comp.content,
                    file_path: comp.file_path, // Kept original comp.file_path for syntactic correctness
                    relevance_score: 1.0,
                    props: vec![],
                    references: vec![],
                });
            }
        }

        // Find design tokens
        for query in search_queries {
            let target_paths = get_target_paths(query);
            let tokens = self.graph.find_design_tokens(query)?;
            for token in tokens {
                if !target_paths.is_empty()
                    && !target_paths
                        .iter()
                        .any(|p| token.file_path.starts_with(p))
                {
                    continue;
                }

                gathered.design_tokens.push(ContextItem {
                    name: token.name,
                    kind: token.token_type,
                    content: token.value.clone(),
                    file_path: token.file_path,
                    relevance_score: 0.7,
                    props: vec![],
                    references: vec![],
                });
            }
        }

        // Find type definitions
        for query in search_queries {
            let target_paths = get_target_paths(query);
            match self.graph.find_type_definitions(query) {
                Ok(types) => {
                    for type_def in types {
                        if !target_paths.is_empty()
                            && !target_paths
                                .iter()
                                .any(|p| type_def.file_path.starts_with(p))
                        {
                            continue;
                        }

                        gathered.types.push(ContextItem {
                            name: type_def.name,
                            kind: type_def.kind,
                            content: type_def.definition,
                            file_path: type_def.file_path,
                            relevance_score: 0.8,
                            props: vec![],
                            references: vec![],
                        });
                    }
                }
                Err(_) => {} // Ignore errors, continue searching
            }
        }

        // Find constants
        for query in search_queries {
            let target_paths = get_target_paths(query);
            match self.graph.find_constants(query) {
                Ok(constants) => {
                    for constant in constants {
                        if !target_paths.is_empty()
                            && !target_paths
                                .iter()
                                .any(|p| constant.file_path.starts_with(p))
                        {
                            continue;
                        }

                        gathered.constants.push(ContextItem {
                            name: constant.name,
                            kind: constant.category,
                            content: constant.value,
                            file_path: constant.file_path,
                            relevance_score: 0.6,
                            props: vec![],
                            references: vec![],
                        });
                    }
                }
                Err(_) => {} // Ignore errors, continue searching
            }
        }

        // Find schemas
        for query in search_queries {
            let target_paths = get_target_paths(query);
            match self.graph.find_schemas(query) {
                Ok(schemas) => {
                    for schema in schemas {
                        if !target_paths.is_empty()
                            && !target_paths
                                .iter()
                                .any(|p| schema.file_path.starts_with(p))
                        {
                            continue;
                        }

                        gathered.schemas.push(ContextItem {
                            name: schema.name,
                            kind: schema.schema_type,
                            content: schema.definition,
                            file_path: schema.file_path,
                            relevance_score: 0.7,
                            props: vec![],
                            references: vec![],
                        });
                    }
                }
                Err(_) => {} // Ignore errors, continue searching
            }
        }

        // Sort by relevance and limit
        gathered.components.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        gathered.helpers.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        gathered.components.truncate(15);
        gathered.helpers.truncate(15);
        gathered.types.truncate(10);
        gathered.design_tokens.truncate(20);
        gathered.constants.truncate(10);
        gathered.schemas.truncate(5);
        gathered.similar_implementations.truncate(5);

        info!(
            "Gathered context: {} components, {} helpers, {} types, {} tokens",
            gathered.components.len(),
            gathered.helpers.len(),
            gathered.types.len(),
            gathered.design_tokens.len()
        );

        Ok(gathered)
    }

    /// Calculate relevance score for a symbol
    fn calculate_relevance(&self, name: &str, kind: &str, query: &str, intent: &str) -> f32 {
        let mut score = 0.5;

        // Name match
        let name_lower = name.to_lowercase();
        let query_lower = query.to_lowercase();
        if name_lower.contains(&query_lower) {
            score += 0.3;
        }

        // Kind match with intent
        let kind_lower = kind.to_lowercase();
        let intent_lower = intent.to_lowercase();
        if intent_lower.contains("component") && kind_lower.contains("component") {
            score += 0.2;
        }

        if score > 1.0f32 {
            1.0f32
        } else {
            score
        }
    }

    /// Convert gathered context to prompt context format
    async fn convert_to_context_data(
        &self,
        gathered: GatheredContext,
        keywords: &[String],
        user_prompt: &str,
    ) -> Result<ContextData> {
        let relevant_symbols: Vec<SymbolInfo> = gathered
            .components
            .iter()
            .chain(gathered.helpers.iter())
            .map(|item| SymbolInfo {
                name: item.name.clone(),
                kind: item.kind.clone(),
                content: item.content.clone(),
                file_path: item.file_path.clone(),
                start_line: 0,
                end_line: 0,
                props: item.props.clone(),
                references: item.references.clone(),
            })
            .collect();

        let similar_symbols: Vec<SymbolInfo> = gathered
            .similar_implementations
            .iter()
            .map(|item| SymbolInfo {
                name: item.name.clone(),
                kind: item.kind.clone(),
                content: item.content.clone(),
                file_path: item.file_path.clone(),
                start_line: 0,
                end_line: 0,
                props: item.props.clone(),
                references: item.references.clone(),
            })
            .collect();

        // PRIMARY: Use vector search for semantic similarity (MOST IMPORTANT!)
        // Vector embeddings capture semantic meaning, not just keyword matches
        let mut vector_symbols_with_scores = Vec::new();
        if let Some(store) = &self.vector_store {
            match store.search_similar(user_prompt, 30).await {
                Ok(results) => {
                    info!("🔍 Vector search found {} semantically similar symbols", results.len());
                    for res in results {
                        vector_symbols_with_scores.push((
                            res.score, // Semantic similarity score from vector search
                            SymbolInfo {
                                name: res.symbol.name,
                                kind: res.symbol.kind,
                                content: res.symbol.content,
                                file_path: res.symbol.file_path,
                                start_line: 0,
                                end_line: 0,
                                props: Vec::new(),
                                references: Vec::new(),
                            },
                        ));
                    }
                }
                Err(err) => {
                    warn!("Vector search failed: {}, using text search only", err);
                }
            }
        }

        // Combine: Vector results FIRST (semantic), then text search as supplement
        let mut all_symbols: Vec<(f32, SymbolInfo)> = vector_symbols_with_scores;

        // Add text search results with lower base score
        for symbol in relevant_symbols {
            let key = format!("{}::{}", symbol.file_path, symbol.name);
            // Only add if not already in vector results (dedupe)
            if !all_symbols.iter().any(|(_, s)| format!("{}::{}", s.file_path, s.name) == key) {
                all_symbols.push((0.0, symbol)); // Lower priority than vector results
            }
        }

        // Rank with vector results getting HIGH priority (semantic similarity > keyword matching)
        let relevant_symbols = self.rank_symbols_with_vector_priority(
            all_symbols,
            keywords,
            user_prompt,
            15,
        );
        let similar_symbols = self.rank_symbols(similar_symbols, keywords, user_prompt, 5);

        let design_tokens = self.collect_design_tokens(&gathered);
        let types = self.collect_type_info(&gathered, 10);
        let constants = self.collect_constant_info(&gathered, 10);
        let schemas = self.collect_schema_info(&gathered, 8);

        Ok(ContextData {
            relevant_symbols,
            similar_symbols,
            design_tokens,
            common_imports: self.extract_common_imports(&gathered),
            types,
            constants,
            schemas,
        })
    }

    fn collect_design_tokens(&self, gathered: &GatheredContext) -> Vec<DesignTokenInfo> {
        let mut seen = HashSet::new();
        let mut tokens = Vec::new();

        for item in &gathered.design_tokens {
            if seen.insert(item.name.clone()) {
                tokens.push(DesignTokenInfo {
                    name: item.name.clone(),
                    value: item.content.clone(),
                    token_type: item.kind.clone(),
                });

                if tokens.len() >= 20 {
                    break;
                }
            }
        }

        tokens
    }

    fn collect_type_info(&self, gathered: &GatheredContext, limit: usize) -> Vec<TypeInfo> {
        let mut seen = HashSet::new();
        let mut items = Vec::new();

        for item in &gathered.types {
            if seen.insert(item.name.clone()) {
                items.push(TypeInfo {
                    name: item.name.clone(),
                    kind: item.kind.clone(),
                    definition: item.content.clone(),
                });

                if items.len() >= limit {
                    break;
                }
            }
        }

        items
    }

    fn collect_constant_info(&self, gathered: &GatheredContext, limit: usize) -> Vec<ConstantInfo> {
        let mut seen = HashSet::new();
        let mut items = Vec::new();

        for item in &gathered.constants {
            if seen.insert(item.name.clone()) {
                items.push(ConstantInfo {
                    name: item.name.clone(),
                    value: item.content.clone(),
                    category: item.kind.clone(),
                });

                if items.len() >= limit {
                    break;
                }
            }
        }

        items
    }

    fn collect_schema_info(&self, gathered: &GatheredContext, limit: usize) -> Vec<SchemaInfo> {
        let mut seen = HashSet::new();
        let mut items = Vec::new();

        for item in &gathered.schemas {
            if seen.insert(item.name.clone()) {
                items.push(SchemaInfo {
                    name: item.name.clone(),
                    schema_type: item.kind.clone(),
                    definition: item.content.clone(),
                });

                if items.len() >= limit {
                    break;
                }
            }
        }

        items
    }

    /// Rank symbols with vector search results getting priority
    /// Vector results already have semantic similarity scores, so we boost those
    fn rank_symbols_with_vector_priority(
        &self,
        symbols_with_scores: Vec<(f32, SymbolInfo)>,
        keywords: &[String],
        user_prompt: &str,
        limit: usize,
    ) -> Vec<SymbolInfo> {
        if symbols_with_scores.is_empty() {
            return Vec::new();
        }

        let keyword_lower: Vec<String> = keywords.iter().map(|k| k.to_lowercase()).collect();
        let prompt_lower = user_prompt.to_lowercase();

        let mut scored: Vec<(f32, SymbolInfo)> = symbols_with_scores
            .into_iter()
            .map(|(vector_score, symbol)| {
                // If vector_score > 0, this came from vector search (semantic similarity)
                // Boost it significantly - semantic similarity is more important than keyword matching
                let base_score = if vector_score > 0.0 {
                    // Vector search result: use semantic score as base, then add keyword bonus
                    vector_score * 10.0 // Scale up semantic similarity (0-1 range to 0-10)
                } else {
                    // Text search result: start from 0, rely on keyword matching
                    0.0
                };

                let keyword_score = self.score_symbol(&symbol, &keyword_lower, &prompt_lower);
                let final_score = base_score + keyword_score;
                (final_score, symbol)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        let mut seen = HashSet::new();
        let mut ranked = Vec::new();
        for (_score, symbol) in scored {
            let key = format!("{}::{}", symbol.file_path, symbol.name);
            if seen.insert(key) {
                ranked.push(symbol);
                if ranked.len() >= limit {
                    break;
                }
            }
        }

        ranked
    }

    fn rank_symbols(
        &self,
        symbols: Vec<SymbolInfo>,
        keywords: &[String],
        user_prompt: &str,
        limit: usize,
    ) -> Vec<SymbolInfo> {
        if symbols.is_empty() {
            return symbols;
        }

        let symbols_with_scores: Vec<(f32, SymbolInfo)> = symbols
            .into_iter()
            .map(|s| (0.0, s)) // No vector score for these
            .collect();

        self.rank_symbols_with_vector_priority(symbols_with_scores, keywords, user_prompt, limit)
    }

    fn score_symbol(&self, symbol: &SymbolInfo, keywords: &[String], prompt_lower: &str) -> f32 {
        let mut score = 0.0;
        let name_lower = symbol.name.to_lowercase();
        let file_lower = symbol.file_path.to_lowercase();
        let content_lower = symbol.content.to_lowercase();

        for keyword in keywords {
            if keyword.is_empty() {
                continue;
            }

            if name_lower.contains(keyword) {
                score += 3.0;
            }

            if file_lower.contains(keyword) {
                score += 2.0;
            }

            if content_lower.contains(keyword) {
                score += 1.0;
            }
        }

        if prompt_lower.contains("login")
            && (name_lower.contains("login")
                || file_lower.contains("login")
                || content_lower.contains("login"))
        {
            score += 4.0;
        }

        if prompt_lower.contains("auth")
            && (file_lower.contains("auth") || content_lower.contains("auth"))
        {
            score += 3.0;
        }

        if prompt_lower.contains("form") && content_lower.contains("form") {
            score += 1.0;
        }

        if symbol.kind.to_lowercase().contains("component") {
            score += 0.5;
        }

        score
    }

    fn refine_keywords(&self, keywords: &[String]) -> Vec<String> {
        const STOP_WORDS: [&str; 16] = [
            "component",
            "components",
            "page",
            "pages",
            "screen",
            "screens",
            "function",
            "functions",
            "make",
            "create",
            "build",
            "add",
            "new",
            "feature",
            "features",
            "please",
        ];

        let mut expanded = Vec::new();

        for keyword in keywords {
            let k = keyword.trim();
            if k.is_empty() {
                continue;
            }

            let lower = k.to_lowercase();
            if STOP_WORDS.contains(&lower.as_str()) {
                continue;
            }

            expanded.push(k.to_string());

            // Expand common UI component keywords
            let expanded_terms = self.expand_ui_keywords(&lower);
            expanded.extend(expanded_terms);
        }

        expanded
    }

    /// Expand keywords to include common UI component patterns
    fn expand_ui_keywords(&self, keyword: &str) -> Vec<String> {
        let mut expanded = Vec::new();

        // Map common terms to UI component names
        if keyword.contains("login") || keyword.contains("auth") || keyword.contains("sign") {
            expanded.push("Button".to_string());
            expanded.push("Input".to_string());
            expanded.push("InputBox".to_string());
            expanded.push("Form".to_string());
            expanded.push("BaseButton".to_string());
        }

        if keyword.contains("form") || keyword.contains("input") || keyword.contains("field") {
            expanded.push("InputBox".to_string());
            expanded.push("Input".to_string());
            expanded.push("DateInput".to_string());
            expanded.push("PhoneNumberInput".to_string());
        }

        if keyword.contains("button") || keyword.contains("submit") || keyword.contains("click") {
            expanded.push("Button".to_string());
            expanded.push("BaseButton".to_string());
        }

        // Always search for common UI primitives
        expanded.push("Button".to_string());
        expanded.push("Input".to_string());
        expanded.push("Form".to_string());

        expanded
    }

    /// Generate implementation plan using LLM
    async fn generate_implementation_plan(
        &self,
        llm: &dyn LLMProvider,
        user_prompt: &str,
        context: &ContextData,
        intent: &str,
    ) -> Result<String> {
        let system_prompt = r#"You are an expert software engineer. Analyze the user's request and the provided context from the codebase.
Generate a detailed, step-by-step implementation plan that:
1. Identifies which existing components, helpers, and utilities to reuse
2. Lists what needs to be created or modified
3. Specifies the order of implementation steps
4. Notes any design tokens, types, or constants to use
5. Includes code examples from similar implementations

Format the plan as a numbered list with clear steps. Be specific about what to reuse and what to create."#;

        let context_summary = format!(
            "Available components: {}\nAvailable helpers: {}\nDesign tokens: {}\nTypes: {}\nConstants: {}",
            context.relevant_symbols.len(),
            context.relevant_symbols.len(),
            context.design_tokens.len(),
            context.types.len(),
            context.constants.len(),
        );

        let user_message = format!(
            "User request: {}\nIntent: {}\n\nContext summary:\n{}\n\nGenerate a detailed implementation plan.",
            user_prompt, intent, context_summary
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

        let response = llm.generate_with_context(messages).await?;
        Ok(response.content)
    }

    /// Generate basic implementation plan without LLM
    fn generate_basic_implementation_plan(&self, context: &ContextData, intent: &str) -> String {
        let mut plan = String::from("## Implementation Plan\n\n");

        if intent.contains("Component") {
            plan.push_str("1. Review similar existing components for patterns\n");
            plan.push_str("2. Identify reusable sub-components\n");
            plan.push_str("3. Use existing design tokens for styling\n");
            plan.push_str("4. Follow the same prop patterns as similar components\n");
            plan.push_str("5. Add proper TypeScript types\n");
        } else if intent.contains("Page") {
            plan.push_str("1. Reuse existing layout components\n");
            plan.push_str("2. Follow the same page structure pattern\n");
            plan.push_str("3. Use existing components for UI elements\n");
            plan.push_str("4. Apply consistent styling with design tokens\n");
        } else {
            plan.push_str("1. Review the provided context\n");
            plan.push_str("2. Identify reusable patterns and components\n");
            plan.push_str("3. Follow existing code conventions\n");
        }

        if !context.relevant_symbols.is_empty() {
            plan.push_str("\n### Components/Functions to Reuse:\n");
            for symbol in context.relevant_symbols.iter().take(10) {
                plan.push_str(&format!("- `{}` ({})\n", symbol.name, symbol.kind));
            }
        }

        plan
    }

    /// Extract common imports from gathered context
    fn extract_common_imports(&self, gathered: &GatheredContext) -> Vec<String> {
        use std::collections::HashMap;

        let mut import_counts: HashMap<String, usize> = HashMap::new();

        // Count imports from components and helpers
        for item in gathered.components.iter().chain(gathered.helpers.iter()) {
            // Try to extract imports from file path
            if let Some(import_path) = self.infer_import_path(&item.file_path) {
                *import_counts.entry(import_path).or_insert(0) += 1;
            }
        }

        // Get top 10 most common imports
        let mut imports: Vec<(String, usize)> = import_counts.into_iter().collect();
        imports.sort_by(|a, b| b.1.cmp(&a.1));
        imports.into_iter().take(10).map(|(path, _)| path).collect()
    }

    /// Infer import path from file path
    fn infer_import_path(&self, file_path: &str) -> Option<String> {
        // Remove common prefixes and extensions
        let path = file_path
            .trim_start_matches("./")
            .trim_start_matches("src/")
            .trim_end_matches(".ts")
            .trim_end_matches(".tsx")
            .trim_end_matches(".js")
            .trim_end_matches(".jsx");

        if path.is_empty() {
            return None;
        }

        // Convert to import path format
        let import_path = path.replace("\\", "/");
        Some(format!("'./{}'", import_path))
    }

    /// Get a reference to the knowledge graph
    pub fn graph(&self) -> &KnowledgeGraph {
        &self.graph
    }

    /// Execute workers in parallel (batched for efficiency)
    async fn execute_workers_sequentially(
        &self,
        llm: Arc<dyn miow_llm::LLMProvider>,
        plan: &miow_agent::SearchPlan,
        user_prompt: &str,
        project_signature: &miow_core::ProjectSignature,
    ) -> Vec<miow_agent::WorkerResult> {
        use miow_agent::{GeminiWorkerAgent, PromptRegistry};
        use futures::future::join_all;

        let registry = Arc::new(PromptRegistry::new());

        // Create tasks for all workers
        let mut tasks = Vec::new();

        for worker_id in &plan.execution_plan {
            if let Some(worker_plan) = plan.workers.iter().find(|w| w.worker_id == *worker_id) {
                let worker_id_clone = worker_id.clone();
                let worker_plan_clone = worker_plan.clone();
                let llm_clone = llm.clone();
                let registry_clone = registry.clone();
                let prompt_clone = user_prompt.to_string();
                let sig_clone = project_signature.clone();

                info!("🔧 Queueing worker for parallel execution: {}", worker_id);

                let task = tokio::spawn(async move {
                    // Create a new worker agent for this task
                    let worker_agent = GeminiWorkerAgent::new(llm_clone.clone(), registry_clone);

                    let search_queries: Vec<miow_agent::SearchQuery> = worker_plan_clone.queries.iter()
                        .map(|q| miow_agent::SearchQuery {
                            query: q.query.clone(),
                            kind: q.kind.clone(),
                            target_paths: q.target_paths.clone(),
                        })
                        .collect();

                    info!("🚀 [WORKER {}] Starting execution...", worker_id_clone);
                    let start = std::time::Instant::now();

                    match worker_agent.execute(
                        &worker_plan_clone.worker_id,
                        &prompt_clone,
                        &sig_clone,
                        &search_queries,
                    ).await {
                        Ok(result) => {
                            let duration = start.elapsed();
                            info!("✅ [WORKER {}] Completed in {:?}: {} chunks, confidence: {:.2}",
                                  worker_id_clone, duration, result.chunks.len(), result.confidence);
                            Some((worker_id_clone, result))
                        }
                        Err(e) => {
                            let duration = start.elapsed();
                            warn!("❌ [WORKER {}] Failed after {:?}: {}", worker_id_clone, duration, e);
                            None
                        }
                    }
                });

                tasks.push(task);
            }
        }

        // Execute all workers in parallel (with concurrency limit via join_all)
        info!("🔄 Executing {} workers in parallel...", tasks.len());
        let results: Vec<_> = join_all(tasks).await;

        // Collect successful results, maintaining order
        let mut worker_results = Vec::new();
        for result in results {
            if let Ok(Some((_worker_id, worker_result))) = result {
                worker_results.push(worker_result);
            }
        }

        info!("✅ Parallel worker execution complete: {}/{} succeeded",
              worker_results.len(), plan.execution_plan.len());

        worker_results
    }

    /// Load project signature from cache or detect it
    fn load_or_detect_signature(&self, project_root: &std::path::Path) -> Result<miow_core::ProjectSignature> {
        let cache_path = project_root.join(".miow_cache.json");

        // Try to load from cache first
        if let Ok(cached_content) = std::fs::read_to_string(&cache_path) {
            if let Ok(signature) = serde_json::from_str(&cached_content) {
                info!("📋 Loaded project signature from cache");
                return Ok(signature);
            }
        }

        // Detect and cache
        info!("📋 Detecting project signature...");
        let signature = miow_core::ProjectSignature::detect(project_root)?;

        // Save to cache
        if let Ok(json) = serde_json::to_string_pretty(&signature) {
            let _ = std::fs::write(&cache_path, json); // Ignore errors, caching is optional
        }

        Ok(signature)
    }

    /// Compile master context by intelligently merging worker results
    async fn compile_master_context(
        &self,
        worker_results: &[miow_agent::WorkerResult],
        base_context: &miow_llm::GatheredContext,
        user_prompt: &str,
        project_signature: &miow_core::ProjectSignature,
    ) -> miow_llm::GatheredContext {
        let mut master_context = base_context.clone();

        // If no workers were executed, return the base context
        if worker_results.is_empty() {
            return master_context;
        }

        // Use LLM to intelligently merge and prioritize results if available
        if let Some(ref llm) = self.llm {
            match self.merge_contexts_with_llm(llm.clone(), worker_results, &master_context, user_prompt, project_signature).await {
                Ok(merged) => {
                    info!("🤖 LLM-based context merging completed");
                    return merged;
                }
                Err(e) => {
                    warn!("LLM context merging failed, using rule-based merging: {}", e);
                }
            }
        }

        // Fallback: rule-based merging
        self.merge_contexts_rule_based(worker_results, &master_context)
    }

    /// Use LLM to intelligently merge and prioritize worker results
    async fn merge_contexts_with_llm(
        &self,
        llm: Arc<dyn miow_llm::LLMProvider>,
        worker_results: &[miow_agent::WorkerResult],
        _base_context: &miow_llm::GatheredContext, // Marked as unused
        user_prompt: &str,
        project_signature: &miow_core::ProjectSignature,
    ) -> Result<miow_llm::GatheredContext> {
        // Create a summary of all worker results
        let worker_summary = worker_results.iter()
            .map(|wr| format!("Worker '{}': {} chunks, confidence {:.2}, summary: {}",
                            wr.worker_id, wr.chunks.len(), wr.confidence, wr.summary))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(r#"You are a Context Compilation Specialist. Analyze worker results and merge them intelligently.

User Task: {}
Project: {}

Worker Results Summary:
{}

Instructions:
1. Prioritize high-confidence, relevant results
2. Remove duplicates across workers
3. Ensure results are directly related to the task
4. Maintain diversity of code patterns
5. Return JSON with merged and filtered context

Respond with JSON containing prioritized context items from all workers."#,
            user_prompt,
            project_signature.to_description(),
            worker_summary
        );

        let messages = vec![
            miow_llm::Message {
                role: miow_llm::Role::System,
                content: "You are an expert at merging and prioritizing code context from multiple specialized agents.".to_string(),
            },
            miow_llm::Message {
                role: miow_llm::Role::User,
                content: prompt,
            },
        ];

        let response = llm.generate_with_context(messages).await?;

        // For now, return the base context enhanced with worker results
        // In a full implementation, parse the LLM response to selectively include items
        Ok(self.merge_contexts_rule_based(worker_results, _base_context))
    }

    /// Rule-based context merging (fallback when LLM fails)
    fn merge_contexts_rule_based(
        &self,
        worker_results: &[miow_agent::WorkerResult],
        base_context: &miow_llm::GatheredContext,
    ) -> miow_llm::GatheredContext {
        let mut master_context = base_context.clone();

        // Track seen items to avoid duplicates
        let mut seen_names = std::collections::HashSet::new();

        // Add base context items to seen set
        for item in &base_context.components {
            seen_names.insert(item.name.clone());
        }
        for item in &base_context.types {
            seen_names.insert(item.name.clone());
        }
        for item in &base_context.schemas {
            seen_names.insert(item.name.clone());
        }
        for item in &base_context.helpers {
            seen_names.insert(item.name.clone());
        }

        // Add worker results, prioritizing by confidence and avoiding duplicates
        for worker_result in worker_results {
            for chunk in &worker_result.chunks {
                if !seen_names.contains(&chunk.id) {
                    seen_names.insert(chunk.id.clone());

                    let item = miow_llm::ContextItem {
                        name: chunk.id.clone(),
                        kind: chunk.kind.clone(),
                        content: chunk.content.clone(),
                        file_path: chunk.file_path.clone(),
                        relevance_score: worker_result.confidence,
                        props: Vec::new(),
                        references: Vec::new(),
                    };

                    // Categorize based on content type
                    if chunk.kind.contains("component") || chunk.kind.contains("function") {
                        master_context.components.push(item);
                    } else if chunk.kind.contains("type") || chunk.kind.contains("interface") {
                        master_context.types.push(item);
                    } else if chunk.kind.contains("schema") || chunk.kind.contains("model") {
                        master_context.schemas.push(item);
                    } else {
                        master_context.helpers.push(item);
                    }
                }
            }
        }

        master_context
    }

    // Enhanced context gathering with smart selection
    #[allow(unused_variables)] // Allow unused variables for now, as the implementation is a placeholder
    async fn gather_smart_context(
        &self,
        user_prompt: &str,
        intent: &str,
        router_plan: Option<&miow_agent::SearchPlan>,
    ) -> Result<ContextData> {
        info!("Gathering smart context with LLM selection");

        // Step 1: Raw context gathering (existing logic)
        let raw_context = self
            .gather_comprehensive_context(user_prompt, &[], "", None)
            .await?;

        // Convert GatheredContext to ContextData
        let mut context_data = ContextData {
            relevant_symbols: raw_context.components.iter().map(|item| SymbolInfo {
                name: item.name.clone(),
                kind: item.kind.clone(),
                content: item.content.clone(),
                file_path: item.file_path.clone(),
                start_line: 0,
                end_line: 0,
                props: item.props.clone(),
                references: item.references.clone(),
            })
            .collect(),
            similar_symbols: raw_context.helpers.iter().map(|item| SymbolInfo {
                name: item.name.clone(),
                kind: item.kind.clone(),
                content: item.content.clone(),
                file_path: item.file_path.clone(),
                start_line: 0,
                end_line: 0,
                props: item.props.clone(),
                references: item.references.clone(),
            })
            .collect(),
            types: raw_context.types.iter().map(|item| TypeInfo {
                name: item.name.clone(),
                kind: item.kind.clone(),
                definition: item.content.clone(),
            }).collect(),
            constants: raw_context.constants.iter().map(|item| ConstantInfo {
                name: item.name.clone(),
                value: item.content.clone(),
                category: "unknown".to_string(),
            }).collect(),
            schemas: raw_context.schemas.iter().map(|item| SchemaInfo {
                name: item.name.clone(),
                schema_type: item.kind.clone(),
                definition: item.content.clone(),
            }).collect(),
            design_tokens: raw_context.design_tokens.iter().map(|item| DesignTokenInfo {
                name: item.name.clone(),
                value: item.content.clone(),
                token_type: item.kind.clone(),
            }).collect(),
            common_imports: vec![],
        };

        // Step 2: LLM-powered context selection if available
        if let Some(llm) = self.llm.as_ref() {
            let selection_prompt = format!(
                "From these {} symbols, select the most relevant 10-15 for '{}'. Prioritize:
                - Direct matches to task (e.g., 'login' → auth components)
                - Framework-specific (Next.js: pages/routes; Rust: handlers/structs)
                - Common patterns (forms → Input/Button; API → routes)
                - High relevance score (>0.8)
                
                Return JSON: {{ selected: [indices], reasons: {{index: reason}} }}",
                context_data.relevant_symbols.len(), user_prompt
            );

            if let Ok(llm_response) = llm.generate(&selection_prompt).await {
                if let Ok(selected_data) = serde_json::from_str::<serde_json::Value>(&llm_response.content) {
                    if let Some(indices) = selected_data.get("selected").and_then(|v| v.as_array()) {
                        let selected_indices: Vec<usize> = indices.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as usize))
                            .collect();
                        
                        if !selected_indices.is_empty() && selected_indices.iter().all(|&i| i < context_data.relevant_symbols.len()) {
                            context_data.relevant_symbols = selected_indices
                                .iter()
                                .filter_map(|&i| context_data.relevant_symbols.get(i))
                                .cloned()
                                .collect();
                        }
                    }
                }
            }
        }

        Ok(context_data)
    }

    // LLM-powered prompt cleaning and variant generation
    async fn generate_cleaned_prompt(
        &self, 
        raw_prompt: &str, 
        task: &str, 
        signature: &ProjectSignature
    ) -> Result<(String, String)> {
        info!("Generating cleaned prompt variant using LLM");

        let cleaning_prompt = format!(
            "Clean and optimize this raw prompt for '{}':
            - Keep structure (TASK, CONTEXT, CONSTRAINTS, STYLE, PLAN)
            - Remove redundant/irrelevant code blocks
            - Prioritize relevant symbols for {}
            - Ensure framework consistency ({}: use App Router patterns)
            - Make concise (under 8k tokens) but comprehensive
            - Add COT for implementation
            
            Raw prompt:
            {}
            
            Return two variants:
            1. RAW_SYSTEM: Original with minor fixes
            2. CLEANED_LLM: Optimized for LLM generation",
            task, task, signature.framework, raw_prompt
        );

        let llm = self.llm.as_ref().ok_or_else(|| anyhow::anyhow!("LLM not available"))?;
        let llm_response = llm.generate(&cleaning_prompt).await?;
        let cleaned_content = llm_response.content;

        // Parse response for variants (simple split for now; could use regex/JSON)
        let parts: Vec<&str> = cleaned_content.split("---").collect();
        let raw_system = if parts.len() > 0 { parts[0].trim().to_string() } else { raw_prompt.to_string() };
        let cleaned_llm = if parts.len() > 1 { parts[1].trim().to_string() } else { cleaned_content };

        Ok((raw_system, cleaned_llm))
    }

    // Autonomous multi-step agent workflow with LLM-driven decision making (DEPRECATED - use the main generate_enhanced_prompt instead)
    #[allow(dead_code)]
    async fn generate_enhanced_prompt_autonomous(&self, user_prompt: &str, project_root: &std::path::Path) -> Result<(String, String)> {
        info!("Starting autonomous multi-step agent workflow for '{}'", user_prompt);

        // Step 1: Detect project signature
        let signature = miow_core::ProjectSignature::detect(project_root)?;

        // Step 2: Autonomous task analysis and planning (LLM decides everything)
        let task_plan = self.generate_task_plan(user_prompt, &signature).await?;

        // Step 3: Gather context based on autonomous plan
        let symbols = self.gather_autonomous_context(&task_plan, &signature).await?;

        // Step 4: Generate raw prompt with plan
        let raw_prompt = self.generate_raw_prompt_with_plan(user_prompt, &symbols, &signature, &task_plan)?;

        // Step 5: Clean and refine autonomously
        let (raw_variant, cleaned_variant) = self.generate_cleaned_prompt(&raw_prompt, user_prompt, &signature).await?;

        // Step 6: Add autonomous COT based on plan
        let cot_prompt = format!(
            "Based on this autonomous plan: {}\nGenerate Chain of Thought for '{}' in {} using {}.\nFocus on decisions made in the plan.",
            task_plan, user_prompt, signature.language, signature.framework
        );
        let llm = self.llm.as_ref().ok_or_else(|| anyhow::anyhow!("LLM not available"))?;
        let cot_response = llm.generate(&cot_prompt).await?;
        let final_cleaned = format!("{}\n\n## AUTONOMOUS CHAIN OF THOUGHT\n{}", cleaned_variant, cot_response.content);

        Ok((raw_variant, final_cleaned))
    }

    // LLM-driven autonomous task planning (no hardcoded biases)
    async fn generate_task_plan(&self, user_prompt: &str, signature: &ProjectSignature) -> Result<String> {
        let plan_prompt = format!(
            "Autonomously analyze task '{}' in this {} project using {} framework.\n
            - Detect task requirements autonomously (e.g., if it needs file upload, look for cloud storage services in codebase)\n
            - Search codebase autonomously: What services/utilities exist? (e.g., scan for AWS, Cloudinary, multer imports)\n
            - Make own decisions: If service exists, plan to reuse; if not, plan to add\n
            - No biases: Adapt to detected patterns, suggest alternatives based on code\n
            - Output JSON: {{ requirements: ['req1'], existing_services: ['service1'], plan: 'detailed plan', decisions: ['decision1'] }}",
            user_prompt, signature.language, signature.framework
        );
        let llm = self.llm.as_ref().ok_or_else(|| anyhow::anyhow!("LLM not available"))?;
        let plan_response = llm.generate(&plan_prompt).await?;
        Ok(plan_response.content)
    }

    // Autonomous context gathering based on plan
    async fn gather_autonomous_context(&self, task_plan: &str, signature: &ProjectSignature) -> Result<Vec<SymbolInfo>> {
        // Use LLM to select relevant symbols based on plan
        let context_prompt = format!(
            "Based on this plan: {}\nAutonomously select relevant symbols from the codebase for {} project.\n
            - Decide what symbols to include (e.g., for upload API, include file handlers, auth)\n
            - No hardcoded lists: Use own judgment\n
            - Return JSON: {{ selected_symbols: ['symbol1'], reasoning: ['why1'] }}",
            task_plan, signature.language
        );
        let llm = self.llm.as_ref().ok_or_else(|| anyhow::anyhow!("LLM not available"))?;
        let context_response = llm.generate(&context_prompt).await?;
        // Parse and map to symbols (simplified; in practice, integrate with vector search)
        let json_value: serde_json::Value = serde_json::from_str(&context_response.content)?;
        let _selected: Vec<String> = json_value["selected_symbols"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|s| s.as_str().map(|s| s.to_string()))
            .collect();
        // For now, return empty; integrate with actual symbol search
        Ok(vec![]) // TODO: Implement symbol filtering based on LLM selection
    }

    // Generate raw prompt with autonomous plan
    fn generate_raw_prompt_with_plan(&self, user_prompt: &str, symbols: &[SymbolInfo], signature: &ProjectSignature, plan: &str) -> Result<String> {
        let mut prompt = format!("# TASK: {}\n\n## PROJECT CONTEXT\n{}\n\n## AUTONOMOUS PLAN\n{}", user_prompt, signature.to_description(), plan);

        if !symbols.is_empty() {
            prompt += "\n\n## AUTONOMOUSLY SELECTED CONTEXT\n";
            for symbol in symbols {
                prompt += &format!("#### `{}` ({})\n**File**: `{}`\n```tsx\n{}\n```\n", symbol.name, symbol.kind, symbol.file_path, symbol.content);
            }
        }

        prompt += "\n\n## IMPLEMENTATION PLAN\nFollow the autonomous plan above, adapting to detected services.";

        Ok(prompt)
    }

    // Update main generation method to use multi-step (DEPRECATED - use generate_enhanced_prompt instead)
    #[allow(dead_code)]
    pub async fn generate(&self, user_prompt: &str, project_root: &std::path::Path) -> Result<String> {
        let (raw, cleaned) = self.generate_enhanced_prompt_autonomous(user_prompt, project_root).await?;
        
        // Log both variants
        info!("Generated raw prompt ({} chars) and cleaned LLM variant ({} chars)", raw.len(), cleaned.len());
        
        // Return cleaned variant as primary
        Ok(cleaned)
    }

    // Helper: Generate raw prompt (existing logic enhanced)
    fn generate_raw_prompt(&self, user_prompt: &str, context: &ContextData, signature: &ProjectSignature) -> Result<String> {
        // Existing raw prompt generation...
        let mut prompt = format!("# TASK: {}\n\n## PROJECT CONTEXT\n{}", user_prompt, signature.to_description());

        // Add intent-based sections
        if !context.relevant_symbols.is_empty() {
            prompt += "\n\n## SELECTED CONTEXT (Smart Selection)\n";
            for symbol in &context.relevant_symbols {
                prompt += &format!("#### `{}` ({})\n**File**: `{}`\n```tsx\n{}\n```\n", 
                    symbol.name, symbol.kind, symbol.file_path, symbol.content);
            }
        }

        // Add framework-specific guidance
        prompt += &format!("\n## FRAMEWORK GUIDANCE\nUse {} patterns: ", signature.framework);
        if signature.framework.contains("Next.js") {
            prompt += "App Router preferred, server components where applicable.";
        }

        prompt += "\n\n## IMPLEMENTATION PLAN\n1. Analyze selected context\n2. Reuse tagged components (UI: common-ui, validation: zod-schema)\n3. Follow detected style and types\n4. Test for framework compatibility";

        Ok(prompt)
    }

    /// Get relevant files for a user prompt (for UI file selection)
    pub async fn get_relevant_files(
        &self,
        user_prompt: &str,
        project_root: &std::path::Path,
    ) -> Result<Vec<ContextItem>> {
        info!("📁 Getting relevant files for prompt: {}", user_prompt);
        
        // Detect project signature
        let _project_signature = self.load_or_detect_signature(project_root)?;
        
        // Analyze prompt
        let analyzed = self.analyzer.analyze_prompt(user_prompt);
        let search_queries = analyzed.keywords.clone();
        
        // Gather context
        let gathered = self.gather_comprehensive_context(
            user_prompt,
            &search_queries,
            &format!("{:?}", analyzed.intent),
            None,
        ).await?;
        
        // Combine all context items
        let mut all_files: Vec<ContextItem> = Vec::new();
        all_files.extend(gathered.components);
        all_files.extend(gathered.helpers);
        all_files.extend(gathered.types);
        all_files.extend(gathered.schemas);
        
        // Deduplicate by file_path
        let mut seen_files = std::collections::HashSet::new();
        let mut unique_files = Vec::new();
        
        for item in all_files {
            if seen_files.insert(item.file_path.clone()) {
                unique_files.push(item);
            }
        }
        
        // Sort by relevance
        unique_files.sort_by(|a, b| {
            b.relevance_score.partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Limit to top 50
        unique_files.truncate(50);
        
        info!("✅ Found {} unique relevant files", unique_files.len());
        Ok(unique_files)
    }

    /// Generate enhanced prompt with user-selected files
    pub async fn generate_enhanced_prompt_with_files(
        &self,
        user_prompt: &str,
        project_root: &std::path::Path,
        selected_files: &[String],
    ) -> Result<String> {
        info!("🚀 Generating prompt with {} selected files", selected_files.len());
        
        // Load project signature
        let project_signature = self.load_or_detect_signature(project_root)?;
        
        // Get symbols from selected files
        let mut selected_symbols = Vec::new();
        for file_path in selected_files {
            if let Ok(symbols) = self.graph.get_file_symbols(file_path) {
                for symbol in symbols {
                    // Parse metadata for props
                    let mut props = Vec::new();
                    if let Some(meta_json) = &symbol.metadata {
                        if let Ok(meta) = serde_json::from_str::<serde_json::Value>(meta_json) {
                             if let Some(props_arr) = meta.get("props").and_then(|p| p.as_array()) {
                                 for p in props_arr {
                                     let name = p.get("name").and_then(|s| s.as_str()).unwrap_or("?");
                                     let type_ann = p.get("type_annotation").and_then(|s| s.as_str()).unwrap_or("any");
                                     props.push(format!("{}: {}", name, type_ann));
                                 }
                             }
                        }
                    }

                    // Get references
                    let references = self.graph.get_symbol_dependencies(symbol.id).unwrap_or_default();

                    selected_symbols.push(SymbolInfo {
                        name: symbol.name,
                        kind: symbol.kind,
                        content: symbol.content,
                        file_path: symbol.file_path,
                        start_line: symbol.start_line as i64,
                        end_line: symbol.end_line as i64,
                        props,
                        references,
                    });
                }
            }
        }
        
        info!("📦 Loaded {} symbols from selected files", selected_symbols.len());
        
        // Build context data with selected files
        let context_data = ContextData {
            relevant_symbols: selected_symbols,
            similar_symbols: vec![],
            design_tokens: vec![],
            common_imports: vec![],
            types: vec![],
            constants: vec![],
            schemas: vec![],
        };
        
        // Generate meta-prompt
        let config = miow_prompt::MetaPromptConfig {
            include_full_code: true,
            include_style_guide: true,
            include_implementation_plan: true,
            max_examples_per_type: 5,
            token_budget: Some(16000),
        };
        
        let project_info = project_signature.to_description();
        let prompt = miow_prompt::MetaPromptGenerator::generate(
            user_prompt,
            &context_data,
            Some(&project_info),
            config,
        )?;
        
        info!("✅ Generated prompt with selected files");
        Ok(prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gather_comprehensive_context_structure() {
        // This test verifies that the method exists and compiles
        // We can't easily mock the vector store without refactoring,
        // but we can verify the graph search path works.
        
        let temp_dir = std::env::temp_dir().join("miow_test_orchestrator");
        let _ = std::fs::create_dir_all(&temp_dir);
        let db_path = temp_dir.join("test.db");
        
        // Clean up previous run
        if db_path.exists() {
            let _ = std::fs::remove_file(&db_path);
        }

        let orchestrator = MiowOrchestrator::new(db_path.to_str().unwrap())
            .expect("Failed to create orchestrator");

        // Should return empty result but not panic
        let result = orchestrator.gather_comprehensive_context(
            "test prompt",
            &["query".to_string()],
            "test_intent",
            None
        ).await;

        assert!(result.is_ok());
        let context = result.unwrap();
        assert!(context.components.is_empty());
    }
}
