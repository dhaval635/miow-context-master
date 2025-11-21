use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Self-monitoring system for agent health
pub struct SelfMonitor {
    execution_history: Vec<ExecutionRecord>,
    health_metrics: HealthMetrics,
    loop_detector: LoopDetector,
}

#[derive(Debug, Clone)]
struct ExecutionRecord {
    step_id: String,
    started_at: Instant,
    completed_at: Option<Instant>,
    success: bool,
    error: Option<String>,
    retry_count: u32,
}

#[derive(Debug, Clone, Default)]
pub struct HealthMetrics {
    pub total_steps: usize,
    pub successful_steps: usize,
    pub failed_steps: usize,
    pub average_step_duration: Duration,
    pub stuck_count: usize,
    pub loop_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthIssue {
    StuckState {
        step_id: String,
        duration: u64, // seconds
    },
    HighFailureRate {
        rate: f32,
    },
    InfiniteLoop {
        pattern: String,
    },
    Timeout {
        step_id: String,
        expected: u64,
        actual: u64,
    },
    ExcessiveRetries {
        step_id: String,
        retry_count: u32,
    },
}

struct LoopDetector {
    recent_steps: Vec<String>,
    max_history: usize,
}

impl SelfMonitor {
    pub fn new() -> Self {
        Self {
            execution_history: Vec::new(),
            health_metrics: HealthMetrics::default(),
            loop_detector: LoopDetector {
                recent_steps: Vec::new(),
                max_history: 20,
            },
        }
    }
    
    /// Record start of step execution
    pub fn record_step_start(&mut self, step_id: String) {
        self.execution_history.push(ExecutionRecord {
            step_id: step_id.clone(),
            started_at: Instant::now(),
            completed_at: None,
            success: false,
            error: None,
            retry_count: 0,
        });
        
        self.loop_detector.add_step(step_id);
        self.health_metrics.total_steps += 1;
    }
    
    /// Record step completion
    pub fn record_step_complete(&mut self, step_id: &str, success: bool, error: Option<String>) {
        if let Some(record) = self.execution_history.iter_mut().rev().find(|r| r.step_id == step_id) {
            record.completed_at = Some(Instant::now());
            record.success = success;
            record.error = error;
            
            if success {
                self.health_metrics.successful_steps += 1;
            } else {
                self.health_metrics.failed_steps += 1;
            }
            
            // Update average duration
            let duration = record.completed_at.unwrap() - record.started_at;
            self.update_average_duration(duration);
        }
    }
    
    /// Record retry attempt
    pub fn record_retry(&mut self, step_id: &str) {
        if let Some(record) = self.execution_history.iter_mut().rev().find(|r| r.step_id == step_id) {
            record.retry_count += 1;
        }
    }
    
    fn update_average_duration(&mut self, new_duration: Duration) {
        let total_completed = self.health_metrics.successful_steps + self.health_metrics.failed_steps;
        if total_completed == 0 {
            self.health_metrics.average_step_duration = new_duration;
        } else {
            let current_total = self.health_metrics.average_step_duration * (total_completed - 1) as u32;
            self.health_metrics.average_step_duration = (current_total + new_duration) / total_completed as u32;
        }
    }
    
    /// Check for health issues
    pub fn check_health(&mut self) -> Vec<HealthIssue> {
        let mut issues = Vec::new();
        
        // Check for stuck states
        if let Some(stuck) = self.check_stuck_state() {
            issues.push(stuck);
            self.health_metrics.stuck_count += 1;
        }
        
        // Check failure rate
        if let Some(failure) = self.check_failure_rate() {
            issues.push(failure);
        }
        
        // Check for infinite loops
        if let Some(loop_issue) = self.loop_detector.detect_loop() {
            issues.push(loop_issue);
            self.health_metrics.loop_count += 1;
        }
        
        // Check for timeouts
        issues.extend(self.check_timeouts());
        
        // Check for excessive retries
        issues.extend(self.check_excessive_retries());
        
        issues
    }
    
    fn check_stuck_state(&self) -> Option<HealthIssue> {
        if let Some(last) = self.execution_history.last() {
            if last.completed_at.is_none() {
                let duration = last.started_at.elapsed();
                if duration > Duration::from_secs(120) {
                    return Some(HealthIssue::StuckState {
                        step_id: last.step_id.clone(),
                        duration: duration.as_secs(),
                    });
                }
            }
        }
        None
    }
    
    fn check_failure_rate(&self) -> Option<HealthIssue> {
        if self.health_metrics.total_steps > 5 {
            let failure_rate = self.health_metrics.failed_steps as f32 
                / self.health_metrics.total_steps as f32;
            
            if failure_rate > 0.5 {
                return Some(HealthIssue::HighFailureRate { rate: failure_rate });
            }
        }
        None
    }
    
    fn check_timeouts(&self) -> Vec<HealthIssue> {
        let mut issues = Vec::new();
        
        for record in &self.execution_history {
            if let Some(completed_at) = record.completed_at {
                let duration = completed_at - record.started_at;
                // Assume 60s is the default timeout
                if duration > Duration::from_secs(60) {
                    issues.push(HealthIssue::Timeout {
                        step_id: record.step_id.clone(),
                        expected: 60,
                        actual: duration.as_secs(),
                    });
                }
            }
        }
        
        issues
    }
    
    fn check_excessive_retries(&self) -> Vec<HealthIssue> {
        let mut issues = Vec::new();
        
        for record in &self.execution_history {
            if record.retry_count > 3 {
                issues.push(HealthIssue::ExcessiveRetries {
                    step_id: record.step_id.clone(),
                    retry_count: record.retry_count,
                });
            }
        }
        
        issues
    }
    
    /// Suggest corrections for issues
    pub fn suggest_corrections(&self, issues: &[HealthIssue]) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        for issue in issues {
            match issue {
                HealthIssue::StuckState { step_id, duration } => {
                    suggestions.push(format!(
                        "Step '{}' stuck for {}s. Suggestions:\n\
                         1. Timeout and retry with different approach\n\
                         2. Skip to fallback step\n\
                         3. Abort and replan with more context",
                        step_id, duration
                    ));
                }
                HealthIssue::HighFailureRate { rate } => {
                    suggestions.push(format!(
                        "High failure rate ({:.1}%). Suggestions:\n\
                         1. Revise execution plan\n\
                         2. Gather more context before proceeding\n\
                         3. Use different tools or approaches\n\
                         4. Break down complex steps into smaller ones",
                        rate * 100.0
                    ));
                }
                HealthIssue::InfiniteLoop { pattern } => {
                    suggestions.push(format!(
                        "Infinite loop detected: {}. Suggestions:\n\
                         1. Break loop with new approach\n\
                         2. Add loop counter and exit condition\n\
                         3. Abort and replan with different strategy",
                        pattern
                    ));
                }
                HealthIssue::Timeout { step_id, expected, actual } => {
                    suggestions.push(format!(
                        "Step '{}' timed out (expected: {}s, actual: {}s). Suggestions:\n\
                         1. Increase timeout for this step\n\
                         2. Split into smaller, faster steps\n\
                         3. Use fallback approach\n\
                         4. Optimize the operation",
                        step_id, expected, actual
                    ));
                }
                HealthIssue::ExcessiveRetries { step_id, retry_count } => {
                    suggestions.push(format!(
                        "Step '{}' retried {} times. Suggestions:\n\
                         1. This approach may not work, try fallback\n\
                         2. Gather more information before retrying\n\
                         3. Adjust parameters or approach\n\
                         4. Skip this step if not critical",
                        step_id, retry_count
                    ));
                }
            }
        }
        
        suggestions
    }
    
    /// Get health metrics
    pub fn get_metrics(&self) -> &HealthMetrics {
        &self.health_metrics
    }
    
    /// Clear history (keep only recent records)
    pub fn cleanup_history(&mut self, keep_recent: usize) {
        if self.execution_history.len() > keep_recent {
            self.execution_history.drain(0..self.execution_history.len() - keep_recent);
        }
    }
}

impl LoopDetector {
    fn add_step(&mut self, step_id: String) {
        self.recent_steps.push(step_id);
        
        if self.recent_steps.len() > self.max_history {
            self.recent_steps.remove(0);
        }
    }
    
    fn detect_loop(&self) -> Option<HealthIssue> {
        // Check for simple loops (same step repeated)
        if self.recent_steps.len() >= 3 {
            let last_three = &self.recent_steps[self.recent_steps.len() - 3..];
            if last_three[0] == last_three[1] && last_three[1] == last_three[2] {
                return Some(HealthIssue::InfiniteLoop {
                    pattern: format!("Repeated step: {}", last_three[0]),
                });
            }
        }
        
        // Check for pattern loops (A-B-A-B-A-B)
        if self.recent_steps.len() >= 6 {
            let pattern1 = &self.recent_steps[self.recent_steps.len() - 6..self.recent_steps.len() - 3];
            let pattern2 = &self.recent_steps[self.recent_steps.len() - 3..];
            
            if pattern1 == pattern2 {
                return Some(HealthIssue::InfiniteLoop {
                    pattern: format!("Repeated pattern: {:?}", pattern1),
                });
            }
        }
        
        None
    }
}

impl Default for SelfMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_loop_detection() {
        let mut detector = LoopDetector {
            recent_steps: Vec::new(),
            max_history: 20,
        };
        
        // Add repeated steps
        detector.add_step("step_1".to_string());
        detector.add_step("step_1".to_string());
        detector.add_step("step_1".to_string());
        
        assert!(detector.detect_loop().is_some());
    }
    
    #[test]
    fn test_pattern_loop_detection() {
        let mut detector = LoopDetector {
            recent_steps: Vec::new(),
            max_history: 20,
        };
        
        // Add pattern A-B-A-B-A-B
        detector.add_step("A".to_string());
        detector.add_step("B".to_string());
        detector.add_step("C".to_string());
        detector.add_step("A".to_string());
        detector.add_step("B".to_string());
        detector.add_step("C".to_string());
        
        assert!(detector.detect_loop().is_some());
    }
    
    #[test]
    fn test_health_metrics() {
        let mut monitor = SelfMonitor::new();
        
        monitor.record_step_start("step_1".to_string());
        std::thread::sleep(Duration::from_millis(10));
        monitor.record_step_complete("step_1", true, None);
        
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_steps, 1);
        assert_eq!(metrics.successful_steps, 1);
        assert_eq!(metrics.failed_steps, 0);
    }
}
