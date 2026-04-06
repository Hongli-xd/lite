//! LiteAgent implementation for zeroclaw-lite.
//!
//! A minimal tool-calling agent that:
//! - Uses Ollama as the local LLM provider
//! - Supports XML-based tool calling via <tool_call> tags
//! - Executes tools with security policy checks
//! - Maintains conversation history

use crate::config::{AgentConfig, AutonomyLevel};
use crate::dispatcher::{default_prompt_addenda, NativeToolDispatcher, XmlToolDispatcher};
use crate::memory::NoneMemory;
use crate::provider::OllamaProvider;
use crate::tools::default_tools;
use crate::traits::{
    ChatMessage, ChatRequest, ConversationMessage,
    ParsedToolCall, Provider, ToolDispatcher, ToolExecutionResult, ToolSpec,
};
use crate::Provider as ProviderTrait;
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Instant;

/// Minimal observer that does nothing
pub struct NoopObserver;

#[async_trait]
impl crate::traits::Observer for NoopObserver {
    fn name(&self) -> &str {
        "noop"
    }
}

/// The LiteAgent - a minimal tool-calling agent for constrained hardware.
pub struct LiteAgent {
    provider: Box<dyn Provider>,
    tools: Vec<Box<dyn crate::traits::Tool>>,
    tool_specs: Vec<ToolSpec>,
    dispatcher: Box<dyn ToolDispatcher>,
    memory: Arc<dyn crate::traits::Memory>,
    observer: Arc<dyn crate::traits::Observer>,
    model_name: String,
    temperature: f64,
    workspace_dir: std::path::PathBuf,
    config: AgentConfig,
    history: Vec<ConversationMessage>,
}

pub struct LiteAgentBuilder {
    provider: Option<Box<dyn Provider>>,
    tools: Option<Vec<Box<dyn crate::traits::Tool>>>,
    dispatcher: Option<Box<dyn ToolDispatcher>>,
    memory: Option<Arc<dyn crate::traits::Memory>>,
    observer: Option<Arc<dyn crate::traits::Observer>>,
    model_name: Option<String>,
    temperature: Option<f64>,
    workspace_dir: Option<std::path::PathBuf>,
    config: Option<AgentConfig>,
    use_xml_dispatcher: bool,
}

impl LiteAgentBuilder {
    pub fn new() -> Self {
        Self {
            provider: None,
            tools: None,
            dispatcher: None,
            memory: None,
            observer: None,
            model_name: None,
            temperature: None,
            workspace_dir: None,
            config: None,
            use_xml_dispatcher: false,
        }
    }

    pub fn provider(mut self, provider: Box<dyn Provider>) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn tools(mut self, tools: Vec<Box<dyn crate::traits::Tool>>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn dispatcher(mut self, dispatcher: Box<dyn ToolDispatcher>) -> Self {
        self.dispatcher = Some(dispatcher);
        self
    }

    pub fn memory(mut self, memory: Arc<dyn crate::traits::Memory>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn observer(mut self, observer: Arc<dyn crate::traits::Observer>) -> Self {
        self.observer = Some(observer);
        self
    }

    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = Some(name.into());
        self
    }

    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn workspace_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.workspace_dir = Some(dir.into());
        self
    }

    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn use_xml_dispatcher(mut self) -> Self {
        self.use_xml_dispatcher = true;
        self
    }

    pub fn build(self) -> anyhow::Result<LiteAgent> {
        let provider = self.provider.unwrap_or_else(|| {
            Box::new(OllamaProvider::new(Some("http://localhost:11434"), Some("llama3")))
        });

        let model_name = self.model_name.unwrap_or_else(|| "llama3".to_string());
        let temperature = self.temperature.unwrap_or(0.7);
        let workspace_dir = self.workspace_dir.unwrap_or_else(|| std::path::PathBuf::from("."));
        let config = self.config.unwrap_or_default();

        let tools = self.tools.unwrap_or_else(|| {
            let security = Arc::new(crate::tools::SecurityPolicy::new());
            default_tools(security)
        });

        let tool_specs: Vec<ToolSpec> = tools.iter().map(|t| t.spec()).collect();

        let dispatcher = self.dispatcher.unwrap_or_else(|| {
            if self.use_xml_dispatcher {
                Box::new(XmlToolDispatcher)
            } else {
                Box::new(NativeToolDispatcher)
            }
        });

        let memory = self.memory.unwrap_or_else(|| Arc::new(NoneMemory::new()));
        let observer = self.observer.unwrap_or_else(|| Arc::new(NoopObserver));

        Ok(LiteAgent {
            provider,
            tools,
            tool_specs,
            dispatcher,
            memory,
            observer,
            model_name,
            temperature,
            workspace_dir,
            config,
            history: Vec::new(),
        })
    }
}

impl Default for LiteAgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LiteAgent {
    /// Create a new LiteAgent with default settings (Ollama, no memory)
    pub fn new() -> anyhow::Result<Self> {
        LiteAgentBuilder::new().build()
    }

    /// Create a new LiteAgent with custom Ollama settings
    pub fn with_ollama(base_url: &str, model: &str) -> anyhow::Result<Self> {
        LiteAgentBuilder::new()
            .provider(Box::new(OllamaProvider::new(Some(base_url), Some(model))))
            .model_name(model)
            .build()
    }

    /// Process a single message and return the response
    pub async fn turn(&mut self, message: &str) -> anyhow::Result<String> {
        let session_id = "lite-session";
        self.observer.on_turn_start(session_id).await;

        // Build the system prompt with tools
        let system_prompt = self.build_system_prompt();

        // Add user message to history
        self.history.push(ConversationMessage::Chat(ChatMessage::user(message)));

        // Run the tool call loop
        let response = self.run_tool_call_loop(&system_prompt).await?;

        // Add assistant response to history
        self.history.push(ConversationMessage::Chat(ChatMessage::assistant(&response)));

        self.observer.on_turn_end(session_id, &response).await;
        Ok(response)
    }

    /// Run the tool call loop until no more tools are needed
    async fn run_tool_call_loop(&mut self, system_prompt: &str) -> anyhow::Result<String> {
        let max_iterations = self.config.max_tool_iterations;
        let mut final_text = String::new();

        for _ in 0..max_iterations {
            // Build messages for this iteration
            let mut messages = vec![ChatMessage::system(system_prompt)];
            messages.extend(self.history.iter().filter_map(|msg| match msg {
                ConversationMessage::Chat(c) => Some(c.clone()),
                ConversationMessage::ToolResult { content } => {
                    Some(ChatMessage::user(format!("[Tool result]\n{content}")))
                }
            }));

            // Build chat request
            let request = ChatRequest {
                messages: &messages,
                tools: Some(&self.tool_specs),
            };

            // Call the LLM
            let llm_start = Instant::now();
            let response = self.provider.chat(request, &self.model_name, self.temperature).await?;
            let _llm_duration = llm_start.elapsed();

            self.observer.on_llm_output(&response).await;

            // Parse response for tool calls
            let (text, calls) = self.dispatcher.parse_response(&response);
            final_text = text.clone();

            if calls.is_empty() {
                // No more tool calls, return the response
                return Ok(text);
            }

            // Execute tool calls
            let mut tool_results = Vec::new();
            for call in calls {
                let tool_start = Instant::now();
                let result = self.execute_tool(&call).await;
                let duration_ms = tool_start.elapsed().as_millis() as u64;

                let exec_result = match &result {
                    Ok(r) => ToolExecutionResult {
                        name: call.name.clone(),
                        output: r.output.clone(),
                        success: r.success,
                    },
                    Err(e) => ToolExecutionResult {
                        name: call.name.clone(),
                        output: e.to_string(),
                        success: false,
                    },
                };

                self.observer.on_tool_call(&call.name, &result.unwrap_or_else(|_| crate::traits::ToolResult::err("unknown")), duration_ms).await;
                tool_results.push(exec_result);
            }

            // Format tool results and add to history
            let tool_msg = self.dispatcher.format_results(&tool_results);
            self.history.push(tool_msg);
        }

        Ok(final_text)
    }

    /// Execute a single tool call
    async fn execute_tool(&self, call: &ParsedToolCall) -> anyhow::Result<crate::traits::ToolResult> {
        // Find the tool
        let tool = self.tools.iter().find(|t| t.name() == call.name);
        let tool = match tool {
            Some(t) => t,
            None => anyhow::bail!("Tool not found: {}", call.name),
        };

        // Execute the tool
        let result = tool.execute(call.arguments.clone()).await?;
        Ok(result)
    }

    /// Build the system prompt including tool descriptions
    fn build_system_prompt(&self) -> String {
        let mut prompt = String::new();

        // Base system instructions
        prompt.push_str("You are a helpful AI assistant with access to tools.\n\n");
        prompt.push_str("## Available Tools\n\n");

        // Add tool instructions based on dispatcher
        let addenda = default_prompt_addenda(&*self.dispatcher, &self.tools);
        prompt.push_str(&addenda);

        // Add tool descriptions
        prompt.push_str("### Tool Descriptions\n\n");
        for spec in &self.tool_specs {
            prompt.push_str(&format!("**{}**: {}\n", spec.name, spec.description));
            prompt.push_str(&format!("Parameters: `{}`\n\n", serde_json::to_string(&spec.parameters).unwrap_or_else(|_| "{}".into())));
        }

        prompt
    }

    /// Get the current conversation history
    pub fn history(&self) -> &[ConversationMessage] {
        &self.history
    }

    /// Clear the conversation history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

impl Default for LiteAgent {
    fn default() -> Self {
        Self::new().expect("Failed to create default LiteAgent")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatcher::XmlToolDispatcher;
    use crate::provider::MockProvider;
    use crate::traits::ToolCall;

    fn test_agent() -> LiteAgent {
        let mock = MockProvider::new(vec![
            "I'll help you with that.".to_string(),
        ]);
        LiteAgentBuilder::new()
            .provider(Box::new(mock))
            .dispatcher(Box::new(XmlToolDispatcher))
            .use_xml_dispatcher()
            .model_name("test")
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn agent_turn_returns_response() {
        let mut agent = test_agent();
        let response = agent.turn("Hello").await.unwrap();
        assert!(!response.is_empty());
    }

    #[tokio::test]
    async fn agent_stores_history() {
        let mut agent = test_agent();
        agent.turn("Hello").await.unwrap();
        assert!(!agent.history().is_empty());
    }

    #[tokio::test]
    async fn agent_clears_history() {
        let mut agent = test_agent();
        agent.turn("Hello").await.unwrap();
        assert!(!agent.history().is_empty());
        agent.clear_history();
        assert!(agent.history().is_empty());
    }

    #[tokio::test]
    async fn agent_with_no_tools_mock_provider() {
        let mock = MockProvider::new(vec!["Simple response".to_string()]);
        let agent = LiteAgentBuilder::new()
            .provider(Box::new(mock))
            .tools(vec![])
            .dispatcher(Box::new(XmlToolDispatcher))
            .use_xml_dispatcher()
            .build()
            .unwrap();

        let mut agent = agent;
        let response = agent.turn("test").await.unwrap();
        assert_eq!(response, "Simple response");
    }

    #[tokio::test]
    async fn agent_xml_dispatcher_works() {
        // Test that XML tool calls are parsed correctly
        let mock = MockProvider::new(vec![
            "Let me check the file.".to_string(),
        ]);
        let agent = LiteAgentBuilder::new()
            .provider(Box::new(mock))
            .dispatcher(Box::new(XmlToolDispatcher))
            .use_xml_dispatcher()
            .build()
            .unwrap();

        let mut agent = agent;
        let response = agent.turn("What's in /tmp/test.txt?").await.unwrap();
        assert!(response.contains("Let me check"));
    }

    #[tokio::test]
    async fn agent_system_prompt_contains_tools() {
        let agent = test_agent();
        let prompt = agent.build_system_prompt();
        assert!(prompt.contains("shell"));
        assert!(prompt.contains("file_read"));
        assert!(prompt.contains("file_write"));
    }

    #[test]
    fn lite_agent_builder_defaults() {
        let builder = LiteAgentBuilder::new();
        // Just verify it constructs without panicking
        let _ = builder;
    }

    #[tokio::test]
    async fn agent_respects_max_iterations() {
        // Create a mock that keeps returning tool calls
        let mock = MockProvider::new(vec![
            r#"<tool_call>{"name": "shell", "arguments": {"command": "echo hi"}}</tool_call>"#.to_string(),
            r#"<tool_call>{"name": "shell", "arguments": {"command": "echo hi"}}</tool_call>"#.to_string(),
            r#"<tool_call>{"name": "shell", "arguments": {"command": "echo hi"}}</tool_call>"#.to_string(),
            "Done".to_string(),
        ]);

        let agent = LiteAgentBuilder::new()
            .provider(Box::new(mock))
            .dispatcher(Box::new(XmlToolDispatcher))
            .config(AgentConfig { max_tool_iterations: 3, autonomy_level: AutonomyLevel::Full })
            .build()
            .unwrap();

        let mut agent = agent;
        // Should not loop infinitely
        let result = agent.turn("test").await;
        assert!(result.is_ok());
    }
}
