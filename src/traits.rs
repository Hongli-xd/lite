//! Core trait definitions for zeroclaw-lite.
//!
//! These traits define the minimal interface for a tool-calling agent.
//! All implementations are self-contained with no external crate dependencies.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

// ─── Tool Traits ─────────────────────────────────────────────────────────────

/// Result of a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

impl ToolResult {
    pub fn ok(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
            error: None,
        }
    }

    pub fn err(error: impl Into<String>) -> Self {
        Self {
            success: false,
            output: String::new(),
            error: Some(error.into()),
        }
    }
}

/// Description of a tool for the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Core tool trait — implement for any capability
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name (used in LLM function calling)
    fn name(&self) -> &str;

    /// Human-readable description
    fn description(&self) -> &str;

    /// JSON schema for parameters
    fn parameters_schema(&self) -> serde_json::Value;

    /// Execute the tool with given arguments
    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult>;

    /// Get the full spec for LLM registration
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters_schema(),
        }
    }
}

// ─── Provider Traits ──────────────────────────────────────────────────────────

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
    pub fn tool(content: impl Into<String>) -> Self {
        Self { role: "tool".into(), content: content.into() }
    }
}

/// A tool call requested by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// An LLM response that may contain text, tool calls, or both.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub reasoning_content: Option<String>,
}

impl ChatResponse {
    pub fn text_response(text: impl Into<String>) -> Self {
        Self { text: Some(text.into()), tool_calls: Vec::new(), reasoning_content: None }
    }

    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    pub fn text_or_empty(&self) -> &str {
        self.text.as_deref().unwrap_or("")
    }
}

/// Request payload for provider chat calls.
#[derive(Debug, Clone, Copy)]
pub struct ChatRequest<'a> {
    pub messages: &'a [ChatMessage],
    pub tools: Option<&'a [ToolSpec]>,
}

/// Provider capabilities declaration.
#[derive(Debug, Clone, Default)]
pub struct ProviderCapabilities {
    pub native_tool_calling: bool,
    pub vision: bool,
    pub prompt_caching: bool,
}

/// Provider trait for LLM backends
#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> ProviderCapabilities { ProviderCapabilities::default() }
    fn supports_native_tools(&self) -> bool { self.capabilities().native_tool_calling }

    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<String>;

    async fn chat(
        &self,
        request: ChatRequest<'_>,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<ChatResponse> {
        let text = if let Some(tools) = request.tools {
            if !tools.is_empty() && !self.supports_native_tools() {
                let tool_instructions = build_tool_instructions_text(tools);
                let mut msgs = request.messages.to_vec();
                if let Some(sys) = msgs.iter_mut().find(|m| m.role == "system") {
                    if !sys.content.is_empty() { sys.content.push_str("\n\n"); }
                    sys.content.push_str(&tool_instructions);
                } else {
                    msgs.insert(0, ChatMessage::system(tool_instructions));
                }
                self.chat_with_history(&msgs, model, temperature).await?
            } else {
                self.chat_with_history(request.messages, model, temperature).await?
            }
        } else {
            self.chat_with_history(request.messages, model, temperature).await?
        };
        Ok(ChatResponse { text: Some(text), tool_calls: Vec::new(), reasoning_content: None })
    }

    async fn chat_with_history(&self, messages: &[ChatMessage], model: &str, temperature: f64) -> anyhow::Result<String> {
        let system = messages.iter().find(|m| m.role == "system").map(|m| m.content.as_str());
        let last_user = messages.iter().rfind(|m| m.role == "user").map(|m| m.content.as_str()).unwrap_or("");
        self.chat_with_system(system, last_user, model, temperature).await
    }

    async fn warmup(&self) -> anyhow::Result<()> { Ok(()) }
}

/// Build tool instructions text for prompt-guided tool calling.
pub fn build_tool_instructions_text(tools: &[ToolSpec]) -> String {
    let mut instructions = String::new();
    instructions.push_str("## Tool Use Protocol\n\n");
    instructions.push_str("To use a tool, respond with JSON in <tool_call></tool_call> tags:\n\n");
    instructions.push_str("<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\n");
    instructions.push_str("### Available Tools\n\n");
    for tool in tools {
        instructions.push_str(&format!("**{}**: {}\n", tool.name, tool.description));
        instructions.push_str(&format!("Parameters: `{}`\n\n", serde_json::to_string(&tool.parameters).unwrap_or_else(|_| "{}".into())));
    }
    instructions
}

// ─── Memory Traits ────────────────────────────────────────────────────────────

/// Memory categories for organization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryCategory {
    Core,
    Daily,
    Conversation,
    Custom(String),
}

impl std::fmt::Display for MemoryCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Core => write!(f, "core"),
            Self::Daily => write!(f, "daily"),
            Self::Conversation => write!(f, "conversation"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// A single memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub key: String,
    pub content: String,
    pub category: MemoryCategory,
    pub timestamp: String,
    pub session_id: Option<String>,
    pub score: Option<f64>,
    pub namespace: String,
    pub importance: Option<f64>,
}

impl MemoryEntry {
    pub fn new(key: impl Into<String>, content: impl Into<String>, category: MemoryCategory) -> Self {
        Self {
            id: uuid_simple(),
            key: key.into(),
            content: content.into(),
            category,
            timestamp: chrono_simple(),
            session_id: None,
            score: None,
            namespace: "default".into(),
            importance: None,
        }
    }
}

/// Memory trait — implement for any persistence backend
#[async_trait]
pub trait Memory: Send + Sync {
    fn name(&self) -> &str;
    async fn store(&self, key: &str, content: &str, category: MemoryCategory, session_id: Option<&str>) -> anyhow::Result<()>;
    async fn recall(&self, query: &str, limit: usize, session_id: Option<&str>, _since: Option<&str>, _until: Option<&str>) -> anyhow::Result<Vec<MemoryEntry>>;
    async fn get(&self, key: &str) -> anyhow::Result<Option<MemoryEntry>>;
    async fn list(&self, _category: Option<&MemoryCategory>, _session_id: Option<&str>) -> anyhow::Result<Vec<MemoryEntry>> { Ok(Vec::new()) }
    async fn forget(&self, _key: &str) -> anyhow::Result<bool> { Ok(false) }
    async fn count(&self) -> anyhow::Result<usize> { Ok(0) }
    async fn health_check(&self) -> bool { true }
}

// ─── Observer Traits ──────────────────────────────────────────────────────────

/// Observer trait for telemetry events (all no-op by default)
#[async_trait]
pub trait Observer: Send + Sync {
    fn name(&self) -> &str { "noop" }
    async fn on_llm_input(&self, _messages: &[ChatMessage], _model: &str) {}
    async fn on_llm_output(&self, _response: &ChatResponse) {}
    async fn on_tool_call(&self, _tool: &str, _result: &ToolResult, _duration_ms: u64) {}
    async fn on_turn_start(&self, _session_id: &str) {}
    async fn on_turn_end(&self, _session_id: &str, _response: &str) {}
}

// ─── Dispatcher Traits ───────────────────────────────────────────────────────

/// A parsed tool call from LLM response
#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
    pub tool_call_id: Option<String>,
}

/// Tool execution result for dispatcher
#[derive(Debug, Clone)]
pub struct ToolExecutionResult {
    pub name: String,
    pub output: String,
    pub success: bool,
}

/// A message in a multi-turn conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum ConversationMessage {
    #[serde(rename = "chat")]
    Chat(ChatMessage),
    #[serde(rename = "tool_result")]
    ToolResult { content: String },
}

/// Tool dispatcher trait
pub trait ToolDispatcher: Send + Sync {
    fn parse_response(&self, response: &ChatResponse) -> (String, Vec<ParsedToolCall>);
    fn format_results(&self, results: &[ToolExecutionResult]) -> ConversationMessage;
    fn prompt_instructions(&self, _tools: &[Box<dyn Tool>]) -> String {
        String::new()
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Simple UUID v4-like string (not cryptographically secure, just for IDs)
pub fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let nanos = now.as_nanos();
    let rand: u64 = (nanos as u64) ^ (std::process::id() as u64).wrapping_mul(0x9e3779b97f4a7c15);
    format!("{:016x}-{:04x}-{:04x}-{:04x}-{:012x}",
        nanos as u64, (rand >> 48) as u16, (rand >> 32) as u16 & 0x0fff | 0x4000,
        rand as u16 & 0x3fff | 0x8000, (rand >> 16) as u64)
}

/// Simple RFC3339-like timestamp
pub fn chrono_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = now.as_secs();
    let days = secs / 86400;
    let remaining = secs % 86400;
    let hours = remaining / 3600;
    let mins = (remaining % 3600) / 60;
    let secs = remaining % 60;
    // Simple epoch-based date (days since 1970-01-01)
    let year = (days / 365 + 1970) as i32;
    let _day_of_year = (days % 365) as i32;
    format!("{:04}-01-01T{:02}:{:02}:{:02}Z", year, hours, mins, secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyTool;
    #[async_trait]
    impl Tool for DummyTool {
        fn name(&self) -> &str { "dummy" }
        fn description(&self) -> &str { "A test tool" }
        fn parameters_schema(&self) -> serde_json::Value { serde_json::json!({}) }
        async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
            Ok(ToolResult::ok("done"))
        }
    }

    #[test]
    fn tool_spec_generation() {
        let tool = DummyTool;
        let spec = tool.spec();
        assert_eq!(spec.name, "dummy");
        assert_eq!(spec.description, "A test tool");
    }

    #[test]
    fn tool_result_ok_and_err() {
        let ok = ToolResult::ok("output");
        assert!(ok.success);
        assert_eq!(ok.output, "output");
        assert!(ok.error.is_none());

        let err = ToolResult::err("failed");
        assert!(!err.success);
        assert!(err.error.is_some());
    }

    #[test]
    fn chat_message_helpers() {
        let sys = ChatMessage::system("be helpful");
        assert_eq!(sys.role, "system");
        assert_eq!(sys.content, "be helpful");

        let user = ChatMessage::user("hello");
        assert_eq!(user.role, "user");
    }

    #[test]
    fn chat_response_helpers() {
        let resp = ChatResponse::text_response("hello");
        assert!(!resp.has_tool_calls());
        assert_eq!(resp.text_or_empty(), "hello");
    }

    #[test]
    fn memory_entry_new() {
        let entry = MemoryEntry::new("key", "content", MemoryCategory::Core);
        assert_eq!(entry.key, "key");
        assert_eq!(entry.content, "content");
        assert!(matches!(entry.category, MemoryCategory::Core));
    }

    #[test]
    fn uuid_is_unique() {
        let id1 = uuid_simple();
        let id2 = uuid_simple();
        assert_ne!(id1, id2);
    }
}
