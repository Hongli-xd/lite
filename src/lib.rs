//! zeroclaw-lite: A minimal tool-calling agent for constrained hardware.
//!
//! This crate provides a lightweight AI agent runtime optimized for:
//! - Limited RAM environments (<50MB overhead)
//! - Local LLM inference via Ollama
//! - Basic tool execution (shell, file read/write)
//! - No persistent storage requirement
//!
//! # Example
//!
//! ```ignore
//! use zeroclaw_lite::{LiteAgent, LiteAgentBuilder};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut agent = LiteAgent::new()?;
//!     let response = agent.turn("Hello!").await?;
//!     println!("{}", response);
//!     Ok(())
//! }
//! ```

pub mod agent;
pub mod config;
pub mod dispatcher;
pub mod memory;
pub mod provider;
pub mod tools;
pub mod traits;

// Re-exports for convenience
pub use agent::{LiteAgent, LiteAgentBuilder};
pub use config::{AgentConfig, AutonomyLevel, LiteConfig, ResolvedConfig, find_config_file, load_config_file, create_default_config};
pub use dispatcher::{NativeToolDispatcher, XmlToolDispatcher};
pub use memory::NoneMemory;
pub use provider::{MockProvider, MiniMaxApiType, MiniMaxConfig, MiniMaxProvider, OllamaConfig, OllamaProvider};
pub use tools::{default_tools, SecurityPolicy};
pub use traits::{
    build_tool_instructions_text, ChatMessage, ChatRequest, ChatResponse, ConversationMessage,
    Memory, MemoryCategory, MemoryEntry, Observer, ParsedToolCall, Provider,
    ProviderCapabilities, Tool, ToolDispatcher, ToolExecutionResult, ToolResult, ToolSpec,
};
