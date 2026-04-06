//! Minimal tool implementations for zeroclaw-lite.
//!
//! Provides the core 3 tools needed for a functional agent:
//! - ShellTool: Execute shell commands
//! - FileReadTool: Read files from disk
//! - FileWriteTool: Write files to disk

use crate::traits::{Tool, ToolResult};
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

/// Security policy for constraining tool operations
#[derive(Clone)]
pub struct SecurityPolicy {
    /// Allowed directories for file operations
    pub allowed_dirs: Vec<String>,
    /// Maximum output size in bytes
    pub max_output_bytes: usize,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self {
            allowed_dirs: vec![".".to_string()],
            max_output_bytes: 1024 * 1024, // 1MB
        }
    }
}

impl SecurityPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a path is within allowed directories
    pub fn is_path_allowed(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        for dir in &self.allowed_dirs {
            let allowed = Path::new(dir);
            // Check if path starts with allowed directory
            if path.starts_with(allowed) || path_str.starts_with(dir.as_str()) {
                return true;
            }
            // Also allow if path has no directory component (just a filename)
            if path.file_name().is_some() && path.parent().map_or(true, |p| p.as_os_str().is_empty()) {
                // This is a bare filename like "test.txt" - allow it
                return true;
            }
        }
        false
    }

    /// Check if a shell command is safe (basic check)
    pub fn is_command_safe(&self, cmd: &str) -> bool {
        let dangerous = ["rm -rf /", "dd if=", ":(){:|:&};:", "mkfs", "dd"];
        !dangerous.iter().any(|d| cmd.contains(d))
    }
}

// ─── Shell Tool ────────────────────────────────────────────────────────────────

pub struct ShellTool {
    security: Arc<SecurityPolicy>,
}

impl ShellTool {
    pub fn new(security: Arc<SecurityPolicy>) -> Self {
        Self { security }
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return its output. Use for running programs, scripts, or system commands."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "timeout_secs": {
                    "type": "number",
                    "description": "Maximum seconds to wait (default: 30)",
                    "default": 30
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let _timeout_secs = args
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(30);

        if !self.security.is_command_safe(command) {
            return Ok(ToolResult::err("Command blocked by security policy"));
        }

        let output = tokio::process::Command::new("sh")
            .args(["-c", command])
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let truncated = |s: &str| -> String {
            let max = self.security.max_output_bytes;
            if s.len() > max {
                format!("{}... [truncated {} bytes]", &s[..max], s.len() - max)
            } else {
                s.to_string()
            }
        };

        if output.status.success() {
            Ok(ToolResult::ok(truncated(&stdout)))
        } else {
            Ok(ToolResult::err(format!("{}\n{}", truncated(&stdout), truncated(&stderr))))
        }
    }
}

// ─── File Read Tool ───────────────────────────────────────────────────────────

pub struct FileReadTool {
    security: Arc<SecurityPolicy>,
}

impl FileReadTool {
    pub fn new(security: Arc<SecurityPolicy>) -> Self {
        Self { security }
    }
}

#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &str {
        "file_read"
    }

    fn description(&self) -> &str {
        "Read the contents of a file from disk. Returns the file contents as a string."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "max_bytes": {
                    "type": "number",
                    "description": "Maximum bytes to read (default: 65536)",
                    "default": 65536
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let max_bytes = args
            .get("max_bytes")
            .and_then(|v| v.as_u64())
            .unwrap_or(65536) as usize;

        let path = Path::new(path_str);

        if !self.security.is_path_allowed(path) {
            return Ok(ToolResult::err("Path not allowed by security policy"));
        }

        if !path.exists() {
            return Ok(ToolResult::err(format!("File not found: {}", path_str)));
        }

        let content = tokio::fs::read(path).await?;
        let len = content.len();

        if len > max_bytes {
            let truncated = String::from_utf8_lossy(&content[..max_bytes]);
            return Ok(ToolResult::ok(format!(
                "{}... [truncated {} bytes]",
                truncated,
                len - max_bytes
            )));
        }

        let text = String::from_utf8_lossy(&content).to_string();
        Ok(ToolResult::ok(text))
    }
}

// ─── File Write Tool ──────────────────────────────────────────────────────────

pub struct FileWriteTool {
    security: Arc<SecurityPolicy>,
}

impl FileWriteTool {
    pub fn new(security: Arc<SecurityPolicy>) -> Self {
        Self { security }
    }
}

#[async_trait]
impl Tool for FileWriteTool {
    fn name(&self) -> &str {
        "file_write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let path = Path::new(path_str);

        if !self.security.is_path_allowed(path) {
            return Ok(ToolResult::err("Path not allowed by security policy"));
        }

        if content.len() > self.security.max_output_bytes {
            return Ok(ToolResult::err("Content too large"));
        }

        // Create parent directories if needed
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(path, content).await?;
        Ok(ToolResult::ok(format!("Wrote {} bytes to {}", content.len(), path_str)))
    }
}

/// Get the default minimal tool set (3 tools: shell, file_read, file_write)
pub fn default_tools(security: Arc<SecurityPolicy>) -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(ShellTool::new(security.clone())),
        Box::new(FileReadTool::new(security.clone())),
        Box::new(FileWriteTool::new(security.clone())),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn shell_tool_executes_command() {
        let security = Arc::new(SecurityPolicy::new());
        let tool = ShellTool::new(security);

        let result = tool.execute(serde_json::json!({"command": "echo hello"})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("hello"));
    }

    #[tokio::test]
    async fn shell_tool_blocks_dangerous_commands() {
        let security = Arc::new(SecurityPolicy::new());
        let tool = ShellTool::new(security);

        let result = tool.execute(serde_json::json!({"command": "rm -rf /"})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[tokio::test]
    async fn file_write_and_read_roundtrip() {
        let mut policy = SecurityPolicy::new();
        policy.allowed_dirs.push("/tmp".to_string());
        let security = Arc::new(policy);
        let write_tool = FileWriteTool::new(security.clone());
        let read_tool = FileReadTool::new(security.clone());

        let test_path = "/tmp/zeroclaw_lite_test.txt";
        let test_content = "Hello, World! 测试内容";

        // Write
        let result = write_tool.execute(serde_json::json!({
            "path": test_path,
            "content": test_content
        })).await.unwrap();
        assert!(result.success);

        // Read
        let result = read_tool.execute(serde_json::json!({"path": test_path})).await.unwrap();
        assert!(result.success);
        assert_eq!(result.output, test_content);

        // Cleanup
        tokio::fs::remove_file(test_path).await.ok();
    }

    #[tokio::test]
    async fn file_read_nonexistent_returns_error() {
        let security = Arc::new(SecurityPolicy::new());
        let tool = FileReadTool::new(security);

        let result = tool.execute(serde_json::json!({"path": "/nonexistent/file/path.txt"})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[tokio::test]
    async fn security_policy_default_allows_current_dir() {
        let policy = SecurityPolicy::new();
        assert!(policy.is_path_allowed(Path::new("./test.txt")));
        assert!(policy.is_path_allowed(Path::new("test.txt")));
    }

    #[test]
    fn security_policy_blocks_dangerous_commands() {
        let policy = SecurityPolicy::new();
        assert!(!policy.is_command_safe("rm -rf /"));
        assert!(!policy.is_command_safe("dd if=/dev/zero of=/dev/sda"));
        assert!(policy.is_command_safe("echo hello"));
        assert!(policy.is_command_safe("ls -la"));
    }

    #[test]
    fn shell_tool_schema() {
        let security = Arc::new(SecurityPolicy::new());
        let tool = ShellTool::new(security);
        let schema = tool.parameters_schema();

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["command"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&serde_json::json!("command")));
    }
}
