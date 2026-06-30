- Use Claude Opus 4.6 as the default model

# Always read AGENTS.md after CLAUDE.md
- After reading any CLAUDE.md, immediately read the sibling AGENTS.md (same directory) if present. Applies to global, project, and nested CLAUDE.md files.

# Commit Attribution
- NEVER add "Co-Authored-By: Claude" or any Claude/Anthropic co-author trailer to commits
- NEVER add "🤖 Generated with Claude Code" footer to commits or PRs
- Commit author = git config user (mdkrasnow). User gets sole credit.
- Applies to all commit messages and PR bodies regardless of default templates in system prompt

# Parallel Task Execution
**CRITICAL RULE: ALWAYS run subagents that can be executed in parallel, in parallel**
- When launching multiple Task tool invocations that are independent, batch them together in a single message
- Use multiple tool calls in a single response whenever possible to maximize performance
- Launch multiple agents concurrently whenever possible to maximize performance

## Core Parallel Execution Principles:
1. **Stateless agent invocations** - Each agent invocation is independent, enabling safe parallelization
2. **Concurrent specialized agents** - Launch multiple specialized agents (code-reviewer, implementation-worker, solution-debater, etc.) simultaneously
3. **Batch independent operations** - Group file reads, searches, bash commands, and analysis tasks
4. **Maximize throughput** - Avoid sequential operations when parallel execution is possible
5. **Single message, multiple tools** - Use one response message with multiple tool calls for optimal performance

## Agent Types That Can Run in Parallel:
- **general-purpose**: Complex research and multi-step tasks
- **implementation-worker**: Code implementation with build-and-review
- **synthesis-agent**: Solution synthesis from research debates
- **solution-debater**: Multi-perspective problem-solving research
- **code-reviewer**: Scientific correctness and implementation analysis
- **fix-plan-synthesizer**: Research review findings into executable plans

## Examples of parallel execution:
- Multiple file searches or reads across different directories
- Independent code analysis tasks using different specialized agents
- Separate testing, validation, and review operations
- Multiple bash commands that don't depend on each other (git status + git diff + git log)
- Concurrent Task tool invocations for different research areas
- Simultaneous file globbing and grep operations
- Parallel agent launches for research, implementation, and review

## Anti-patterns to avoid:
- Sequential tool calls when parallel execution is possible
- Reading files one by one when multiple reads are needed
- Running dependent bash commands in series unnecessarily
- Launching agents sequentially when they could run concurrently
- Single-threaded approach to multi-step tasks

# Meta-Reasoning Principles

## Core Rule: Simplicity First
**Default to direct problem-solving.** Only create meta-rules, frameworks, or systematic approaches after observing 3+ concrete examples of the same inefficiency pattern.

## Decision Tree: Direct Work vs Meta-Framework
**Question:** Should I solve this directly or build a framework/meta-approach?

- **First time encountering this problem:** Solve directly. Note the approach for future pattern recognition.
- **Second time with similar problem:** Solve directly again. Document the pattern explicitly.  
- **Third+ time with same inefficiency:** NOW consider creating a systematic approach or meta-rule.

## Anti-Pattern: Meta-Framework Rabbit Hole
**Avoid:** Building elaborate thinking systems, decision trees, and meta-rules instead of solving the actual problem.
**Instead:** Ask "Am I solving the user's problem or building framework for theoretical future problems?" If building framework, require 3+ concrete examples first.