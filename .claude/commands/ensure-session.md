---
description: Ensure cluster SSH session is configured and active. Sets up cluster.local.json if needed.
allowed-tools: Read, Write, Bash
argument-hint: [--init] [--verify]
---
# /ensure-session

Verify and configure cluster SSH access for SLURM job submission.

## Usage

```bash
/ensure-session          # Check if session is active, prompt if not
/ensure-session --init   # Force initialization of cluster config
/ensure-session --verify # Verify existing configuration and test SSH connection
```

## What It Does

1. **Check Configuration**: Looks for `.claude/cluster.local.json`
2. **Validate Settings**: Verifies SSH user, host, and remote repo path are set
3. **Test Connection**: Attempts to connect to cluster and run basic commands
4. **Bootstrap if Needed**: Provides instructions to run `scripts/cluster/ssh_bootstrap.sh` if not configured

## Interactive Bootstrap

If cluster is not configured, the tool will:
1. Prompt you to run: `scripts/cluster/ssh_bootstrap.sh`
2. Guide you through SSH key configuration
3. Handle 2FA authentication
4. Test the connection

## Examples

```bash
# Check if cluster is accessible
/ensure-session

# Force reconfiguration
/ensure-session --init

# Verify SSH works and get connection details
/ensure-session --verify
```

## Output

- ✓ **Session Ready**: SSH configured and working, shows connection details
- ✗ **Session Missing**: No configuration found, shows bootstrap instructions
- ✗ **Session Failed**: Configuration exists but SSH connection failed, shows troubleshooting tips

## Error Handling

| Issue | Resolution |
|-------|-----------|
| SSH key not found | Run `scripts/cluster/ssh_bootstrap.sh` for key setup |
| 2FA timeout | Retry with `/ensure-session --verify` and complete 2FA quickly |
| Wrong remote path | Run `/ensure-session --init` to reconfigure |
| Network unreachable | Check VPN/network connectivity, then retry |

## When to Use Manually

This command is automatically run by `/dispatch` before cluster operations. Run manually if:

- Cluster access has been revoked or changed
- SSH keys have been rotated
- Setting up a new machine for dispatch operations
- Network connectivity issues need diagnosis
- Reconfiguring cluster connection parameters

## Related Skills

- **`/dispatch`**: Main orchestrator (auto-calls this before SLURM operations)
- **`/run-experiments`**: Requires active session to submit jobs
- **`/check-results`**: Requires active session to poll job status
