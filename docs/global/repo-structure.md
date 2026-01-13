# Repo structure

.claude/commands/          # global commands (versioned)
.claude/hooks/             # Stop hook for Ralph loop
docs/global/               # global process docs
scripts/ralph/             # hook + helpers (locks)
scripts/cluster/           # optional SLURM wrappers
projects/<slug>/           # each project (isolated state + outputs)
templates/project/         # project scaffold
