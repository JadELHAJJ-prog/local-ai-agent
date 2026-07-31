# Contributing to local-ai-agent

This project tracks its improvement backlog as [GitHub Issues](https://github.com/JadELHAJJ-prog/local-ai-agent/issues). This guide describes how to pick up an issue, branch, and get your change merged.

## Branch strategy

There are two long-lived branches:

| Branch | Purpose | Direct pushes |
|--------|---------|----------------|
| `master` | Stable, production-ready state | **Never.** Only ever updated by merging a reviewed PR from `dev`. |
| `dev` | Integration branch — where finished issue work lands | **Never.** Only ever updated by merging a reviewed PR from an issue branch. |

Every piece of work happens on a short-lived **issue branch** cut from `dev`, and merges back into `dev` through a pull request. Nobody — including the person who opened the PR — pushes directly to `master` or `dev`, and nobody merges their own PR (see [Review and merge rules](#review-and-merge-rules)).

```
master  ←── (reviewed release PR, occasional) ──  dev
                                                     ↑
                                    (reviewed PR)    │
                                                     │
dev  ──cut──▶ feat/12-coding-subagent ──▶ PR into dev
dev  ──cut──▶ fix/2-double-approval-bug ──▶ PR into dev
dev  ──cut──▶ security/22-docker-hardening ──▶ PR into dev
```

`master` only moves forward via its own separate, reviewed PR from `dev` — treat that as a deliberate release step, not something that happens as a side effect of finishing an issue.

## Branch naming convention

Every issue branch name encodes the issue it resolves, so it's traceable at a glance:

```
<type>/<issue-number>-<short-kebab-slug>
```

- `<issue-number>` — the GitHub issue number this branch resolves (e.g. `12`).
- `<short-kebab-slug>` — a few words from the issue title, lowercase, hyphen-separated.
- `<type>` — pick based on the issue's label:

| Issue label | Branch type prefix |
|-------------|--------------------|
| `security` | `security/` |
| `performance` | `perf/` |
| `bug` | `fix/` |
| `memory-rag`, `new-tool`, `multi-agent`, `observability`, `core-quality`, `enhancement` | `feat/` |

Examples, matching the current backlog:
- `feat/1-llm-input-routing` (issue #1)
- `fix/2-double-approval-bug` (issue #2)
- `feat/12-coding-subagent` (issue #12)
- `perf/15-kv-cache-quant` (issue #15)
- `security/22-docker-hardening` (issue #22)

One branch, one issue, one PR. If an issue turns out to need splitting into multiple PRs, branch off the same base with a `-part2` suffix rather than mixing unrelated issues into one branch.

## Step-by-step workflow

1. **Find an issue.** Browse the [open issues](https://github.com/JadELHAJJ-prog/local-ai-agent/issues) and pick one that isn't already assigned or in progress.
2. **Claim it.** Comment on the issue (or assign yourself) so two people don't duplicate work on the same task.
3. **Sync `dev` locally:**
   ```bash
   git checkout dev
   git pull origin dev
   ```
4. **Cut your issue branch from `dev`** using the naming convention above:
   ```bash
   git checkout -b feat/12-coding-subagent
   ```
5. **Do the work.** Follow the issue's description — context, tech stack, implementation guide, and acceptance criteria are all spelled out there. Commit in small, logical steps; reference the issue number in commit messages (e.g. `Add self-debug loop to code_generation_node (#12)`).
6. **Push your branch:**
   ```bash
   git push -u origin feat/12-coding-subagent
   ```
7. **Open a pull request targeting `dev`** — not `master`. In the PR description:
   - Link the issue with `Closes #12` (or `Refs #12` if it only partially addresses it) so GitHub auto-links and auto-closes it on merge.
   - Summarize what changed and why.
   - Confirm the issue's acceptance criteria are met (copy the checklist from the issue and check items off).
   - Note how you tested it (which commands you ran, what you verified manually).
8. **Request review** — do not merge your own PR, even if it's green and you're confident in it (see below).
9. **Address feedback** with additional commits on the same branch (avoid force-pushing once a review is in progress, so the reviewer can see what changed since their last pass).
10. **Once approved, the reviewer merges it** into `dev`.
11. **Delete the branch** after merge (GitHub will offer to do this automatically).

## Review and merge rules

- **Only the reviewer merges a PR — the author never self-merges,** regardless of approval status. This applies to every PR into both `dev` and `master`.
- Every PR needs at least one approval before it can be merged.
- `master` is only ever updated via a dedicated, separately reviewed PR from `dev` (a "release" PR) — not automatically, and not as a side effect of any individual issue PR.
- Nobody pushes directly to `master` or `dev` under any circumstance, including small fixes or docs — everything goes through a branch and a PR.

## Commit messages

- Reference the issue number so history stays traceable: `Fix double approval prompt on rejection replay (#2)`.
- Prefer explaining *why* over *what* — the diff already shows what changed.

## Before opening a PR, check

- [ ] Branch was cut from an up-to-date `dev`, not `master`.
- [ ] Branch name follows the `<type>/<issue-number>-<slug>` convention.
- [ ] The issue's acceptance criteria (from its GitHub description) are met.
- [ ] Relevant tests pass: `pytest tests/ -v`.
- [ ] The PR description links the issue (`Closes #N`) and summarizes the change.
