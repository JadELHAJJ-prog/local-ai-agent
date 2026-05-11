# Docker Sandbox

---
<- [Code Generation Node](code-generation-node.md) | [Home](../README.md) | *(end)* ->

---

## What it is

The Docker sandbox is an isolated environment where the agent executes Python code. Instead of running the generated code directly on the host system, the code runs inside a Docker container with strict resource limits and no network access.

## Why Docker

This is not optional. Without isolation, an LLM agent that executes arbitrary code is a security disaster waiting to happen.

Consider what can go wrong if you run `exec(code)` on the host:
- Code reads your SSH keys and sends them somewhere
- Code deletes your home directory
- Code installs a backdoor
- Code runs an infinite loop and hangs the process

With Docker:
- **Network disabled**: the container can't make any outbound connections
- **Memory capped**: can't OOM the host
- **CPU capped**: can't starve the system
- **Filesystem isolated**: only the temp file is mounted, read-only
- **Timeout**: 30 seconds max, then killed

None of these protections are optional. The agent will eventually generate dangerous code, either because the LLM makes a mistake or because a user asks it to. The sandbox exists to make these failures harmless.

## The Dockerfile

```dockerfile
# docker/Dockerfile.sandbox
FROM python:3.11-slim

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    scipy \
    requests \
    pillow \
    sympy \
    statsmodels

WORKDIR /app
```

A minimal Python 3.11 image with commonly needed packages pre-installed. The agent's code can use numpy, pandas, matplotlib etc. without any package installation step.

Build the image:
```bash
docker build -f docker/Dockerfile.sandbox -t agent-sandbox .
```

## The execute_code tool

```python
# src/tools.py
@tool
def execute_code(code: str) -> str:
    """Execute Python code safely in an isolated Docker container."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        result = subprocess.run(
            [
                "docker", "run",
                "--rm",              # delete container after exit
                "--network", "none", # no internet
                "--memory", "128m",  # 128MB RAM limit
                "--cpus", "0.5",     # half a CPU core
                "-v", f"{tmp_path}:/app/code.py:ro",  # mount code file, read-only
                SANDBOX_IMAGE,       # "agent-sandbox"
                "python", "/app/code.py",
            ],
            capture_output=True,
            text=True,
            timeout=30,             # kill after 30 seconds
        )

        if result.returncode == 0:
            return result.stdout or "Code executed successfully with no output."
        else:
            return f"Error:\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 30 seconds."
    except Exception as e:
        return f"Error: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
```

The code is:
1. Written to a temporary `.py` file on the host
2. Mounted into the container at `/app/code.py` as read-only
3. Executed with `python /app/code.py`
4. stdout/stderr captured and returned as a string
5. Temp file deleted after execution (finally block)

## Security flags explained

| Flag | Value | Purpose |
|------|-------|---------|
| `--rm` | - | Container deleted immediately after exit. No persistent state. |
| `--network none` | - | No network interfaces. Cannot make HTTP requests, DNS lookups, or connect to anything. |
| `--memory 128m` | 128 MB | Container killed if it tries to allocate more than 128MB RAM. |
| `--cpus 0.5` | 0.5 cores | Container gets at most half a CPU core. Can't starve the host. |
| `-v path:/app/code.py:ro` | read-only | Code file is mounted read-only. Container can't modify it or write to the host filesystem at that path. |

The `requests` package in the sandbox image might seem dangerous given `--network none`. But with no network interfaces, any HTTP request will immediately fail with a connection error - the container can't reach anywhere regardless of what packages are installed.

## Gotchas and lessons learned

- **Docker must be running.** If Docker daemon isn't running when execute_code is called, you get a `FileNotFoundError: [Errno 2] No such file or directory: 'docker'` or similar. The agent returns this as an error string - it doesn't crash the graph.
- **The sandbox has no access to the host filesystem.** Code that tries to read files from the host (e.g., `open("/home/user/data.csv")`) will fail with `FileNotFoundError`. This is by design.
- **matplotlib requires a display.** If code tries to show a plot with `plt.show()`, it will hang or error because there's no display in the container. Tell the model to save figures to a file instead: `plt.savefig("/app/output.png")`. But `/app/` is also not mounted back to the host, so the file won't be accessible. This is a known limitation.
- **The 30-second timeout is a soft deadline.** `subprocess.run(timeout=30)` sends SIGKILL after 30 seconds, but the container process might not die immediately. In practice, it terminates within a second of the timeout.
- **SANDBOX_IMAGE is configurable.** The image name `"agent-sandbox"` comes from `config.py` via the `.env` file. If you rename the image when building, update `SANDBOX_IMAGE` in `.env`.

---

<- [Code Generation Node](code-generation-node.md) | [Home](../README.md) | *(end)* ->

**All pages:** [Home](../README.md) · [Introduction](../00-introduction/README.md) · [01 Foundations](../01-foundations/README.md) · [Quantization](../01-foundations/quantization.md) · [LLM Basics](../01-foundations/llm-basics.md) · [02 LangChain](../02-langchain/README.md) · [Messages](../02-langchain/messages.md) · [Prompt Templates](../02-langchain/prompt-templates.md) · [Tools](../02-langchain/tools.md) · [Pipe Operator](../02-langchain/pipe-operator.md) · [Runnables](../02-langchain/runnables.md) · [03 LangGraph](../03-langgraph/README.md) · [State](../03-langgraph/state.md) · [Nodes & Edges](../03-langgraph/nodes-edges.md) · [Conditional Edges](../03-langgraph/conditional-edges.md) · [Memory](../03-langgraph/memory.md) · [Human-in-the-Loop](../03-langgraph/human-in-the-loop.md) · [04 Building the Agent](README.md) · [V1 Basic Agent](v1-basic-agent.md) · [Input Router](input-router.md) · [Tools & ReAct Loop](tools-react-loop.md) · [Code Generation Node](code-generation-node.md) · [Docker Sandbox](docker-sandbox.md)
