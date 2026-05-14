"""
Unit tests for every graph flow in the local-ai-agent.

Each test class covers one routing or node function in isolation.
All external calls (LLMs, Docker, DuckDuckGo, OpenCV, filesystem) are mocked
so the suite runs without Ollama, Docker, or any real files on disk.

Flows covered:
  Flow 1  - input_router_node + should_route      (entry point dispatch)
  Flow 2  - agent_node + should_use_tool          (general / tool / code paths)
  Flow 3  - code_generation_node + _strip_markdown + _build_code_prompt
  Flow 4  - human_approval_node + should_execute_tool
  Flow 5  - output_parser_node + should_retry
  Flow 6  - search_web tool
  Flow 7  - execute_code tool
  Flow 8  - analyze_image tool
  Flow 9  - analyze_video tool
  Flow 10 - analyze_document tool
  Flow 11 - trim_messages_window
  Flow 12 - parse_user_input (main.py utility)
"""

import io
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from nodes import (
    _build_code_prompt,
    _strip_markdown,
    code_generation_node,
    human_approval_node,
    input_router_node,
    output_parser_node,
    should_execute_tool,
    should_retry,
    should_route,
    should_use_tool,
    trim_messages_window,
)


# ---------------------------------------------------------------------------
# Shared helper: build a minimal AgentState dict
# ---------------------------------------------------------------------------

def make_state(**overrides):
    base = {
        "messages": [],
        "input_type": "general",
        "retry_count": 0,
        "is_valid": True,
        "human_feedback": None,
        "approved": None,
        "thread_id": "test-thread",
    }
    base.update(overrides)
    return base


# ===========================================================================
# Flow 1 — input_router_node + should_route
# ===========================================================================

class TestInputRouterNode:
    """input_router_node classifies the last message and writes input_type to state."""

    def test_greeting_routes_to_general(self):
        state = make_state(messages=[HumanMessage(content="Hello, how are you?")])
        assert input_router_node(state)["input_type"] == "general"

    def test_math_question_routes_to_general(self):
        state = make_state(messages=[HumanMessage(content="what is 2 plus 2?")])
        assert input_router_node(state)["input_type"] == "general"

    def test_write_keyword_routes_to_code(self):
        state = make_state(messages=[HumanMessage(content="write a python function to sort a list")])
        assert input_router_node(state)["input_type"] == "code"

    def test_generate_keyword_routes_to_code(self):
        state = make_state(messages=[HumanMessage(content="generate a script to parse JSON")])
        assert input_router_node(state)["input_type"] == "code"

    def test_implement_keyword_routes_to_code(self):
        state = make_state(messages=[HumanMessage(content="implement a binary search")])
        assert input_router_node(state)["input_type"] == "code"

    def test_image_path_marker_routes_to_media(self):
        msg = "[image provided at path: /tmp/photo.png] what is in this image?"
        state = make_state(messages=[HumanMessage(content=msg)])
        assert input_router_node(state)["input_type"] == "media"

    def test_pdf_file_marker_routes_to_document_pdf(self):
        msg = "summarize this [file provided at path: /tmp/report.pdf]"
        state = make_state(messages=[HumanMessage(content=msg)])
        assert input_router_node(state)["input_type"] == "document_pdf"

    def test_docx_file_marker_routes_to_document_docx(self):
        msg = "read [file provided at path: /tmp/notes.docx]"
        state = make_state(messages=[HumanMessage(content=msg)])
        assert input_router_node(state)["input_type"] == "document_docx"

    def test_xlsx_file_marker_routes_to_document_xlsx(self):
        msg = "analyze [file provided at path: /data/sheet.xlsx]"
        state = make_state(messages=[HumanMessage(content=msg)])
        assert input_router_node(state)["input_type"] == "document_xlsx"

    def test_csv_file_marker_routes_to_document_csv(self):
        msg = "analyze [file provided at path: /data/data.csv]"
        state = make_state(messages=[HumanMessage(content=msg)])
        assert input_router_node(state)["input_type"] == "document_csv"


class TestShouldRoute:
    """should_route reads input_type and returns the correct node name."""

    def test_code_type_goes_to_code_generation_node(self):
        assert should_route(make_state(input_type="code")) == "code_generation_node"

    def test_general_type_goes_to_agent_node(self):
        assert should_route(make_state(input_type="general")) == "agent_node"

    def test_media_type_goes_to_agent_node(self):
        assert should_route(make_state(input_type="media")) == "agent_node"

    def test_document_type_goes_to_agent_node(self):
        assert should_route(make_state(input_type="document_pdf")) == "agent_node"

    def test_missing_input_type_defaults_to_agent_node(self):
        state = make_state()
        del state["input_type"]
        assert should_route(state) == "agent_node"


# ===========================================================================
# Flow 2 — agent_node + should_use_tool
# ===========================================================================

class TestShouldUseTool:
    """should_use_tool inspects the last AIMessage and picks the next node."""

    def test_no_tool_call_goes_to_output_parser(self):
        msg = AIMessage(content="Here is your answer.")
        assert should_use_tool(make_state(messages=[msg])) == "output_parser_node"

    def test_execute_code_call_goes_to_code_generation(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "execute_code", "args": {"code": "print(1)"}, "id": "c1", "type": "tool_call"}],
        )
        assert should_use_tool(make_state(messages=[msg])) == "code_generation_node"

    def test_search_web_call_goes_to_tool_node(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "search_web", "args": {"query": "AI"}, "id": "c2", "type": "tool_call"}],
        )
        assert should_use_tool(make_state(messages=[msg])) == "tool_node"

    def test_analyze_image_call_goes_to_tool_node(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "analyze_image", "args": {"image_path": "/x.png"}, "id": "c3", "type": "tool_call"}],
        )
        assert should_use_tool(make_state(messages=[msg])) == "tool_node"

    def test_analyze_video_call_goes_to_tool_node(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "analyze_video", "args": {"video_path": "/x.mp4"}, "id": "c4", "type": "tool_call"}],
        )
        assert should_use_tool(make_state(messages=[msg])) == "tool_node"

    def test_analyze_document_call_goes_to_tool_node(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "analyze_document", "args": {"file_path": "/x.pdf"}, "id": "c5", "type": "tool_call"}],
        )
        assert should_use_tool(make_state(messages=[msg])) == "tool_node"


# ===========================================================================
# Flow 3 — code_generation_node + helpers
# ===========================================================================

class TestBuildCodePrompt:
    """_build_code_prompt composes the instruction header for the coder LLM."""

    def test_action_appears_in_output(self):
        result = _build_code_prompt("Write Python code for this task", "sort a list")
        assert "Write Python code for this task" in result

    def test_content_appears_in_output(self):
        result = _build_code_prompt("Improve and optimize this code", "print('hi')")
        assert "print('hi')" in result

    def test_no_markdown_instruction_is_present(self):
        result = _build_code_prompt("Write Python code for this task", "anything")
        assert "no markdown" in result

    def test_no_explanation_instruction_is_present(self):
        result = _build_code_prompt("Write Python code for this task", "anything")
        assert "no explanation" in result


class TestStripMarkdown:
    """_strip_markdown removes fenced code blocks added by the model."""

    def test_strips_python_fenced_block(self):
        assert _strip_markdown("```python\nprint('hi')\n```") == "print('hi')"

    def test_strips_generic_fenced_block(self):
        assert _strip_markdown("```\nprint('hi')\n```") == "print('hi')"

    def test_returns_raw_text_when_no_fences(self):
        assert _strip_markdown("print('hi')") == "print('hi')"

    def test_prefers_python_fence_over_generic_when_both_present(self):
        text = "some text\n```python\nprint('hi')\n```\nmore"
        assert _strip_markdown(text) == "print('hi')"

    def test_strips_surrounding_whitespace(self):
        assert _strip_markdown("```python\n  x = 1  \n```") == "x = 1"


class TestCodeGenerationNode:
    """code_generation_node has two entry paths: from the router and from the agent."""

    def _mock_coder_response(self, code):
        mock = MagicMock()
        mock.content = code
        return mock

    @patch("nodes.coder_llm")
    def test_router_entry_creates_execute_code_tool_call(self, mock_coder):
        mock_coder.invoke.return_value = self._mock_coder_response("print('hello')")
        state = make_state(messages=[HumanMessage(content="write a hello world script")])
        result = code_generation_node(state)
        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)
        assert msg.tool_calls[0]["name"] == "execute_code"
        assert msg.tool_calls[0]["args"]["code"] == "print('hello')"

    @patch("nodes.coder_llm")
    def test_router_entry_strips_markdown_from_coder_output(self, mock_coder):
        mock_coder.invoke.return_value = self._mock_coder_response("```python\nprint('hello')\n```")
        state = make_state(messages=[HumanMessage(content="write a hello world script")])
        result = code_generation_node(state)
        assert result["messages"][0].tool_calls[0]["args"]["code"] == "print('hello')"

    @patch("nodes.coder_llm")
    def test_agent_entry_improves_rough_code(self, mock_coder):
        mock_coder.invoke.return_value = self._mock_coder_response("x = [i for i in range(10)]")
        rough_msg = AIMessage(
            id="msg-1",
            content="",
            tool_calls=[{"name": "execute_code", "args": {"code": "x = list(range(10))"}, "id": "call-1", "type": "tool_call"}],
        )
        state = make_state(messages=[rough_msg])
        result = code_generation_node(state)
        updated = result["messages"][0]
        assert updated.tool_calls[0]["args"]["code"] == "x = [i for i in range(10)]"

    @patch("nodes.coder_llm")
    def test_agent_entry_preserves_original_message_id(self, mock_coder):
        mock_coder.invoke.return_value = self._mock_coder_response("pass")
        rough_msg = AIMessage(
            id="original-id",
            content="",
            tool_calls=[{"name": "execute_code", "args": {"code": "pass"}, "id": "call-99", "type": "tool_call"}],
        )
        state = make_state(messages=[rough_msg])
        result = code_generation_node(state)
        assert result["messages"][0].id == "original-id"
        assert result["messages"][0].tool_calls[0]["id"] == "call-99"


# ===========================================================================
# Flow 4 — human_approval_node + should_execute_tool
# ===========================================================================

class TestHumanApprovalNode:
    """human_approval_node interrupts the graph and records the human decision."""

    def _state_with_execute_code(self, code="print('hello')"):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "execute_code", "args": {"code": code}, "id": "call-1", "type": "tool_call"}],
        )
        return make_state(messages=[msg])

    @patch("nodes.interrupt", return_value="yes")
    def test_yes_sets_approved_true(self, _):
        result = human_approval_node(self._state_with_execute_code())
        assert result["approved"] is True

    @patch("nodes.interrupt", return_value="looks good")
    def test_approval_phrase_sets_approved_true(self, _):
        result = human_approval_node(self._state_with_execute_code())
        assert result["approved"] is True

    @patch("nodes.interrupt", return_value="run it")
    def test_run_it_phrase_sets_approved_true(self, _):
        result = human_approval_node(self._state_with_execute_code())
        assert result["approved"] is True

    @patch("nodes.interrupt", return_value="no, use a function")
    def test_no_response_sets_approved_false(self, _):
        result = human_approval_node(self._state_with_execute_code())
        assert result["approved"] is False

    @patch("nodes.interrupt", return_value="no, use a function")
    def test_rejection_injects_human_message_with_feedback(self, _):
        result = human_approval_node(self._state_with_execute_code())
        assert "messages" in result
        msg = result["messages"][0]
        assert isinstance(msg, HumanMessage)
        assert "no, use a function" in msg.content

    @patch("nodes.interrupt", return_value="yes")
    def test_approval_does_not_inject_any_message(self, _):
        result = human_approval_node(self._state_with_execute_code())
        assert "messages" not in result

    @patch("nodes.interrupt", return_value="yes")
    def test_human_feedback_stored_in_result(self, _):
        result = human_approval_node(self._state_with_execute_code())
        assert result["human_feedback"] == "yes"


class TestShouldExecuteTool:
    """should_execute_tool reads the approved flag and picks the next node."""

    def test_approved_true_goes_to_tool_node(self):
        assert should_execute_tool(make_state(approved=True)) == "tool_node"

    def test_approved_false_goes_to_agent_node(self):
        assert should_execute_tool(make_state(approved=False)) == "agent_node"

    def test_approved_none_goes_to_agent_node(self):
        assert should_execute_tool(make_state(approved=None)) == "agent_node"

    def test_missing_approved_defaults_to_agent_node(self):
        state = make_state()
        del state["approved"]
        assert should_execute_tool(state) == "agent_node"


# ===========================================================================
# Flow 5 — output_parser_node + should_retry
# ===========================================================================

class TestOutputParserNode:
    """output_parser_node validates the last message and tracks retries."""

    def test_non_empty_response_is_valid(self):
        state = make_state(messages=[AIMessage(content="Hello!")], retry_count=0)
        result = output_parser_node(state)
        assert result["is_valid"] is True
        assert result["retry_count"] == 0

    def test_empty_response_is_invalid_and_increments_retry(self):
        state = make_state(messages=[AIMessage(content="")], retry_count=0)
        result = output_parser_node(state)
        assert result["is_valid"] is False
        assert result["retry_count"] == 1

    def test_whitespace_only_response_is_invalid(self):
        state = make_state(messages=[AIMessage(content="   \n  ")], retry_count=1)
        result = output_parser_node(state)
        assert result["is_valid"] is False
        assert result["retry_count"] == 2

    def test_retry_count_not_incremented_on_valid_response(self):
        state = make_state(messages=[AIMessage(content="answer")], retry_count=2)
        result = output_parser_node(state)
        assert result["retry_count"] == 2


class TestShouldRetry:
    """should_retry decides whether to loop back to the agent or end the graph."""

    def test_valid_response_ends_the_graph(self):
        assert should_retry(make_state(is_valid=True, retry_count=0)) == "end"

    def test_invalid_response_retries_when_budget_remains(self):
        assert should_retry(make_state(is_valid=False, retry_count=1)) == "retry"

    def test_invalid_response_ends_when_retry_count_reaches_three(self):
        assert should_retry(make_state(is_valid=False, retry_count=3)) == "end"

    def test_invalid_response_ends_when_retry_count_exceeds_three(self):
        assert should_retry(make_state(is_valid=False, retry_count=5)) == "end"

    def test_invalid_response_retries_at_count_zero(self):
        assert should_retry(make_state(is_valid=False, retry_count=0)) == "retry"

    def test_invalid_response_retries_at_count_two(self):
        assert should_retry(make_state(is_valid=False, retry_count=2)) == "retry"


# ===========================================================================
# Flow 6 — search_web tool
# ===========================================================================

class TestSearchWeb:
    """search_web wraps DuckDuckGo and formats results as readable text."""

    @patch("tools.DDGS")
    def test_formats_results_as_title_url_summary(self, mock_ddgs_class):
        from tools import search_web
        mock_ctx = MagicMock()
        mock_ddgs_class.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ddgs_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ctx.text.return_value = [
            {"title": "AI News", "href": "https://example.com", "body": "Latest AI"}
        ]
        result = search_web.invoke({"query": "AI news"})
        assert "AI News" in result
        assert "https://example.com" in result
        assert "Latest AI" in result

    @patch("tools.DDGS")
    def test_returns_no_results_message_on_empty_response(self, mock_ddgs_class):
        from tools import search_web
        mock_ctx = MagicMock()
        mock_ddgs_class.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ddgs_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ctx.text.return_value = []
        result = search_web.invoke({"query": "nothing"})
        assert result == "No results found."

    @patch("tools.DDGS")
    def test_multiple_results_are_all_included(self, mock_ddgs_class):
        from tools import search_web
        mock_ctx = MagicMock()
        mock_ddgs_class.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ddgs_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ctx.text.return_value = [
            {"title": "Result 1", "href": "https://a.com", "body": "Body 1"},
            {"title": "Result 2", "href": "https://b.com", "body": "Body 2"},
        ]
        result = search_web.invoke({"query": "test"})
        assert "Result 1" in result
        assert "Result 2" in result


# ===========================================================================
# Flow 7 — execute_code tool
# ===========================================================================

class TestExecuteCode:
    """execute_code writes code to a temp file, runs it in Docker, and returns output."""

    def _setup_tmpfile_mock(self, mock_tmpfile, path="/tmp/test_code.py"):
        mock_tmp = MagicMock()
        mock_tmp.__enter__ = MagicMock(return_value=mock_tmp)
        mock_tmp.__exit__ = MagicMock(return_value=False)
        mock_tmp.name = path
        mock_tmpfile.return_value = mock_tmp
        return mock_tmp

    @patch("tools.os.remove")
    @patch("tools.os.path.exists", return_value=True)
    @patch("tools.subprocess.run")
    @patch("tools.tempfile.NamedTemporaryFile")
    def test_success_returns_stdout(self, mock_tmpfile, mock_run, mock_exists, mock_remove):
        from tools import execute_code
        self._setup_tmpfile_mock(mock_tmpfile)
        mock_run.return_value = MagicMock(returncode=0, stdout="Hello\n", stderr="")
        result = execute_code.invoke({"code": "print('Hello')"})
        assert "Hello" in result

    @patch("tools.os.remove")
    @patch("tools.os.path.exists", return_value=True)
    @patch("tools.subprocess.run")
    @patch("tools.tempfile.NamedTemporaryFile")
    def test_nonzero_exit_returns_stderr(self, mock_tmpfile, mock_run, mock_exists, mock_remove):
        from tools import execute_code
        self._setup_tmpfile_mock(mock_tmpfile)
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="NameError: x")
        result = execute_code.invoke({"code": "print(x)"})
        assert "Error" in result
        assert "NameError" in result

    @patch("tools.os.remove")
    @patch("tools.os.path.exists", return_value=True)
    @patch("tools.subprocess.run", side_effect=subprocess.TimeoutExpired("docker", 30))
    @patch("tools.tempfile.NamedTemporaryFile")
    def test_timeout_returns_timeout_message(self, mock_tmpfile, mock_run, mock_exists, mock_remove):
        from tools import execute_code
        self._setup_tmpfile_mock(mock_tmpfile)
        result = execute_code.invoke({"code": "import time; time.sleep(100)"})
        assert "timed out" in result.lower()

    @patch("tools.os.remove")
    @patch("tools.os.path.exists", return_value=True)
    @patch("tools.subprocess.run", side_effect=Exception("Docker not found"))
    @patch("tools.tempfile.NamedTemporaryFile")
    def test_unexpected_exception_returns_error(self, mock_tmpfile, mock_run, mock_exists, mock_remove):
        from tools import execute_code
        self._setup_tmpfile_mock(mock_tmpfile)
        result = execute_code.invoke({"code": "print(1)"})
        assert "Error" in result

    @patch("tools.os.remove")
    @patch("tools.os.path.exists", return_value=True)
    @patch("tools.subprocess.run")
    @patch("tools.tempfile.NamedTemporaryFile")
    def test_no_stdout_returns_success_message(self, mock_tmpfile, mock_run, mock_exists, mock_remove):
        from tools import execute_code
        self._setup_tmpfile_mock(mock_tmpfile)
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = execute_code.invoke({"code": "x = 1"})
        assert "no output" in result.lower()


# ===========================================================================
# Flow 8 — analyze_image tool
# ===========================================================================

class TestAnalyzeImage:
    """analyze_image encodes the file and sends it to the VLM with the correct MIME type."""

    @patch("tools.os.path.exists", return_value=False)
    def test_missing_file_returns_error(self, _):
        from tools import analyze_image
        result = analyze_image.invoke({"image_path": "/nonexistent/photo.jpg"})
        assert "Error" in result and "not found" in result

    @patch("tools.vlm")
    @patch("tools.os.path.exists", return_value=True)
    @patch("builtins.open", create=True)
    def test_successful_analysis_returns_vlm_content(self, mock_open, _, mock_vlm):
        from tools import analyze_image
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"bytes")))
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_file
        mock_vlm.invoke.return_value = MagicMock(content="A cat on a table")
        result = analyze_image.invoke({"image_path": "/tmp/photo.jpg"})
        assert result == "A cat on a table"

    @patch("tools.vlm")
    @patch("tools.os.path.exists", return_value=True)
    @patch("builtins.open", create=True)
    def test_png_file_uses_image_png_mime_type(self, mock_open, _, mock_vlm):
        from tools import analyze_image
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"bytes")))
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_file
        mock_vlm.invoke.return_value = MagicMock(content="description")
        analyze_image.invoke({"image_path": "/tmp/photo.png"})
        call_args = mock_vlm.invoke.call_args[0][0]
        url = call_args[0].content[1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    @patch("tools.vlm")
    @patch("tools.os.path.exists", return_value=True)
    @patch("builtins.open", create=True)
    def test_jpg_file_uses_image_jpeg_mime_type(self, mock_open, _, mock_vlm):
        from tools import analyze_image
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"bytes")))
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_file
        mock_vlm.invoke.return_value = MagicMock(content="description")
        analyze_image.invoke({"image_path": "/tmp/photo.jpg"})
        call_args = mock_vlm.invoke.call_args[0][0]
        url = call_args[0].content[1]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    @patch("tools.vlm")
    @patch("tools.os.path.exists", return_value=True)
    @patch("builtins.open", create=True)
    def test_empty_vlm_response_returns_fallback_message(self, mock_open, _, mock_vlm):
        from tools import analyze_image
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"bytes")))
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_file
        mock_vlm.invoke.return_value = MagicMock(content="")
        result = analyze_image.invoke({"image_path": "/tmp/photo.jpg"})
        assert "No response" in result


# ===========================================================================
# Flow 9 — analyze_video tool
# ===========================================================================

class TestAnalyzeVideo:
    """analyze_video samples frames via OpenCV and sends them in a single VLM call."""

    @patch("tools.os.path.exists", return_value=False)
    def test_missing_file_returns_error(self, _):
        from tools import analyze_video
        result = analyze_video.invoke({"video_path": "/nonexistent/video.mp4"})
        assert "Error" in result and "not found" in result

    @patch("tools.cv2.VideoCapture")
    @patch("tools.os.path.exists", return_value=True)
    def test_unopenable_file_returns_error(self, _, mock_cap_class):
        from tools import analyze_video
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_class.return_value = mock_cap
        result = analyze_video.invoke({"video_path": "/tmp/video.mp4"})
        assert "Error" in result and "Could not open" in result

    @patch("tools.cv2.VideoCapture")
    @patch("tools.os.path.exists", return_value=True)
    def test_no_extractable_frames_returns_error(self, _, mock_cap_class):
        from tools import analyze_video
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 10
        # Every cap.read() call returns ret=False so no frames are collected
        mock_cap.read.return_value = (False, None)
        mock_cap_class.return_value = mock_cap
        result = analyze_video.invoke({"video_path": "/tmp/video.mp4"})
        assert "No frames" in result


# ===========================================================================
# Flow 10 — analyze_document tool
# ===========================================================================

class TestAnalyzeDocument:
    """analyze_document parses PDF, DOCX, XLSX, or CSV and answers a question via the LLM."""

    @patch("tools.os.path.exists", return_value=False)
    def test_missing_file_returns_error(self, _):
        from tools import analyze_document
        result = analyze_document.invoke({"file_path": "/nonexistent/doc.pdf"})
        assert "Error" in result and "not found" in result

    @patch("tools.os.path.exists", return_value=True)
    def test_unsupported_extension_returns_error(self, _):
        from tools import analyze_document
        result = analyze_document.invoke({"file_path": "/tmp/file.txt"})
        assert "Unsupported file type" in result

    @patch("tools.llm")
    @patch("tools.os.path.exists", return_value=True)
    def test_csv_analysis_calls_llm_with_extracted_text(self, _, mock_llm):
        from tools import analyze_document
        import pandas as pd
        mock_llm.invoke.return_value = MagicMock(content="3 rows found")
        csv_data = pd.DataFrame({"name": ["Alice", "Bob", "Carol"], "age": [30, 25, 28]})
        with patch("pandas.read_csv", return_value=csv_data):
            result = analyze_document.invoke({"file_path": "/tmp/data.csv", "question": "How many rows?"})
        assert result == "3 rows found"

    @patch("tools.os.path.exists", return_value=True)
    def test_empty_extracted_text_returns_error(self, _):
        from tools import analyze_document
        # Patch pdfplumber to return a pdf with pages that have no extractable text
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [MagicMock(extract_text=MagicMock(return_value=None))]
        with patch("pdfplumber.open", return_value=mock_pdf):
            result = analyze_document.invoke({"file_path": "/tmp/empty.pdf"})
        assert "Could not extract" in result


# ===========================================================================
# Flow 11 — trim_messages_window
# ===========================================================================

class TestTrimMessagesWindow:
    """trim_messages_window keeps only the most recent messages up to the limit."""

    def test_does_not_trim_when_within_limit(self):
        messages = [HumanMessage(content=f"msg {i}") for i in range(10)]
        assert len(trim_messages_window(messages, max_messages=20)) == 10

    def test_trims_to_last_n_messages_when_over_limit(self):
        messages = [HumanMessage(content=f"msg {i}") for i in range(25)]
        result = trim_messages_window(messages, max_messages=20)
        assert len(result) == 20
        assert result[-1].content == "msg 24"
        assert result[0].content == "msg 5"

    def test_exactly_at_limit_is_not_trimmed(self):
        messages = [HumanMessage(content=f"msg {i}") for i in range(20)]
        assert len(trim_messages_window(messages, max_messages=20)) == 20

    def test_empty_list_returns_empty_list(self):
        assert trim_messages_window([], max_messages=20) == []


# ===========================================================================
# Flow 12 — parse_user_input (main.py utility)
# ===========================================================================

class TestParseUserInput:
    """parse_user_input extracts an attached file or media path from the raw message."""

    def setup_method(self):
        from main import parse_user_input
        self.parse = parse_user_input

    def test_plain_text_returns_unchanged_text_and_none_path(self):
        text, path = self.parse("what is the weather?")
        assert text == "what is the weather?"
        assert path is None

    def test_extracts_jpg_path(self):
        text, path = self.parse("analyze this /home/user/photo.jpg please")
        assert path == "/home/user/photo.jpg"
        assert "photo.jpg" not in text

    def test_extracts_pdf_path(self):
        _, path = self.parse("summarize /tmp/report.pdf")
        assert path == "/tmp/report.pdf"

    def test_extracts_mp4_path(self):
        _, path = self.parse("what happens in /data/clip.mp4?")
        assert path == "/data/clip.mp4"

    def test_extracts_csv_path(self):
        _, path = self.parse("analyze /data/sales.csv")
        assert path == "/data/sales.csv"

    def test_remaining_text_is_stripped_of_whitespace(self):
        text, _ = self.parse("describe /tmp/img.png")
        assert text == "describe"

    def test_no_path_when_extension_is_not_supported(self):
        text, path = self.parse("read the file /tmp/notes.txt")
        assert path is None
        assert text == "read the file /tmp/notes.txt"
