#!/usr/bin/env python3
"""
Llama.cpp AI Code Agent - Ultimate Monolithic Edition
Features: PySide6 GUI, LangGraph, Network Flow Control, Episodic Memory, 
          Human-in-the-Loop, Diff Viewer, Trace Logs, Multi-Tab Chat, @file mentions.
"""

import sys
import os
import json
import time
import re
import ast
import traceback
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, Any, TypedDict, Annotated, Optional
from typing_extensions import NotRequired

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QSplitter, QTabWidget, QDockWidget, QToolBar,
    QStatusBar, QMenuBar, QMenu, QAction, QMessageBox, QFileDialog, QLabel,
    QProgressBar, QListWidget, QListWidgetItem, QDialog, QFormLayout,
    QSpinBox, QDoubleSpinBox, QCompleter, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsPathItem, QPainterPath,
    QPainter, QSizePolicy, QInputDialog, QScrollBar
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QTimer, QEventLoop, QPointF, QRectF, QSize, QRegExp
)
from PySide6.QtGui import (
    QFont, QPalette, QColor, QSyntaxHighlighter, QTextCharFormat, QPen, QBrush,
    QTextCursor, QKeySequence, QLinearGradient, QIcon
)

# ==============================================================================
# 1. UTILITIES & PERSISTENCE
# ==============================================================================
CONFIG_DIR = Path.home() / ".llama_code_agent"
CONFIG_FILE = CONFIG_DIR / "config.json"
SESSIONS_DIR = CONFIG_DIR / "sessions"

def ensure_dirs():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

def load_config() -> Dict[str, Any]:
    ensure_dirs()
    if CONFIG_FILE.exists():
        try: return json.load(open(CONFIG_FILE, 'r', encoding='utf-8'))
        except Exception: pass
    return {}

def save_config(config: Dict[str, Any]):
    ensure_dirs()
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def save_session(thread_id: str, messages: list):
    ensure_dirs()
    with open(SESSIONS_DIR / f"{thread_id}.json", 'w', encoding='utf-8') as f:
        json.dump([m if isinstance(m, dict) else {"role": str(type(m)), "content": str(m)} for m in messages], f, indent=4, default=str)

# ==============================================================================
# 2. NETWORK FLOW ALGORITHMS (Token Buckets)
# ==============================================================================
class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now - self.last_refill) * self.refill_rate)
        self.last_refill = now

    def get_wait_time(self, tokens: int = 1) -> float:
        self._refill()
        return max(0.0, (tokens - self.tokens) / self.refill_rate) if self.refill_rate > 0 else 999.0

    def get_state(self) -> Dict[str, Any]:
        self._refill()
        return {"available": self.tokens, "capacity": self.capacity, "utilization": 1.0 - (self.tokens / self.capacity)}

class FlowNetwork:
    def __init__(self, global_max_flow: int = 15, max_llm_calls_per_min: int = 12):
        self.global_max_flow = global_max_flow
        self.current_global_flow = 0
        self.llm_inference_bucket = TokenBucket(capacity=max_llm_calls_per_min, refill_rate=1.0/5.0)
        self.current_llm_flow = 0
        self.tool_buckets: Dict[str, TokenBucket] = {
            "execute_code": TokenBucket(5, 0.1), "run_shell_command": TokenBucket(3, 0.05),
            "write_file": TokenBucket(10, 0.2), "apply_diff": TokenBucket(10, 0.2),
            "read_file": TokenBucket(20, 0.5), "search_files": TokenBucket(10, 0.3),
            "analyze_code": TokenBucket(10, 0.3), "list_directory": TokenBucket(15, 0.4),
        }

    def check_llm_flow(self) -> Dict[str, Any]:
        if self.current_llm_flow >= self.llm_inference_bucket.capacity:
            return {"allowed": False, "reason": "llm_max_flow", "message": "LLM inference limit reached. Halting."}
        if self.llm_inference_bucket.consume(1):
            self.current_llm_flow += 1
            return {"allowed": True, "llm_state": self.llm_inference_bucket.get_state(), "llm_flow": self.current_llm_flow}
        wait = self.llm_inference_bucket.get_wait_time(1)
        return {"allowed": False, "reason": "llm_rate_limited", "message": f"LLM rate limited. Wait {wait:.1f}s.", "wait_time": wait}

    def check_tool_flow(self, tool_name: str) -> Dict[str, Any]:
        if self.current_global_flow >= self.global_max_flow:
            return {"allowed": False, "reason": "global_max_flow", "message": "Global tool max flow reached."}
        bucket = self.tool_buckets.get(tool_name)
        if not bucket: return {"allowed": True, "reason": "unmonitored"}
        if bucket.consume(1):
            self.current_global_flow += 1
            return {"allowed": True, "tool_state": bucket.get_state(), "global_flow": self.current_global_flow}
        wait = bucket.get_wait_time(1)
        return {"allowed": False, "reason": "rate_limited", "message": f"Rate limited on '{tool_name}'. Wait {wait:.1f}s.", "wait_time": wait}

    def get_network_status(self) -> Dict[str, Any]:
        return {
            "llm_flow": f"{self.current_llm_flow}/{self.llm_inference_bucket.capacity}",
            "llm_state": self.llm_inference_bucket.get_state(),
            "global_tool_flow": f"{self.current_global_flow}/{self.global_max_flow}",
            "interfaces": {n: b.get_state() for n, b in self.tool_buckets.items()}
        }

# ==============================================================================
# 3. AGENT TOOLS
# ==============================================================================
_workspace_dir = Path.home() / "llama_code_agent_workspace"
_workspace_dir.mkdir(exist_ok=True)

def set_workspace(path: str):
    global _workspace_dir
    _workspace_dir = Path(path)
    _workspace_dir.mkdir(exist_ok=True)

def get_workspace() -> Path:
    return _workspace_dir

def tool_decorator(func):
    from langchain_core.tools import tool
    return tool(func)

@tool_decorator
def execute_code(code: str) -> str:
    """Execute Python code and return output."""
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, cwd=_workspace_dir)
        out = []
        if r.stdout: out.append(f"STDOUT:\n{r.stdout}")
        if r.stderr: out.append(f"STDERR:\n{r.stderr}")
        if r.returncode != 0: out.append(f"Return code: {r.returncode}")
        return "\n".join(out) if out else "Code executed successfully (no output)"
    except Exception as e: return f"Error: {e}"

@tool_decorator
def run_shell_command(command: str) -> str:
    """Execute a shell command (e.g., pip install, git)."""
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60, cwd=_workspace_dir)
        out = []
        if r.stdout: out.append(r.stdout)
        if r.stderr: out.append(f"STDERR:\n{r.stderr}")
        if r.returncode != 0: out.append(f"Return code: {r.returncode}")
        return "\n".join(out) if out else "Done."
    except Exception as e: return f"Error: {e}"

@tool_decorator
def read_file(filepath: str) -> str:
    """Read contents of a file."""
    try:
        p = Path(filepath) if Path(filepath).is_absolute() else _workspace_dir / filepath
        if not p.exists(): return f"Error: File not found: {p}"
        try: content = p.read_text(encoding='utf-8')
        except UnicodeDecodeError: content = p.read_text(encoding='latin-1')
        lines = content.split('\n')
        return f"File: {p}\n{'='*50}\n" + '\n'.join([f"{i+1:4d} | {l}" for i, l in enumerate(lines)])
    except Exception as e: return f"Error: {e}"

@tool_decorator
def write_file(filepath: str, content: str) -> str:
    """Write content to a file. Creates if not exists."""
    try:
        p = Path(filepath) if Path(filepath).is_absolute() else _workspace_dir / filepath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
        return f"Successfully wrote to {p}"
    except Exception as e: return f"Error: {e}"

@tool_decorator
def apply_diff(filepath: str, old_string: str, new_string: str) -> str:
    """Apply a targeted diff to a file. Use this INSTEAD of write_file to modify code."""
    try:
        p = Path(filepath) if Path(filepath).is_absolute() else _workspace_dir / filepath
        if not p.exists(): return f"Error: File not found. Use write_file for new files."
        content = p.read_text(encoding='utf-8')
        if old_string not in content:
            norm_content, norm_old = content.replace("\t", "    "), old_string.replace("\t", "    ")
            if norm_old in norm_content: content, old_string, new_string = norm_content, norm_old, new_string.replace("\t", "    ")
            else: return "Error: `old_string` not found exactly in file."
        p.write_text(content.replace(old_string, new_string, 1), encoding='utf-8')
        return f"✅ Patched {p}"
    except Exception as e: return f"Error: {e}"

@tool_decorator
def list_directory(path: str = ".") -> str:
    """List files and directories."""
    try:
        dp = Path(path) if Path(path).is_absolute() else _workspace_dir / path
        if not dp.exists(): return f"Error: Not found: {dp}"
        entries = []
        for e in sorted(dp.iterdir()):
            if e.is_dir(): entries.append(f"📁 {e.name}/")
            else: entries.append(f"📄 {e.name} ({e.stat().st_size} B)")
        return "\n".join(entries) if entries else "Empty"
    except Exception as e: return f"Error: {e}"

@tool_decorator
def search_files(pattern: str, path: str = ".") -> str:
    """Search for files matching a pattern recursively."""
    try:
        sp = Path(path) if Path(path).is_absolute() else _workspace_dir / path
        matches = list(sp.rglob(pattern))
        return "\n".join([str(m.relative_to(sp)) for m in matches[:50]]) if matches else "No matches."
    except Exception as e: return f"Error: {e}"

@tool_decorator
def analyze_code(code: str) -> str:
    """Analyze Python code for syntax, imports, functions, classes."""
    try:
        tree = ast.parse(code)
        res = ["✅ Syntax Valid"]
        imports, funcs, classes = [], [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imports.extend([a.name for a in node.names])
            elif isinstance(node, ast.ImportFrom): imports.extend([f"{node.module}.{a.name}" for a in node.names])
            elif isinstance(node, ast.FunctionDef): funcs.append(f"- {node.name}() [L{node.lineno}]")
            elif isinstance(node, ast.ClassDef): classes.append(f"- {node.name} [L{node.lineno}]")
        if imports: res.append(f"📦 Imports: {', '.join(imports)}")
        if classes: res.append(f"🏗️ Classes:\n" + "\n".join(classes))
        if funcs: res.append(f"⚙️ Functions:\n" + "\n".join(funcs))
        return "\n".join(res)
    except SyntaxError as e: return f"❌ Syntax Error L{e.lineno}: {e.msg}"

def get_tools():
    return [execute_code, run_shell_command, read_file, apply_diff, write_file, list_directory, search_files, analyze_code]

# ==============================================================================
# 4. LANGGRAPH AGENT (Memory, Flow Control, Safe Tools)
# ==============================================================================
SYSTEM_PROMPT = """You are an expert AI coding assistant. You have tools to execute code, read/write files, and analyze code.
CRITICAL RULES:
1. When modifying existing code, ALWAYS use `apply_diff`. NEVER use `write_file` to overwrite existing files.
2. Always test your code using `execute_code` if possible.
3. Break complex tasks into smaller steps."""

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration_count: int
    flow_report: NotRequired[dict]
    is_throttled: NotRequired[bool]
    force_end: NotRequired[bool]

class SafeToolNode:
    def __init__(self, tools): self.tools_by_name = {t.name: t for t in tools}
    def invoke(self, state, config=None):
        msgs = state["messages"]
        last = msgs[-1]
        results = []
        for tc in last.tool_calls:
            tool = self.tools_by_name.get(tc["name"])
            try:
                res = tool.invoke(tc["args"])
                results.append(ToolMessage(content=str(res), tool_call_id=tc["id"], name=tc["name"]))
            except Exception as e:
                results.append(ToolMessage(content=f"TOOL EXECUTION ERROR: {e}. Fix your args.", tool_call_id=tc["id"], name=tc["name"], status="error"))
        return {"messages": results}

def compress_context(llm, messages: list, max_tokens: int) -> list:
    try: total = sum(len(llm.client.tokenize(str(m.content).encode('utf-8'))) for m in messages)
    except: total = sum(len(str(m.content))//4 for m in messages)
    if total <= max_tokens * 0.8: return messages
    
    sys_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    recent, old, cur_tok = [], [], 0
    for m in reversed(messages[1:]):
        tok = len(str(m.content))//4
        if cur_tok + tok < max_tokens * 0.6: recent.insert(0, m); cur_tok += tok
        else: old.insert(0, m)
    if not old: return messages
    try:
        summary = llm.invoke([HumanMessage(content=f"Summarize this concisely for memory (files changed, errors fixed):\n{old}")], max_tokens=300).content
        return sys_msgs + [SystemMessage(content=f"[EPISODIC MEMORY]\n{summary}\n[END MEMORY]")] + recent
    except: return sys_msgs + recent

def parse_tool_fallback(text: str, tools) -> list:
    matches = re.findall(r'```json\s*(\{.*?"name"\s*:\s*".*?"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\})\s*```', text, re.DOTALL)
    if not matches: matches = re.findall(r'(\{"name"\s*:\s*".*?"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\})', text, re.DOTALL)
    calls = []
    names = [t.name for t in tools]
    for m in matches:
        try:
            d = json.loads(m)
            if d.get("name") in names: calls.append({"name": d["name"], "args": d.get("arguments", {}), "id": f"call_fb_{len(calls)}"})
        except: pass
    return calls

def create_code_agent(llm, max_global_flow=15, max_llm_calls=12):
    from langgraph.graph import StateGraph, END, START
    from langgraph.graph.message import add_messages
    from langgraph.checkpoint.memory import MemorySaver
    
    tools = get_tools()
    tool_node = SafeToolNode(tools)
    llm_with_tools = llm.bind_tools(tools)
    flow_net = FlowNetwork(max_global_flow, max_llm_calls)

    def agent_node(state: AgentState):
        msgs = compress_context(llm, state["messages"], getattr(llm, 'n_ctx', 4096))
        if len(msgs) == 1 and isinstance(msgs[0], HumanMessage): msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
        resp = llm_with_tools.invoke(msgs)
        if not resp.tool_calls and resp.content:
            fb = parse_tool_fallback(resp.content, tools)
            if fb: resp.tool_calls = fb; resp.content = "Using tools..."
        return {"messages": [resp], "iteration_count": state.get("iteration_count", 0) + 1, "force_end": False, "is_throttled": False}

    def llm_flow_node(state: AgentState):
        r = flow_net.check_llm_flow()
        if not r["allowed"]: return {"messages": [AIMessage(content=f"⚠️ {r['message']}")], "force_end": True, "flow_report": r}
        return {"force_end": False, "flow_report": r}

    def tool_flow_node(state: AgentState):
        last = state["messages"][-1]
        if not hasattr(last, 'tool_calls') or not last.tool_calls: return {"is_throttled": False}
        for tc in last.tool_calls:
            r = flow_net.check_tool_flow(tc["name"])
            if not r["allowed"]: return {"messages": [SystemMessage(content=f"FLOW CONTROL: {r['message']} Respond without using {tc['name']}.")], "is_throttled": True, "flow_report": r}
        return {"is_throttled": False}

    workflow = StateGraph(AgentState)
    workflow.add_node("llm_flow", llm_flow_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tool_flow", tool_flow_node)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "llm_flow")
    workflow.add_conditional_edges("llm_flow", lambda s: "end" if s.get("force_end") else "agent", {"agent": "agent", "end": END})
    workflow.add_edge("agent", "tool_flow")
    workflow.add_conditional_edges("tool_flow", lambda s: "agent" if s.get("is_throttled") else "tools", {"tools": "tools", "agent": "agent"})
    workflow.add_edge("tools", "llm_flow")
    
    compiled = workflow.compile(checkpointer=MemorySaver())
    compiled.flow_network = flow_net
    compiled.raw_llm = llm
    return compiled

# Required imports for agent definitions
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

# ==============================================================================
# 5. BACKGROUND WORKERS & HUMAN-IN-THE-LOOP
# ==============================================================================
class ApprovalHandler(QObject):
    approved = Signal(bool)
    def __init__(self): super().__init__(); self._is_approved = False
    def approve(self): self._is_approved = True; self.approved.emit(True)
    def deny(self): self._is_approved = False; self.approved.emit(False)

class ModelLoadWorker(QThread):
    progress = Signal(str); finished = Signal(object); error = Signal(str)
    def __init__(self, config): super().__init__(); self.config = config
    def run(self):
        try:
            self.progress.emit("Loading model...")
            from langchain_community.llms import LlamaCpp
            llm = LlamaCpp(model_path=self.config["model_path"], n_ctx=self.config.get("n_ctx", 4096), n_gpu_layers=self.config.get("n_gpu_layers", -1), temperature=self.config.get("temperature", 0.7), top_p=self.config.get("top_p", 0.95), repeat_penalty=self.config.get("repeat_penalty", 1.1), max_tokens=self.config.get("max_tokens", 2048), verbose=False)
            self.progress.emit("Building graph...")
            agent = create_code_agent(llm, self.config.get("max_iterations", 15), self.config.get("max_llm_calls", 12))
            self.finished.emit(agent)
        except Exception as e: self.error.emit(str(e))

class AgentStreamWorker(QThread):
    token_received = Signal(str); tool_call_start = Signal(str, str); tool_call_end = Signal(str, str)
    tool_stream_chunk = Signal(str); flow_status = Signal(dict); flow_throttled = Signal(str)
    llm_rate_limited = Signal(str); diff_applied = Signal(str, str, str); trace_log = Signal(str, str)
    response_complete = Signal(); error = Signal(str); thinking = Signal(str)
    request_approval = Signal(str, str, ApprovalHandler)

    def __init__(self, agent, message, thread_id):
        super().__init__(); self.agent = agent; self.message = message; self.thread_id = thread_id; self._cancelled = False
        self.dangerous_tools = {"run_shell_command", "write_file", "execute_code"}

    def cancel(self): self._cancelled = True

    def _ask_approval(self, t_name, args):
        loop = QEventLoop(); handler = ApprovalHandler()
        self.request_approval.emit(t_name, args, handler)
        handler.approved.connect(loop.quit); loop.exec_()
        return handler._is_approved

    def run(self):
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            state = {"messages": [HumanMessage(content=self.message)]}
            self.trace_log.emit("System", "Starting agent loop...")
            
            while not self._cancelled:
                self.trace_log.emit("FlowControl", "Checking LLM bandwidth...")
                r = self.agent.flow_network.check_llm_flow()
                self.flow_status.emit(self.agent.flow_network.get_network_status())
                if not r["allowed"]:
                    self.llm_rate_limited.emit(r["message"])
                    state["messages"].append(AIMessage(content=f"⚠️ {r['message']}"))
                    break

                self.trace_log.emit("AgentNode", "Invoking LLM...")
                self.thinking.emit("Thinking...")
                last_idx = len(state["messages"])
                
                for event in self.agent.stream(state, config=config, stream_mode="updates"):
                    if self._cancelled: break
                    for node, update in event.items():
                        if node == "agent":
                            for msg in update.get("messages", []):
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        t_name, args = tc["name"], json.dumps(tc.get("args", {}), default=str)
                                        if t_name in self.dangerous_tools:
                                            if not self._ask_approval(t_name, args):
                                                state["messages"].append(AIMessage(content=f"User denied {t_name}."))
                                                self.tool_call_end.emit(t_name, "DENIED BY USER")
                                                self.response_complete.emit(); return
                                        self.tool_call_start.emit(t_name, args)
                                        self.trace_log.emit("ToolFlow", f"Approved {t_name}")
                                elif msg.content and "FLOW CONTROL" not in str(msg.content):
                                    self.token_received.emit(str(msg.content))
                        elif node == "tools":
                            for msg in update.get("messages", []):
                                self.tool_call_end.emit(getattr(msg, 'name', 'tool'), str(msg.content)[:500])
                                self.trace_log.emit("ToolNode", f"Finished {getattr(msg, 'name', 'tool')}")
                
                state["messages"].extend(state["messages"][last_idx:]) # Update local state
                last_msg = state["messages"][-1]
                if not hasattr(last_msg, 'tool_calls') or not last_msg.tool_calls: break
                
                # Handle apply_diff UI trigger
                if hasattr(last_msg, 'tool_calls'):
                    for tc in last_msg.tool_calls:
                        if tc["name"] == "apply_diff":
                            a = tc["args"]
                            self.diff_applied.emit(a.get("old_string", ""), a.get("new_string", ""), a.get("filepath", ""))

            self.response_complete.emit()
        except Exception as e: self.error.emit(traceback.format_exc())

# ==============================================================================
# 6. CUSTOM WIDGETS (Editor, Diff, Trace, Chat, Graph)
# ==============================================================================
class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, doc):
        super().__init__(doc); self.rules = []
        kw_fmt = QTextCharFormat(); kw_fmt.setForeground(QColor("#CC7832")); kw_fmt.setFontWeight(QFont.Weight.Bold)
        for w in ['def', 'class', 'return', 'if', 'else', 'elif', 'for', 'while', 'import', 'from', 'try', 'except', 'finally', 'with', 'as', 'async', 'await', 'yield', 'lambda', 'pass', 'raise', 'True', 'False', 'None']:
            self.rules.append((QRegExp(r'\b'+w+r'\b'), kw_fmt))
        str_fmt = QTextCharFormat(); str_fmt.setForeground(QColor("#6A8759"))
        self.rules.append((QRegExp(r'".*?"'), str_fmt)); self.rules.append((QRegExp(r"'.*?'"), str_fmt))
        com_fmt = QTextCharFormat(); com_fmt.setForeground(QColor("#808080")); com_fmt.setFontItalic(True)
        self.rules.append((QRegExp(r'#.*'), com_fmt))

    def highlightBlock(self, text):
        for pattern, fmt in self.rules:
            idx = pattern.indexIn(text)
            while idx >= 0: self.setFormat(idx, pattern.matchedLength(), fmt); idx = pattern.indexIn(text, idx + pattern.matchedLength())

class CodeEditor(QTextEdit):
    run_requested = Signal(str)
    def __init__(self): 
        super().__init__(); self.setFont(QFont("Consolas", 11))
        self.setStyleSheet("background-color: #1e1e1e; color: #a9b7c6; border: none;")
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(' ') * 4)
        PythonHighlighter(self.document())

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Tab: self.textCursor().insertText("    "); return
        if e.key() == Qt.Key.Key_Return and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.run_requested.emit(self.toPlainText()); return
        super().keyPressEvent(e)

class DiffViewer(QTextEdit):
    def __init__(self): super().__init__(); self.setReadOnly(True); self.setFont(QFont("Consolas", 10)); self.setStyleSheet("background-color: #111; border: none;")
    def display_diff(self, old, new):
        self.clear(); c = self.textCursor()
        fm_r, fm_a, fm_n = self._fmt("#ffcccc", "#4a1515"), self._fmt("#ccffcc", "#154a15"), self._fmt("#ddd", "transparent")
        for o, n in zip(old.splitlines(), new.splitlines()):
            if o != n:
                c.insertText(f"- {o}\n", fm_r); c.insertText(f"+ {n}\n", fm_a)
            else: c.insertText(f"  {o}\n", fm_n)
    def _fmt(self, t, b): f = QTextCharFormat(); f.setForeground(QColor(t)); f.setBackground(QColor(b)); return f

class TracePanel(QTextEdit):
    def __init__(self): 
        super().__init__(); self.setReadOnly(True); self.setFont(QFont("Consolas", 9))
        self.setStyleSheet("background-color: #111; color: #aaa; border: none;"); self.setMaximumBlockCount(500)
        self.fmt = {"sys": self._c("#888", True), "agent": self._c("#bb86fc", True), "tools": self._c("#03dac6"), "flow": self._c("#ff9800", True), "err": self._c("#cf6679", True)}
    def _c(self, col, b=False): f=QTextCharFormat(); f.setForeground(QColor(col)); if b: f.setFontWeight(QFont.Weight.Bold); return f
    def log_event(self, node, msg):
        cat = "sys"
        if "agent" in node.lower(): cat="agent"
        elif "tool" in node.lower(): cat="tools"
        elif "flow" in node.lower(): cat="flow"
        elif "error" in node.lower(): cat="err"
        self.moveCursor(self.textCursor().MoveOperation.End)
        self.setCurrentCharFormat(self.fmt["sys"]); self.insertPlainText(f"[{time.strftime('%H:%M:%S')}] ")
        self.setCurrentCharFormat(self.fmt[cat]); self.insertText(f"<{node}> "); self.setCurrentCharFormat(self.fmt["sys"]); self.insertText(f"{msg}\n")

class ChatWidget(QTextBrowser):
    message_sent = Signal(str); code_action = Signal(str, str); retry_requested = Signal(int)
    def __init__(self): 
        super().__init__(); self.messages = []; self.code_blocks = {}; self._cid = 0
        self.setOpenLinks(False); self.anchorClicked.connect(self._on_link)
        self.setStyleSheet("QTextBrowser { background-color: #1e1e1e; color: #ddd; border: none; padding: 10px; font-size: 13px; } pre { background-color: #282c34; padding: 10px; border-radius: 5px; } code { font-family: Consolas; font-size: 12px; color: #abb2bf; } a { color: #61afef; }")
        self.input_field = QLineEdit(); self.input_field.setPlaceholderText("Type message... (@filename for context)")
        self.input_field.setFont(QFont("Segoe UI", 11))
        self.input_field.setStyleSheet("QLineEdit { background-color: #3a3a3a; border: 1px solid #555; border-radius: 20px; padding: 10px 20px; color: #ddd; } QLineEdit:focus { border: 1px solid #6CB4EE; }")
        self.input_field.returnPressed.connect(self._send)

    def _send(self):
        t = self.input_field.text().strip()
        if t: self.input_field.clear(); self.add_message("user", t); self.message_sent.emit(self._parse_mentions(t))

    def _parse_mentions(self, t):
        matches = re.findall(r'@([\w\-./]+\.\w+)', t)
        if not matches: return t
        ctx = "\n\n[USER CONTEXT]:\n"
        for m in matches:
            r = read_file.invoke({"filepath": m})
            if not r.startswith("Error:"): ctx += f"--- {m} ---\n{r}\n---\n"
            t = t.replace(f"@{m}", "")
        return t + ctx

    def _on_link(self, url):
        p = url.toString()
        if p.startswith("action:"): parts = p.split("|"); self.code_action.emit(parts[0].replace("action:", ""), self.code_blocks.get(parts[1], ""))

    def _fmt_code(self, text):
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        def repl(m):
            self._cid += 1; cid = str(self._cid); self.code_blocks[cid] = m.group(2).strip()
            return f"<div style='background:#21252b; padding:5px 10px; border-radius:5px 5px 0 0;'><span style='color:#888;'>{m.group(1).upper()}</span> <a href='action:run|{cid}'>▶ Run</a> <a href='action:insert|{cid}'>📝 Insert</a></div><pre><code>{m.group(2)}</code></pre>"
        text = re.sub(r"```(\w*)\n?(.*?)```", repl, text, flags=re.DOTALL)
        return text.replace("\n", "<br>").replace("**", "<b>", 1).replace("**", "</b>", 1)

    def add_message(self, role, content, is_error=False):
        col = {"user": "#6CB4EE", "assistant": "#98D8AA", "tool": "#F7DC6F", "system": "#BB8FCE"}.get(role, "#ddd")
        retry = f"<br><a href='action:retry|{len(self.messages)}' style='color:#e74c3c;'>🔄 Retry</a>" if is_error else ""
        html = f"<div style='margin-bottom:10px;'><b style='color:{col};'>{role.upper()}</b><br>{self._fmt_code(content) if role=='assistant' else content.replace(chr(10), '<br>')}{retry}</div>"
        self.append(html); self.messages.append({"role": role, "content": content}); self.scrollToBottom()

    def append_to_last(self, t): self.moveCursor(self.textCursor().MoveOperation.End); self.insertText(t); self.ensureCursorVisible()
    def set_thinking(self, t): pass # Simplified for monolith

class ChatTabWidget(QTabWidget):
    message_sent = Signal(str); code_action = Signal(str, str)
    def __init__(self): 
        super().__init__(); self.setTabsClosable(True); self.setMovable(True)
        self.setStyleSheet("QTabWidget::pane { border: none; } QTabBar::tab { background: #2a2a2a; color: #aaa; padding: 8px 15px; } QTabBar::tab:selected { background: #1e1e1e; color: #fff; }")
        self.add_btn = QPushButton("+"); self.add_btn.setFixedSize(25, 25); self.add_btn.setStyleSheet("QPushButton { border: none; background: #3a3a3a; color: white; border-radius: 3px; }")
        self.add_btn.clicked.connect(self.create_tab); self.tabBar().setCornerWidget(self.add_btn); self.tabCloseRequested.connect(self.close_tab)

    def create_tab(self, tid=None):
        if not tid: tid = f"t_{int(time.time())}"
        c = ChatWidget(); c.message_sent.connect(self.message_sent.emit); c.code_action.connect(self.code_action.emit)
        self.addTab(c, f"Chat {self.count()+1}"); self.setCurrentIndex(self.count()-1); return tid, c

    def close_tab(self, i):
        if self.count() > 1: self.removeTab(i)

    def current_chat(self) -> ChatWidget: return self.currentWidget()

class GraphViewer(QGraphicsView):
    def __init__(self):
        super().__init__(); self.setScene(QGraphicsScene()); self.scene().setBackgroundBrush(QBrush(QColor("#1e1e1e")))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        nodes = [("START", "#2ecc71", 100, 20), ("LLM Flow", "#f39c12", 75, 120), ("Agent", "#3498db", 75, 220), ("Tool Flow", "#f39c12", 75, 320), ("Tools", "#e67e22", 75, 420), ("END", "#e74c3c", 100, 520)]
        items = []
        for name, col, x, y in nodes:
            r = QGraphicsRectItem(0, 0, 150, 50); r.setPos(x, y); r.setBrush(QBrush(QColor(col))); t = QGraphicsTextItem(name, r); t.setDefaultTextColor(QColor("white"))
            self.scene().addItem(r); items.append(r)
        for i in range(len(items)-1): self.scene().addItem(self._edge(items[i], items[i+1]))
        self.scene().addItem(self._edge(items[3], items[1])) # Throttle loop back
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _edge(self, a, b):
        p = QGraphicsPathItem()
        path = QPainterPath(); start = QPointF(a.pos().x()+75, a.pos().y()+50); end = QPointF(b.pos().x()+75, b.pos().y())
        path.moveTo(start); path.cubicTo(start.x(), start.y()+50, end.x(), end.y()-50, end.x(), end.y())
        p.setPath(path); p.setPen(QPen(QColor("#7f8c8d"), 2)); return p

class ConfigDialog(QDialog):
    config_saved = Signal(dict)
    def __init__(self, cfg=None, parent=None):
        super().__init__(parent); self.cfg = cfg or {}; self.setWindowTitle("Config"); self.resize(400, 400); self.setStyleSheet("QDialog{background:#2a2a2a;color:#ddd;} QLineEdit,QSpinBox,QDoubleSpinBox{background:#3a3a3a;border:1px solid #555;padding:5px;color:#ddd;} QPushButton{background:#3a7bd5;border:none;padding:8px;color:white;}")
        l = QFormLayout(self)
        self.path_e = QLineEdit(self.cfg.get("model_path", "")); bh = QHBoxLayout(); bh.addWidget(self.path_e); b = QPushButton("Browse"); b.clicked.connect(lambda: self.path_e.setText(QFileDialog.getOpenFileName(self, "Model", "", "GGUF (*.gguf)")[0] or self.path_e.text())); bh.addWidget(b); l.addRow("Model:", bh)
        self.ctx = QSpinBox(); self.ctx.setRange(512, 32768); self.ctx.setValue(self.cfg.get("n_ctx", 4096)); l.addRow("Context:", self.ctx)
        self.temp = QDoubleSpinBox(); self.temp.setRange(0.0, 2.0); self.temp.setValue(self.cfg.get("temperature", 0.7)); l.addRow("Temp:", self.temp)
        self.gpu = QSpinBox(); self.gpu.setRange(-1, 100); self.gpu.setValue(self.cfg.get("n_gpu_layers", -1)); l.addRow("GPU Layers:", self.gpu)
        sb = QPushButton("Save"); sb.clicked.connect(self._save); l.addRow(sb)

    def _save(self):
        self.cfg.update({"model_path": self.path_e.text(), "n_ctx": self.ctx.value(), "temperature": self.temp.value(), "n_gpu_layers": self.gpu.value()})
        self.config_saved.emit(self.cfg); self.accept()

# ==============================================================================
# 7. MAIN WINDOW (Orchestrator)
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.agent = None; self.worker = None; self.config = load_config()
        if self.config.get("workspace"): set_workspace(self.config["workspace"])
        self.setWindowTitle("Llama.cpp Agent Suite"); self.resize(1400, 900)
        self._build_ui(); self._build_menus(); self._build_toolbar(); self._build_statusbar()

    def _build_ui(self):
        cw = QWidget(); self.setCentralWidget(cw); ml = QHBoxLayout(cw); ml.setContentsMargins(0,0,0,0)
        sp = QSplitter(Qt.Orientation.Horizontal); ml.addWidget(sp)
        
        self.tabs = ChatTabWidget(); self.tabs.create_tab("main"); self.tabs.message_sent.connect(self._on_msg)
        self.tabs.code_action.connect(self._on_code_action); sp.addWidget(self.tabs)
        
        rsp = QSplitter(Qt.Orientation.Vertical); sp.addWidget(rsp)
        self.editor = CodeEditor(); self.editor.run_requested.connect(self._run_code); rsp.addWidget(self.editor)
        
        bt = QTabWidget(); bt.setStyleSheet("QTabWidget::pane{border:none;background:#1e1e1e;} QTabBar::tab{background:#2a2a2a;color:#aaa;padding:8px;} QTabBar::tab:selected{background:#1e1e1e;color:#fff;border-bottom:2px solid #3a7bd5;}")
        self.output = QTextEdit(); self.output.setReadOnly(True); self.output.setFont(QFont("Consolas", 10)); self.output.setStyleSheet("background:#1e1e1e;color:#ccc;border:none;"); bt.addTab(self.output, "Output")
        bt.addTab(TracePanel(), "Traces"); bt.addTab(DiffViewer(), "Diffs"); bt.addTab(GraphViewer(), "Workflow")
        
        fw = QWidget(); fl = QVBoxLayout(fw); fl.setContentsMargins(0,0,0,0)
        fb = QPushButton("🔄 Refresh"); fb.clicked.connect(self._refresh_files); fl.addWidget(fb)
        self.files = QListWidget(); self.files.setStyleSheet("QListWidget{background:#1e1e1e;color:#ccc;border:none;}"); self.files.itemDoubleClicked.connect(self._open_file); fl.addWidget(self.files)
        bt.addTab(fw, "Files"); rsp.addWidget(bt)
        sp.setSizes([500, 900]); rsp.setSizes([400, 300])

    def _build_menus(self):
        mb = self.menuBar(); mb.setStyleSheet("QMenuBar{background:#2a2a2a;color:#ddd;} QMenu{background:#2a2a2a;color:#ddd;} QMenu::item:selected{background:#3a7bd5;}")
        fm = mb.addMenu("File"); fm.addAction("Load Model", self._load_model, QKeySequence("Ctrl+O"))
        fm.addSeparator(); fm.addAction("Open File", self._open_file_dlg, QKeySequence("Ctrl+Shift+O"))
        fm.addAction("Save File", self._save_file_dlg, QKeySequence("Ctrl+S"))
        em = mb.addMenu("Edit"); em.addAction("Config", self._show_config, QKeySequence("Ctrl+,"))
        em.addAction("Send Selection", self._send_sel, QKeySequence("Ctrl+E"))
        am = mb.addMenu("Agent"); am.addAction("New Thread", self._new_thread, QKeySequence("Ctrl+N"))
        am.addAction("Stop", self._stop, QKeySequence("Ctrl+."))

    def _build_toolbar(self):
        t = QToolBar(); t.setMovable(False); t.setStyleSheet("QToolBar{background:#2a2a2a;border-bottom:1px solid #444;padding:5px;} QToolButton{color:#ddd;background:transparent;border:none;padding:5px;} QToolButton:hover{background:#3a3a3a;}")
        self.addToolBar(t)
        self.model_lbl = QLabel("No Model"); self.model_lbl.setStyleSheet("color:#e74c3c;padding:0 10px;"); t.addWidget(self.model_lbl)
        t.addAction("📂 Load", self._load_model); t.addAction("⚙️ Config", self._show_config)
        t.addAction("🛑 Stop", self._stop); t.addAction("▶ Run Editor", lambda: self._run_code(self.editor.toPlainText()))

    def _build_statusbar(self):
        sb = QStatusBar(); self.setStatusBar(sb); sb.setStyleSheet("QStatusBar{background:#2a2a2a;color:#aaa;}")
        self.status_lbl = QLabel("Ready"); sb.addWidget(self.status_lbl)
        self.flow_frame = QWidget(); fh = QHBoxLayout(self.flow_frame); fh.setContentsMargins(5,0,5,0)
        self.llm_flow_lbl = QLabel("🧠 LLM: 0/0"); self.llm_flow_lbl.setStyleSheet("color:#bb86fc;font-weight:bold;"); fh.addWidget(self.llm_flow_lbl)
        self.tool_flow_lbl = QLabel("🔧 Tools: 0/0"); self.tool_flow_lbl.setStyleSheet("color:#03dac6;font-weight:bold;"); fh.addWidget(self.tool_flow_lbl)
        self.flow_det_lbl = QLabel("Idle"); self.flow_det_lbl.setStyleSheet("color:#666;"); fh.addWidget(self.flow_det_lbl)
        sb.addPermanentWidget(self.flow_frame)

    # --- SLOTS ---
    def _load_model(self):
        p = QFileDialog.getOpenFileName(self, "GGUF Model", "", "GGUF (*.gguf)")[0]
        if p: self.config["model_path"] = p; self._start_load()

    def _start_load(self):
        if not self.config.get("model_path"): return
        self.status_lbl.setText("Loading..."); self.model_lbl.setText("Loading..."); self.model_lbl.setStyleSheet("color:#f39c12;padding:0 10px;")
        self.loader = ModelLoadWorker(self.config); self.loader.finished.connect(self._on_loaded); self.loader.error.connect(self._on_load_err); self.loader.start()

    def _on_loaded(self, agent):
        self.agent = agent; self.status_lbl.setText("Ready"); self.model_lbl.setText(f"✅ {Path(self.config['model_path']).name}"); self.model_lbl.setStyleSheet("color:#2ecc71;padding:0 10px;")
        self.tabs.current_chat().add_message("system", "Model loaded. Ready.")

    def _on_load_err(self, e): self.status_lbl.setText("Error"); self.model_lbl.setText("❌ Error"); QMessageBox.critical(self, "Error", e)

    def _on_msg(self, msg):
        if not self.agent: self.tabs.current_chat().add_message("system", "Load a model first."); return
        if self.worker and self.worker.isRunning(): return
        self.worker = AgentStreamWorker(self.agent, msg, f"tab_{self.tabs.currentIndex()}")
        self.worker.token_received.connect(self.tabs.current_chat().append_to_last)
        self.worker.tool_call_start.connect(lambda n, a: self.output.append(f"\n🔧 {n}:\n{a}"))
        self.worker.tool_call_end.connect(lambda n, r: self.output.append(f"   -> {r[:200]}"))
        self.worker.diff_applied.connect(self._show_diff)
        self.worker.flow_status.connect(self._update_flow)
        self.worker.llm_rate_limited.connect(lambda m: self.tabs.current_chat().add_message("system", f"🛑 {m}", True))
        self.worker.request_approval.connect(self._show_approval)
        self.worker.response_complete.connect(lambda: self.tabs.current_chat().add_message("system", "Done."))
        self.worker.error.connect(lambda e: self.tabs.current_chat().add_message("system", f"❌ {e}", True))
        
        # Connect trace logs to the trace tab
        trace_tab = self.centralWidget().findChild(QSplitter).findChild(QSplitter).findChild(QTabWidget).widget(1)
        if isinstance(trace_tab, TracePanel): self.worker.trace_log.connect(trace_tab.log_event)
        
        self.worker.start()

    def _show_approval(self, t_name, args, handler):
        d = QDialog(self); d.setWindowTitle(f"Approve {t_name}?"); d.resize(400, 200); d.setStyleSheet("background:#2a2a2a;color:#ddd;")
        l = QVBoxLayout(d); l.addWidget(QLabel(f"Allow <b style='color:#e74c3c'>{t_name}</b>?")); e = QTextEdit(); e.setPlainText(args); e.setReadOnly(True); l.addWidget(e)
        bl = QHBoxLayout()
        dn = QPushButton("🚫 Deny"); dn.setStyleSheet("background:#e74c3c;color:white;padding:10px;font-weight:bold;"); dn.clicked.connect(handler.deny); dn.clicked.connect(d.close)
        ap = QPushButton("✅ Approve"); ap.setStyleSheet("background:#2ecc71;color:white;padding:10px;font-weight:bold;"); ap.clicked.connect(handler.approve); ap.clicked.connect(d.close)
        bl.addWidget(dn); bl.addWidget(ap); l.addLayout(bl); d.show()

    def _update_flow(self, s):
        self.llm_flow_lbl.setText(f"🧠 LLM: {s.get('llm_flow', '0/0')}")
        self.tool_flow_lbl.setText(f"🔧 Tools: {s.get('global_tool_flow', '0/0')}")
        ut = s.get('llm_state', {}).get('utilization', 0)
        self.llm_flow_lbl.setStyleSheet(f"color:{'#cf6679' if ut>0.9 else '#f9a825' if ut>0.6 else '#bb86fc'};font-weight:bold;")

    def _show_diff(self, old, new, fname):
        dt = self.centralWidget().findChild(QSplitter).findChild(QSplitter).findChild(QTabWidget).widget(2)
        if isinstance(dt, DiffViewer): dt.display_diff(old, new)

    def _on_code_action(self, act, code):
        if act == "run": self._run_code(code)
        elif act == "insert": self.editor.setPlainText(code)

    def _run_code(self, code):
        if not code.strip(): return
        self.output.append(f"\n▶ Running..."); r = execute_code.invoke({"code": code}); self.output.append(r); self.output.verticalScrollBar().setValue(self.output.verticalScrollBar().maximum())

    def _show_config(self):
        d = ConfigDialog(self.config); d.config_saved.connect(self._save_cfg); d.exec()

    def _save_cfg(self, c): self.config = c; save_config(c)
    def _stop(self):
        if self.worker: self.worker.cancel(); self.status_lbl.setText("Stopped")
    def _new_thread(self): self.tabs.create_tab()
    def _refresh_files(self):
        self.files.clear()
        for f in get_workspace().iterdir(): self.files.addItem(f"{'📁' if f.is_dir() else '📄'} {f.name}")
    def _open_file(self, item):
        p = get_workspace() / item.text().split(" ", 1)[-1]
        if p.exists(): self.editor.setPlainText(p.read_text(encoding='utf-8'))
    def _open_file_dlg(self):
        p = QFileDialog.getOpenFileName(self, "Open", str(get_workspace()))[0]
        if p: self.editor.setPlainText(Path(p).read_text(encoding='utf-8'))
    def _save_file_dlg(self):
        p = QFileDialog.getSaveFileName(self, "Save", str(get_workspace()), "Py (*.py)")[0]
        if p: Path(p).write_text(self.editor.toPlainText(), encoding='utf-8')
    def _send_sel(self):
        sel = self.editor.textCursor().selectedText().replace('\u2029', '\n')
        if sel: self.tabs.current_chat().input_field.setText(f"Analyze:\n```python\n{sel}\n```\n")

    def closeEvent(self, e):
        if self.worker: self.worker.cancel(); self.worker.wait(1000)
        e.accept()

# ==============================================================================
# 8. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(30,30,30)); p.setColor(QPalette.ColorRole.WindowText, QColor(220,220,220))
    p.setColor(QPalette.ColorRole.Base, QColor(25,25,25)); p.setColor(QPalette.ColorRole.Text, QColor(220,220,220))
    p.setColor(QPalette.ColorRole.Button, QColor(45,45,45)); p.setColor(QPalette.ColorRole.ButtonText, QColor(220,220,220))
    p.setColor(QPalette.ColorRole.Highlight, QColor(60,120,200)); app.setPalette(p)
    
    w = MainWindow(); w.show()
    sys.exit(app.exec())