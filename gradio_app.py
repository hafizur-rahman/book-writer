from __future__ import annotations
import json
import re
import html
from pathlib import Path
import difflib

import gradio as gr

from chapters import CHAPTERS, Chapter
from models import build_cpu_model, build_gpu_model
from workflow import (
    RESEARCH_SYS,
    OUTLINE_SYS,
    DRAFT_SYS,
    REVIEW_SYS,
    EDIT_SYS,
    _build,
)

PROGRESS = Path(".book_progress")


# =====================================================================
# HYBRID STREAMING RUNNER (per-node + thinking + cleaned final)
# =====================================================================

class HybridStreamingRunner:
    def __init__(self, thread_id: str, cpu_model, gpu_model, progress_dir: str = str(PROGRESS)):
        self.thread_id = thread_id
        self.cpu_model = cpu_model
        self.gpu_model = gpu_model
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        m = self.progress_dir / "manifest.json"
        return json.loads(m.read_text()) if m.exists() else {}

    def _save_manifest(self):
        (self.progress_dir / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2)
        )

    @staticmethod
    def _render_thinking_html(thinking_raw: str) -> str:
        if not thinking_raw:
            return ""
        escaped = html.escape(thinking_raw)
        escaped = escaped.replace(
            "<think>",
            "<span style='color:#ffcc00;font-weight:bold;'>⟪ THINK START ⟫</span>\n"
        )
        escaped = escaped.replace(
            "</think>",
            "\n<span style='color:#ff4444;font-weight:bold;'>⟪ THINK END ⟫</span>"
        )
        return (
            "<div style='background:#111;padding:10px;border-radius:6px;'>"
            "<div style='color:#ffcc00;font-weight:bold;margin-bottom:4px;'>Model reasoning</div>"
            f"<pre style='color:#ccc;font-size:0.9em;white-space:pre-wrap;'>{escaped}</pre>"
            "</div>"
        )

    def stream_chapter(self, chapter: Chapter, max_iterations: int = 2):
        """
        Yields per-node streaming updates:

        (node_tag, chunk_text, progress_float,
         original_draft, edited_draft, thinking_html, cleaned_final)
        """

        state = {
            "chapter": chapter,
            "research": "",
            "outline": "",
            "draft": "",
            "review": "",
            "final": "",
            "iterations": 0,
            "decision": "pass",
        }

        total_phases = 5
        phase_index = 0

        original_draft = ""
        edited_draft = ""
        thinking_raw = ""
        thinking_html = ""
        in_think = False
        cleaned_final = ""

        # ---------------- RESEARCH (CPU streaming) ----------------
        phase_index += 1
        progress = phase_index / total_phases
        research_prompt = f"{RESEARCH_SYS}\n\n{_build(state)}\n\nResearch notes:"
        research_text = ""
        yield "[research]", "", progress, original_draft, edited_draft, thinking_html, cleaned_final

        for chunk in self.cpu_model.stream(research_prompt):
            piece = getattr(chunk, "content", None)
            if not piece:
                continue

            if "<think>" in piece or "</think>" in piece or in_think:
                thinking_raw += piece
                if "<think>" in piece:
                    in_think = True
                if "</think>" in piece:
                    in_think = False
                thinking_html = self._render_thinking_html(thinking_raw)
                yield "[research-thinking]", "", progress, original_draft, edited_draft, thinking_html, cleaned_final
                continue

            research_text += piece
            state["research"] = research_text
            yield "[research]", piece, progress, original_draft, edited_draft, thinking_html, cleaned_final

        # ---------------- OUTLINE (CPU streaming) ----------------
        phase_index += 1
        progress = phase_index / total_phases
        outline_prompt = (
            f"{OUTLINE_SYS}\n\n{_build(state)}\n\nResearch notes:\n{state['research']}"
        )
        outline_text = ""
        yield "[outline]", "", progress, original_draft, edited_draft, thinking_html, cleaned_final

        for chunk in self.cpu_model.stream(outline_prompt):
            piece = getattr(chunk, "content", None)
            if not piece:
                continue

            if "<think>" in piece or "</think>" in piece or in_think:
                thinking_raw += piece
                if "<think>" in piece:
                    in_think = True
                if "</think>" in piece:
                    in_think = False
                thinking_html = self._render_thinking_html(thinking_raw)
                yield "[outline-thinking]", "", progress, original_draft, edited_draft, thinking_html, cleaned_final
                continue

            outline_text += piece
            state["outline"] = outline_text
            yield "[outline]", piece, progress, original_draft, edited_draft, thinking_html, cleaned_final


        # ---------------- DRAFT (CPU streaming) ----------------
        phase_index += 1
        progress = phase_index / total_phases
        draft_prompt = (
            f"{DRAFT_SYS}\n\n{state['outline'] or _build(state)}\n\n"
            f"Notes:\n{state['research']}\n\n"
            f"Write the full chapter now."
        )
        draft_text = ""
        yield "[draft]", "", progress, original_draft, edited_draft, thinking_html, cleaned_final

        for chunk in self.cpu_model.stream(draft_prompt):
            piece = getattr(chunk, "content", None)
            if not piece:
                continue

            if "<think>" in piece or "</think>" in piece or in_think:
                thinking_raw += piece
                if "<think>" in piece:
                    in_think = True
                if "</think>" in piece:
                    in_think = False
                thinking_html = self._render_thinking_html(thinking_raw)
                yield "[draft-thinking]", "", progress, original_draft, edited_draft, thinking_html, cleaned_final
                continue

            draft_text += piece
            state["draft"] = draft_text
            yield "[draft]", piece, progress, original_draft, edited_draft, thinking_html, cleaned_final

        original_draft = draft_text

        # ---------------- REVIEW + EDIT LOOP ----------------
        phase_index += 1
        progress = phase_index / total_phases

        iterations = 0
        decision = "improve"
        review_text = ""

        while decision == "improve" and iterations < max_iterations:
            review_prompt = f"{REVIEW_SYS}\n\nDRAFT:\n{state['draft']}"
            review_resp = self.gpu_model.invoke(review_prompt).content
            review_text = review_resp
            state["review"] = review_text

            try:
                data = json.loads(review_resp.split("```")[0].strip().lstrip("```json"))
            except Exception:
                data = {"score": 5, "pass": False, "feedback": ["Could not parse JSON"]}

            decision = "pass" if data.get("pass") and data.get("score", 0) >= 7 else "improve"
            state["decision"] = decision

            yield "[review]", review_resp, progress, original_draft, edited_draft, thinking_html, cleaned_final

            if decision == "pass":
                break

            iterations += 1
            state["iterations"] = iterations

            edit_prompt = (
                f"{DRAFT_SYS}\n{EDIT_SYS}\n\nDraft:\n{state['draft']}\n\nFeedback:\n{state['review']}"
            )
            edited_text = ""
            yield "[edit]", "", progress, original_draft, edited_draft, thinking_html, cleaned_final

            for chunk in self.cpu_model.stream(edit_prompt):
                piece = getattr(chunk, "content", None)
                if not piece:
                    continue

                if "<think>" in piece or "</think>" in piece or in_think:
                    thinking_raw += piece
                    if "<think>" in piece:
                        in_think = True
                    if "</think>" in piece:
                        in_think = False
                    thinking_html = self._render_thinking_html(thinking_raw)
                    yield "[edit-thinking]", "", progress, original_draft, edited_draft, thinking_html, cleaned_final
                    continue

                edited_text += piece
                state["draft"] = edited_text
                edited_draft = edited_text
                yield "[edit]", piece, progress, original_draft, edited_draft, thinking_html, cleaned_final

        # ---------------- FINALIZE (cleaned output) ----------------
        phase_index += 1
        progress = phase_index / total_phases

        cleaned_final = re.sub(r"<think>.*?</think>", "", state["draft"], flags=re.DOTALL)
        state["final"] = cleaned_final

        yield "[finalize]", "", progress, original_draft, edited_draft, thinking_html, cleaned_final

        chapter_md = state["final"]
        if chapter_md:
            (self.progress_dir / f"chapter_{chapter.num}.md").write_text(chapter_md)
            self.manifest[str(chapter.num)] = {"status": "done"}
            self._save_manifest()

        yield f"[done] Chapter {chapter.num}: {chapter.title}", "", 1.0, original_draft, edited_draft, thinking_html, cleaned_final


# =====================================================================
# GRADIO UI — Multi-tab per step
# =====================================================================

def build_chapter_map():
    return {ch.num: ch for ch in CHAPTERS}


def make_app():
    chapter_map = build_chapter_map()

    with gr.Blocks(title="DL/LLM Book Author — Multi-step Streaming") as demo:
        gr.Markdown(
            "# 📘 Deep Learning & LLM Book Author\n"
            "Per‑step streaming, thinking tokens, cleaned final, and diff viewer."
        )

        with gr.Row():
            thread_id = gr.Textbox(
                label="Thread ID",
                value="book-thread-1",
                placeholder="Unique thread id",
            )
            cpu_model_name = gr.Textbox(
                label="CPU model (Ollama)",
                value="ornith-1.5:35b",
            )

        chapter_num = gr.Dropdown(
            choices=[str(ch.num) for ch in CHAPTERS],
            label="Chapter number",
            value=str(CHAPTERS[0].num),
        )

        backend_prefer = gr.Radio(
            choices=["gpu", "cpu", "auto"],
            value="gpu",
            label="Backend preference (informational)",
        )

        # --- Research tab ---
        with gr.Tab("Research"):
            research_status = gr.Textbox(label="Status", interactive=False)
            research_output = gr.Markdown(label="Research Output", interactive=False)
            research_thinking = gr.HTML(label="Thinking Stream")

        # --- Outline tab ---
        with gr.Tab("Outline"):
            outline_status = gr.Textbox(label="Status", interactive=False)
            outline_output = gr.Markdown(label="Outline Output", interactive=False)
            outline_thinking = gr.HTML(label="Thinking Stream")

        # --- Draft tab ---
        with gr.Tab("Draft"):
            draft_status = gr.Textbox(label="Status", interactive=False)
            draft_output = gr.Markdown(label="Draft Output", interactive=False)
            draft_thinking = gr.HTML(label="Thinking Stream")

        # --- Review tab ---
        with gr.Tab("Review"):
            review_status = gr.Textbox(label="Status", interactive=False)
            review_output = gr.Markdown(label="Review JSON", interactive=False)
            review_thinking = gr.HTML(label="Thinking Stream")

        # --- Edit tab ---
        with gr.Tab("Edit"):
            edit_status = gr.Textbox(label="Status", interactive=False)
            edit_output = gr.Markdown(label="Edited Draft", interactive=False)
            edit_thinking = gr.HTML(label="Thinking Stream")

        # --- Final tab ---
        with gr.Tab("Final"):
            final_output = gr.Markdown(label="Cleaned Final Chapter", interactive=False)

        # --- Diff tab ---
        with gr.Tab("Diff Viewer"):
            diff_box = gr.HTML(label="Diff (original vs edited)")

        # --- Full chapter tab ---
        with gr.Tab("Full Chapter"):
            full_chapter = gr.Markdown(label="Complete Chapter", interactive=False)

        # --- Progress tab ---
        with gr.Tab("Progress"):
            progress_status = gr.Textbox(label="Current Node", interactive=False)
            progress_bar = gr.Slider(
                label="Progress",
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.01,
                interactive=False,
            )

        def gradio_multi_step_stream(thread_id_val, cpu_model_name_val, chapter_num_val, backend_pref):
            cpu_model = build_cpu_model(cpu_model_name_val)
            gpu_model = build_gpu_model()
            ch = chapter_map[int(chapter_num_val)]

            runner = HybridStreamingRunner(thread_id_val, cpu_model, gpu_model)

            # initial state
            rs, ro, rt = "", "", ""
            os, oo, ot = "", "", ""
            ds, do, dt = "", "", ""
            rvs, rvo, rvt = "", "", ""
            es, eo, et = "", "", ""
            fo = ""
            diff_html = ""
            full = ""
            p_status = ""
            p_bar = 0.0

            yield (
                rs, ro, rt,
                os, oo, ot,
                ds, do, dt,
                rvs, rvo, rvt,
                es, eo, et,
                fo,
                diff_html,
                full,
                p_status,
                p_bar,
            )

            original_draft = ""
            edited_draft = ""

            for node, chunk, prog, orig, edited, thinking_html, cleaned_final in runner.stream_chapter(ch):
                p_status = node
                p_bar = prog

                if orig:
                    original_draft = orig
                if edited:
                    edited_draft = edited

                if node.startswith("[research"):
                    rs = node
                    ro += chunk
                    rt = thinking_html

                elif node.startswith("[outline"):
                    os = node
                    oo += chunk
                    ot = thinking_html

                elif node.startswith("[draft"):
                    ds = node
                    do += chunk
                    dt = thinking_html

                elif node.startswith("[review"):
                    rvs = node
                    rvo = chunk
                    rvt = thinking_html

                elif node.startswith("[edit"):
                    es = node
                    eo += chunk
                    et = thinking_html

                elif node.startswith("[finalize"):
                    fo = cleaned_final

                elif node.startswith("[done"):
                    full = cleaned_final

                if original_draft and edited_draft:
                    diff_html = difflib.HtmlDiff().make_table(
                        original_draft.splitlines(),
                        edited_draft.splitlines(),
                        fromdesc="Original",
                        todesc="Edited",
                        context=True,
                        numlines=3,
                    )

                yield (
                    rs, ro, rt,
                    os, oo, ot,
                    ds, do, dt,
                    rvs, rvo, rvt,
                    es, eo, et,
                    fo,
                    diff_html,
                    full,
                    p_status,
                    p_bar,
                )

        run_btn = gr.Button("Run Agentic Chapter (Multi-step streaming)")

        run_btn.click(
            fn=gradio_multi_step_stream,
            inputs=[thread_id, cpu_model_name, chapter_num, backend_prefer],
            outputs=[
                research_status, research_output, research_thinking,
                outline_status, outline_output, outline_thinking,
                draft_status, draft_output, draft_thinking,
                review_status, review_output, review_thinking,
                edit_status, edit_output, edit_thinking,
                final_output,
                diff_box,
                full_chapter,
                progress_status,
                progress_bar,
            ],
        )

    return demo


if __name__ == "__main__":
    app = make_app()
    app.queue().launch(server_name="0.0.0.0", server_port=7860)
