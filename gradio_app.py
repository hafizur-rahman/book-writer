from __future__ import annotations
import json
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
# HYBRID STREAMING RUNNER
# GPU: research, outline, edit
# CPU: draft
# =====================================================================

class HybridStreamingRunner:
    def __init__(self, thread_id: str, cpu_model, gpu_model, progress_dir: str = str(PROGRESS)):
        self.thread_id = thread_id
        self.cpu_model = cpu_model
        self.gpu_model = gpu_model
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_manifest()

    # ---------------- manifest ----------------
    def _load_manifest(self) -> dict:
        m = self.progress_dir / "manifest.json"
        return json.loads(m.read_text()) if m.exists() else {}

    def _save_manifest(self):
        (self.progress_dir / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2)
        )

    # ---------------- hybrid streaming chapter ----------------
    def stream_chapter(self, chapter: Chapter, max_iterations: int = 2):
        """
        Agentic loop implemented as a streaming runner:

        - research: GPU streaming
        - outline: GPU streaming
        - draft: CPU streaming
        - review: GPU (JSON, non-streaming)
        - edit: GPU streaming
        - finalize: pass draft as final

        Yields:
          (status_label, incremental_chunk, progress_float, original_draft, edited_draft)
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

        total_phases = 5  # research, outline, draft, review+edit loop, finalize
        phase_index = 0

        original_draft = ""
        edited_draft = ""

        # ---------------- RESEARCH (GPU streaming) ----------------
        phase_index += 1
        progress = phase_index / total_phases
        research_prompt = f"{RESEARCH_SYS}\n\n{_build(state)}\n\nResearch notes:"
        research_text = ""
        yield "[research]", "", progress, original_draft, edited_draft

        for chunk in self.cpu_model.stream(research_prompt):
            piece = getattr(chunk, "content", None)
            if not piece:
                continue
            research_text += piece
            state["research"] = research_text
            yield "[research]", piece, progress, original_draft, edited_draft

        research_text += '\n'

        # ---------------- OUTLINE (GPU streaming) ----------------
        phase_index += 1
        progress = phase_index / total_phases
        outline_prompt = (
            f"{OUTLINE_SYS}\n\n{_build(state)}\n\nResearch notes:\n{state['research']}"
        )
        outline_text = ""
        yield "[outline]", "", progress, original_draft, edited_draft

        for chunk in self.cpu_model.stream(outline_prompt):
            piece = getattr(chunk, "content", None)
            if not piece:
                continue
            outline_text += piece
            state["outline"] = outline_text
            yield "[outline]", piece, progress, original_draft, edited_draft
        
        outline_text += '\n'

        # ---------------- DRAFT (CPU streaming) ----------------
        phase_index += 1
        progress = phase_index / total_phases
        draft_prompt = (
            f"{DRAFT_SYS}\n\n{state['outline'] or _build(state)}\n\n"
            f"Notes:\n{state['research']}\n\n"
            f"Write the full chapter now."
        )
        draft_text = ""
        yield "[draft]", "", progress, original_draft, edited_draft

        for chunk in self.cpu_model.stream(draft_prompt):
            piece = getattr(chunk, "content", None)
            if not piece:
                continue
            draft_text += piece
            state["draft"] = draft_text
            yield "[draft]", piece, progress, original_draft, edited_draft

        draft_text += '\n'

        original_draft = draft_text

        # ---------------- REVIEW + EDIT LOOP ----------------
        phase_index += 1
        progress = phase_index / total_phases

        iterations = 0
        decision = "improve"
        review_text = ""

        while decision == "improve" and iterations < max_iterations:
            # REVIEW (GPU, non-streaming JSON)
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

            yield "[review]", review_resp, progress, original_draft, edited_draft

            if decision == "pass":
                break

            # EDIT (GPU streaming)
            iterations += 1
            state["iterations"] = iterations

            edit_prompt = (
                f"{DRAFT_SYS}\n{EDIT_SYS}\n\nDraft:\n{state['draft']}\n\nFeedback:\n{state['review']}"
            )
            edited_text = ""
            yield "[edit]", "", progress, original_draft, edited_draft

            for chunk in self.cpu_model.stream(edit_prompt):
                piece = getattr(chunk, "content", None)
                if not piece:
                    continue
                edited_text += piece
                state["draft"] = edited_text
                edited_draft = edited_text
                yield "[edit]", piece, progress, original_draft, edited_draft
            
            edited_text += '\n'
        
        # ---------------- FINALIZE ----------------
        phase_index += 1
        progress = phase_index / total_phases
        state["final"] = state["draft"]
        yield "[finalize]", "", progress, original_draft, edited_draft

        # Persist final chapter
        chapter_md = state["final"]
        if chapter_md:
            (self.progress_dir / f"chapter_{chapter.num}.md").write_text(chapter_md)
            self.manifest[str(chapter.num)] = {"status": "done"}
            self._save_manifest()

        yield f"[done] Chapter {chapter.num}: {chapter.title}", "", 1.0, original_draft, edited_draft


# =====================================================================
# GRADIO UI (Hybrid streaming + diff + progress)
# =====================================================================

def build_chapter_map():
    return {ch.num: ch for ch in CHAPTERS}


def make_app():
    chapter_map = build_chapter_map()

    with gr.Blocks(title="DL/LLM Book Author — Hybrid Streaming + Diff + Progress") as demo:
        gr.Markdown(
            "# 📘 Deep Learning & LLM Book Author\n"
            "Hybrid GPU/CPU streaming per node, side‑by‑side draft vs edited, and real‑time progress."
        )

        with gr.Row():
            thread_id = gr.Textbox(
                label="Thread ID",
                value="book-thread-1",
                placeholder="Unique thread id (for consistency with your LangGraph setup)",
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

        with gr.Tab("Hybrid Agentic Chapter (per‑node streaming)"):
            status = gr.Textbox(
                label="Status / Node",
                lines=2,
                interactive=False,
            )

            output = gr.Markdown(
                label="Streaming Chapter Markdown",
                lines=30,
                interactive=False,
            )

            progress_bar = gr.Slider(
                label="Progress",
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.01,
                interactive=False,
            )

            with gr.Row():
                original_box = gr.Markdown(
                    label="Original Draft (before edit)",
                    lines=20,
                    interactive=False,
                )
                edited_box = gr.Markdown(
                    label="Edited Draft (after edit)",
                    lines=20,
                    interactive=False,
                )

            diff_box = gr.HTML(
                label="Diff (original vs edited)",
            )

            def gradio_hybrid_stream(thread_id_val, cpu_model_name_val, chapter_num_val, backend_pref):
                cpu_model = build_cpu_model(cpu_model_name_val)
                gpu_model = build_gpu_model()
                ch = chapter_map[int(chapter_num_val)]

                runner = HybridStreamingRunner(thread_id_val, cpu_model, gpu_model)

                status_text = (
                    f"Hybrid streaming for Chapter {ch.num}: {ch.title} "
                    f"(backend={backend_pref}, thread_id={thread_id_val})"
                )
                full_output = ""
                original_draft = ""
                edited_draft = ""
                progress = 0.0

                # First yield: initial status
                yield status_text, full_output, progress, original_draft, edited_draft, ""

                for node_prefix, chunk, prog, orig, edited in runner.stream_chapter(ch):
                    status_text = node_prefix
                    progress = prog
                    full_output += chunk
                    if orig:
                        original_draft = orig
                    if edited:
                        edited_draft = edited

                    diff_html = ""
                    if original_draft and edited_draft:
                        diff_html = difflib.HtmlDiff().make_table(
                            original_draft.splitlines(),
                            edited_draft.splitlines(),
                            fromdesc="Original",
                            todesc="Edited",
                            context=True,
                            numlines=3,
                        )

                    yield status_text, full_output, progress, original_draft, edited_draft, diff_html

            run_btn = gr.Button("Generate Chapter (Hybrid per‑node streaming)")

            run_btn.click(
                fn=gradio_hybrid_stream,
                inputs=[thread_id, cpu_model_name, chapter_num, backend_prefer],
                outputs=[status, output, progress_bar, original_box, edited_box, diff_box],
            )

    return demo


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    app = make_app()
    app.queue().launch(server_name="0.0.0.0", server_port=7860)
