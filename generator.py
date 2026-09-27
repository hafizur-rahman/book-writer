"""Loop that generates every chapter and writes book.md."""
from __future__ import annotations
import json
from pathlib import Path
from chapters import CHAPTERS
from workflow import build_workflow

PROGRESS = Path(".book_progress")

class BookGenerator:
    def __init__(self, chapters, cpu_model, gpu_model, progress_dir: str = str(PROGRESS)):
        self.chapters = chapters
        self.cpu_model = cpu_model
        self.gpu_model = gpu_model
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_manifest()

    # ---- manifest (resume / skip-done) -------------------------
    def _load_manifest(self) -> dict:
        m = self.progress_dir / "manifest.json"
        return json.loads(m.read_text()) if m.exists() else {}

    def _save_manifest(self):
        (self.progress_dir / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2))

    # ---- single chapter ----------------------------------------
    def run_chapter(self, chapter) -> str:
        graph = build_workflow(self.cpu_model, self.gpu_model)
        initial = {
            "chapter": chapter, "research": "", "outline": "",
            "draft": "", "review": "", "final": "",
            "iterations": 0, "decision": "pass", "messages": [],
        }
        result = graph.invoke(initial)
        chapter_md = result["final"]
        (self.progress_dir / f"chapter_{chapter.num}.md").write_text(chapter_md)
        self.manifest[str(chapter.num)] = {"status": "done"}
        self._save_manifest()
        return chapter_md

    # ---- full loop ---------------------------------------------
    def run(self, resume: bool = True, only: int = None,
            regenerate: list[int] = None):
        regenerate = regenerate or []
        for ch in self.chapters:
            idx = ch.num
            if only is not None and idx != only:
                continue
            done = self.manifest.get(str(idx), {}).get("status") == "done"
            if idx in regenerate:
                done = False
            if resume and done:
                print(f"[skip]   Chapter {idx}: {ch.title}")
                continue
            print(f"[run]    Chapter {idx}: {ch.title}")
            self.run_chapter(ch)
        self.build_book()

    # ---- assemble book.md --------------------------------------
    def build_book(self):
        front = (
            "# Deep Learning & Large Language Models\n\n"
            "A progressive textbook from CNNs to agentic LLMs.\n"
            "Generated with a LangGraph agentic author over local Ollama/llama.cpp models.\n\n"
        )
        parts = [front]
        for ch in self.chapters:
            f = self.progress_dir / f"chapter_{ch.num}.md"
            if f.exists():
                parts.append(f.read_text().lstrip("#").lstrip())
        book = "\n\n".join(parts)
        Path("book.md").write_text(book)
        print(f"\n[done] Wrote book.md ({len(book)} chars, {len(self.chapters)} chapters)")


if __name__ == "__main__":
    pass  # main.py owns model resolution + CLI