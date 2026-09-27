import argparse, time
from models import build_cpu_model, build_gpu_model
from chapters import CHAPTERS
from generator import BookGenerator

def main():
    p = argparse.ArgumentParser(description="Generate a DL/LLM book with a LangGraph author.")
    p.add_argument("--cpu-model", default="ornith-1.5:35b", help="Ollama model (CPU).")
    p.add_argument("--prefer", choices=["gpu", "cpu", "auto"], default="gpu",
                   help="Where to do the heavy drafting.")
    p.add_argument("--all", action="store_true", help="Generate all chapters.")
    p.add_argument("--chapter", type=int, help="Only generate this single chapter.")
    p.add_argument("--regenerate", type=int, help="Regenerate a specific chapter number.")
    p.add_argument("--resume", action="store_true", default=True, help="Skip finished chapters.")
    p.add_argument("--no-resume", action="store_false", dest="resume")
    args = p.parse_args()

    cpu_model = build_cpu_model(args.cpu_model)
    gpu_model = build_gpu_model()
    
    print(f"Authoring with: (backend={args.prefer})")

    gen = BookGenerator(CHAPTERS, cpu_model, gpu_model)
    t0 = time.time()
    gen.run(resume=args.resume, only=args.chapter, regenerate=[args.regenerate] if args.regenerate else None)
    print(f"Elapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()