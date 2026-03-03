"""
Extract text and render pages from project PDFs.

Usage:
    uv run python scripts/extract_pdfs.py                    # List all PDFs
    uv run python scripts/extract_pdfs.py --extract-text     # Extract text from all PDFs
    uv run python scripts/extract_pdfs.py --render A-10      # Render pages matching pattern to PNG
    uv run python scripts/extract_pdfs.py --render-all       # Render all PDF pages to PNG
"""

import argparse
import sys
from pathlib import Path

import fitz  # pymupdf


DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def list_pdfs() -> list[Path]:
    """List all PDF files in the data directory."""
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    for pdf in pdfs:
        size_kb = pdf.stat().st_size / 1024
        doc = fitz.open(pdf)
        print(f"  {pdf.name:<60} {doc.page_count:>3} pages  {size_kb:>8.1f} KB")
        doc.close()
    return pdfs


def extract_text(pattern: str | None = None) -> None:
    """Extract text from PDFs, optionally filtering by pattern."""
    output_dir = OUTPUT_DIR / "text"
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if pattern:
        pdfs = [p for p in pdfs if pattern.lower() in p.name.lower()]

    for pdf_path in pdfs:
        doc = fitz.open(pdf_path)
        text_parts = []
        has_text = False

        for page in doc:
            text = page.get_text().strip()
            if text:
                has_text = True
                text_parts.append(f"--- Page {page.number + 1} ---\n{text}")

        if has_text:
            out_file = output_dir / f"{pdf_path.stem}.txt"
            out_file.write_text("\n\n".join(text_parts), encoding="utf-8")
            print(f"  Extracted text: {out_file.name}")
        else:
            print(f"  No text (image-based): {pdf_path.name}")

        doc.close()


def render_pages(pattern: str | None = None, dpi: int = 200) -> None:
    """Render PDF pages to PNG images."""
    output_dir = OUTPUT_DIR / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if pattern:
        pdfs = [p for p in pdfs if pattern.lower() in p.name.lower()]

    zoom = dpi / 72  # 72 is the default PDF DPI
    mat = fitz.Matrix(zoom, zoom)

    for pdf_path in pdfs:
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            out_file = output_dir / f"{pdf_path.stem}_p{page.number + 1}.png"
            pix.save(str(out_file))
            print(f"  Rendered: {out_file.name} ({pix.width}x{pix.height})")
        doc.close()


def main():
    parser = argparse.ArgumentParser(description="PDF extraction utility for Nygårdsvikveien 38 project")
    parser.add_argument("--extract-text", action="store_true", help="Extract text from all PDFs")
    parser.add_argument("--render", type=str, nargs="?", const="", help="Render PDF pages to PNG (optional: filter pattern)")
    parser.add_argument("--render-all", action="store_true", help="Render all PDF pages to PNG")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for rendering (default: 200)")
    parser.add_argument("--pattern", type=str, help="Filter PDFs by name pattern")

    args = parser.parse_args()

    if not any([args.extract_text, args.render is not None, args.render_all]):
        print(f"PDFs in {DATA_DIR}:\n")
        list_pdfs()
        print(f"\nUse --extract-text, --render, or --render-all to process files.")
        return

    if args.extract_text:
        print("Extracting text from PDFs...\n")
        extract_text(args.pattern)

    if args.render is not None or args.render_all:
        pattern = args.render if args.render else args.pattern
        print(f"\nRendering PDF pages to PNG (DPI={args.dpi})...\n")
        render_pages(pattern, args.dpi)


if __name__ == "__main__":
    main()
