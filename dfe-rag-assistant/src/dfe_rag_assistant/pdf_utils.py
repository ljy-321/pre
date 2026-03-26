from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Element:
    element_type: str  # "text" | "table"
    content: str
    page_num: int
    # bbox: (x0, y0, x1, y1) in PDF coordinate space
    bbox: Optional[List[float]] = None
    # 排序辅助字段（来自多栏版面）
    col_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_type": self.element_type,
            "content": self.content,
            "page_num": self.page_num,
            "bbox": self.bbox,
            "col_id": self.col_id,
        }


_HEADER_FOOTER_RE = re.compile(r"(^第\s*\d+\s*页$)|(^Page\s*\d+$)", re.IGNORECASE)


def _clean_text_block(text: str) -> str:
    # 去掉可能存在的页眉页脚行
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _HEADER_FOOTER_RE.match(s):
            continue
        lines.append(s)
    return "\n".join(lines).strip()


def _table_to_markdown(table: List[List[Any]]) -> str:
    # table: rows x cols
    # 取第一行作为表头，其余行作为数据行（简单启发式）
    # 注意：pdfplumber 的表格提取并不保证一定干净，所以这里尽量做容错。
    cleaned_rows: List[List[str]] = []
    for row in table:
        cleaned_row: List[str] = []
        for cell in row:
            if cell is None:
                cleaned_row.append("")
            else:
                cleaned_row.append(str(cell).strip())
        cleaned_rows.append(cleaned_row)

    # 找到第一个非全空行当表头
    header = None
    for r in cleaned_rows:
        if any(c for c in r):
            header = r
            break
    if header is None:
        return ""

    # 统一列数
    n_cols = len(header)
    header = (header + [""] * n_cols)[:n_cols]

    data_rows = []
    for r in cleaned_rows:
        r = (r + [""] * n_cols)[:n_cols]
        if r == header:
            continue
        if not any(c for c in r):
            continue
        data_rows.append(r)

    md = []
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * n_cols) + " |")
    for dr in data_rows:
        md.append("| " + " | ".join(dr) + " |")
    return "\n".join(md).strip()


def extract_elements_from_pdf(
    pdf_path: str,
    max_pages: Optional[int] = None,
    assume_two_columns: bool = True,
) -> List[Element]:
    """
    产出可检索的“证据元素”：文本块 + 表格块。

    说明：
    - 多栏排序：当前用一个简单假设（两栏则按 x_center 分左/右，非两栏则退化成单栏顺序）
    - 表格保护：表格作为原子 element，不参与普通文本切分。
    """
    pdf_path = str(pdf_path)
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(pdf_path)

    elements: List[Element] = []

    # 1) 文本块：用 PyMuPDF 的 blocks（带 bbox）
    try:
        import fitz  # PyMuPDF
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少依赖 PyMuPDF（pymupdf），请先 pip install pymupdf") from e

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    n_pages = total_pages if max_pages is None else min(max_pages, total_pages)

    for page_idx in range(n_pages):
        page = doc.load_page(page_idx)
        # page.rect: (0,0,w,h)
        page_w = float(page.rect.width)
        blocks = page.get_text("blocks")  # List[tuple]

        # block 元组常见结构：x0, y0, x1, y1, text, block_no, block_type
        text_blocks: List[Element] = []
        for b in blocks:
            x0, y0, x1, y1, text, _block_no, _block_type = b
            if not text or not str(text).strip():
                continue
            cleaned = _clean_text_block(str(text))
            if not cleaned:
                continue
            if assume_two_columns:
                x_center = (float(x0) + float(x1)) / 2.0
                col_id = 0 if x_center < page_w / 2.0 else 1
            else:
                col_id = 0
            text_blocks.append(
                Element(
                    element_type="text",
                    content=cleaned,
                    page_num=page_idx,
                    bbox=[float(x0), float(y0), float(x1), float(y1)],
                    col_id=col_id,
                )
            )

        # 多栏阅读顺序：先列后行
        text_blocks.sort(key=lambda e: (e.col_id, e.bbox[1] if e.bbox else 0))
        elements.extend(text_blocks)

    doc.close()

    # 2) 表格块：用 pdfplumber 提取并转成 markdown
    try:
        import pdfplumber
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少依赖 pdfplumber，请先 pip install pdfplumber") from e

    with pdfplumber.open(pdf_path) as pdf:
        n_pages2 = len(pdf.pages) if max_pages is None else min(max_pages, len(pdf.pages))
        for page_idx in range(n_pages2):
            page = pdf.pages[page_idx]
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []

            for t in tables or []:
                md = _table_to_markdown(t)
                if not md:
                    continue
                # bbox 在 pdfplumber 里不总是直接返回；这里先置空
                elements.append(
                    Element(
                        element_type="table",
                        content=md,
                        page_num=page_idx,
                        bbox=None,
                        col_id=0,
                    )
                )

    return elements


def iter_pdf_paths(pdf_dir: str, exts: Iterable[str] = (".pdf",)) -> List[str]:
    root = Path(pdf_dir)
    if not root.exists():
        raise FileNotFoundError(f"pdf_dir not found: {pdf_dir}")
    out = []
    for f in root.rglob("*"):
        if f.suffix.lower() in set(exts):
            out.append(str(f))
    return sorted(out)

