import re
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator

app = FastAPI(title="AI Document Processing System")


FIELD_ALIASES = {
    "cr_no": "CR No",
    "registered_in_the_grade": "Registered in the grade",
    "occi_no": "OCCI No",
    "date_of_issue": "Date of issue",
    "date_of_expiry": "Date of expiry",
    "head_office": "Head Office",
}


class InvoiceExtraction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cr_no: str | None = Field(default=None, alias="CR No")
    registered_in_the_grade: str | None = Field(
        default=None, alias="Registered in the grade"
    )
    occi_no: str | None = Field(default=None, alias="OCCI No")
    date_of_issue: date | None = Field(default=None, alias="Date of issue")
    date_of_expiry: date | None = Field(default=None, alias="Date of expiry")
    head_office: str | None = Field(default=None, alias="Head Office")

    @field_validator("cr_no", "registered_in_the_grade", "occi_no", "head_office")
    @classmethod
    def clean_blank_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split()).strip(" :-")
        return value or None

    @field_validator("date_of_issue", "date_of_expiry", mode="before")
    @classmethod
    def parse_document_dates(cls, value: str | date | None) -> date | None:
        if value in (None, "") or isinstance(value, date):
            return value

        text = str(value).strip()
        formats = (
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%Y-%m-%d",
            "%d.%m.%Y",
            "%d %b %Y",
            "%d %B %Y",
        )
        for fmt in formats:
            try:
                return (
                    date.fromisoformat(text)
                    if fmt == "%Y-%m-%d"
                    else datetime.strptime(text, fmt).date()
                )
            except ValueError:
                continue

        raise ValueError(f"Unsupported date format: {text}")


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Upload a PDF to POST /extract-invoice"}


@app.post("/extract-invoice", response_model=InvoiceExtraction, response_model_by_alias=True)
async def extract_invoice(
    file: Annotated[UploadFile, File(description="PDF document to scan")]
) -> InvoiceExtraction:
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        shutil.copyfileobj(file.file, temp_pdf)
        pdf_path = Path(temp_pdf.name)

    try:
        text = extract_text_from_pdf(pdf_path)
    finally:
        pdf_path.unlink(missing_ok=True)

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "No text could be extracted. Install OCR dependencies "
                "(pymupdf, pytesseract, pillow, and the Tesseract binary) for scanned PDFs."
            ),
        )

    return extract_structured_fields(text)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract selectable PDF text first; use open-source OCR as a fallback."""
    selectable_text = extract_selectable_text(pdf_path)
    if selectable_text.strip():
        return selectable_text
    return extract_text_with_ocr(pdf_path)


def extract_selectable_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    try:
        reader = PdfReader(str(pdf_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def extract_text_with_ocr(pdf_path: Path) -> str:
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""

    pages_text: list[str] = []
    try:
        document = fitz.open(pdf_path)
        for page in document:
            pixmap = page.get_pixmap(dpi=220)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            pages_text.append(pytesseract.image_to_string(image))
    except Exception:
        return "\n".join(pages_text)

    return "\n".join(pages_text)


def extract_structured_fields(text: str) -> InvoiceExtraction:
    normalized = normalize_ocr_text(text)
    values = {
        "cr_no": find_value(normalized, [r"CR\s*No\.?", r"Commercial\s*Registration\s*No\.?"]),
        "registered_in_the_grade": find_value(
            normalized, [r"Registered\s*in\s*the\s*grade", r"Grade"]
        ),
        "occi_no": find_value(normalized, [r"OCCI\s*No\.?", r"Chamber\s*No\.?"]),
        "date_of_issue": find_date_value(normalized, [r"Date\s*of\s*issue", r"Issue\s*Date"]),
        "date_of_expiry": find_date_value(normalized, [r"Date\s*of\s*expiry", r"Expiry\s*Date"]),
        "head_office": find_value(normalized, [r"Head\s*Office", r"Head\s*Office\s*Address"]),
    }
    return InvoiceExtraction.model_validate(values)


def normalize_ocr_text(text: str) -> str:
    replacements = {
        "\r": "\n",
        "|": " ",
        "\uff1a": ":",
        "\u2013": "-",
        "\u2014": "-",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"[ \t]+", " ", text)


def find_value(text: str, labels: list[str]) -> str | None:
    for label in labels:
        patterns = [
            rf"{label}\s*[:\-]\s*(?P<value>[^\n]+)",
            rf"{label}\s+(?P<value>[^\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return cleanup_extracted_value(match.group("value"))
    return None


def find_date_value(text: str, labels: list[str]) -> str | None:
    date_pattern = (
        r"(?P<value>\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|"
        r"\d{4}-\d{1,2}-\d{1,2}|"
        r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})"
    )
    for label in labels:
        match = re.search(rf"{label}\s*[:\-]?\s*{date_pattern}", text, flags=re.IGNORECASE)
        if match:
            return match.group("value")
    return None


def cleanup_extracted_value(value: str) -> str:
    value = value.split("  ")[0].strip()
    return re.sub(
        r"\s+(CR\s*No|Registered\s*in\s*the\s*grade|OCCI\s*No|Date\s*of\s*issue|Date\s*of\s*expiry|Head\s*Office)\b.*$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip(" :-")
