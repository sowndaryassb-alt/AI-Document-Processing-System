import json
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Annotated
import ollama
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

app = FastAPI(title="AI Document Processing System")


OLLAMA_MODEL = "llama3.2"


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
    values = extract_fields_with_ollama(normalized)
    try:
        return InvoiceExtraction.model_validate(values)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Ollama returned data that did not match the expected schema: {exc}",
        ) from exc


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
    return "\n".join(" ".join(line.split()) for line in text.splitlines())


def extract_fields_with_ollama(text: str) -> dict[str, str | None]:
    prompt = build_extraction_prompt(text)
    try:
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            format="json",
            options={"temperature": 0},
        )
    except ollama.ResponseError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama model error: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not connect to Ollama. Start Ollama locally and make sure "
                f"model '{OLLAMA_MODEL}' is available."
            ),
        ) from exc

    ollama_response = response.get("response", "")
    return parse_ollama_json(ollama_response)


def build_extraction_prompt(text: str) -> str:
    return (
        "Extract document fields from the text below. "
        "Return only one valid JSON object. Do not include markdown or explanation. "
        "Use null when a field is missing. "
        "Return dates as YYYY-MM-DD when possible, otherwise keep the date exactly as found. "
        "The JSON keys must be exactly: "
        '"CR No", "Registered in the gr'
        'ade", "OCCI No", '
        '"Date of issue", "Date of expiry", "Head Office".\n\n'
        "Document text:\n"
        f"{text}"
    )


def parse_ollama_json(raw_response: str) -> dict[str, str | None]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise HTTPException(
                status_code=422,
                detail="Ollama did not return a JSON object.",
            )
        parsed = json.loads(raw_response[start : end + 1])

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail="Ollama JSON output must be an object.")

    return {
        "CR No": normalize_optional_value(parsed.get("CR No")),
        "Registered in the grade": normalize_optional_value(
            parsed.get("Registered in the grade")
        ),
        "OCCI No": normalize_optional_value(parsed.get("OCCI No")),
        "Date of issue": normalize_optional_value(parsed.get("Date of issue")),
        "Date of expiry": normalize_optional_value(parsed.get("Date of expiry")),
        "Head Office": normalize_optional_value(parsed.get("Head Office")),
    }


def normalize_optional_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "not found", "n/a"}:
        return None
    return text
