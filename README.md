# AI-Document-Processing-System
FastAPI based AI document processing API that extracts invoice fields from PDFs using PDF text extraction, validates the result with Pydantic, and returns structured JSON.


# AI Document Processing System

This project is a FastAPI-based API that extracts structured information from uploaded PDF documents. The API accepts a PDF file, extracts text using PDF parsing or OCR, maps the extracted text into required fields, validates the output with Pydantic, and returns clean JSON.

## API Endpoint

### POST `/extract-invoice`

Upload a PDF document using multipart form-data with the field name `file`.

The API returns:

```json
{
  "CR No": "1234567",
  "Registered in the grade": "Excellent",
  "OCCI No": "OCCI-98765",
  "Date of issue": "2026-01-15",
  "Date of expiry": "2027-01-14",
  "Head Office": "Muscat, Sultanate of Oman"
}
