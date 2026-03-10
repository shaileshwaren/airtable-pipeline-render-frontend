#!/usr/bin/env python3
"""utils.py

Shared utility functions for the recruitment pipeline.
Includes resume extraction, hashing, and filename sanitization.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
import docx


# =========================
# Resume Text Extraction
# =========================

def extract_text_from_pdf(path: Path) -> str:
    """Extract text from PDF file.
    
    Args:
        path: Path to PDF file
        
    Returns:
        Extracted text as string, or empty string on failure
    """
    try:
        with open(str(path), 'rb') as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        print(f"[WARN] PDF extraction failed for {path.name}: {e}")
        return ""


def extract_text_from_docx(path: Path) -> str:
    """Extract text from DOCX file.
    
    Args:
        path: Path to DOCX file
        
    Returns:
        Extracted text as string, or empty string on failure
    """
    try:
        doc = docx.Document(str(path))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        print(f"[WARN] DOCX extraction failed for {path.name}: {e}")
        return ""


def extract_resume_text(path: Path) -> str:
    """Extract text from resume file (PDF or DOCX).
    
    Automatically detects file type based on extension and uses
    appropriate extraction method.
    
    Args:
        path: Path to resume file
        
    Returns:
        Extracted text as string, or empty string if unsupported/failed
    """
    ext = path.suffix.lower()
    
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(path)
    else:
        print(f"[WARN] Unsupported file type: {path.name}")
        return ""


# =========================
# Hashing
# =========================

def sha256_text(s: str) -> str:
    """Generate SHA256 hash of text.
    
    Args:
        s: Text to hash
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


# =========================
# Filename Sanitization
# =========================

def safe_filename(s: str, max_len: int = 180) -> str:
    """Sanitize string for use as filename.
    
    Removes special characters, normalizes whitespace, and truncates to max length.
    
    Args:
        s: String to sanitize
        max_len: Maximum length (default: 180)
        
    Returns:
        Safe filename string
    """
    s = (s or "").strip()
    # Replace unsafe characters with underscore
    s = re.sub(r"[^\w\-. ]+", "_", s)
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len] if s else "export"


# =========================
# Text Clipping
# =========================

def clip(s: str, n: int) -> str:
    """Truncate string to maximum length.
    
    Args:
        s: String to clip
        n: Maximum length
        
    Returns:
        Clipped string
    """
    s = s or ""
    return s[:n]
