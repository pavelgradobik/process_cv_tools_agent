import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from bs4 import BeautifulSoup

from backend.config import (
    DEFAULT_CSV_PATH,
    ROLE_SYNONYMS,
    SKILL_CATEGORIES,
)

logger = logging.getLogger(__name__)


class ResumeProcessor:
    PATTERNS = {
        "email": re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            re.IGNORECASE
        ),
        "phone": re.compile(
            r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,5}[-\s\.]?[0-9]{1,5}'
        ),
        "years": re.compile(
            r'(?<!\d)(\d{1,2}(?:\.\d+)?)\s*(?:\+?\s*)?(?:years?|yrs?)',
            re.IGNORECASE
        ),
        "linkedin": re.compile(
            r'(?:linkedin\.com/in/|linkedin:?\s*)([a-zA-Z0-9-]+)',
            re.IGNORECASE
        ),
        "github": re.compile(
            r'(?:github\.com/|github:?\s*)([a-zA-Z0-9-]+)',
            re.IGNORECASE
        ),
    }

    def __init__(self):
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "warnings": [],
        }

    def load_file(
            self,
            file_path: str,
            file_type: Optional[str] = None
    ) -> pd.DataFrame:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_type is None:
            file_type = "excel" if path.suffix in [".xlsx", ".xls"] else "csv"

        logger.info(f"Loading {file_type} file: {path.name}")

        if file_type == "excel":
            return self._load_excel(path)
        else:
            return self._load_csv(path)

    def _load_csv(self, path: Path) -> pd.DataFrame:
        encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1", "iso-8859-1"]

        for encoding in encodings:
            try:
                df = pd.read_csv(
                    path,
                    encoding=encoding,
                    on_bad_lines="skip",
                    engine="python",
                )
                logger.info(f"Successfully loaded CSV with {encoding} encoding")
                return df
            except Exception as e:
                continue

        raise ValueError(f"Failed to load CSV with any encoding: {path}")

    def _load_excel(self, path: Path) -> pd.DataFrame:
        try:
            df = pd.read_excel(path, engine="openpyxl")
            logger.info(f"Successfully loaded Excel file")
            return df
        except Exception as e:
            raise ValueError(f"Failed to load Excel file: {e}")

    def process_resumes(
            self,
            file_path: Optional[str] = None,
            limit: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        file_path = file_path or DEFAULT_CSV_PATH

        df = self.load_file(file_path)

        df = self._normalize_columns(df)

        if not self._validate_columns(df):
            return [], {
                "error": "Missing required columns (ID and Resume content)",
                "columns_found": list(df.columns)
            }

        resumes = []
        total_rows = len(df) if limit is None else min(limit, len(df))

        for idx, row in df.head(total_rows).iterrows():
            try:
                resume = self._process_single_resume(row, idx)
                if resume:
                    resumes.append(resume)
                    self.stats["successful"] += 1
            except Exception as e:
                logger.warning(f"Failed to process row {idx}: {e}")
                self.stats["failed"] += 1

        self.stats["total_processed"] = total_rows

        return resumes, self.stats

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]

        # Map common variations
        column_mapping = {
            "candidate_id": "id",
            "resumehtml": "resume_html",
            "resumestr": "resume_str",
            "resume_text": "resume_str",
            "job_title": "title",
            "position": "title",
            "skillset": "skills",
            "tech_stack": "skills",
        }

        df = df.rename(columns=column_mapping)

        return df

    def _validate_columns(self, df: pd.DataFrame) -> bool:
        required = ["id"]
        content_columns = ["resume_html", "resume_str", "resume_text"]

        has_id = "id" in df.columns
        has_content = any(col in df.columns for col in content_columns)

        return has_id and has_content

    def _process_single_resume(
            self,
            row: pd.Series,
            idx: int
    ) -> Optional[Dict[str, Any]]:
        resume_id = str(row.get("id", f"resume_{idx}"))

        text = self._extract_text(row)

        if not text or len(text.strip()) < 50:
            logger.warning(f"Skipping resume {resume_id}: insufficient content")
            return None

        metadata = self._extract_metadata(row, text)

        return {
            "id": resume_id,
            "text": text,
            "metadata": metadata,
        }

    def _extract_text(self, row: pd.Series) -> str:
        text_parts = []

        if pd.notna(row.get("resume_str")):
            text_parts.append(str(row["resume_str"]))

        if pd.notna(row.get("resume_html")):
            html_text = self._parse_html(row["resume_html"])
            if html_text:
                text_parts.append(html_text)

        for field in ["title", "skills", "summary", "experience"]:
            if field in row and pd.notna(row[field]):
                text_parts.append(str(row[field]))

        combined_text = "\n".join(text_parts)
        return self._clean_text(combined_text)

    def _parse_html(self, html_content: str) -> str:
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            for element in soup(["script", "style", "meta", "link"]):
                element.decompose()

            text = soup.get_text(separator=" ", strip=True)

            text = " ".join(text.split())

            return text
        except Exception as e:
            logger.warning(f"HTML parsing failed: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)

        text = re.sub(r'[^\w\s\-\.\,\;\:\@\#\+\/]', ' ', text)

        text = re.sub(r'\s+([,.:;])', r'\1', text)

        return text.strip()

    def _extract_metadata(
            self,
            row: pd.Series,
            text: str
    ) -> Dict[str, Any]:
        metadata = {}

        metadata["title"] = self._extract_title(row, text)
        metadata["skills"] = self._extract_skills(row, text)
        metadata["category"] = row.get("category", "Unknown")

        metadata["years_experience"] = self._extract_years_experience(text)
        metadata["email"] = self._extract_email(text)
        metadata["phone"] = self._extract_phone(text)
        metadata["linkedin"] = self._extract_linkedin(text)
        metadata["github"] = self._extract_github(text)
        metadata["roles"] = self._extract_roles(text)

        metadata["education"] = self._extract_education(text)
        metadata["certifications"] = self._extract_certifications(text)

        return metadata

    def _extract_title(self, row: pd.Series, text: str) -> str:
        if pd.notna(row.get("title")) and row["title"].strip():
            return row["title"].strip()

        lines = text.split("\n")
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if 10 < len(line) < 100:
                if any(role in line.lower() for role in ["engineer", "developer", "manager", "analyst", "designer"]):
                    return line
        return "Professional"

    def _extract_skills(self, row: pd.Series, text: str) -> str:
        skills_text = ""

        if pd.notna(row.get("skills")):
            skills_text = str(row["skills"])

        text_lower = text.lower()
        found_skills = set()

        for category, skills in SKILL_CATEGORIES.items():
            for skill in skills:
                if skill.lower() in text_lower:
                    found_skills.add(skill)

        if found_skills:
            additional = ", ".join(sorted(found_skills))
            skills_text = f"{skills_text}, {additional}" if skills_text else additional

        return skills_text[:500]

    def _extract_years_experience(self, text: str) -> float:
        matches = self.PATTERNS["years"].findall(text)

        if matches:
            years = [float(m) for m in matches]
            return max(years)

        year_range_pattern = re.compile(r'\b(19|20)\d{2}\s*[-–—]\s*(19|20)\d{2}\b')
        range_matches = year_range_pattern.findall(text)

        if range_matches:
            max_diff = 0
            for start, end in range_matches:
                diff = int(end) - int(start)
                if diff > max_diff:
                    max_diff = diff
            return float(max_diff) if max_diff > 0 else 0.0

        return 0.0

    def _extract_email(self, text: str) -> str:
        match = self.PATTERNS["email"].search(text)
        return match.group(0) if match else ""

    def _extract_phone(self, text: str) -> str:
        match = self.PATTERNS["phone"].search(text)
        return match.group(0) if match else ""

    def _extract_linkedin(self, text: str) -> str:
        match = self.PATTERNS["linkedin"].search(text)
        return f"linkedin.com/in/{match.group(1)}" if match else ""

    def _extract_github(self, text: str) -> str:
        match = self.PATTERNS["github"].search(text)
        return f"github.com/{match.group(1)}" if match else ""

    def _extract_roles(self, text: str) -> List[str]:
        text_lower = text.lower()
        found_roles = set()

        for base_role, synonyms in ROLE_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in text_lower:
                    found_roles.add(base_role)
                    break

        return sorted(list(found_roles))

    def _extract_education(self, text: str) -> List[str]:
        education = []

        # Common degree patterns
        degree_patterns = [
            r'\b(?:B\.?S\.?|Bachelor)\s+(?:of\s+)?(?:Science|Arts|Engineering)\b',
            r'\b(?:M\.?S\.?|Master)\s+(?:of\s+)?(?:Science|Arts|Engineering|Business)\b',
            r'\b(?:Ph\.?D\.?|Doctorate)\b',
            r'\b(?:MBA|MCA|BCA|BE|BTech|MTech)\b',
        ]

        for pattern in degree_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            education.extend(matches)

        return education[:5]  # Limit to 5 degrees

    def _extract_certifications(self, text: str) -> List[str]:
        certifications = []

        cert_patterns = [
            r'\b(?:AWS|Azure|GCP)\s+(?:Certified|Associate|Professional)\b',
            r'\b(?:PMP|ITIL|Scrum Master|CISSP|CCNA|CCNP)\b',
            r'\b(?:Oracle|Microsoft|Google|Cisco)\s+Certified\b',
        ]

        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            certifications.extend(matches)

        return certifications[:10]


def load_resumes(
        file_path: Optional[str] = None,
        limit: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    processor = ResumeProcessor()
    return processor.process_resumes(file_path, limit)


def process_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    processor = ResumeProcessor()
    processor.df = df
    processor._normalize_columns(df)

    resumes = []
    for idx, row in df.iterrows():
        resume = processor._process_single_resume(row, idx)
        if resume:
            resumes.append(resume)

    return resumes
