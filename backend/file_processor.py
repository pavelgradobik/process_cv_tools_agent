import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

ROLE_SYNONYMS = {
    "software engineer": [
        "software developer", "swe", "software engineer", "programmer",
        "developer", "coder", "software architect", "software dev"
    ],
    "data scientist": [
        "data scientist", "ml engineer", "machine learning engineer",
        "ai engineer", "data analyst", "research scientist", "ml specialist"
    ],
    "product manager": [
        "product manager", "pm", "product owner", "po",
        "product lead", "program manager", "product management"
    ],
    "designer": [
        "designer", "ux designer", "ui designer", "product designer",
        "ux/ui designer", "user experience designer", "visual designer",
        "ui/ux designer", "graphic designer", "web designer"
    ],
    "hr": [
        "hr", "human resources", "recruiter", "talent acquisition",
        "people operations", "hr manager", "talent partner", "hr specialist",
        "hr generalist", "hrbp", "hr business partner", "people ops"
    ],
    "qa": [
        "qa", "quality assurance", "test engineer", "sdet",
        "qa engineer", "tester", "automation engineer", "qa analyst",
        "quality engineer", "test automation"
    ],
    "devops": [
        "devops", "site reliability engineer", "sre", "cloud engineer",
        "infrastructure engineer", "platform engineer", "devops engineer",
        "systems engineer", "cloud architect"
    ],
    "frontend": [
        "frontend", "front-end", "frontend developer", "ui developer",
        "react developer", "angular developer", "vue developer",
        "frontend engineer", "web developer", "javascript developer"
    ],
    "backend": [
        "backend", "back-end", "backend developer", "server developer",
        "api developer", "microservices developer", "backend engineer",
        "server-side developer"
    ],
    "fullstack": [
        "fullstack", "full-stack", "full stack developer",
        "generalist engineer", "web developer", "full-stack engineer"
    ],
    "data engineer": [
        "data engineer", "etl developer", "data pipeline engineer",
        "big data engineer", "data architect", "analytics engineer"
    ],
    "business analyst": [
        "business analyst", "ba", "systems analyst", "product analyst",
        "business systems analyst", "requirements analyst"
    ],
    "project manager": [
        "project manager", "pm", "technical project manager",
        "program manager", "delivery manager", "scrum master",
        "agile coach", "project lead"
    ],
    "sales": [
        "sales", "sales executive", "account executive", "sales manager",
        "business development", "sales representative", "account manager"
    ],
    "marketing": [
        "marketing", "marketing manager", "digital marketing",
        "growth manager", "marketing specialist", "brand manager",
        "content marketing", "marketing executive"
    ],
}

SKILL_CATEGORIES = {
    "programming": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go",
        "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab"
    ],
    "frontend": [
        "react", "angular", "vue", "html", "css", "sass", "webpack",
        "redux", "next.js", "nuxt.js", "svelte", "tailwind", "bootstrap"
    ],
    "backend": [
        "django", "flask", "fastapi", "spring", "spring boot", "express",
        "node.js", "rails", ".net", "laravel", "nestjs"
    ],
    "database": [
        "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "cassandra", "dynamodb", "oracle", "sql server", "neo4j"
    ],
    "cloud": [
        "aws", "gcp", "azure", "kubernetes", "docker", "terraform",
        "ansible", "jenkins", "ci/cd", "cloudformation", "serverless"
    ],
    "data": [
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
        "spark", "hadoop", "tableau", "power bi", "looker", "airflow"
    ],
    "mobile": [
        "ios", "android", "react native", "flutter", "swift", "kotlin",
        "xamarin", "ionic"
    ],
    "testing": [
        "selenium", "jest", "pytest", "junit", "cypress", "postman",
        "jmeter", "testng", "mocha", "jasmine"
    ],
    "soft_skills": [
        "leadership", "communication", "teamwork", "problem solving",
        "analytical", "project management", "agile", "scrum"
    ],
}


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
        logger.info("Initialized ResumeProcessor")

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
        if not file_path:
            file_path = "data/Resume.xlsx"

        df = self.load_file(file_path)

        df = self._normalize_columns(df)

        if not self._validate_columns(df):
            return [], {
                "error": "Missing required columns (ID and Resume content)",
                "columns_found": list(df.columns)
            }

        resumes = []
        total_rows = len(df) if limit is None else min(limit, len(df))

        logger.info(f"Processing {total_rows} resumes...")

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

        logger.info(f"Processed {len(resumes)} resumes successfully")

        return resumes, self.stats

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]

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
        content_columns = ["resume_html", "resume_str"]

        has_id = "id" in df.columns
        has_content = any(col in df.columns for col in content_columns)

        if not has_id:
            logger.warning(f"Missing ID column. Available columns: {list(df.columns)}")
        if not has_content:
            logger.warning(f"Missing content columns. Need one of: {content_columns}")

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

        if "category" in row and pd.notna(row["category"]):
            metadata["category"] = str(row["category"])

        return {
            "id": resume_id,
            "text": text,
            "metadata": metadata,
        }

    def _extract_text(self, row: pd.Series) -> str:
        text_parts = []

        if "resume_str" in row.index and pd.notna(row["resume_str"]):
            text_parts.append(str(row["resume_str"]))

        if not text_parts and "resume_html" in row.index and pd.notna(row["resume_html"]):
            html_text = self._parse_html(row["resume_html"])
            if html_text:
                text_parts.append(html_text)

        for field in ["title", "skills", "summary", "experience"]:
            if field in row.index and pd.notna(row[field]):
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

        text = re.sub(r'[^\w\s\-\.\,\;\:\@\#\+\/\(\)]', ' ', text)

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
        if "title" in row.index and pd.notna(row["title"]) and str(row["title"]).strip():
            return str(row["title"]).strip()

        text_lower = text.lower()
        for role, synonyms in ROLE_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in text_lower:
                    return role.title()

        title_patterns = [
            r'\b(Senior|Junior|Lead|Principal|Staff)\s+([\w\s]+(?:Engineer|Developer|Manager|Analyst|Designer))\b',
            r'\b([\w\s]+(?:Engineer|Developer|Manager|Analyst|Designer|Architect|Consultant))\b',
        ]

        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip().title()

        # Fallback
        return "Professional"

    def _extract_skills(self, row: pd.Series, text: str) -> str:
        skills_text = ""

        if "skills" in row.index and pd.notna(row["skills"]):
            skills_text = str(row["skills"])

        text_lower = text.lower()
        found_skills = set()

        for category, skills in SKILL_CATEGORIES.items():
            for skill in skills:
                # Use word boundaries for more accurate matching
                if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                    found_skills.add(skill)

        if found_skills:
            additional = ", ".join(sorted(found_skills))
            skills_text = f"{skills_text}, {additional}" if skills_text else additional

        return skills_text[:500]  # Limit length

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

        degree_patterns = [
            r'\b(?:B\.?S\.?|Bachelor)\s+(?:of\s+)?(?:Science|Arts|Engineering|Technology)\b',
            r'\b(?:M\.?S\.?|Master)\s+(?:of\s+)?(?:Science|Arts|Engineering|Business|Technology)\b',
            r'\b(?:Ph\.?D\.?|Doctorate|Doctor of Philosophy)\b',
            r'\b(?:MBA|MCA|BCA|BE|BTech|MTech|BBA|BA|MA)\b',
        ]

        for pattern in degree_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            education.extend(matches)

        education = list(set(education))[:5]

        return education

    def _extract_certifications(self, text: str) -> List[str]:
        certifications = []

        cert_patterns = [
            r'\b(?:AWS|Azure|GCP|Google Cloud)\s+(?:Certified|Associate|Professional|Expert)\b',
            r'\b(?:PMP|ITIL|Scrum Master|CISSP|CCNA|CCNP|CEH|CompTIA)\b',
            r'\b(?:Oracle|Microsoft|Google|Cisco|VMware)\s+Certified\b',
            r'\b(?:Certified\s+[\w\s]+(?:Professional|Expert|Associate|Specialist))\b',
        ]

        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            certifications.extend(matches)

        certifications = list(set(certifications))[:10]

        return certifications


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


ResumeLoader = ResumeProcessor
