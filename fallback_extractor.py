# fallback_extractor.py
"""
Fallback resume extractor that works without AI
"""
import re
from typing import Dict, Any, List

def extract_email(text: str) -> str:
    """Extract email from text"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else ""

def extract_phone(text: str) -> str:
    """Extract phone number from text"""
    phone_patterns = [
        r'\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        r'\(\d{3}\)\s?\d{3}[-.]?\d{4}'
    ]
    
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        if phones:
            if isinstance(phones[0], tuple):
                return ''.join(phones[0])
            return phones[0]
    return ""

def extract_name(text: str) -> str:
    """Extract name from text (first few words, usually)"""
    lines = text.strip().split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if line and len(line.split()) <= 4 and len(line) > 2:
            # Skip lines that look like emails, phones, or addresses
            if '@' not in line and not re.search(r'\d{3}', line) and 'http' not in line.lower():
                return line
    return ""

def extract_skills(text: str) -> List[str]:
    """Extract common skills from text"""
    common_skills = [
        'Python', 'JavaScript', 'Java', 'C++', 'React', 'Node.js', 'HTML', 'CSS',
        'SQL', 'MongoDB', 'PostgreSQL', 'Git', 'Docker', 'AWS', 'Azure',
        'Machine Learning', 'Data Analysis', 'Project Management', 'Leadership',
        'Communication', 'Problem Solving', 'Teamwork', 'Microsoft Office',
        'Excel', 'PowerPoint', 'Photoshop', 'Figma', 'UI/UX', 'Marketing',
        'Sales', 'Customer Service', 'Research', 'Writing', 'Editing'
    ]
    
    found_skills = []
    text_lower = text.lower()
    
    for skill in common_skills:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    
    return found_skills[:10]  # Limit to 10 skills

def extract_education(text: str) -> List[str]:
    """Extract education information"""
    education_keywords = [
        'bachelor', 'master', 'phd', 'degree', 'university', 'college',
        'bs', 'ms', 'ba', 'ma', 'mba', 'bsc', 'msc', 'diploma',
        'certification', 'certificate'
    ]
    
    education = []
    lines = text.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        for keyword in education_keywords:
            if keyword in line_lower and len(line.strip()) > 5:
                education.append(line.strip())
                break
    
    return education[:5]  # Limit to 5 entries

def fallback_extract_resume_data(text: str) -> Dict[str, Any]:
    """
    Fallback resume extraction without AI
    """
    return {
        "name": extract_name(text),
        "tag": "Professional",  # Default tag
        "email": extract_email(text),
        "location": "",  # Hard to extract without AI
        "number": extract_phone(text),
        "summary": "Experienced professional seeking new opportunities to contribute skills and expertise.",
        "websites": [],
        "skills": extract_skills(text),
        "education": extract_education(text),
        "experience": [],  # Complex to extract without AI
        "student": [],
        "courses": [],
        "internships": [],
        "extracurriculars": [],
        "hobbies": [],
        "references": [],
        "languages": ["English"]  # Default
    }