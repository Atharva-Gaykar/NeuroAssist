# app/utils/presidio_worker.py
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine

# Instantiate directly at the module level for clean, thread-safe reading
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

username_pattern = Pattern(
    name="username_regex",
    regex=r"@\w+|\b(?:user|username):\s*\w+\b",
    score=0.85
)
username_recognizer = PatternRecognizer(
    supported_entity="USERNAME", 
    patterns=[username_pattern]
)
analyzer.registry.add_recognizer(username_recognizer)

def execute_sanitize(text: str) -> str:
    target_entities = ["PERSON", "USERNAME", "EMAIL_ADDRESS", "PHONE_NUMBER"]
    results = analyzer.analyze(text=text, language="en", entities=target_entities)
    anonymized_result = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized_result.text
