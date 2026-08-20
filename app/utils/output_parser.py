import re
from langchain_core.output_parsers import BaseOutputParser

class CleanQueryParser(BaseOutputParser[str]):
    def parse(self, text: str) -> str:
        # Erase everything inside <think>...</think> blocks safely
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned.strip()