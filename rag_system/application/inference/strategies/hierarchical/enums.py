from enum import Enum

class QuestionType(str, Enum):
    FACT = "fact-question"
    SCHEMA = "schema-question"
    GENERAL = "general-question"
