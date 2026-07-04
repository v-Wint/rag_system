from ..state import RAGState, QuestionType

def route_after_classify(state: RAGState) -> str:
    match state.question_type:
        case QuestionType.GENERAL:
            return "final_prompt"
        case _:
            return "retrieve"
