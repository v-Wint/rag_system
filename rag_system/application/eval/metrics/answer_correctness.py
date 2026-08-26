from pydantic import BaseModel, Field
from ragas.metrics.result import MetricResult
from ragas.metrics.collections.base import BaseMetric
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.prompt import Prompt


class EntailmentInput(BaseModel):
    response: str
    reference_facts: list[str]


class IndividualClaimVerdict(BaseModel):
    claim: str = Field(description="The exact claim being evaluated")
    reason: str = Field(description="A brief logical explanation for the given verdict: 1 sentence")
    verdict: str = Field(description="Must be exactly one of: 'entailed', 'neutral', or 'contradicted'")


class EntailmentOutput(BaseModel):
    evaluations: list[IndividualClaimVerdict] = Field(description="List of factual verification verdicts")


class EntailmentPrompt(Prompt):
    def __init__(self):
        super().__init__(
            instruction=(
                "You are an expert logical entailment and fact-verification judge.\n"
                "Evaluate whether each given claim is logically entailed, neutral, "
                "or contradicted by the provided reference text (response).\n"
                "Do not penalize minor phrasing differences unless they explicitly conflict with the facts."
            )
        )

    def to_string(self, data: EntailmentInput) -> str:
        claims_str = "\n".join(f"- {claim}" for claim in data.reference_facts)
        return (
            f"{self.instruction}\n\n"
            f"REFERENCE TEXT (Premise):\n{data.response}\n\n"
            f"CLAIMS TO EVALUATE:\n{claims_str}"
        )

class CustomAnswerCorrectness(BaseMetric):
    """
    Calculate answer correctness by scoring each atomic fact against the llm output
    """
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "custom_answer_correctness",
        **kwargs,
    ):
        self.llm = llm
        self.entailment_prompt = EntailmentPrompt()
        
        super().__init__(name=name, **kwargs)

    async def ascore( # type: ignore
        self, response: str, reference_facts: list[str]
    ) -> MetricResult:
        """
        Calculate correctness by evaluating each individual claim.

        Args:
            response: The response to evaluate for answer correctness
            reference_facts: The facts against which to evaluate

        Returns:
            MetricResult with answer correctness score (0.0-1.0, higher is better)
        """
        if not response:
            raise ValueError("response is missing from input sample.")
        if not reference_facts:
            raise ValueError("reference_facts list is missing or empty.")

        input_data = EntailmentInput(response=response, reference_facts=reference_facts)
        prompt_str = self.entailment_prompt.to_string(input_data)
        
        structured_result: EntailmentOutput = await self.llm.agenerate(
            prompt_str, 
            EntailmentOutput,
        )

        score = self._compute_score(structured_result)

        return MetricResult(
            value=float(score),
            reason=self._compute_reason(structured_result)
        )

    def _compute_score(self, output: EntailmentOutput) -> float:
        if not output.evaluations:
            return 0.0
            
        entailed_count = sum(
            1 for item in output.evaluations if item.verdict.lower() == "entailed"
        )
        return entailed_count / len(output.evaluations)

    def _compute_reason(self, output: EntailmentOutput) -> str:
        return str([item.verdict.lower() for item in output.evaluations])
