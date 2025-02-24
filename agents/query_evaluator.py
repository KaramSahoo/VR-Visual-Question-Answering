from langchain.schema import SystemMessage, HumanMessage
from utils.logger import logger
from prompts.system_message import QUERY_EVALUATOR_PROMPT_SYSTEM

class QueryEvaluator:
    def __init__(self, llm):
        """Initialize the user query evaluator agent."""
        self.llm = llm

    def evaluate_query(self, question: str):
        """Evaluates the given user query and provides the tools to call."""
        logger.info(f"Deciding tool calls for user query...")

        decision = self.llm.invoke([
            SystemMessage(content=QUERY_EVALUATOR_PROMPT_SYSTEM),
            HumanMessage(content=f"User query:\n\n{question}")
        ])

        logger.success(f"Query Evaluation Complete! Tools to use: \"{decision.decision}\"")  # Logs first 100 chars

        return {"tool_decision": decision.decision, 
                "tool_decision_reasoning": decision.reasoning,                }
