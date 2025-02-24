from langchain.schema import SystemMessage, HumanMessage
from utils.logger import logger
from prompts.system_message import STORY_EVALUATOR_PROMPT_SYSTEM

class StoryEvaluator:
    def __init__(self, llm):
        """Initialize the story evaluator agent."""
        self.llm = llm

    def evaluate_story(self, story: str):
        """Evaluates the given story and provides feedback."""
        logger.info(f"Evaluating story quality...")

        feedback = self.llm.invoke([
            SystemMessage(content=STORY_EVALUATOR_PROMPT_SYSTEM),
            HumanMessage(content=f"Here is the story:\n\n{story}")
        ])

        logger.success(f"Story Evaluation Complete! Feedback: \"{feedback.grade}\"")  # Logs first 100 chars

        return {"story_feedback": feedback.feedback, "feedback_grade": feedback.grade}
