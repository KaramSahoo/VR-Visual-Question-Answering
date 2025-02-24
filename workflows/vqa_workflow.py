from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from utils.schemas import Decision
from agents.query_evaluator import QueryEvaluator
from utils.logger import logger
# Graph state
class VQAState(TypedDict):
    question: str  # User Question
    answer: str  # LLM Answer
    tool_decision: str  # Tools to use
    ocr_text: str  # OCR text from the image
    od_results: list[str]  # Object detection results
    img_path: str  # Image path
    tool_decision_reasoning: str  # Reasoning for tool decision

class VQAWorkflow:
    def __init__(self, llm):
        """
        Initializes the Workflow with the LLM model and instructions.

        Args:
            llm: The language model used for structured output.
        """
        self.llm = llm
        self.query_evaluator_llm = llm.with_structured_output(Decision)  # Augment LLM with Team schema
        
        self.query_evaluator_agent = QueryEvaluator(self.query_evaluator_llm)

        # Initialize the workflow state
        self.state: VQAState = {
            "question": "",
            "answer": "",
            "od_results": [],
            "tool_decision": "",
            "ocr_text": "",
            "img_path": ""
        }

        self.orchestrator_worker_builder = StateGraph(VQAState)
    
    def query_evaluator(self, state: VQAState):
        result = self.query_evaluator_agent.evaluate_query(state["question"])
        self.state.update(result)
        return result

    def build_workflow(self):
        """
        Constructs the workflow by adding nodes and edges.
        """
        logger.info("Building workflow...")
        # self.orchestrator_worker_builder.add_node("team_generator", self.team_generator)
        # self.orchestrator_worker_builder.add_node("story_generator", self.story_generator)
        # self.orchestrator_worker_builder.add_node("story_evaluator", self.story_evaluator)
        self.orchestrator_worker_builder.add_node("query_evaluator", self.query_evaluator)

        # Add edges to connect nodes
        self.orchestrator_worker_builder.add_edge(START, "query_evaluator")
        self.orchestrator_worker_builder.add_edge("query_evaluator", END)
        # self.orchestrator_worker_builder.add_edge("team_generator", "story_generator")
        # self.orchestrator_worker_builder.add_edge("story_generator", "story_evaluator")
        # self.orchestrator_worker_builder.add_conditional_edges(
        #     "story_evaluator", self.route_story_feedback, {
        #         "Accepted": END,
        #         "Rejected + Feedback": "story_generator"
        #     }
        # )

    def compile_workflow(self):
        """
        Compiles the workflow for execution.
        """
        logger.info("Compiling workflow...")
        self.orchestrator_worker = self.orchestrator_worker_builder.compile()

    def invoke_workflow(self, question: str):
        """
        Runs the workflow with the initialized mission.

        Returns:
            dict: The final state after execution.
        """
        logger.info(f"Invoking workflow for question: [yellow]{question}[/yellow]")
        self.state["question"] = question
        return self.orchestrator_worker.invoke({"question": self.state["question"]})
