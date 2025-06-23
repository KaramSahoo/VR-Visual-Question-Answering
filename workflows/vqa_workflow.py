from typing import TypedDict, Annotated, List
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from utils.schemas import Decision
from agents.query_evaluator import QueryEvaluator
from agents.florence_agent import FlorenceAgent
from agents.write_answer import AnswerWriter
from utils.logger import logger
from langgraph.graph.message import add_messages
from gtts import gTTS

# Graph state
class VQAState(TypedDict):
    messages: Annotated[list[HumanMessage | AIMessage], add_messages]
    question: str  # User Question
    answer: str  # LLM Answer
    tool_decision: str  # Tools to use
    ocr_text: list[str]  # OCR text from the image
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
        self.ocr_agent = FlorenceAgent()
        self.answer_writer_agent = AnswerWriter(self.llm)

        # Initialize the workflow state
        self.state: VQAState = {
            "messages": [],  # List of messages
            "question": "",
            "answer": "",
            "od_results": [],
            "tool_decision": "",
            "ocr_text": [],
            "img_path": "",
            "tool_decision_reasoning": "",
        }

        self.orchestrator_worker_builder = StateGraph(VQAState)
    
    def query_evaluator(self, state: VQAState):
        result = self.query_evaluator_agent.evaluate_query(state["question"])
        self.state.update(result)
        return result
    
    def ocr_node(self, state: VQAState):
        result = self.ocr_agent.perform_ocr(state["img_path"])
        self.state.update(result)
        return result
    
    def od_node(self, state: VQAState):
        result = self.ocr_agent.perform_odetection(state["img_path"])
        self.state.update(result)
        return result
    
    def store_audio(self, message):
        # Text-to-speech conversion
        language = 'en'
        myobj = gTTS(text=message, lang=language, slow=False)
        audio_file_path = r"C:/Users/ksahoo1.ASURITE/AppData/LocalLow/DefaultCompany/My project/Audio/audio_answer.mp3"
        print("####################### Stored audio file")
        myobj.save(audio_file_path)
    
    def answer_node(self, state: VQAState):
        result = self.answer_writer_agent.generate_answer(self.state["question"], self.state["img_path"], self.state["ocr_text"], self.state['od_results'])
        self.state.update(result)

        self.store_audio(result['answer'])
        return result
    
    # Conditional edge function to route back to joke generator or end based upon feedback from the evaluator
    def route_story_feedback(self, state: VQAState):
        """Route back to story generator or continue based upon feedback from the evaluator"""
        if state.get("tool_decision") == "ocr":
            return "ocr_node"
        elif state.get("tool_decision") == "od":
            return "od_node"
        else:
            return "answer_node"

    def build_workflow(self):
        """
        Constructs the workflow by adding nodes and edges.
        """
        logger.info("Building workflow...")
        # self.orchestrator_worker_builder.add_node("query_evaluator", self.query_evaluator)
        # self.orchestrator_worker_builder.add_node("ocr_node", self.ocr_node)
        # self.orchestrator_worker_builder.add_node("od_node", self.od_node)
        self.orchestrator_worker_builder.add_node("answer_node", self.answer_node)

        # Add edges to connect nodes
        self.orchestrator_worker_builder.add_edge(START, "answer_node")
        # self.orchestrator_worker_builder.add_conditional_edges(
        #     "query_evaluator", self.route_story_feedback, {
        #         "ocr_node": "ocr_node",
        #         "od_node": "od_node",
        #         "answer_node": "answer_node"
        #     }
        # )
        # self.orchestrator_worker_builder.add_edge("ocr_node", "answer_node")
        # self.orchestrator_worker_builder.add_edge("od_node", "answer_node")
        self.orchestrator_worker_builder.add_edge("answer_node", END)

    def compile_workflow(self):
        """
        Compiles the workflow for execution.
        """
        logger.info("Compiling workflow...")
        self.orchestrator_worker = self.orchestrator_worker_builder.compile()

    def invoke_workflow(self, question: str, img_path: str):
        """
        Runs the workflow with the initialized mission.

        Returns:
            dict: The final state after execution.
        """
        logger.info(f"Invoking workflow for question: [yellow]{question}[/yellow] and image: [yellow]{img_path}[/yellow]")
        self.state["question"] = question
        self.state["img_path"] = img_path
        return self.orchestrator_worker.invoke({"question": self.state["question"], "img_path": self.state["img_path"]})

    def get_state(self):
        """
        Returns the current state of the workflow.

        Returns:
            dict: The current state of the workflow.
        """
        return {"state": self.state}
    
    def call_tools(self, img_path: str):
        result = self.ocr_agent.perform_ocr(img_path)
        self.state.update(result)
        result = self.ocr_agent.perform_odetection(img_path)
        self.state.update(result)
        return {"state": self.state}