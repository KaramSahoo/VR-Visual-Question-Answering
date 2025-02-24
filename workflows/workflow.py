from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from utils.schemas import SuperHero, Team, Story, StoryFeedback, Decision
from agents.story_evaluator import StoryEvaluator
from agents.story_writer import StoryGenerator
from agents.query_evaluator import QueryEvaluator
from agents.team_creator import TeamGenerator
from utils.logger import logger

# Graph state
class State(TypedDict):
    mission: str  # Mission
    team_name: str  # Name of the superhero team
    team: list[SuperHero]  # List of superheroes
    story: str  # Story of the superhero team's mission
    story_feedback: str  # Feedback on the story
    feedback_grade: str  # Grade for the story feedback


# Graph state
class VQAState(TypedDict):
    question: str  # User Question
    answer: str  # LLM Answer
    tool_use: list[str]  # Tools to use
    story: str  # Story of the superhero team's mission
    ansewer_feedback: str  # Feedback on the story
    answer_grade: str  # Grade for the story feedback
    ocr_text: str  # OCR text from the image
    od_results: list[str]  # Object detection results
    img_path: str  # Image path

class Workflow:
    def __init__(self, llm):
        """
        Initializes the Workflow with the LLM model and instructions.

        Args:
            llm: The language model used for structured output.
        """
        self.llm = llm
        self.query_evaluator_llm = llm.with_structured_output(Decision)  # Augment LLM with Team schema
        self.team_creator_llm = llm.with_structured_output(Team)  # Augment LLM with Team schema
        self.story_creator_llm = llm.with_structured_output(Story)  # Augment LLM with Story schema
        self.story_evaluator_llm = llm.with_structured_output(StoryFeedback)  # Augment LLM with StoryFeedback schema
        
        self.team_generator_agent = TeamGenerator(self.team_creator_llm)
        self.story_generator_agent = StoryGenerator(self.story_creator_llm)
        self.story_evaluator_agent = StoryEvaluator(self.story_evaluator_llm)
        self.qu_evaluator_agent = QueryEvaluator(self.story_evaluator_llm)

        # Initialize the workflow state
        self.state: State = {
            "mission": "",
            "team_name": "",
            "team": [],
            "story": "",
            "story_feedback": "",
            "feedback_grade": ""
        }

        self.orchestrator_worker_builder = StateGraph(State)

    def team_generator(self, state: State):
        """Wrapper function to call the team generator agent."""
        result = self.team_generator_agent.generate_team(state["mission"])
        self.state.update(result)
        return result
    
    def story_generator(self, state: State):
        if self.state["feedback_grade"] == "Rejected + Feedback":
            result = self.story_generator_agent.improve_story(state["mission"], state["team_name"], state["team"], state["story_feedback"], state["story"])
        else:
            result = self.story_generator_agent.generate_story(state["mission"], state["team_name"], state["team"])
        self.state.update(result)
        return result
    
    def story_evaluator(self, state: State):
        result = self.story_evaluator_agent.evaluate_story(state["story"])
        self.state.update(result)
        return result
    
    def query_evaluator(self, state: VQAState):
        result = self.story_evaluator_agent.evaluate_story(state["mission"])
        self.state.update(result)
        return result

    # Conditional edge function to route back to joke generator or end based upon feedback from the evaluator
    def route_story_feedback(self, state: State):
        """Route back to story generator or continue based upon feedback from the evaluator"""

        if state.get("feedback_grade") == "accepted":
            self.state["feedback_grade"] = "Accepted"
            return "Accepted"
        self.state["feedback_grade"] = "Rejected + Feedback"
        return "Rejected + Feedback"

    def build_workflow(self):
        """
        Constructs the workflow by adding nodes and edges.
        """
        logger.info("Building workflow...")
        self.orchestrator_worker_builder.add_node("team_generator", self.team_generator)
        self.orchestrator_worker_builder.add_node("story_generator", self.story_generator)
        self.orchestrator_worker_builder.add_node("story_evaluator", self.story_evaluator)

        # Add edges to connect nodes
        self.orchestrator_worker_builder.add_edge(START, "team_generator")
        self.orchestrator_worker_builder.add_edge("team_generator", "story_generator")
        self.orchestrator_worker_builder.add_edge("story_generator", "story_evaluator")
        self.orchestrator_worker_builder.add_conditional_edges(
            "story_evaluator", self.route_story_feedback, {
                "Accepted": END,
                "Rejected + Feedback": "story_generator"
            }
        )

    def compile_workflow(self):
        """
        Compiles the workflow for execution.
        """
        logger.info("Compiling workflow...")
        self.orchestrator_worker = self.orchestrator_worker_builder.compile()

    def invoke_workflow(self, mission: str):
        """
        Runs the workflow with the initialized mission.

        Returns:
            dict: The final state after execution.
        """
        logger.info(f"Invoking workflow for mission: [yellow]{mission}[/yellow]")
        self.state["mission"] = mission
        return self.orchestrator_worker.invoke({"mission": self.state["mission"]})
