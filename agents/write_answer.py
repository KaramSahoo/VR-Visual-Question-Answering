from langchain.schema import SystemMessage, HumanMessage
from utils.logger import logger
from prompts.system_message import ANSWER_PROMPT_SYSTEM
from prompts.user_prompts import ANSWER_PROMPT_USER, IMPROVE_STORY_PROMPT_USER
import base64



class AnswerWriter:
    def __init__(self, llm):
        """Initialize the story generator agent."""
        self.llm = llm

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def query_llm(self, question, screenshot_full_path):
        # print(llm_chain.invoke({'question': question})['text'])
        base64_image = self.encode_image(screenshot_full_path)

        messages=[
                {"role": "system", "content": "You are a specialized and helpful assistant for Blind and Low Vision User's named ThirdAI. Your expertise is to analyse the scenes in front of the user which will be passed as image input. The user will have a question about that virtual reality scene they are navigating, and you need to answer or provide help with that question. Since they are blind or low vision user you need to provide information that is concise, descriptive and not overwhelming to while navigating these 3D Virtual Reality Scenes. You do not provide information that is incorrect or not asked for. If you cannot answer confidently say so to the user."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"{question}"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]}
            ]
        ai_message = self.llm.invoke(messages)
        return ai_message.content

    def format_detection(self, objects: list[str], texts: list[str]) -> str:
        objects_str = ", ".join(objects) if objects else "None"
        texts_str = ", ".join(texts) if texts else "None"
        
        return f"Detected objects are: {objects_str}\nDetected text in the scene: {texts_str}"

    def generate_answer_prompt(self, question: str, objects: list[str], texts: list[str]):
        """Formats the answer prompt with question and tool results."""
        formatted_info = self.format_detection(objects, texts)

        return ANSWER_PROMPT_USER.format(
            question=question,
            formatted_info=formatted_info,
        )

    def generate_answer(self, question: str, img_path: str, texts: list[str], objects: list[str]):
        """Generates a answer for the question."""
        logger.info("Generating final answer...")
        answer_prompt_user = self.generate_answer_prompt(question, objects, texts)

        base64_image = self.encode_image(img_path)

        messages=[
                SystemMessage(content=ANSWER_PROMPT_SYSTEM),
                {"role": "user", "content": [
                    {"type": "text", "text": answer_prompt_user},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]}
            ]
        ai_message = self.llm.invoke(messages)

        logger.success(f"[green]Answer generated successfully: {ai_message.content}![/green]")

        return {"answer": ai_message.content}
