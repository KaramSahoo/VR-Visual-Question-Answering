from langchain.schema import SystemMessage, HumanMessage
from utils.logger import logger
from transformers import AutoProcessor, AutoModelForCausalLM  
from prompts.system_message import QUERY_EVALUATOR_PROMPT_SYSTEM
from tools.ocr import run_example

class FlorenceAgent:
    def __init__(self):
        """Initialize the user query evaluator agent."""
        model_id = 'microsoft/Florence-2-large'
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        

    def perform_ocr(self, img_path: str):
        """Evaluates the given user query and provides the tools to call."""
        logger.info(f"Performing OCR...")

        ocr_results = run_example('<OCR>', img_path=img_path, processor=self.processor, model=self.model)
        ocr_results = ocr_results["<OCR>"].split('\n')
        logger.success(f"OCR Performed!")  # Logs first 100 chars

        return {"ocr_text": ocr_results}
    
    def perform_odetection(self, img_path: str):
        """Evaluates the given user query and provides the tools to call."""
        logger.info(f"Performing Object Detection...")

        od_results = run_example('<OD>', img_path=img_path, processor=self.processor, model=self.model)
        od_results = od_results['<OD>']['labels']
        logger.success(f"Object Detection Performed!")  # Logs first 100 chars

        return {"od_results": od_results}
