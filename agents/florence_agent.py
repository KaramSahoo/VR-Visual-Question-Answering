from langchain.schema import SystemMessage, HumanMessage
from utils.logger import logger
from transformers import AutoProcessor, AutoModelForCausalLM  
from prompts.system_message import QUERY_EVALUATOR_PROMPT_SYSTEM
from tools.ocr import run_example
import time

class FlorenceAgent:
    def __init__(self):
        """Initialize the user query evaluator agent."""
        model_id = 'microsoft/Florence-2-large'
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        

    def perform_ocr(self, img_path: str):
        """Evaluates the given user query and provides the tools to call."""
        logger.info(f"Performing OCR...")

        start_time = time.time()  # Start time logging

        ocr_results = run_example('<OCR>', img_path=img_path, processor=self.processor, model=self.model)
        ocr_results = ocr_results["<OCR>"].split('\n')

        end_time = time.time()  # End time logging
        elapsed_time = end_time - start_time  # Calculate time taken

        logger.success(f"OCR performed in {elapsed_time:.2f} seconds!")  

        return {"ocr_text": ocr_results}
    
    def perform_odetection(self, img_path: str):
        """Evaluates the given user query and provides the tools to call."""
        logger.info(f"Performing Object Detection...")
        start_time = time.time()  # Start time logging

        od_results = run_example('<OD>', img_path=img_path, processor=self.processor, model=self.model)
        od_results = od_results['<OD>']['labels']

        end_time = time.time()  # End time logging
        elapsed_time = end_time - start_time  # Calculate time taken

        logger.success(f"Object Detection performed in {elapsed_time:.2f} seconds!")  # Logs first 100 chars

        return {"od_results": od_results}
