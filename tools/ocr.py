from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image
import requests
import copy


def run_example(task_prompt, text_input=None, img_path=None, processor=None, model=None):
	image = Image.open(img_path)

	if text_input is None:
		prompt = task_prompt
	else:
		prompt = task_prompt + text_input
	inputs = processor(text=prompt, images=image, return_tensors="pt")
	generated_ids = model.generate(
  	input_ids=inputs["input_ids"].cuda(),
  	pixel_values=inputs["pixel_values"].cuda(),
  	max_new_tokens=1024,
  	early_stopping=False,
  	do_sample=False,
  	num_beams=3,
	)
	generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
	parsed_answer = processor.post_process_generation(
    	generated_text,
    	task=task_prompt,
    	image_size=(image.width, image.height)
	)

	return parsed_answer