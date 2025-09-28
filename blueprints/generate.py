from flask import Blueprint, request, jsonify, g
from utils.logger import logger
import os
from helper.audio import speech_to_text

# Create a Blueprint named 'prompts'
generate_bp = Blueprint('generate', __name__)

# Route 1: POST request to handle mission input
@generate_bp.route('/', methods=['POST'])
def generate_answer():
    try:
        screenshot_path = "screenshot.png"
        audio_path = "audio.wav"

        # Construct the full paths for the files
        screenshot_full_path = os.path.join(r"C:/Users/ksahoo1.ASURITE/AppData/LocalLow/DefaultCompany/My project/Screenshots", screenshot_path)
        audio_full_path = os.path.join(r"C:/Users/ksahoo1.ASURITE/AppData/LocalLow/DefaultCompany/My project/Audio", audio_path)

        # Check if the screenshot exists
        if not os.path.exists(screenshot_full_path):
            return jsonify({"error": f"Screenshot file not found at {screenshot_full_path}"}), 404
        
        # Check if the audio exists
        if not os.path.exists(audio_full_path):
            return jsonify({"error": f"Audio file not found at {audio_full_path}"}), 404

        user_prompt = speech_to_text(audio_full_path, 'en-US')

        # Initialize LLM and workflow
        workflow = g.workflow

        # Invoke the workflow
        state = workflow.invoke_workflow(question=user_prompt, img_path=screenshot_full_path)
        logger.success("[green]Workflow execution completed![/green]")


        return jsonify({
            "state": {
                **state,
            }
        }), 200


    except Exception as e:
        # Error handling for any other issues (e.g., invalid JSON format)
        return jsonify({"error": str(e)}), 500
    
# For Pilot Study, can be ignored if all images are stored in the same folder
@generate_bp.route('/pilot', methods=['POST'])
def generate_pilot_answer():
    try:
        # Extract paths for screenshot and audio from the JSON payload
        screenshot_path = "screenshot.png"
        audio_path = "audio.wav"

        # Construct the full paths for the files
        screenshot_full_path = os.path.join(r"C:/Users/ksahoo1.ASURITE/AppData/LocalLow/DefaultCompany/My Project/Screenshots", screenshot_path)
        audio_full_path = os.path.join(r"C:/Users/ksahoo1.ASURITE/AppData/LocalLow/DefaultCompany/My Project/Audio", audio_path)

        # Check if the screenshot exists
        if not os.path.exists(screenshot_full_path):
            return jsonify({"error": f"Screenshot file not found at {screenshot_full_path}"}), 404
        
        # Check if the audio exists
        if not os.path.exists(audio_full_path):
            return jsonify({"error": f"Audio file not found at {audio_full_path}"}), 404

        user_prompt = speech_to_text(audio_full_path, 'en-US')

        # Initialize LLM and workflow
        workflow = g.workflow

        # Invoke the workflow
        state = workflow.invoke_workflow(question=user_prompt, img_path=screenshot_full_path)
        logger.success("[green]Workflow execution completed![/green]")


        return jsonify({
            "state": {
                **state,
            }
        }), 200


    except Exception as e:
        # Error handling for any other issues (e.g., invalid JSON format)
        return jsonify({"error": str(e)}), 500


@generate_bp.route('/state', methods=['GET'])
def get_state():
    try:

        # Initialize workflow
        workflow = g.workflow

        # Invoke the OCR
        state = workflow.get_state()
        # logger.success("[green]Workflow execution completed![/green]")

        # print(f"Workflow state: {state}")

        return jsonify({
            "state": {
                **state,
            }
        }), 200
    except Exception as e:
        # Error handling for any other issues (e.g., invalid JSON format)
        return jsonify({"error": str(e)}), 500
    
    
@generate_bp.route('/tools', methods=['GET'])
def call_tools():
    try:
        # Get the JSON data from the request
        # data = request.get_json()

        screenshot_path = "screenshot.png"

        # Construct the full paths for the files
        screenshot_full_path = os.path.join(r"C:/Users/ksahoo1.ASURITE/AppData/LocalLow/DefaultCompany/My project/Screenshots", screenshot_path)
        
        # Check if the screenshot exists
        if not os.path.exists(screenshot_full_path):
            return jsonify({"error": f"Screenshot file not found at {screenshot_full_path}"}), 404

        # Initialize workflow
        workflow = g.workflow

        # Invoke the OCR
        state = workflow.call_tools(screenshot_full_path)
        # logger.success("[green]Workflow execution completed![/green]")

        # print(f"Workflow state: {state}")

        return jsonify({
            "state": {
                **state,
            }
        }), 200
    except Exception as e:
        # Error handling for any other issues (e.g., invalid JSON format)
        return jsonify({"error": str(e)}), 500


# For Pilot Study, can be ignored if all images are stored in the same folder
@generate_bp.route('/pilot/tools', methods=['GET'])
def call_pilot_tools():
    try:
        # Get the JSON data from the request
        # data = request.get_json()

        screenshot_path = "screenshot.png"

        # Construct the full paths for the files
        screenshot_full_path = os.path.join(r"C:/Users/ksahoo1.ASURITE/AppData/LocalLow/DefaultCompany/My Project/Screenshots", screenshot_path)
        
        # Check if the screenshot exists
        if not os.path.exists(screenshot_full_path):
            return jsonify({"error": f"Screenshot file not found at {screenshot_full_path}"}), 404

        # Initialize workflow
        workflow = g.workflow

        # Invoke the OCR
        state = workflow.call_tools(screenshot_full_path)
        # logger.success("[green]Workflow execution completed![/green]")

        # print(f"Workflow state: {state}")

        return jsonify({
            "state": {
                **state,
            }
        }), 200
    except Exception as e:
        # Error handling for any other issues (e.g., invalid JSON format)
        return jsonify({"error": str(e)}), 500
