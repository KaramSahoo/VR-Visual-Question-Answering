from flask import Blueprint, request, jsonify, g
from workflows.vqa_workflow import VQAWorkflow
from utils.logger import logger

# Create a Blueprint named 'prompts'
generate_bp = Blueprint('generate', __name__)

# Route 1: POST request to handle mission input
@generate_bp.route('/', methods=['POST'])
def generate_answer():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Check if 'mission' key is in the JSON data
        if 'question' not in data:
            return jsonify({"error": "Missing 'question' key in request data"}), 400
        
        # Check if 'img_path' key is in the JSON data
        if 'img_path' not in data:
            return jsonify({"error": "Missing 'img_path' key in request data"}), 400

        # Extract the 'question' value from the request
        question = data['question']
        img_path = data['img_path']

        # Initialize LLM and workflow
        llm = g.llm_model
        workflow = VQAWorkflow(llm)

        # Build and compile the workflow
        workflow.build_workflow()
        workflow.compile_workflow()

        # Invoke the workflow
        state = workflow.invoke_workflow(question=question, img_path=img_path)
        logger.success("[green]Workflow execution completed![/green]")

        # print(f"Workflow state: {state}")

        return jsonify({
            "state": {
                **state,
            }
        }), 200


    except Exception as e:
        # Error handling for any other issues (e.g., invalid JSON format)
        return jsonify({"error": str(e)}), 500

# Route 2: Dummy mission response for frontend testing
@generate_bp.route('/test', methods=['POST'])
def test_mission():
    """Returns a dummy mission response for frontend testing."""
    dummy_state = {
        "team_name": "Guardians of Code",
        "team": [
            {
                "name": "CodeMaster",
                "alias": "The Debugger",
                "power": "Can fix any bug instantly",
                "origin": "Silicon Valley",
                "weapon": {
                    "weapon_type": "Keyboard",
                    "weapon_name": "Bug Slayer",
                    "weapon_lore": "A mystical keyboard that can auto-correct any syntax error."
                }
            },
            {
                "name": "CyberKnight",
                "alias": "The Encrypter",
                "power": "Unbreakable cyber defenses",
                "origin": "Dark Web",
                "weapon": {
                    "weapon_type": "Shield",
                    "weapon_name": "Firewall Shield",
                    "weapon_lore": "A shield forged from the strongest encryption algorithms."
                }
            }
        ],
        "story": "The Guardians of Code unite to battle the evil forces of malware and cyber threats!",
        "story_feedback": "An epic tale of heroism and cybersecurity!",
        "feedback_grade": "A+"
    }

    return jsonify({
        "message": f"Team {dummy_state['team_name']} has been summoned!",
        "state": {
            **dummy_state,  
            "team": [hero for hero in dummy_state["team"]]  # Convert to ensure frontend compatibility
        }
    }), 200
