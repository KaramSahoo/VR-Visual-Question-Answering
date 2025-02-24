ANSWER_PROMPT_USER = """
User Question: {question}

Additional information from object detection and OCR tools to help you answer the question:
{formatted_info}
"""

IMPROVE_STORY_PROMPT_USER = """
- Story: {story}

- Mission: {mission}
- Team Name: {team_name}
- Heroes:
{heroes_details}

- Feedback: {feedback}
"""