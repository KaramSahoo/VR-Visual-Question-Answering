ANSWER_PROMPT_USER = """
User Question: {question}

Additional information about the VR Scene from object detection and OCR tools to help you answer the question:
{formatted_info}

Use the information above as additional information. Try to answer the question based on the image first, and then use the additional information to improve your answer if needed. If you cannot answer confidently, say so to the user.
The user does not want any special characters or formatting in the answer.
The user also does not want to know what additional information you used to answer the question, just the final answer.
"""
