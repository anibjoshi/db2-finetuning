import json
from typing import List, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


interactions = [
    # Direct queries about the SQL code
    {"user": "What is {sql_code}?", "assistant": "{sql_code} means {message}"},
    {"user": "Explain {sql_code}.", "assistant": "{sql_code} means {message}"},
    {"user": "What does {sql_code} mean?", "assistant": "{sql_code} means {message}"},
    {"user": "Can you provide details about {sql_code}?", "assistant": "{sql_code} means {message}"},
    {"user": "I'm seeing {sql_code}. What does it mean?", "assistant": "{sql_code} means {message}"},

    # Queries combining the SQL code and its message
    {"user": "{sql_code} {message} What does this mean?", "assistant": "The code '{sql_code} {message}' indicates: {explanation}"},
    {"user": "Can you explain the message: '{sql_code} {message}'?", "assistant": "The code '{sql_code} {message}' indicates: {explanation}"},
    {"user": "What should I understand from {sql_code} {message}?", "assistant": "The code '{sql_code} {message}' indicates: {explanation}"},

    # Resolution-focused queries
    {"user": "How do I resolve {sql_code}?", "assistant": "{response}"},
    {"user": "What steps should I take to fix {sql_code}?", "assistant": "{response}"},
    {"user": "What is the recommended response for {sql_code}?", "assistant": "{response}"},

    # Explanation-related queries
    {"user": "What does the explanation mean for {sql_code}?", "assistant": "{explanation}"},
    {"user": "Can you provide more details on the explanation for {sql_code}?", "assistant": "{explanation}"},
    {"user": "Can you break down the explanation of {sql_code} for me?", "assistant": "{explanation}"},
]


def generate_first_turn_conversations(input_path: str, output_path: str) -> None:
    """Generate first-turn conversations from SQL code data in JSONL format.
    
    For each SQL code, generates multiple conversation variations using predefined
    interaction templates. Each template is filled with the SQL code's details.
    
    Args:
        input_path: Path to input JSONL file containing SQL codes
        output_path: Path to output JSONL file for generated conversations
        
    Raises:
        JSONDecodeError: If JSONL parsing fails
        IOError: If file operations fail
    """
    try:
        # Read JSONL file line by line
        sql_codes = []
        with open(input_path, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        sql_codes.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line: {e}")
                        continue

        if not sql_codes:
            raise ValueError("No valid SQL code entries found in input file")

        # Generate conversations
        conversations = []
        for code in sql_codes:
            # Extract fields from the code documentation
            sql_code = code.get("id")
            message = code.get("message", "")
            explanation = code.get("explanation", "")
            response = code.get("response", "")
            
            # Skip if required fields are missing
            if not sql_code:
                logger.warning(f"Skipping code with missing ID: {code}")
                continue
            
            # Generate variations using interaction templates
            for interaction in interactions:
                try:
                    # Format the user query and assistant response using the template
                    user_content = interaction["user"].format(
                        sql_code=sql_code,
                        message=message,
                        explanation=explanation
                    )
                    
                    assistant_content = interaction["assistant"].format(
                        sql_code=sql_code,
                        message=message,
                        explanation=explanation,
                        response=response
                    )

                    # Create the conversation
                    conversation = {
                        "dialogue": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content}
                        ]
                    }
                    conversations.append(conversation)
                    
                except KeyError as e:
                    logger.warning(f"Failed to format interaction for code {sql_code}: {e}")
                    continue

        # Write conversations in JSONL format
        with open(output_path, "w") as f:
            for conversation in conversations:
                f.write(json.dumps(conversation) + "\n")

        logger.info(
            f"Generated {len(conversations)} conversations "
            f"({len(sql_codes)} SQL codes × {len(interactions)} templates)"
        )

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise
    except IOError as e:
        logger.error(f"File operation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during conversation generation: {e}")
        raise


