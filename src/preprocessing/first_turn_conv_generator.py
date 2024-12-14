import json
from typing import List, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_first_turn_conversations(input_path: str, output_path: str) -> None:
    """Generate first-turn conversations from SQL code data in JSONL format.
    
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
            message_id = code.get("id")
            message = code.get("message")
            explanation = code.get("explanation")
            user_response = code.get("response")
            
            # Skip if required fields are missing
            if not all([message_id, message, explanation, user_response]):
                logger.warning(f"Skipping code with missing fields: {code}")
                continue
            
            # User's input (real-world query format)
            user_query = f"{message_id} {message}"
            
            # Assistant response
            assistant_response = (
                f"The code '{user_query}' indicates: {explanation} "
                f"{user_response}"
            )

            # Create the single-turn conversation
            conversation = {
                "dialogue": [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": assistant_response}
                ]
            }
            conversations.append(conversation)

        # Write conversations in JSONL format
        with open(output_path, "w") as f:
            for conversation in conversations:
                f.write(json.dumps(conversation) + "\n")

        logger.info(f"Generated {len(conversations)} conversations from {len(sql_codes)} SQL codes")

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise
    except IOError as e:
        logger.error(f"File operation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during conversation generation: {e}")
        raise


