import json


# Function to generate first-turn conversations
def generate_first_turn_conversations(input_file):

    # Load SQL error data
    with open(input_file, "r") as f:
        sql_errors = json.load(f)

    conversations = []
    for error in sql_errors:
        # Extract fields from the error documentation
        message_id = error.get("SQL Message ID")
        message = error.get("Message")
        explanation = error.get("Explanation")
        user_response = error.get("User Response")
        
        # User's input (real-world query format)
        user_query = f"{message_id} {message}"
        
        # Assistant response
        assistant_response = (
            f"The error '{user_query}' indicates: {explanation} "
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
    
    with open("first_turn_conversations.jsonl", "w") as f:
        for conversation in conversations:
            f.write(json.dumps(conversation) + "\n")

    print("First-turn conversations generated and saved to first_turn_conversations.jsonl")


