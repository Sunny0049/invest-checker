import os
import json
import datetime
import uuid

import os
hf_token = os.environ.get("HF_TOKEN")


def save_user_story_json(user_story, session_id=None, base_dir="user_data"):
    # Generate filename based on current date
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    os.makedirs(base_dir, exist_ok=True)
    filepath = os.path.join(base_dir, f"user_stories_{date_str}.json")

    # Load existing data if file exists
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    # Use session_id or generate a new one
    session_id = session_id or str(uuid.uuid4())

    if session_id not in data:
        data[session_id] = []

    # Add story
    if isinstance(user_story, list):
        data[session_id].extend(user_story)
    else:
        data[session_id].append(user_story)

    # Save updated data
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    st.success(f"Saved under session {session_id} in {filepath}")
    return session_id
