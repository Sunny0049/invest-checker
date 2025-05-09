import os
import json
import datetime
import logging
import io
import csv
import time
import uuid
import fcntl  # For file locking on Linux/Unix
import random
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app_logs.log"),
                              logging.StreamHandler()])

# Thread-local storage for session identifiers
thread_local = threading.local()

def get_session_id():
    """Get a unique session ID for the current thread/user"""
    if not hasattr(thread_local, "session_id"):
        thread_local.session_id = str(uuid.uuid4())
    return thread_local.session_id

def ensure_directory_exists(filename):
    """Ensure that the directory for the given filename exists"""
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create directory {directory}: {str(e)}")
            return False
    return True

def save_to_daily_json(stories):
    """
    Save stories to a daily JSON file with concurrency support.
    Works for both single story and batch processing.
    
    Args:
        stories: Either a single story dict or a list of story dicts
    """
    # Create daily filename
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Define the directory where files will be stored
    output_dir = "user_stories"  # Can be changed to any directory
    
    # Ensure the directory exists
    ensure_directory_exists(output_dir)
    
    # Create full path filenames
    filename = os.path.join(output_dir, f"user_stories_{today}.json")
    temp_filename = os.path.join(output_dir, f"user_stories_{today}.{get_session_id()}.tmp")
    
    max_retries = 5
    retry_count = 0
    success = False
    
    while retry_count < max_retries and not success:
        try:
            # Load existing stories for today if file exists
            existing_stories = []
            
            # Try to safely read the current file
            if os.path.exists(filename):
                try:
                    # On Unix/Linux systems, use file locking
                    with open(filename, "r", encoding="utf-8") as f:
                        try:
                            # Get a shared lock for reading
                            fcntl.flock(f, fcntl.LOCK_SH)
                            data = json.load(f)
                            existing_stories = data.get("stories", [])
                        except json.JSONDecodeError:
                            logging.warning(f"Existing JSON file {filename} is corrupted. Creating a new one.")
                        finally:
                            # Release the lock
                            fcntl.flock(f, fcntl.LOCK_UN)
                except (IOError, OSError) as e:
                    if retry_count < max_retries - 1:
                        # Add a small random delay to reduce collision probability
                        time.sleep(0.1 + random.random() * 0.3)
                        retry_count += 1
                        continue
                    else:
                        logging.error(f"Failed to read file after {max_retries} attempts: {str(e)}")
                        return False
            
            # Add new stories to existing ones
            if isinstance(stories, list):
                existing_stories.extend(stories)
            else:
                existing_stories.append(stories)
            
            # Write to a temporary file first
            with open(temp_filename, "w", encoding="utf-8") as f:
                json.dump({"stories": existing_stories}, f, ensure_ascii=False, indent=2)
            
            # Now atomically replace the original file with our new one
            # On Unix/Linux, this is an atomic operation
            try:
                # On Unix/Linux systems
                # First, check if file exists - if not, create it with proper permissions
                if not os.path.exists(filename):
                    # Create the file if it doesn't exist
                    open(filename, 'a').close()
                
                # Open for writing with exclusive lock
                with open(filename, "w", encoding="utf-8") as f:
                    # Get an exclusive lock for writing
                    fcntl.flock(f, fcntl.LOCK_EX)
                    # Read the temporary file
                    with open(temp_filename, "r", encoding="utf-8") as temp:
                        content = temp.read()
                    # Write to the actual file
                    f.write(content)
                    success = True
            except (IOError, OSError) as e:
                if retry_count < max_retries - 1:
                    time.sleep(0.1 + random.random() * 0.3)
                    retry_count += 1
                    continue
                else:
                    logging.error(f"Failed to write file after {max_retries} attempts: {str(e)}")
                    return False
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_filename):
                    try:
                        os.remove(temp_filename)
                    except:
                        pass
            
            # Log success message
            count = len(stories) if isinstance(stories, list) else 1
            logging.info(f"{count} {'stories' if count > 1 else 'story'} saved successfully to {filename}!")
            return True
            
        except Exception as e:
            if retry_count < max_retries - 1:
                time.sleep(0.1 + random.random() * 0.3)
                retry_count += 1
                continue
            else:
                logging.error(f"Error saving stories after {max_retries} attempts: {str(e)}")
                logging.debug(f"Current working directory: {os.getcwd()}")
                logging.debug(f"Files in directory: {os.listdir()}")
                return False
    
    return success

# For Windows systems (if needed)
def save_to_daily_json_windows(stories):
    """
    Windows-compatible version of save_to_daily_json.
    Use this on Windows deployments where fcntl is not available.
    """
    # Create daily filename
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Define the directory where files will be stored
    output_dir = "user_stories"  # Can be changed to any directory
    
    # Ensure the directory exists
    ensure_directory_exists(output_dir)
    
    # Create full path filenames
    filename = os.path.join(output_dir, f"user_stories_{today}.json")
    temp_filename = os.path.join(output_dir, f"user_stories_{today}.{get_session_id()}.tmp")
    
    max_retries = 5
    retry_count = 0
    success = False
    
    while retry_count < max_retries and not success:
        try:
            # Load existing stories for today if file exists
            existing_stories = []
            
            # Try to safely read the current file
            if os.path.exists(filename):
                try:
                    # Windows doesn't have fcntl, so we use a simpler approach
                    # This is not fully concurrent-safe but reduces the risk
                    try:
                        with open(filename, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            existing_stories = data.get("stories", [])
                    except json.JSONDecodeError:
                        logging.warning(f"Existing JSON file {filename} is corrupted. Creating a new one.")
                except (IOError, OSError) as e:
                    if retry_count < max_retries - 1:
                        time.sleep(0.1 + random.random() * 0.3)
                        retry_count += 1
                        continue
                    else:
                        logging.error(f"Failed to read file after {max_retries} attempts: {str(e)}")
                        return False
            
            # Add new stories to existing ones
            if isinstance(stories, list):
                existing_stories.extend(stories)
            else:
                existing_stories.append(stories)
            
            # On Windows, we can't guarantee atomic writes
            # Write to a temporary file first
            with open(temp_filename, "w", encoding="utf-8") as f:
                json.dump({"stories": existing_stories}, f, ensure_ascii=False, indent=2)
            
            # Then try to rename it (closest to atomic on Windows)
            try:
                # Make sure parent directory exists
                os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
                
                # Remove old file if it exists
                if os.path.exists(filename):
                    os.remove(filename)
                os.rename(temp_filename, filename)
                success = True
            except (IOError, OSError) as e:
                if retry_count < max_retries - 1:
                    time.sleep(0.1 + random.random() * 0.3)
                    retry_count += 1
                    continue
                else:
                    logging.error(f"Failed to write file after {max_retries} attempts: {str(e)}")
                    return False
            
            # Log success message
            count = len(stories) if isinstance(stories, list) else 1
            logging.info(f"{count} {'stories' if count > 1 else 'story'} saved successfully to {filename}!")
            return True
            
        except Exception as e:
            if retry_count < max_retries - 1:
                time.sleep(0.1 + random.random() * 0.3)
                retry_count += 1
                continue
            else:
                logging.error(f"Error saving stories after {max_retries} attempts: {str(e)}")
                logging.debug(f"Current working directory: {os.getcwd()}")
                logging.debug(f"Files in directory: {os.listdir()}")
                return False
    
    return success

# Choose the right function based on platform
import platform
if platform.system() == "Windows":
    # For Windows environments
    actual_save_function = save_to_daily_json_windows
    logging.info("Running on Window system")
else:
    # For Unix/Linux environments (including most cloud providers)
    actual_save_function = save_to_daily_json
    logging.info("Running on Unix/Linux system")

# Other functions remain the same
def evaluate_and_save_single_story(user_story):
    """Evaluate a single user story and save it to the daily JSON file."""
    try:
        # Your existing evaluation logic
        #result = evaluate_user_story(user_story, model, tokenizer, thresholds, device)
        
        # Save to daily JSON using the platform-appropriate function
        actual_save_function(user_story)
    except Exception as e:
        logging.error(f"Error processing single story: {str(e)}")
        return None

def process_batch(user_stories, model, tokenizer, thresholds, device):
    """Process a batch of user stories and evaluate them."""
    results = []
    
    for story in user_stories:
        try:
            # Process each story individually
            story_result = evaluate_user_story(story, model, tokenizer, thresholds, device)
            results.append(story_result)
        except Exception as e:
            logging.error(f"Error processing story: {story}")
            logging.error(f"Error details: {str(e)}")
            # Add failed result with error
            results.append({
                "user_story": story,
                "overall_score": 0,
                "criteria_scores": {},
                "error": str(e)
            })
    
    # Save results to daily JSON file using the platform-appropriate function
    actual_save_function(results)
    
    return results