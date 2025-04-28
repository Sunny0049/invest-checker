import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import re
import os
import json
import datetime
import csv
import io
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="INVEST User Story Evaluator",
    page_icon="✅",
    layout="wide"
)

# Model definition - same as your training code
class MultiLabelBERT(nn.Module):
    def __init__(self, hidden_dim=768, output_dim=6):
        super(MultiLabelBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.pooler_output)
        return self.sigmoid(self.fc(x))

# Function to load the model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiLabelBERT().to(device)
    
    # Load model parameters
    # Replace this path with the actual path where you saved your model
    model_path = "invest_bert_model.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    else:
        st.error(f"Model file not found at {model_path}. Please check the file path.")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Default thresholds in case the file is not found
    default_thresholds = {'I': 0.5, 'N': 0.5, 'V': 0.5, 'E': 0.5, 'S': 0.5, 'T': 0.5}
    
    # Try to load optimized thresholds
    try:
        with open("thresholds_log.json", "r") as f:
            thresholds_data = f.readlines()[-1]  # Get the last entry
            thresholds = json.loads(thresholds_data.split(":", 1)[1].strip().rstrip("}").strip())
    except Exception as e:
        print(f"Error loading thresholds: {e}")
        thresholds = default_thresholds
    
    return model, tokenizer, thresholds, device

# Criteria descriptions and enhancement suggestions
INVEST_DESCRIPTIONS = {
    'I': {
        'name': 'Independent',
        'description': 'The user story should be self-contained and not dependent on other stories.',
        'suggestions': [
            'Break down the story into smaller, independent pieces',
            'Remove dependencies on other stories',
            'Make sure the story can be implemented and delivered on its own'
        ],
        'examples': {
            'good': 'As a user, I want to log in with my credentials so I can access my account.',
            'bad': 'As a user, I want to log in after the admin has created my account and assigned permissions.'
        }
    },
    'N': {
        'name': 'Negotiable',
        'description': 'The story should be flexible and open to discussion, not a rigid contract.',
        'suggestions': [
            'Focus on the "what" and "why", not the "how"',
            'Avoid specific implementation details',
            'Use more general terms that allow for discussion and alternatives'
        ],
        'examples': {
            'good': 'As a customer, I want to receive order confirmations so I know my purchase was successful.',
            'bad': 'As a customer, I want to receive an email with order details using the SendGrid API with specific template ID #12345.'
        }
    },
    'V': {
        'name': 'Valuable',
        'description': 'The story should deliver value to users or stakeholders.',
        'suggestions': [
            'Clearly state the benefit to the user or business',
            'Make sure the "so that" part explains the value',
            'Consider if the story aligns with business goals'
        ],
        'examples': {
            'good': 'As a shopper, I want to save my favorite items so that I can quickly find them when I return.',
            'bad': 'As a developer, I want to refactor the shopping cart code so it uses the repository pattern.'
        }
    },
    'E': {
        'name': 'Estimable',
        'description': 'The team should be able to estimate the size of the story.',
        'suggestions': [
            'Add more details about scope and requirements',
            'Break down complex stories into smaller pieces',
            'Provide context necessary for estimation'
        ],
        'examples': {
            'good': 'As a user, I want to reset my password by receiving a reset link via email.',
            'bad': 'As a user, I want the system to be more secure.'
        }
    },
    'S': {
        'name': 'Small',
        'description': 'The story should be small enough to be completed in one iteration.',
        'suggestions': [
            'Split large stories into multiple smaller stories',
            'Focus on a single functionality or capability',
            'Remove scope that could be delivered separately'
        ],
        'examples': {
            'good': 'As a user, I want to update my profile picture.',
            'bad': 'As a user, I want a complete profile management system with profile pictures, personal information, privacy settings, and social media integration.'
        }
    },
    'T': {
        'name': 'Testable',
        'description': 'The story should have clear acceptance criteria that can be verified.',
        'suggestions': [
            'Add specific acceptance criteria',
            'Ensure criteria can be verified objectively',
            'Include test scenarios for the functionality'
        ],
        'examples': {
            'good': 'As a user, I want to search for products by keyword so that I can find relevant items quickly.',
            'bad': 'As a user, I want the system to be user-friendly.'
        }
    }
}

# Function to predict INVEST criteria
def predict_invest_criteria(user_story, model, tokenizer, thresholds, device):
    # Prepare the input
    encoding = tokenizer(
        user_story,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    # Get predicted labels using thresholds
    labels = ['I', 'N', 'V', 'E', 'S', 'T']
    predictions = {}
    scores = outputs.cpu().numpy()[0]
    
    for i, label in enumerate(labels):
        threshold = thresholds.get(label, 0.5)
        predictions[label] = {
            'score': float(scores[i]),
            'passed': scores[i] >= threshold,
            'threshold': threshold
        }
    
    return predictions

# Function to analyze and provide feedback
def analyze_user_story(user_story, predictions):
    analysis = {}
    
    # Check common patterns of a good user story
    user_story_pattern = re.compile(r'As a\s+(.+?),\s+I want\s+(.+?)\s+so that\s+(.+)', re.IGNORECASE)
    match = user_story_pattern.search(user_story)
    
    if match:
        analysis['format'] = {
            'passed': True,
            'message': 'Your user story follows the standard format: "As a [role], I want [goal] so that [benefit]"'
        }
        analysis['role'] = match.group(1)
        analysis['goal'] = match.group(2)
        analysis['benefit'] = match.group(3)
    else:
        analysis['format'] = {
            'passed': False,
            'message': 'Your user story does not follow the standard format: "As a [role], I want [goal] so that [benefit]"'
        }
    
    # Analyze each INVEST criterion
    for criterion, result in predictions.items():
        analysis[criterion] = {
            'passed': result['passed'],
            'score': result['score'],
            'threshold': result['threshold'],
            'name': INVEST_DESCRIPTIONS[criterion]['name'],
            'description': INVEST_DESCRIPTIONS[criterion]['description'],
            'suggestions': INVEST_DESCRIPTIONS[criterion]['suggestions'] if not result['passed'] else []
        }
    
    # Calculate overall score (average of all criteria)
    scores = [predictions[criterion]['score'] for criterion in "INVEST"]
    analysis['overall_score'] = sum(scores) / len(scores)
    
    # Count passed criteria
    passed_count = sum(1 for criterion in "INVEST" if predictions[criterion]['passed'])
    analysis['passed_count'] = passed_count
    analysis['total_count'] = len("INVEST")
    
    return analysis

# Generate a suggestion to improve the user story
def suggest_improvement(analysis, user_story):
    suggestions = []
    
    # Suggest format if needed
    if not analysis['format']['passed']:
        suggestions.append("Rewrite your user story using the format: 'As a [role], I want [goal] so that [benefit]'")
    
    # Add suggestions for failed criteria
    failed_criteria = [criterion for criterion in "INVEST" if criterion in analysis and not analysis[criterion]['passed']]
    
    if failed_criteria:
        for criterion in failed_criteria:
            criterion_name = INVEST_DESCRIPTIONS[criterion]['name']
            random_suggestion = np.random.choice(INVEST_DESCRIPTIONS[criterion]['suggestions'])
            suggestions.append(f"To improve {criterion_name}: {random_suggestion}. Example: {INVEST_DESCRIPTIONS[criterion]['examples']['good']}")
    
    if suggestions:
        return "Your user story could be improved:\n- " + "\n- ".join(suggestions)
    else:
        return "Great job! Your user story meets all INVEST criteria."

# Function to initialize session state variables
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None
    if 'show_batch_results' not in st.session_state:
        st.session_state.show_batch_results = False

# Function to save results to history
def save_to_history(user_story, analysis):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract scores for each criterion
    scores = {criterion: analysis[criterion]['score'] for criterion in "INVEST"}
    
    # Create history entry
    history_entry = {
        'timestamp': timestamp,
        'user_story': user_story,
        'overall_score': analysis['overall_score'],
        'passed_count': analysis['passed_count'],
        'total_count': analysis['total_count'],
        'format_passed': analysis['format']['passed'],
        'scores': scores
    }
    
    # Add to history
    st.session_state.history.append(history_entry)
    
    # Keep only the last 50 entries to avoid excessive memory usage
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[-50:]

# Function to export history to CSV
def export_history_to_csv():
    if not st.session_state.history:
        return None
    
    csv_buffer = StringIO()
    csv_writer = csv.writer(csv_buffer)
    
    # Write header
    header = ["Timestamp", "User Story", "Overall Score", "Passed Criteria", "Format Passed", 
              "I Score", "N Score", "V Score", "E Score", "S Score", "T Score"]
    csv_writer.writerow(header)
    
    # Write data
    for entry in st.session_state.history:
        row = [
            entry['timestamp'],
            entry['user_story'],
            f"{entry['overall_score']:.2f}",
            f"{entry['passed_count']}/{entry['total_count']}",
            "Yes" if entry['format_passed'] else "No"
        ]
        
        # Add individual scores
        for criterion in "INVEST":
            row.append(f"{entry['scores'][criterion]:.2f}")
        
        csv_writer.writerow(row)
    
    return csv_buffer.getvalue()

# Function to process batch of user stories
def process_batch(user_stories, model, tokenizer, thresholds, device):
    results = []
    
    for story in user_stories:
        # Skip empty stories
        if not story.strip():
            continue
            
        # Predict and analyze
        predictions = predict_invest_criteria(story, model, tokenizer, thresholds, device)
        analysis = analyze_user_story(story, predictions)
        suggestion = suggest_improvement(analysis, story)
        
        # Create result entry
        result = {
            'user_story': story,
            'analysis': analysis,
            'suggestion': suggestion
        }
        
        # Add to results
        results.append(result)
        
        # Add to history
        save_to_history(story, analysis)
    
    return results

# Function to parse CSV or text input for batch processing
def parse_batch_input(file_content):
    try:
        # Try parsing as CSV
        df = pd.read_csv(StringIO(file_content))
        
        # Check if there's a column that might contain user stories
        possible_columns = ['user_story', 'story', 'description', 'text', 'content']
        found_column = None
        
        for col in possible_columns:
            if col in df.columns:
                found_column = col
                break
        
        if found_column:
            return df[found_column].fillna('').tolist()
        else:
            # If no specific column found, use the first column
            return df.iloc[:, 0].fillna('').tolist()
    except:
        # If not CSV, treat as plain text with one story per line
        return [line.strip() for line in file_content.strip().split('\n') if line.strip()]

# Main Streamlit app
def main():
    # Initialize session state
    initialize_session_state()
    
    st.title("INVEST User Story Evaluator")
    
    # Create tabs for different features
    tab1, tab2, tab3 = st.tabs(["Single Evaluation", "Batch Processing", "History"])
    
    # Load model and tokenizer
    with st.spinner("Loading model..."):
        model, tokenizer, thresholds, device = load_model()
    
    # Tab 1: Single Evaluation
    with tab1:
        st.markdown("""
        ### Evaluate a single user story based on INVEST criteria
        
        INVEST is an acronym that defines a set of criteria to assess the quality of a user story:
        - **I**ndependent
        - **N**egotiable
        - **V**aluable
        - **E**stimable
        - **S**mall
        - **T**estable
        
        Enter your user story below to evaluate how well it meets these criteria and get suggestions for improvement.
        """)
        
        # User story input
        user_story = st.text_area(
            "Enter your user story:",
            value="As a user, I want to log in so that I can access my account.",
            height=120,
            key="single_story_input"
        )
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Evaluate User Story", key="evaluate_single"):
                if user_story.strip():
                    with st.spinner("Analyzing your user story..."):
                        # Predict INVEST criteria
                        predictions = predict_invest_criteria(user_story, model, tokenizer, thresholds, device)
                        
                        # Analyze the user story
                        analysis = analyze_user_story(user_story, predictions)
                        
                        # Get suggestions for improvement
                        suggestion = suggest_improvement(analysis, user_story)
                        
                        # Save to history
                        save_to_history(user_story, analysis)
                        
                        # Store in session state for display
                        st.session_state.single_analysis = analysis
                        st.session_state.single_suggestion = suggestion
                        st.session_state.show_single_results = True
                else:
                    st.error("Please enter a user story to evaluate.")
        
        # Display evaluation results (if available)
        if st.session_state.get('show_single_results', False):
            analysis = st.session_state.single_analysis
            suggestion = st.session_state.single_suggestion
            
            st.markdown("### Evaluation Results")
            
            # Create a progress bar for overall score
            overall_score = analysis['overall_score']
            st.markdown(f"**Overall Score: {overall_score:.2f}/1.00**")
            st.progress(overall_score)
            st.markdown(f"**Criteria Passed: {analysis['passed_count']}/{analysis['total_count']}**")
            
            # Display format analysis
            if analysis['format']['passed']:
                st.success(analysis['format']['message'])
                st.markdown(f"**Role:** {analysis.get('role', 'Not identified')}")
                st.markdown(f"**Goal:** {analysis.get('goal', 'Not identified')}")
                st.markdown(f"**Benefit:** {analysis.get('benefit', 'Not identified')}")
            else:
                st.error(analysis['format']['message'])
            
            # Display INVEST criteria results
            st.markdown("### INVEST Criteria Scores")
            
            cols = st.columns(3)
            for i, criterion in enumerate("INVEST"):
                if criterion in analysis:
                    col = cols[i % 3]
                    with col:
                        st.markdown(f"**{analysis[criterion]['name']}**")
                        score = analysis[criterion]['score']
                        threshold = analysis[criterion]['threshold']
                        
                        # Progress bar for score
                        progress_color = "green" if analysis[criterion]['passed'] else "red"
                        st.progress(score)
                        st.caption(f"Score: {score:.2f} | Threshold: {threshold:.2f}")
                        
                        if analysis[criterion]['passed']:
                            st.success("Passed ✓")
                        else:
                            st.error("Failed ✗")
            
            # Display suggestions
            st.markdown("### Suggestions for Improvement")
            st.markdown(suggestion)
            
            # Show detailed analysis in an expandable section
            with st.expander("Detailed Analysis"):
                for criterion in "INVEST":
                    if criterion in analysis:
                        st.markdown(f"#### {analysis[criterion]['name']}")
                        st.markdown(analysis[criterion]['description'])
                        
                        if not analysis[criterion]['passed']:
                            st.markdown("**Suggestions:**")
                            for sugg in analysis[criterion]['suggestions']:
                                st.markdown(f"- {sugg}")
                            
                            st.markdown("**Good Example:**")
                            st.markdown(f"> {INVEST_DESCRIPTIONS[criterion]['examples']['good']}")
                            
                            st.markdown("**Bad Example:**")
                            st.markdown(f"> {INVEST_DESCRIPTIONS[criterion]['examples']['bad']}")
            
            # Provide an enhanced version if there are failures
            if any(not analysis[c]['passed'] for c in "INVEST" if c in analysis) or not analysis['format']['passed']:
                with st.expander("Get an Enhanced User Story"):
                    st.markdown("Based on the analysis, here's a suggested improved version of your user story:")
                    
                    # Generate enhanced user story (placeholder - you could implement a more sophisticated version)
                    if not analysis['format']['passed']:
                        # Basic format correction
                        st.markdown("> As a [user role], I want to [goal/action] so that [benefit/value].")
                        st.markdown("(Fill in the appropriate details based on your specific requirements)")
                    else:
                        # Keep the same story but highlight that improvements should be made according to suggestions
                        st.markdown(f"> {user_story}")
                        st.caption("Modify the user story based on the suggestions above to improve its quality.")
    
    # Tab 2: Batch Processing
    with tab2:
        st.markdown("""
        ### Batch Process Multiple User Stories
        
        Upload a CSV file or paste multiple user stories to evaluate them all at once.
        Each story will be analyzed against the INVEST criteria and saved to your history.
        """)
        
        upload_method = st.radio(
            "Choose input method:",
            ["Upload CSV", "Paste Text"],
            key="batch_input_method"
        )
        
        batch_input = None
        
        if upload_method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV file with user stories", type=["csv"])
            if uploaded_file is not None:
                batch_input = uploaded_file.getvalue().decode()
                st.success(f"File uploaded successfully")
        else:  # Paste Text
            batch_input = st.text_area(
                "Paste multiple user stories (one per line):",
                height=150,
                key="batch_text_input"
            )
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Process Batch", key="process_batch"):
                if batch_input and batch_input.strip():
                    with st.spinner("Processing batch of user stories..."):
                        # Parse input
                        user_stories = parse_batch_input(batch_input)
                        
                        if not user_stories:
                            st.error("No valid user stories found in the input.")
                        else:
                            # Process batch
                            results = process_batch(user_stories, model, tokenizer, thresholds, device)
                            
                            # Store in session state
                            st.session_state.batch_results = results
                            st.session_state.show_batch_results = True
                            
                            st.success(f"Processed {len(results)} user stories successfully.")
                else:
                    st.error("Please provide input for batch processing.")
        
        # Display batch results (if available)
        if st.session_state.get('show_batch_results', False) and st.session_state.batch_results:
            results = st.session_state.batch_results
            
            st.markdown("### Batch Processing Results")
            st.markdown(f"Processed {len(results)} user stories")
            
            # Create dataframe for summary
            summary_data = []
            for i, result in enumerate(results):
                analysis = result['analysis']
                story_text = result['user_story']
                
                # Truncate long stories for display
                if len(story_text) > 50:
                    display_text = story_text[:47] + "..."
                else:
                    display_text = story_text
                
                # Calculate pass/fail status
                status = []
                for criterion in "INVEST":
                    if analysis[criterion]['passed']:
                        status.append(criterion)
                
                status_str = ", ".join(status)
                
                summary_data.append({
                    "No.": i+1,
                    "User Story": display_text,
                    "Overall Score": f"{analysis['overall_score']:.2f}",
                    "Passed": f"{analysis['passed_count']}/{analysis['total_count']}",
                    "Criteria Passed": status_str
                })
            
            # Display summary dataframe
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                csv_data = StringIO()
                summary_df.to_csv(csv_data, index=False)
                st.download_button(
                    label="Download Summary CSV",
                    data=csv_data.getvalue(),
                    file_name="invest_batch_summary.csv",
                    mime="text/csv"
                )
            
            # Detailed results in expander
            with st.expander("View Detailed Results"):
                for i, result in enumerate(results):
                    st.markdown(f"### Story {i+1}")
                    st.markdown(f"> {result['user_story']}")
                    
                    analysis = result['analysis']
                    
                    # Format check
                    if analysis['format']['passed']:
                        st.success("Format: Passed ✓")
                    else:
                        st.error("Format: Failed ✗")
                    
                    # Criteria results
                    cols = st.columns(6)
                    for j, criterion in enumerate("INVEST"):
                        with cols[j]:
                            st.markdown(f"**{criterion}**")
                            score = analysis[criterion]['score']
                            if analysis[criterion]['passed']:
                                st.success(f"{score:.2f} ✓")
                            else:
                                st.error(f"{score:.2f} ✗")
                    
                    # Suggestions
                    st.markdown("**Suggestions:**")
                    st.markdown(result['suggestion'])
                    st.divider()
    
    # Tab 3: History
    with tab3:
        st.markdown("""
        ### Evaluation History
        
        View the history of your previously evaluated user stories.
        The history is preserved during your current session.
        """)
        
        if not st.session_state.history:
            st.info("No evaluation history yet. Evaluate some user stories to see them here.")
        else:
            # Display history count
            st.markdown(f"**{len(st.session_state.history)} evaluations in history**")
            
            # Create history dataframe
            history_data = []
            for i, entry in enumerate(st.session_state.history):
                # Truncate long stories for display
                story_text = entry['user_story']
                if len(story_text) > 50:
                    display_text = story_text[:47] + "..."
                else:
                    display_text = story_text
                
                # Format scores
                scores_formatted = ", ".join([f"{c}: {entry['scores'][c]:.2f}" for c in "INVEST"])
                
                history_data.append({
                    "Time": entry['timestamp'],
                    "User Story": display_text,
                    "Overall Score": f"{entry['overall_score']:.2f}",
                    "Passed": f"{entry['passed_count']}/{entry['total_count']}",
                    "Format": "✓" if entry['format_passed'] else "✗",
                    "Scores": scores_formatted
                })
            
            # Display history table
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
            
            # Export history
            csv_export = export_history_to_csv()
            if csv_export:
                st.download_button(
                    label="Export History to CSV",
                    data=csv_export,
                    file_name=f"invest_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Clear history button
            if st.button("Clear History"):
                st.session_state.history = []
                st.experimental_rerun()

if __name__ == "__main__":
    main()