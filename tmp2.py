import pandas as pd

# Create sample user stories related to the leasing industry with INVEST criteria
leasing_user_stories = [
    {
        "user_story": "As a customer, I want to view available leasing options for cars based on my budget. acceptance_criteria: Leasing options must be filtered by price range and availability.",
        "I": 1, "N": 1, "V": 1, "E": 1, "S": 1, "T": 1
    },
    {
        "user_story": "As a leasing agent, I want to generate lease agreement documents automatically.acceptance_criteria: The agreement should include customer and vehicle details and monthly terms.",
        "I": 1, "N": 1, "V": 1, "E": 1, "S": 1, "T": 1
    },
    {
        "user_story": "As a user, I want to compare leasing vs buying options. acceptance_criteria: The system displays monthly payments and total cost over time.",
        "I": 1, "N": 1, "V": 1, "E": 1, "S": 1, "T": 1
    },
    {
        "user_story": "As a customer, I want to lease a vehicle without submitting all documents again. acceptance_criteria: If already verified, users can skip redundant submissions.",
        "I": 1, "N": 1, "V": 1, "E": 1, "S": 0, "T": 1
    },
    {
        "user_story": "As a finance manager, I want to get notified if lease payments are overdue. acceptance_criteria: Notifications should trigger after 3 missed payments.",
        "I": 1, "N": 1, "V": 1, "E": 1, "S": 0, "T": 1
    },
    {
        "user_story": "As a leasing partner, I want to upload new vehicle inventory in bulk. acceptance_criteria: A CSV upload should update multiple records at once.",
        "I": 1, "N": 0, "V": 1, "E": 1, "S": 1, "T": 1
    },
    {
        "user_story": "As a customer, I want to terminate my lease early online. acceptance_criteria: Early termination fee and final billing should be shown.",
        "I": 1, "N": 1, "V": 1, "E": 1, "S": 1, "T": 1
    },
    {
        "user_story": "As a user, I want to log in. acceptance_criteria: ",
        "I": 1, "N": 1, "V": 0, "E": 0, "S": 1, "T": 1
    },
    {
        "user_story": "As a system admin, I want to archive old lease contracts acceptance_criteria: Contracts older than 7 years should be archived automatically.",
        "I": 1, "N": 0, "V": 1, "E": 1, "S": 1, "T": 1
    },
    {
        "user_story": "As a leasing agent, I want to resend leasing agreements to customers.acceptance_criteria: Agreements must be resent with version history.",
        "I": 1, "N": 1, "V": 1, "E": 1, "S": 0, "T": 1
    },
    {
        "user_story": "As a customer, I want to select lease terms by kilometers driven annually. acceptance_criteria: Lease pricing should update dynamically based on selection.",
        "I": 1, "N": 1, "V": 1, "E": 1, "S": 1, "T": 1
    },
    {
        "user_story": "As a user, I want to know what is leasing. acceptance_criteria: Basic explanation must be displayed on request.",
        "I": 0, "N": 1, "V": 0, "E": 1, "S": 1, "T": 1
    },
    {
        "user_story": "As a business client, I want to lease multiple vehicles in one request. acceptance_criteria: Form should allow adding multiple vehicle preferences.",
        "I": 1, "N": 1, "V": 1, "E": 1, "S": 0, "T": 1
    },
    {
        "user_story": "As a customer, I want a reminder before my lease ends. acceptance_criteria: Reminders sent 30 days before lease expiration.",
        "I": 1, "N": 1, "V": 1, "E": 1, "S": 1, "T": 1
    },
    {
        "user_story": "I want lease calculator acceptance_criteria: ",
        "I": 0, "N": 0, "V": 0, "E": 0, "S": 1, "T": 0
    },
]

# Convert to DataFrame
df_leasing = pd.DataFrame(leasing_user_stories)

# Save to CSV
csv_path = "leasing_user_stories_invest.csv"
df_leasing.to_csv(csv_path, index=False)

csv_path
