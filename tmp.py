"""
import pandas as pd

# Define the data
data = [
    # Banking
    ["As a customer, I want to transfer funds between my accounts so that I can manage my finances better. Acceptance: Transfers over $10,000 must be verified by SMS.",1,1,1,1,1,1],
    ["As a customer, I want to view my transaction history.",1,1,1,1,1,1],
    ["As a customer, I want to receive notifications after every transaction. Acceptance: Email and SMS alerts should be sent within 1 minute.",1,1,1,1,1,1],
    ["As a user, I want to reset my password.",1,1,1,1,1,1],
    ["As a bank manager, I want to generate daily transaction reports.",1,1,1,1,1,0],
    ["As a user, I want to apply for a credit card.",1,1,1,1,1,1],
    ["Enable loan calculator on the homepage.",0,0,1,0,1,1],
    ["Fix overdraft issue",0,0,1,0,0,0],
    ["User can log in",0,0,1,0,0,0],
    ["As a user, I want to upload KYC documents. Acceptance: Supported formats are PDF, PNG, JPG; file size limit is 2MB.",1,1,1,1,1,1],
    ["As a user, I want to set up auto-debit for my utility bills. Acceptance: Auto-debit must trigger on the 5th of each month.",1,1,1,1,1,1],
    ["As a user, I want to link my Aadhaar to my account.",1,1,1,1,1,1],
    ["As a user, I want to download my account statement. Acceptance: Statement should be available in PDF and Excel.",1,1,1,1,1,1],
    ["As a user, I want to schedule a fund transfer. Acceptance: Should allow future dates up to 1 year.",1,1,1,1,1,1],
    ["As a customer, I want to request a new cheque book.",1,1,1,1,1,1],
    
    # Insurance
    ["As a policyholder, I want to view my policy documents online. Acceptance: Must be downloadable in PDF.",1,1,1,1,1,1],
    ["As a user, I want to renew my car insurance. Acceptance: Show premium breakdown and confirm before payment.",1,1,1,1,1,1],
    ["As a customer, I want to update my address.",1,1,1,1,1,1],
    ["As an agent, I want to view pending claims.",1,1,1,1,1,1],
    ["As a user, I want to compare health insurance plans.",1,1,1,1,1,1],
    ["Enable chatbot on claims page.",0,0,1,0,1,1],
    ["As a customer, I want to submit a claim for my stolen phone.",1,1,1,1,1,1],
    ["As a user, I want to upload claim documents. Acceptance: Allow JPEG, PNG, and PDF only.",1,1,1,1,1,1],
    ["As a customer, I want to receive SMS updates on claim status.",1,1,1,1,1,1],
    ["As a user, I want to set up a nominee for my policy. Acceptance: Validate relationship and age.",1,1,1,1,1,1],
    ["Process policies faster",0,0,1,0,0,0],
    ["Submit policy info",0,0,1,0,0,0],
    ["As a user, I want to cancel my insurance policy.",1,1,1,1,1,1],
    ["As a user, I want to download premium payment receipts.",1,1,1,1,1,1],
    ["As a user, I want to access policy renewal options. Acceptance: Allow renewals within 30 days of expiry only.",1,1,1,1,1,1],
]

# Create DataFrame
df = pd.DataFrame(data, columns=["user_story", "I", "N", "V", "E", "S", "T"])

# Save to CSV
output_path = ".\\diverse_invest_user_stories.csv"
df.to_csv(output_path, index=False)

output_path

#---------------------------------------------------------------------------
"As a customer, I want to transfer funds between my accounts so that I can manage my finances better. Acceptance: Transfers over $10,000 must be verified by SMS.",1,1,1,1,1,1
"As a customer, I want to view my transaction history.",1,1,1,1,1,1
"As a customer, I want to receive notifications after every transaction. Acceptance: Email and SMS alerts should be sent within 1 minute.",1,1,1,1,1,1
"As a user, I want to reset my password.",1,1,1,1,1,1
"As a bank manager, I want to generate daily transaction reports.",1,1,1,1,1,0
"As a user, I want to apply for a credit card.",1,1,1,1,1,1
"Enable loan calculator on the homepage.",0,0,1,0,1,1
"Fix overdraft issue",0,0,1,0,0,0
"User can log in",0,0,1,0,0,0
"As a user, I want to upload KYC documents. Acceptance: Supported formats are PDF, PNG, JPG; file size limit is 2MB.",1,1,1,1,1,1
"As a user, I want to set up auto-debit for my utility bills. Acceptance: Auto-debit must trigger on the 5th of each month.",1,1,1,1,1,1
"As a user, I want to link my Aadhaar to my account.",1,1,1,1,1,1
"As a user, I want to download my account statement. Acceptance: Statement should be available in PDF and Excel.",1,1,1,1,1,1
"As a user, I want to schedule a fund transfer. Acceptance: Should allow future dates up to 1 year.",1,1,1,1,1,1
"As a customer, I want to request a new cheque book.",1,1,1,1,1,1

#-----------------------------------------------------------------------------
"As a policyholder, I want to view my policy documents online. Acceptance: Must be downloadable in PDF.",1,1,1,1,1,1
"As a user, I want to renew my car insurance. Acceptance: Show premium breakdown and confirm before payment.",1,1,1,1,1,1
"As a customer, I want to update my address.",1,1,1,1,1,1
"As an agent, I want to view pending claims.",1,1,1,1,1,1
"As a user, I want to compare health insurance plans.",1,1,1,1,1,1
"Enable chatbot on claims page.",0,0,1,0,1,1
"As a customer, I want to submit a claim for my stolen phone.",1,1,1,1,1,1
"As a user, I want to upload claim documents. Acceptance: Allow JPEG, PNG, and PDF only.",1,1,1,1,1,1
"As a customer, I want to receive SMS updates on claim status.",1,1,1,1,1,1
"As a user, I want to set up a nominee for my policy. Acceptance: Validate relationship and age.",1,1,1,1,1,1
"Process policies faster",0,0,1,0,0,0
"Submit policy info",0,0,1,0,0,0
"As a user, I want to cancel my insurance policy.",1,1,1,1,1,1
"As a user, I want to download premium payment receipts.",1,1,1,1,1,1
"As a user, I want to access policy renewal options. Acceptance: Allow renewals within 30 days of expiry only.",1,1,1,1,1,1
"""
#----------------------------------------------------------------
"""
import pandas as pd

# Re-defining the user stories after kernel reset
more_user_stories = [
    {
        "User Story": "As a customer, I want to receive an SMS alert for each transaction.",
        "Acceptance Criteria": "An SMS is sent immediately after each debit or credit over $10.",
        "INVEST - I": 1, "INVEST - N": 1, "INVEST - V": 1, "INVEST - E": 1, "INVEST - S": 1, "INVEST - T": 1,
        "Note": ""
    },
    {
        "User Story": "As a user, I want to check my loan eligibility.",
        "Acceptance Criteria": "Eligibility check must return results in under 5 seconds.",
        "INVEST - I": 1, "INVEST - N": 1, "INVEST - V": 1, "INVEST - E": 1, "INVEST - S": 1, "INVEST - T": 1,
        "Note": ""
    },
    {
        "User Story": "As a policyholder, I want to renew my policy online, so that I don't need to visit the office.",
        "Acceptance Criteria": "",
        "INVEST - I": 1, "INVEST - N": 1, "INVEST - V": 1, "INVEST - E": 0, "INVEST - S": 1, "INVEST - T": 0,
        "Note": "No acceptance criteria; not testable and lacks detail for estimation."
    },
    {
        "User Story": "I want savings.",
        "Acceptance Criteria": "",
        "INVEST - I": 0, "INVEST - N": 0, "INVEST - V": 0, "INVEST - E": 0, "INVEST - S": 0, "INVEST - T": 0,
        "Note": "Too vague and short to evaluate any INVEST criteria."
    },
    {
        "User Story": "As an agent, I want to access all customer data in one click.",
        "Acceptance Criteria": "Agent should see account, transaction, and contact details on a single page.",
        "INVEST - I": 0, "INVEST - N": 1, "INVEST - V": 1, "INVEST - E": 0, "INVEST - S": 1, "INVEST - T": 1,
        "Note": "May depend on other teams/systems; not fully independent or estimable."
    },
    {
        "User Story": "As a banker, I want to view loan application trends to identify growth areas.",
        "Acceptance Criteria": "System must generate trend reports for past 6 months, grouped by region.",
        "INVEST - I": 1, "INVEST - N": 1, "INVEST - V": 1, "INVEST - E": 1, "INVEST - S": 1, "INVEST - T": 1,
        "Note": ""
    },
    {
        "User Story": "As a user, I want to get notified for failed login attempts.",
        "Acceptance Criteria": "Send notification after 3 failed attempts within 10 minutes.",
        "INVEST - I": 1, "INVEST - N": 1, "INVEST - V": 1, "INVEST - E": 1, "INVEST - S": 1, "INVEST - T": 1,
        "Note": ""
    },
    {
        "User Story": "As an underwriter, I want to filter policies by risk category.",
        "Acceptance Criteria": "",
        "INVEST - I": 1, "INVEST - N": 1, "INVEST - V": 0, "INVEST - E": 0, "INVEST - S": 1, "INVEST - T": 0,
        "Note": "Missing acceptance criteria; difficult to validate or test."
    },
    {
        "User Story": "As a user, I want to upload KYC documents.",
        "Acceptance Criteria": "User can upload PDF or JPEG files up to 5MB; rejection message shown otherwise.",
        "INVEST - I": 1, "INVEST - N": 1, "INVEST - V": 1, "INVEST - E": 1, "INVEST - S": 1, "INVEST - T": 1,
        "Note": ""
    },
    {
        "User Story": "As a client, I want to receive annual investment summary.",
        "Acceptance Criteria": "Summary should include gains, losses, and current value by category.",
        "INVEST - I": 1, "INVEST - N": 1, "INVEST - V": 1, "INVEST - E": 1, "INVEST - S": 1, "INVEST - T": 1,
        "Note": ""
    }
]

# Create a DataFrame
df_more_stories = pd.DataFrame(more_user_stories)

# Save to CSV
file_path = ".\\banking_insurance_user_stories.csv"
df_more_stories.to_csv(file_path, index=False)

file_path
"""
#-------------------------------------------------------------------------------------------------------
import pandas as pd

# More diverse user stories in banking and insurance with and without acceptance criteria
more_user_stories = [
    {
        "User Story": "As a customer, I want to receive an SMS alert for each transaction.",
        "Acceptance Criteria": "An SMS is sent immediately after each debit or credit over $10.",
        "INVEST - I": 1, "INVEST - N": 1, "INVEST - V": 1, "INVEST - E": 1, "INVEST - S": 1, "INVEST - T": 1,
        "Note": ""
    },
    {
        "User Story": "As a user, I want to check my loan eligibility.",
        "Acceptance Criteria": "Eligibility check must return results in under 5 seconds.",
        "INVEST - I": 1, "INVEST - N": 1, "INVEST - V": 1, "INVEST - E": 1, "INVEST - S": 1, "INVEST - T": 1,
        "Note": ""
    },
    {
        "User Story": "As a policyholder, I want to renew my policy online, so that I don't need to visit the office.",
        "Acceptance Criteria": "",
        "INVEST - I": 1, "INVEST - N": 1, "INVEST - V": 1, "INVEST - E": 0, "INVEST - S": 1, "INVEST - T": 0,
        "Note": "No acceptance criteria; not testable and lacks detail for estimation."
    },
    {
        "User Story": "I want savings.",
        "Acceptance Criteria": "",
        "INVEST - I": 0, "INVEST - N": 0, "INVEST - V": 0, "INVEST - E": 0, "INVEST - S": 0, "INVEST - T": 0,
        "Note": "Too vague and short to evaluate any INVEST criteria."
    },
    {
        "User Story": "As an agent, I want to access all customer data in one click.",
        "Acceptance Criteria": "Agent should see account, transaction, and contact details on a single page.",
        "INVEST - I": 0, "INVEST - N": 1, "INVEST - V": 1, "INVEST - E": 0, "INVEST - S": 1, "INVEST - T": 1,
        "Note": "May depend on other teams/systems; not fully independent or estimable."
    },
    {
        "User Story": "As a banker, I want to view loan application trends to identify growth areas.",
        "Acceptance Criteria": "System must generate trend reports for past 6 months, grouped by region.",
        "INVEST - I": 1, "INVEST - N": 1, "INVEST - V": 1, "INVEST - E": 1, "INVEST - S": 1, "INVEST - T": 1,
        "Note": ""
    },
    {
        "User Story": "As a user, I want to get notified for failed login attempts.",
        "Acceptance Criteria": "Send notification after 3 failed attempts within 10 minutes.",
        "INVEST - I": 1, "INVEST - N": 1, "INVEST - V": 1, "INVEST - E": 1, "INVEST - S": 1, "INVEST - T": 1,
        "Note": ""
    },
    {
        "User Story": "As an underwriter, I want to filter policies by risk category.",
        "Acceptance Criteria": "",
        "INVEST - I": 1, "INVEST - N": 1, "INVEST - V": 0, "INVEST - E": 0, "INVEST - S": 1, "INVEST - T": 0,
        "Note": "Missing acceptance criteria; difficult to validate or test."
    },
    {
        "User Story": "As a user, I want to upload KYC documents.",
        "Acceptance Criteria": "User can upload PDF or JPEG files up to 5MB; rejection message shown otherwise.",
        "INVEST - I": 1, "INVEST - N": 1, "INVEST - V": 1, "INVEST - E": 1, "INVEST - S": 1, "INVEST - T": 1,
        "Note": ""
    },
    {
        "User Story": "As a client, I want to receive annual investment summary.",
        "Acceptance Criteria": "Summary should include gains, losses, and current value by category.",
        "INVEST - I": 1, "INVEST - N": 1, "INVEST - V": 1, "INVEST - E": 1, "INVEST - S": 1, "INVEST - T": 1,
        "Note": ""
    }
]

# Convert to DataFrame
df_more_stories = pd.DataFrame(more_user_stories)

# Save as CSV
file_path = ".\\02banking_insurance_user_stories.csv"
df_more_stories.to_csv(file_path, index=False)

file_path
