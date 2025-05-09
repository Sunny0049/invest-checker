import pandas as pd
import datetime


print(f"Start Time: {datetime.datetime.now()}")

# Load your CSV file
filepath= "D:\\Program Files\\Apache Software Foundation\\Apache24\\0_Backup_projects\\Projects\\invest\\01-US-Backup\\09_invest_user_stories.xlsx"
df = pd.read_excel(filepath)


# Check for exact duplicates based on raw 'story' column
duplicates = df[df.duplicated(subset='story', keep=False)]

# Print duplicates for manual verification
print(f"Found {len(duplicates)} duplicate entries:")
print(duplicates[['story']])

print(f"End Time: {datetime.datetime.now()}")

# Optional: remove duplicates, keeping the first occurrence
#df_unique = df.drop_duplicates(subset='story', keep='first')

# Save cleaned file
#df_unique.to_csv("cleaned_user_stories.csv", index=False)
#print("Saved cleaned user stories to 'cleaned_user_stories.csv'")
