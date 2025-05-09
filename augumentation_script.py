#!D:/Program Files/Apache Software Foundation/Apache24/cgi-bin/SimpleFlaskApp/venv/Scripts/python.exe

import pandas as pd
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import torch

# Load paraphrasing model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
para_model_name = "Vamsi/T5_Paraphrase_Paws"
para_tokenizer = T5Tokenizer.from_pretrained(para_model_name, legacy=False)
para_model = AutoModelForSeq2SeqLM.from_pretrained(para_model_name).to(device)

# Paraphrasing function
def paraphrase(text, num_return_sequences=3):
    input_text = f"paraphrase: {text} </s>"
    encoding = para_tokenizer.encode_plus(
        input_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length"
    ).to(device)
    
    outputs = para_model.generate(
        **encoding,
        max_length=256,
        num_return_sequences=num_return_sequences,
        num_beams=4,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.5,
        early_stopping=True
    )
    
    return list(set([para_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]))

# Load your original labeled dataset (example CSV)
df = pd.read_csv("08_invest_user_stories.csv")  # should include 'story', 'I', 'N', 'V', 'E', 'S', 'T'

# Augment data
augmented_rows = []

for idx, row in df.iterrows():
    try:
        original_text = row['story']
        labels = row[['I', 'N', 'V', 'E', 'S', 'T']].to_dict()
        paraphrases = paraphrase(original_text, num_return_sequences=3)
        
        for new_text in paraphrases:
            if new_text.strip() and new_text != original_text:
                new_row = {"story": new_text, **labels}
                augmented_rows.append(new_row)
    except Exception as e:
        print(f"Skipping row {idx} due to error: {e}")

# Create augmented DataFrame
augmented_df = pd.DataFrame(augmented_rows)

# Merge and shuffle
final_df = pd.concat([df, augmented_df], ignore_index=True).sample(frac=1).reset_index(drop=True)

# Save to file
final_df.to_csv("augmented_invest_dataset.csv", index=False)
