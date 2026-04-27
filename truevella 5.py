import torch
import torch.nn as nn
import spacy
import re

class UnbiasedScorer:
    def __init__(self):
        # Load spaCy for Named Entity Recognition (Anonymization)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # PyTorch Tensor defining the exact weighted criteria
        # Skills (30%), Experience (25%), Education (15%), Achievements (20%), Comm (10%)
        self.weights = torch.tensor([0.30, 0.25, 0.15, 0.20, 0.10], dtype=torch.float32)

    def anonymize_text(self, text):
        """Removes Names, Genders, and Religions to prevent bias."""
        doc = self.nlp(text)
        clean_text = text
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "NORP", "GPE"]: # Removes names, nationalities, locations (like Agra or Dehradun)
                clean_text = clean_text.replace(ent.text, "[REDACTED]")
        
        # Strip common gendered pronouns
        clean_text = re.sub(r'\b(he|him|his|she|her|hers)\b', '[PRONOUN]', clean_text, flags=re.IGNORECASE)
        return clean_text

    def extract_features(self, text):
        """
        In production, use a fine-tuned BERT model to score these out of 100.
        For this deployable architecture, we use keyword density as a proxy metric.
        """
        text = text.lower()
        skills_score = min(100, text.count("python")*10 + text.count("react")*10 + text.count("manage")*5 + 50)
        exp_score = min(100, text.count("year")*10 + text.count("developed")*10 + 40)
        edu_score = min(100, text.count("degree")*20 + text.count("university")*20 + text.count("school")*15 + 40)
        achieve_score = min(100, text.count("award")*25 + text.count("won")*20 + 30)
        comm_score = 85 # Baseline clarity score for parsed text

        return torch.tensor([skills_score, exp_score, edu_score, achieve_score, comm_score], dtype=torch.float32)

    def calculate_score(self, feature_tensor):
        """Uses PyTorch dot product to calculate the final weighted evaluation."""
        final_score = torch.dot(feature_tensor, self.weights).item()
        breakdown = {
            "Skills (30%)": feature_tensor[0].item(),
            "Experience (25%)": feature_tensor[1].item(),
            "Education (15%)": feature_tensor[2].item(),
            "Achievements (20%)": feature_tensor[3].item(),
            "Communication (10%)": feature_tensor[4].item()
        }
        return final_score, breakdown