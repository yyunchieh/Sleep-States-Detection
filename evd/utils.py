import torch
import torch.nn.functional as F

def calculate_confidence_score(output):
  
    probabilities = F.softmax(output, dim=-1)
    confidence_score = torch.max(probabilities).item()  
    return confidence_score