from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


# Resultaterne fra modellernes klassifikationsrapport og confusion matrix
prompt = """
The classification reports of the two models are as follows:

Random Forest:
- Confusion Matrix: [[79, 20], [18, 37]]
- Precision for class 0: 0.81, class 1: 0.65
- Recall for class 0: 0.80, class 1: 0.67
- F1-score for class 0: 0.81, class 1: 0.66
- Accuracy: 0.75
- Macro Avg F1-score: 0.73
- Weighted Avg F1-score: 0.75

XGBoost:
- Confusion Matrix: [[78, 21], [17, 38]]
- Precision for class 0: 0.82, class 1: 0.64
- Recall for class 0: 0.79, class 1: 0.69
- F1-score for class 0: 0.80, class 1: 0.67
- Accuracy: 0.75
- Macro Avg F1-score: 0.74
- Weighted Avg F1-score: 0.76

Generate a comparative analysis of the performance of both models based on these metrics.
"""


response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a data scientist that explains model performance clearly, in a fun and engaging way."},
        {"role": "user", "content": prompt}
    ]
)


print(response.choices[0].message['content'])
