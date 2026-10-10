import os
from groq import Groq

# Initialize Groq client
# This requires the GROQ_API_KEY environment variable to be set
try:
    client = Groq()
except Exception:
    client = None

def generate_drift_explanation(mhs: float, drift_confirmed: bool, alert_level: str, metrics: dict) -> str:
    """
    Calls the Groq API to generate a plain-English explanation of the drift.
    """
    if not client:
        return "Groq API key not configured. Unable to generate AI explanation."
    
    if alert_level.lower() == "none" or not drift_confirmed:
        return "Model behavior is normal. No significant drift detected."

    prompt = f"""
    You are an AI assistant for a Model Decay Radar platform. Your job is to explain machine learning drift to a non-technical user in a short, easy-to-understand paragraph.
    
    Current System Status:
    - Model Health Score (MHS): {mhs:.2f}/1.0
    - Alert Level: {alert_level}
    - Drift Confirmed: {drift_confirmed}
    
    Detailed Metrics:
    {metrics}
    
    Explain what might have caused this and what the user should do. Keep it under 3 sentences and professional but accessible.
    """
    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful data science assistant."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=150,
        )
        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        print(f"Error generating Groq explanation: {e}")
        return "Failed to generate AI explanation due to an API error."
