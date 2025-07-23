#!/usr/bin/env python3
"""
Simple test of question classification using Azure OpenAI without structured output.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_classification():
    """Test question classification with Azure OpenAI."""
    
    # Configure client for Azure OpenAI
    client = OpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        base_url="https://aiendpointsswe5672148277.cognitiveservices.azure.com/openai/deployments/gpt-4.1-mini-gobal/",
        default_query={"api-version": "2025-01-01-preview"}
    )
    
    # Test questions
    test_questions = [
        "Find the characteristic of the ring 2Z.",  # Should be answerable in free-form
        "Which of the following statements is true?",  # Should NOT be answerable in free-form
        "What is the capital of France?",  # Should be answerable in free-form
        "Select the best answer from the options below:"  # Should NOT be answerable in free-form
    ]
    
    classification_prompt = """You are analyzing a question to determine if it can be answered in free-form format without seeing multiple choice answer options.

A question is answerable in free-form if:
1. It is self-contained and doesn't reference answer choices
2. It can be answered with specific knowledge without needing to compare given options
3. It doesn't require choosing between provided alternatives

A question is NOT answerable in free-form if:
1. It references answer options (e.g., "Which of the following...")
2. It asks to select from given choices
3. The answer requires comparing multiple provided options

Please respond with a JSON object containing:
- "is_answerable_freeform": true or false
- "reasoning": your explanation
- "confidence": a number between 0.0 and 1.0

Question to analyze: "{question}"
"""
    
    print("Testing question classification with Azure OpenAI...")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- QUESTION {i} ---")
        print(f"Question: {question}")
        
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini-gobal",
                messages=[
                    {"role": "user", "content": classification_prompt.format(question=question)}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content
            print(f"Raw response: {result_text}")
            
            # Try to parse JSON
            try:
                result = json.loads(result_text)
                print(f"✅ Answerable in free-form: {result.get('is_answerable_freeform')}")
                print(f"📝 Reasoning: {result.get('reasoning')}")
                print(f"🎯 Confidence: {result.get('confidence')}")
            except json.JSONDecodeError:
                print("⚠️  Could not parse JSON response")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_classification()