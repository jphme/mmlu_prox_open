#!/usr/bin/env python3
"""
Simple test to verify Azure OpenAI configuration works with the standard OpenAI API.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables
load_dotenv()


class BasicIntAnswer(BaseModel):
    answer: int


def test_azure_openai():
    """Test Azure OpenAI endpoint with standard OpenAI client."""

    # Check if API key is available
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("Error: AZURE_OPENAI_API_KEY not found in environment variables")
        return

    # Configure client for Azure OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://aiendpointsswe5672148277.cognitiveservices.azure.com/openai/deployments/gpt-4.1-mini-gobal/",
        default_query={"api-version": "2025-01-01-preview"},
    )

    # Test message
    test_message = "Hello! Can you tell me what 2+2 equals?"

    print("Testing Azure OpenAI connection...")
    print("Endpoint: https://aiendpointsswe5672148277.cognitiveservices.azure.com/")
    print("Model: gpt-4.1-mini-gobal")
    print("API Version: 2025-01-01-preview")
    print(f"Test message: {test_message}")
    print("-" * 50)

    try:
        response = client.chat.completions.parse(
            model="gpt-4.1-mini-gobal",  # This should match the deployment name
            messages=[{"role": "user", "content": test_message}],
            max_tokens=100,
            temperature=0.1,
            response_format=BasicIntAnswer,
        )

        print("✅ SUCCESS! Azure OpenAI is working correctly.")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")
        print(f"Tokens used: {response.usage.total_tokens}")

    except Exception as e:
        print("❌ ERROR: Failed to connect to Azure OpenAI")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_azure_openai()
