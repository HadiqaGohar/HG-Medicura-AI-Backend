#!/usr/bin/env python3

# test_gemini.py
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("🔍 Testing Gemini API Configuration")
print("=" * 50)
print(f"API Key: {GEMINI_API_KEY[:10]}...{GEMINI_API_KEY[-5:] if GEMINI_API_KEY else 'None'}")

# Test direct Gemini API call
def test_direct_gemini():
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": "Hello, this is a test message."
            }]
        }]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"\n📡 Direct Gemini API Test:")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API Key is valid!")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

# Test OpenAI-compatible endpoint
def test_openai_compatible():
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    
    payload = {
        "model": "gemini-2.0-flash",
        "messages": [
            {"role": "user", "content": "Hello, this is a test."}
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"\n🔄 OpenAI-Compatible API Test:")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ OpenAI-compatible endpoint works!")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        print("❌ No valid API key found!")
        print("Please set GEMINI_API_KEY in your .env file")
    else:
        direct_works = test_direct_gemini()
        openai_works = test_openai_compatible()
        
        print(f"\n📊 Results:")
        print(f"Direct Gemini API: {'✅' if direct_works else '❌'}")
        print(f"OpenAI Compatible: {'✅' if openai_works else '❌'}")
        
        if not direct_works and not openai_works:
            print("\n💡 Suggestions:")
            print("1. Check if your API key is correct")
            print("2. Verify API key permissions")
            print("3. Try generating a new API key")
            print("4. Check if Gemini API is enabled in your Google Cloud project")