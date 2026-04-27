"""
Test script to verify API connectivity - Direct approach
"""

print("=" * 60)
print("TESTING LLM APIS")
print("=" * 60)

# Read from secrets.toml directly
import configparser
import os

config = configparser.ConfigParser()
secrets_path = r'c:\Users\isra9\Desktop\ai-bias-app\.streamlit\secrets.toml'

# Parse TOML manually (simple version)
secrets = {}
if os.path.exists(secrets_path):
    with open(secrets_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"')
                secrets[key] = value

gemini_key = secrets.get("GEMINI_API_KEY")
groq_key = secrets.get("GROQ_API_KEY")

print(f"\n✓ Gemini Key loaded: {bool(gemini_key)}")
if gemini_key:
    print(f"  Key preview: {gemini_key[:30]}...")
print(f"✓ Groq Key loaded: {bool(groq_key)}")
if groq_key:
    print(f"  Key preview: {groq_key[:30]}...")

# Test 2: Try Gemini API
print("\n" + "=" * 60)
print("Testing GEMINI API...")
print("=" * 60)

try:
    import google.generativeai as genai
    
    if gemini_key:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        test_prompt = "Say 'Hello from Gemini' in exactly 3 words"
        response = model.generate_content(test_prompt)
        
        print(f"✅ GEMINI WORKS!")
        print(f"Response: {response.text}")
    else:
        print("❌ No Gemini key found")
        
except Exception as e:
    print(f"❌ GEMINI ERROR: {type(e).__name__}")
    print(f"   Details: {str(e)[:200]}")

# Test 3: Try Groq API
print("\n" + "=" * 60)
print("Testing GROQ API...")
print("=" * 60)

try:
    from groq import Groq
    
    if groq_key:
        client = Groq(api_key=groq_key)
        
        test_prompt = "Say 'Hello from Groq' in exactly 3 words"
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": test_prompt}],
            max_tokens=50,
            temperature=0.7
        )
        
        print(f"✅ GROQ WORKS!")
        print(f"Response: {response.choices[0].message.content}")
    else:
        print("❌ No Groq key found")
        
except Exception as e:
    print(f"❌ GROQ ERROR: {type(e).__name__}")
    print(f"   Details: {str(e)[:200]}")

print("\n" + "=" * 60)
print("Testing complete!")
print("=" * 60)
