"""
Quick test script to verify backend is working
"""
import backend
import os

print("Testing backend setup...\n")

# Test 1: Check environment
print("1. Checking GEMINI_API_KEY...")
if os.getenv("GEMINI_API_KEY"):
    print("   ✅ API Key found")
else:
    print("   ❌ API Key not found in environment")
    print("   Create a .env file with: GEMINI_API_KEY=your_key")

# Test 2: Check FAISS files
print("\n2. Checking FAISS database files...")
if os.path.exists("D:/tutorial/vectorstores/db_faiss/index.faiss"):
    print("   ✅ index.faiss exists")
else:
    print("   ❌ index.faiss not found")
    print("   Run: python create_db.py")

if os.path.exists("D:/tutorial/vectorstores/db_faiss/texts.npy"):
    print("   ✅ texts.npy exists")
else:
    print("   ❌ texts.npy not found")
    print("   Run: python create_db.py")

# Test 3: Try a simple query
print("\n3. Testing backend query...")
try:
    response = backend.get_response("What is the leave policy?")
    print(f"   ✅ Backend response: {response[:100]}...")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n✅ All tests completed!")