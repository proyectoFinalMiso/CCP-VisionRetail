import json
import requests

API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA1MzM2MTE1NywiaWF0IjoxNzQ2MTYxMTU3LCJqdGkiOiI3YzMwZTMxNjQwZGU0OWMxYWIwNGEyYjkyZGJlNTlkMSIsInVzZXJfaWQiOjF9.Q6zXHHtZ96JBcUxcWvohFBN8JMf7IBdDZOZ0oKsfNWg'
PROJECT_ID = 2
FILE = 'label_studio_tasks.json'  # Or use a split file

URL = f'http://localhost:8080/api/projects/{PROJECT_ID}/import'
HEADERS = {
    'Authorization': f'Token {API_KEY}'
}

with open(FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

response = requests.post(URL, headers=HEADERS, json=data)

if response.status_code == 200:
    print("✅ Import successful!")
else:
    print(f"❌ Failed: {response.status_code}")
    print(response.text)