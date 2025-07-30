import google.generativeai as genai

genai.configure(api_key="your_actual_key")

model = genai.GenerativeModel("gemini-pro")
response = model.generate_content("Explain deductible in health insurance.")
print(response.text)