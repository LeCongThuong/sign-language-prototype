import os
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv('.env')
import json
from tqdm import tqdm

class TextByGemini:
    def __init__(self) -> None:
        print(os.getenv('GOOGLE_API_KEY'))
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-pro')

    def generate_text(self, prompt):
        return self.model.generate_content(prompt,    
                                           generation_config=genai.types.GenerationConfig(
                                            # Only one candidate for now.
                                            candidate_count=1,
                                            temperature=.7)).text

class TextByGPT4:
    def __init__(self) -> None:
        print(os.getenv('GPT4_API_KEY'))
        self.model = OpenAI(
            api_key=os.getenv('GPT4_API_KEY')
        )

    def generate_text(self, prompt):
        response = self.model.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4",
        )

        status = response['choices'][0]['finish_reason']
        if status == "stop":
            return response['choices'][0]['message']['content']
        
        return None



def load_folder_txt(folder_path) -> list:
    file_list = os.listdir(folder_path)
    sentence_list = []
    for file_name in tqdm(file_list, desc="Processing files", unit="file"):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                sentence_list.extend(line.strip() for line in lines if line.strip())
    sentence_list = list(sorted(sentence_list))
    sentence_list = [s.strip() for s in sentence_list]
    print(f"Success") 
    print(f"Number of sentences : {len(sentence_list)}")
    return sentence_list

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        content = json.load(json_file)
    return content
    
def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)