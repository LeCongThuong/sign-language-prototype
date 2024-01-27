import utils
from Gemini.gemini_simplification import GeminiSimplification
from GPT4.gpt4_simplification import GPT4Simplification

gemini = GeminiSimplification()
gpt4 = GPT4Simplification()
sentence_list = utils.load_folder_txt("test-folder")
gpt4.generate_simplification(sentence_list=sentence_list)

