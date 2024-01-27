from utils import TextByGPT4, load_json, save_json
import os 
import math
from tqdm import tqdm

class GPT4Simplification(TextByGPT4):
    def __init__(self) -> None:
        super().__init__()
        self.tracker_path = "GPT4/tracker.txt"
        self.result_path = "GPT4/gpt4_result.json"

    def generate_simplification(self, sentence_list, batch_size=10) -> None:
        
        if not os.path.exists(self.tracker_path):
            with open(self.tracker_path, 'w', encoding='utf-8') as file:
                file.write("0")
            file.close()

        end = len(sentence_list)
        with open(self.tracker_path, 'r', encoding='utf-8') as file:
            start = file.read()
            if start.isdigit():
                start = int(start)
                if start < 0 or start >= (end-1):
                    print("Index in tracker is over the length of list.")
                    return

        gpt4_result = load_json(self.result_path)

        unprocessed_sentence_list = sentence_list[start:end]
        batch_nums = math.ceil(len(unprocessed_sentence_list) / batch_size)

        for index in range(batch_nums):
            print(f"Batch {index}:")
            batch_start = batch_size * index
            batch_end = batch_size * (index + 1)
            batch = unprocessed_sentence_list[batch_start:batch_end]
            simplified_sents = []
            count = 0
            for sent in tqdm(batch, desc="Simplify sentences", unit=" sentence"):
                prompt = self.gen_zero_shot_prompt(sent)
                result = self.generate_text(prompt)
                # result = prompt
                # if count == 2 and index == 2:
                #     result = None
                if result == None:
                    # update result
                    gpt4_result.extend(simplified_sents)
                    save_json(self.result_path, gpt4_result)

                    # update tracker
                    start = start + len(simplified_sents)
                    with open(self.tracker_path, 'w', encoding='utf-8') as tracker_file:
                        tracker_file.write(str(start))
                    tracker_file.close()
                    return
                
                simplified_sents.append(result)
                # count += 1
        
            # update result
            gpt4_result.extend(simplified_sents)
            save_json(self.result_path, gpt4_result)

            # update tracker
            start = start + len(simplified_sents)
            with open(self.tracker_path, 'w', encoding='utf-8') as tracker_file:
                tracker_file.write(str(start))
            tracker_file.close()


    def gen_zero_shot_prompt(self, sentence) -> str:
        return f"""
                Simplify this sentence to make it more understandable . You can keep sentence the same if you think there is not thing to simplify .
                Sentence : {sentence}
                Simplified sentence : 
                """


