
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5_QA:
    def __init__(self, model_path):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model.eval()
        
    
    def answer(self, question, max_length=200):
        
        input_text = f"answer question: {question}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length)
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer



if __name__ == "__main__":
    
    qa = T5_QA("./output_model/t5_financial_model.bin")
    
    
    question = "Is it wise to sell Apple stock at this stage?"
    answer = qa.answer(question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}\n")
    
    