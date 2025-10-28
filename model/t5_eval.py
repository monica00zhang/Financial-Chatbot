import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn.functional as F

"""
QA Evaluation Framework
Evaluate generated answer quality without ground truth
"""

class Evaluator:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model.eval()
    
    def get_embedding(self, text):
        input_text = f"answer question: {text}"
        inputs = self.tokenizer(input_text, 
                                return_tensors="pt", 
                                max_length=config.MAX_LENGTH, 
                            )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            encoder_output = self.model.encoder(**inputs).last_hidden_state
            # mean pooling
            embedding = encoder_output.mean(dim=1)
        
        return embedding
    
    def compute_consistency(self, question, answer):
        """
        Compute consistency : Uses T5 encoder embeddings and cosine similarity
        """
        q_emb = self.get_embedding(question)
        a_emb = self.get_embedding(answer)
        
        # cosine similarity
        consistency = F.cosine_similarity(q_emb, a_emb, dim=1).item()
        
        # normalize to [0, 1]
        consistency = (consistency + 1) / 2
        
        return consistency
    
    def compute_confidence(self, question, answer):
        """
        Compute generation confidence: Calculate probability of each generated token
        """
        input_text = f"answer question: {question}"
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=128, 
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # convert answer to token ids
        answer_ids = self.tokenizer(
            answer, 
            return_tensors="pt", 
            add_special_tokens=False
        ).input_ids.to(self.device)
        
        with torch.no_grad():
            # get model output logits
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=answer_ids
            )
            
            logits = outputs.logits  # (batch, seq_len, vocab_size)
            
            # compute probability for each token
            probs = F.softmax(logits, dim=-1)
            
            # get probability of actual generated tokens
            token_probs = []
            for i in range(answer_ids.shape[1]):
                if i < logits.shape[1]:
                    token_id = answer_ids[0, i].item()
                    prob = probs[0, i, token_id].item()
                    token_probs.append(prob)
            
            # average probability as confidence
            if token_probs:
                confidence = sum(token_probs) / len(token_probs)
            else:
                confidence = 0.0
        
        return confidence
    
    
    def evaluate(self, question, answer, weights=None):

        if weights is None:
            weights = { 'consistency': 0.6,   
                        'confidence': 0.4,    # generation confidence second
                        }
        
        # calculate each metric
        consistency = self.compute_consistency(question, answer)
        confidence = self.compute_confidence(question, answer)
        
        # weighted final score
        final_score = weights['consistency'] * consistency + weights['confidence'] * confidence
        
        result = {'consistency': consistency,
                  'confidence': confidence,
                  'final_score': final_score,
                  'is_valid': final_score >= 0.8
                }
        
        return result
    
    def evaluate_batch(self, qa_pairs):
        """
        Batch evaluation for test
        """
        results = []
        for question, answer in qa_pairs:
            result = self.evaluate(question, answer)
            result['question'] = question
            result['answer'] = answer
            results.append(result)
        
        return results



if __name__ == "__main__":

    evaluator = T5Evaluator("./output_model/t5_financial_model.bin")
    
    question = " If Tech firms keep interest expenses low, what would Buffett think about their capital structure?"
    
    answer = """When I see tech companies with 40%+ gross margins for three straight years, I first ask myself: "Do I understand why?" Tech isn't primarily my playground, but I recognize patterns. Sustainable high margins often signal pricing power – something special customers can't easily find elsewhere.
                High margins are like a good fishing spot – they attract crowds. The real question isn't whether they've had good margins in the past, but whether they have the moat to keep competitors at bay for the next decade.
                I'm less concerned with the exact percentage and more focused on durability. Can they maintain these margins for the next 10 years? In investing, you're not picking a company for a quarter – you're finding a business that can compound value for years. High margins attract competition like honey attracts bears. 
                Unless there's a genuine moat protecting those margins, Mr. Market's initial enthusiasm will eventually face a reality check."""
    
    result = evaluator.evaluate(question, answer)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")


    print(f"\nEvaluation Results:")

    print(f"  Consistency (QA semantic similarity): {result['consistency']:.2f}")
    print(f"  Confidence (generation certainty):    {result['confidence']:.3f}")
    print(f"  Final Score (weighted):               {result['final_score']:.3f}")

    print(f"  Is Valid (>0.8):                      {result['is_valid']}")
    