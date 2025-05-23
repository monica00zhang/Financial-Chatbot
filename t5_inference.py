import tensorflow as tf
from transformers import pipeline, T5ForConditionalGeneration, AutoTokenizer
import transformers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


model_path = "model/Financial_chatbot_T5_Fine_tuned"

tokenizer = AutoTokenizer.from_pretrained(model_path)#rT5Tokenizer
print("Tokenizer loaded!")



def build_chatbot_pipeline():
    finetuned_model = T5ForConditionalGeneration.from_pretrained(model_path)
    pipeline_task = "text2text-generation"
    pipeline_min_length = 15
    pipeline_temperature = 0.2
    pipeline_max_length = 193

    return pipeline(
        task=pipeline_task,
        model=finetuned_model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=pipeline_max_length,
        min_length=pipeline_min_length,
        temperature=pipeline_temperature,
        device=0  # Set device to 0 for GPU, -1 for CPU
    )


def get_response(user_input):
    text_generation_pipeline = build_chatbot_pipeline()

    prefix = "Answer this question: "  # prefix our task

    transfer_questions = [prefix +  user_input]
    generated_texts = text_generation_pipeline(transfer_questions, do_sample=True)
    predictions = [output_text["generated_text"] for output_text in generated_texts]

    return predictions

