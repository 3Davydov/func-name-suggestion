from collections.abc import Iterable
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline # type: ignore
import pandas as pd # type: ignore
from typing import List, Dict, Any
import re

import datasets # type: ignore
import evaluate # type: ignore

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codeT5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codeT5-base")
mask_filler = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def run_evaluate(results: List[Dict]) -> Dict[str, float]:
	exact_match_metric = evaluate.load('exact_match')
	rouge_metric = evaluate.load('rouge')

	references = [item['function_name'] for item in results]
	predictions = [item['prediction'] for item in results]

	em_results = exact_match_metric.compute(
		predictions=predictions,
		references=references,
		ignore_case=True,
		ignore_punctuation=True
	)
	exact_match_score = em_results['exact_match']

	rouge_results = rouge_metric.compute(
		predictions=predictions,
		references=references
	)

	average_rouge_score = {k: round(v, 4) for k, v in rouge_results.items()}

	print(f"Exact Match : {round(exact_match_score, 4)}")
	print(f"ROUGE-Scores : {average_rouge_score}")

def predict_function_name(function_body: str, function_name: str) -> list[dict]:
	masked_code = f"def <extra_id_0>():\n\t{function_body}"

	predictions = mask_filler(masked_code, max_length=20, num_return_sequences=1, truncation=True)
	generated_text = predictions[0]['generated_text']

	start_index = generated_text.find('<extra_id_0>')
	end_index = generated_text.find('<extra_id_1>')
	if start_index != -1 and end_index != -1 and end_index > start_index:
		predicted_name = generated_text[start_index + len('<extra_id_0>'):end_index].strip()
	else:
		words = generated_text.replace('<extra_id_0>', '').strip().split()
		if words:
			predicted_name = words[0]

	cleaned_name = re.sub(r'[^a-zA-Z0-9_]', '', predicted_name)
	return [{'token_str': cleaned_name}]


def predict(dataset: datasets.Dataset, model: str) -> None:
	results = []

	for example in dataset:
		fbody = example['extracted_body_without_comments']
		fname = example['extracted_function_name']

		if not fbody or not fname:
			continue

		prediction_list = predict_function_name(fbody, fname)

		if not prediction_list:
			raise LookupError("Cannot predict function name by body")

		# Проверяем, есть ли истинное имя функции среди топ-5 предсказаний
		predicted_tokens = [p['token_str'].strip() for p in prediction_list]

		is_correct = (fname == predicted_tokens[0])
		if (is_correct):
			print(f"real name {fname} vs prediction {predicted_tokens[0]}")

		results.append({
			'function_name': fname,
			'prediction': predicted_tokens[0],
			'is_correct': is_correct
		})

	run_evaluate(results)
