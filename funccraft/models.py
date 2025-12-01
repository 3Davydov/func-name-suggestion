from collections.abc import Iterable
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline # type: ignore
import pandas as pd # type: ignore
from typing import List, Dict, Any

import datasets # type: ignore
import evaluate # type: ignore

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
mask_filler = pipeline("fill-mask", model="roberta-base", tokenizer=tokenizer)

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
	masked_code = f"def <mask>:\n\t{function_body}"
	max_length = 512
	# tokenized_input = mask_filler.tokenizer(masked_code, truncation=True, max_length=max_length, return_tensors="pt")

	# predictions = mask_filler(masked_code, top_k=5)
	predictions = mask_filler(masked_code[:max_length], top_k=1)
	return predictions


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
		print(f"real name {fname} vs prediction {predicted_tokens[0]} is correct {is_correct}")

		results.append({
			'function_name': fname,
			'prediction': predicted_tokens[0],
			'is_correct': is_correct
		})

	run_evaluate(results)
