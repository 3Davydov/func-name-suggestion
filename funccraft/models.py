from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline # type: ignore
from typing import List, Dict

import datasets # type: ignore
import evaluate # type: ignore

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5p-220m")
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

def predict_function_name(function_body: str, language: str) -> dict:
	# truncate to 512 symbols
	function_body = function_body[:512]

	if (language == "python"):
		inputs = tokenizer.encode(f"def <extra_id_0>():\n\t{function_body}", return_tensors="pt")
	elif (language == "java"):
		inputs = tokenizer.encode(f"static void <extra_id_0>()\n\t{function_body}", return_tensors="pt")
	else:
		raise NotImplementedError(f"cannot use {language} language")

	outputs = model.generate(inputs, max_length=20, num_return_sequences=1)

	generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

	start_index = generated_text.find('<extra_id_0>')
	end_index = generated_text.find('<extra_id_1>')

	if start_index != -1 and end_index != -1 and end_index > start_index:
		predicted_name = generated_text[start_index + len('<extra_id_0>'):end_index].strip()
	else:
		words = generated_text.replace('<extra_id_0>', '').strip().split()
		predicted_name = words[0] if words else ''

	cleaned_name = predicted_name.split('(')[0]
	return {'token_str': cleaned_name}

def predict(dataset: datasets.Dataset, language_str: str, model: str) -> None:
	only_body_results = []
	body_with_comments_results = []

	if (model != "codeT5"):
		raise NotImplementedError("can use only codeT5 model")

	for example in dataset:
		fbody = example['extracted_body_without_comments']
		fbody_with_comments = example['extracted_body_with_comments']
		fname = example['extracted_function_name']

		only_body_prediction = predict_function_name(fbody, language_str)
		only_body_results.append({
			'function_name': fname,
			'prediction': only_body_prediction['token_str'],
		})

		body_with_comments_prediction = predict_function_name(fbody_with_comments, language_str)
		body_with_comments_results.append({
			'function_name': fname,
			'prediction': body_with_comments_prediction['token_str'],
		})

	run_evaluate(only_body_results)
	run_evaluate(body_with_comments_results)
