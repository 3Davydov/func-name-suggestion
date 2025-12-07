import datasets # type: ignore
import pandas as pd # type: ignore
import tree_sitter_python as tspython # type: ignore
from pathlib import Path
from tree_sitter import Language, Parser, Node # type: ignore
from tree_sitter_languages import get_language, get_parser

def parse_single_function(source_code: str, language_str: str) -> dict | None:
	if language_str == "python":
		language = get_language('python')
		parser = get_parser('python')
		query_string = """
		(function_definition
			name: (identifier) @function_name
			body: (block) @func_code_string
		)
		"""
	elif language_str == "java":
		language = get_language('java')
		parser = get_parser('java')
		query_string = """
		(method_declaration
			name: (identifier) @function_name
			body: (block) @func_code_string
		)
		"""
	else:
		raise NotImplementedError(f"cannot use {language_str} language")

	# parser = Parser()
	# parser.set_language(language)
	code_bytes = source_code.encode('utf8')
	tree = parser.parse(code_bytes)
	root_node = tree.root_node

	query = language.query(query_string)
	captures = query.captures(root_node) # executes query on the tree

	if not captures:
		return None

	# Must init it with 'None' values in order to keep dataset structure
	parsed_function = {
		'extracted_function_name': None,
		'extracted_body_with_comments': None,
		'extracted_body_without_comments': None
	}
	body_node: Node = None

	for node, name in captures:
		if name == 'function_name':
			parsed_function['extracted_function_name'] = node.text.decode('utf8')
		elif name == 'func_code_string':
			body_node = node
			# Тело с комментариями/docstring - просто текст узла body
			parsed_function['extracted_body_with_comments'] = node.text.decode('utf8').strip()

	if body_node:
		parsed_function['extracted_body_without_comments'] = extract_body_without_comments(body_node, code_bytes)

	return parsed_function

# In case of java code this is more about removing annotations...
def extract_body_without_comments(body_node: Node, source_bytes: bytes) -> str:
	filtered_body_parts = []

	for child in body_node.children:
		if child.type == 'comment':
			continue

		# Assume that documentation is the symbols that start the function body
		is_docstring = (child.type == 'expression_statement' and
			len(child.children) > 0 and
			child.children[0].type == 'string')

		if is_docstring and filtered_body_parts == []:
			continue

		node_text = source_bytes[child.start_byte:child.end_byte].decode('utf8')
		filtered_body_parts.append(node_text)

	# Join parts of function body + delete empty lines and redundant indents
	return "\n".join(part.strip() for part in filtered_body_parts if part.strip())

def prepare(language: str) -> datasets.Dataset:
	if language != "python" and language != "java":
		raise NotImplementedError(f"cannot use {language} language")

	dataset = datasets.load_dataset(
		'code_search_net',
		language,
		split='test',
		trust_remote_code=True
	)

	dataset = dataset.select(range(1000))

	processed_dataset = dataset.map(
		lambda example: parse_single_function(example['whole_func_string'], language)
	)

	print("\nExample of extracted features\n")
	df = pd.DataFrame(processed_dataset[:1])
	print(df[['func_code_string', 'extracted_function_name', 'extracted_body_with_comments', 'extracted_body_without_comments']].to_markdown(index=False))

	return processed_dataset


def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))
