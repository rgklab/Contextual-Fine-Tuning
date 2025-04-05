import anthropic
import json
import time
import os
from tqdm import tqdm
from datasets import load_dataset
import google.generativeai as genai
import argparse

def load_incorrect_indices(indices_file):
    """Load incorrect indices from a text file."""
    with open(indices_file, 'r') as f:
        content = f.read()
        # Skip the header line if it exists
        if 'Total incorrect predictions:' in content:
            indices_line = content.split('Indices:\n')[-1]
        else:
            indices_line = content
        
        # Parse the indices
        if ',' in indices_line:
            indices = [int(idx.strip()) for idx in indices_line.split(',') if idx.strip().isdigit()]
        else:
            indices = [int(idx.strip()) for idx in indices_line.split() if idx.strip().isdigit()]
    
    return indices, len(indices)

def format_prompt_for_educational_content(question, options, answer, answer_idx):
    """Format prompt for generating educational content for a question."""
    if isinstance(options, str):
        options = json.loads(options)
    
    prompt = f"I have the following medical question from MedQA USMLE exam to prepare for:\n\nQ: {question}\nA: {options['A']}\nB: {options['B']}\nC: {options['C']}\nD: {options['D']}\nAnswer: {answer_idx}. {answer}\n\n"
    prompt += "Please provide:\nA thorough textbook-style explanation written in clear, connected paragraphs that build upon each other. Include relevant clinical correlations and physiological mechanisms throughout the text.\n"
    prompt += "Present this as a cohesive educational yet concise resource similar to what I might find in a high-quality medical textbook structure. The output should have less than 3000 characters."
    return prompt

def format_prompt_for_contextual_prompt(question, options, answer, answer_idx, education_content):
    """Format prompt for generating contextual prompts based on educational content."""
    if isinstance(options, str):
        options = json.loads(options)
    
    prompt = f"""
Based on the following question-answer pair and its related educational content:

QUESTION: {question}
A: {options['A']}
B: {options['B']}
C: {options['C']}
D: {options['D']}
ANSWER: {answer_idx}. {answer}
EDUCATIONAL CONTENT: {education_content}

Generate a very concise contextual prompt that will enhance learning effectiveness. The prompt should:

1. Follow the style of [select one learning theory approach: Application of Knowledge/In-Depth Exploration/Reflective Thinking/Creative Interpretation/Summarization and Synthesis/Focus on Key Concepts/Contextual Understanding/Critical Analysis/Question-Based Learning/Comparative Learning]
2. Identify:
   • The fundamental concepts that must be understood
   • Critical facts that require focus for mastery
   • How these elements connect to clinical reasoning or application
3. Be formatted as a directive that encourages active engagement with the material (approximately 1-2 sentences)
4. Frame the learning in a way that facilitates long-term retention

The contextual prompt should help the learner not just memorize information but develop a deeper, more applicable understanding of the medical concept to correctly answer the question using the educational content.
The output should only contain the concise contextual prompt with 1-2 sentences.
"""
    return prompt

def generate_with_claude(client, prompt, max_tokens=2048):
    """Generate content using Claude API."""
    try:
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        print(f"Error generating content: {e}")
        return None

def generate_with_batch(client, prompts, max_tokens=2048, model="claude-3-7-sonnet-20250219"):
    """Generate content using Claude batch API."""
    requests = []
    for idx, prompt in enumerate(prompts):
        requests.append({
            "custom_id": f"prompt-{idx}",
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            },
        })
    
    try:
        batch = client.messages.batches.create(requests=requests)
        return batch.id
    except Exception as e:
        print(f"Error creating batch: {e}")
        return None

def get_batch_results(client, batch_id):
    """Get results from a batch request."""
    results = []
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        response = result.result.message.content[0].text
        results.append({'custom_id': custom_id, 'response': response})
    
    return results

def format_dataset_for_training(contextual_prompts, educational_contents, test_set, output_path, max_row_length=5000):
    """Format the data into a CSV file for training."""
    import csv
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['input', 'output'])
        writer.writeheader()
        total_excess_count = 0
        
        for idx, (contextual, educational) in enumerate(zip(contextual_prompts, educational_contents)):
            input_text = contextual.strip()

            # Get corresponding test question data
            question = test_set[idx]['question']
            options = test_set[idx]['options']
            if isinstance(options, str):
                options = json.loads(options)
            answer_idx = test_set[idx]['answer_idx']
            answer = test_set[idx]['answer']
            
            # Format question and options
            question_answer_text = f"Question: {question} (A) {options['A']} (B) {options['B']} (C) {options['C']} (D) {options['D']}\nAnswer: ({answer_idx}) {answer}"

            # Combine educational content with question data
            output_text = f"{educational.strip()}\n\n{question_answer_text}"
            
            # Calculate total length
            total_length = len(input_text) + len(output_text)
            
            # If total length exceeds limit, truncate the output
            if total_length > max_row_length:
                excess = total_length - max_row_length
                total_excess_count += 1
                print(f"Excess: {excess}, Total excess count: {total_excess_count}")
                output_text = output_text[:len(output_text) - excess - 3] + "..."
            
            row = {
                'input': input_text,
                'output': output_text
            }
            writer.writerow(row)
    
    print(f"CSV file has been created at: {output_path}")
    print(f"Rows are limited to {max_row_length} characters total (input + output combined)")

def evaluate_model_medqa(model, output_file, custom_prompt=None):
    """Evaluate a Gemini model on MedQA dataset."""
    # Load MedQA dataset
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    
    correct = 0
    total = 0
    results = []
    
    for row in tqdm(dataset):
        try:
            # Format question
            question = row['question']
            options = {
                'A': row['options']['A'],
                'B': row['options']['B'],
                'C': row['options']['C'],
                'D': row['options']['D']
            }
            
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = f"""You are a medical expert answering a multiple choice question about medical knowledge.
Think carefully before providing your final answer. Explain your reasoning process explicitly
"""

            prompt += f"\nQuestion: {question}"
            for opt, text in options.items():
                prompt += f" ({opt}) {text}"
            prompt += "\nAnswer:"
            
            # Get model response
            response = model.generate_content(prompt, safety_settings=[
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
            ])
            
            predicted_answer = response.text
            
            # Clean up prediction (take first letter if model outputs more)
            if predicted_answer and predicted_answer[0] in ["A", "B", "C", "D"]:
                predicted_answer = predicted_answer[0]
            else:
                for char in predicted_answer:
                    if char in ["A", "B", "C", "D"]:
                        predicted_answer = char
                        break
            
            # Store result
            result = {
                "idx": total,
                "question": row['question'],
                "predicted_answer": predicted_answer,
                "model_response": response.text,
                "answer_idx": row['answer_idx'],
                "answer": row['options'][row['answer_idx']],
                "prompt": prompt,
            }
            results.append(result)
            
            # Check if correct
            if predicted_answer == row['answer_idx']:
                correct += 1
            total += 1
            
            # Print progress
            if total % 10 == 0:
                print(f"\nCurrent accuracy: {(correct/total)*100:.2f}% ({correct}/{total})")
                
        except Exception as e:
            print(f"Error processing question {total}: {str(e)}")
            continue
    
    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    final_accuracy = (correct/total)*100 if total > 0 else 0
    print("\nEvaluation Results:")
    print(f"Total questions: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {final_accuracy:.2f}%")
    
    return final_accuracy

def process_model_responses_with_claude(eval_results_file, claude_client, output_file):
    """Process model responses with Claude to extract predicted answers."""
    # Load the dataset
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    test_set = dataset["test"]
    
    # Load the evaluation results
    with open(eval_results_file, 'r') as f:
        eval_data = json.load(f)
    
    results = []
    correct_count = 0
    total_count = 0
    
    for idx, item in enumerate(tqdm(eval_data)):
        try:
            question = item['question']
            options = test_set[idx]['options']
            model_response = item['model_response']
            answer_idx = item['answer_idx']
            
            # Format options if they're in string format
            if isinstance(options, str):
                options = json.loads(options)
            
            # Construct prompt for Claude
            prompt = f"""Given the following medical question and model's response, determine which option (A, B, C, or D) the response is indicating as the answer. Output ONLY the letter.

QUESTION: {question}
OPTIONS:
A: {options['A']}
B: {options['B']}
C: {options['C']}
D: {options['D']}

MODEL'S RESPONSE: {model_response}

Output the single letter (A, B, C, or D) that best matches the model's response:"""
            
            # Get Claude's interpretation
            message = claude_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            
            claude_option = message.content[0].text.strip()
            # Extract just the letter if there's any additional text
            for letter in ['A', 'B', 'C', 'D']:
                if letter in claude_option:
                    claude_option = letter
                    break
            
            # Check if correct
            is_correct = claude_option == answer_idx
            if is_correct:
                correct_count += 1
            total_count += 1
            
            results.append({
                'question': question,
                'options': options,
                'model_response': model_response,
                'answer_idx': answer_idx,
                'claude_interpretation': claude_option,
                'is_correct': is_correct
            })
            
            # Save progress after every 10 items
            if total_count % 10 == 0:
                accuracy = correct_count / total_count
                print(f"\nCurrent accuracy: {accuracy:.2%}")
                
                with open(output_file, 'w') as f:
                    json.dump({
                        'results': results,
                        'accuracy': accuracy,
                        'processed_count': total_count
                    }, f, indent=2)
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    
    # Final accuracy
    final_accuracy = correct_count / total_count
    print(f"\nFinal accuracy: {final_accuracy:.2%}")
    
    # Save final results
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'accuracy': final_accuracy,
            'total_processed': total_count
        }, f, indent=2)
    
    return final_accuracy

def generate_dataset_pipeline(claude_api_key, indices_file, output_csv_path, batch_size=50):
    """Pipeline to generate dataset from hard questions."""
    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=claude_api_key)
    
    # Load incorrect indices from file
    incorrect_indices, total_indices = load_incorrect_indices(indices_file)
    print(f"Loaded {total_indices} incorrect indices from {indices_file}")
    
    # Load the dataset
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    test_set = dataset["test"]
    test_set = test_set.select(incorrect_indices)
    
    # Generate educational content
    print("Generating educational content...")
    educational_contents = []
    
    for i in range(0, len(test_set), batch_size):
        batch_questions = test_set['question'][i:i + batch_size]
        batch_options = test_set['options'][i:i + batch_size]
        batch_answers = test_set['answer'][i:i + batch_size]
        batch_answer_idx = test_set['answer_idx'][i:i + batch_size]
        
        prompts = []
        for idx, question in enumerate(batch_questions):
            prompt = format_prompt_for_educational_content(
                question,
                batch_options[idx],
                batch_answers[idx],
                batch_answer_idx[idx]
            )
            prompts.append(prompt)
        
        # Submit batch request
        batch_id = generate_with_batch(client, prompts, max_tokens=2048)
        if not batch_id:
            print("Error creating batch for educational content")
            continue
        
        print(f"Submitted batch {batch_id}, waiting for results...")
        time.sleep(30)  # Wait for batch to complete
        
        # Get batch results
        batch_results = get_batch_results(client, batch_id)
        for result in batch_results:
            educational_contents.append(result['response'])
    
    # Generate contextual prompts
    print("Generating contextual prompts...")
    contextual_prompts = []
    
    for i in range(0, len(test_set), batch_size):
        batch_questions = test_set['question'][i:i + batch_size]
        batch_options = test_set['options'][i:i + batch_size]
        batch_answers = test_set['answer'][i:i + batch_size]
        batch_answer_idx = test_set['answer_idx'][i:i + batch_size]
        batch_educational = educational_contents[i:i + batch_size]
        
        prompts = []
        for idx, question in enumerate(batch_questions):
            prompt = format_prompt_for_contextual_prompt(
                question,
                batch_options[idx],
                batch_answers[idx],
                batch_answer_idx[idx],
                batch_educational[idx]
            )
            prompts.append(prompt)
        
        # Submit batch request
        batch_id = generate_with_batch(client, prompts, max_tokens=1024)
        if not batch_id:
            print("Error creating batch for contextual prompts")
            continue
        
        print(f"Submitted batch {batch_id}, waiting for results...")
        time.sleep(30)  # Wait for batch to complete
        
        # Get batch results
        batch_results = get_batch_results(client, batch_id)
        for result in batch_results:
            contextual_prompts.append(result['response'])
    
    # Format dataset for training
    print("Formatting dataset for training...")
    format_dataset_for_training(
        contextual_prompts,
        educational_contents,
        test_set,
        output_csv_path
    )
    
    return output_csv_path

def evaluate_pipeline(gemini_api_key, model_name, output_eval_file, claude_api_key, output_interp_file):
    """Pipeline to evaluate model and process results with Claude."""
    # Initialize clients
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name=model_name, generation_config=genai.GenerationConfig(max_output_tokens=1024))
    
    claude_client = anthropic.Anthropic(api_key=claude_api_key)
    
    # Evaluate model
    print(f"Evaluating model {model_name}...")
    evaluate_model_medqa(model, output_eval_file)
    
    # Process results with Claude
    print("Processing results with Claude...")
    process_model_responses_with_claude(output_eval_file, claude_client, output_interp_file)
    
    return output_interp_file

def main():
    parser = argparse.ArgumentParser(description='Medical Question Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Dataset generation parser
    dataset_parser = subparsers.add_parser('generate-dataset', help='Generate dataset from hard questions')
    dataset_parser.add_argument('--claude-api-key', required=True, help='Anthropic API key')
    dataset_parser.add_argument('--indices-file', required=True, help='Text file with incorrect prediction indices')
    dataset_parser.add_argument('--output-csv', required=True, help='Output CSV path')
    dataset_parser.add_argument('--batch-size', type=int, default=50, help='Batch size for API requests')
    
    # Evaluation parser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model and process results')
    eval_parser.add_argument('--gemini-api-key', required=True, help='Google AI API key')
    eval_parser.add_argument('--claude-api-key', required=True, help='Anthropic API key')
    eval_parser.add_argument('--model-name', required=True, help='Gemini model name')
    eval_parser.add_argument('--output-eval-file', required=True, help='Output evaluation JSON file')
    eval_parser.add_argument('--output-interp-file', required=True, help='Output interpretation JSON file')
    
    args = parser.parse_args()
    
    if args.command == 'generate-dataset':
        generate_dataset_pipeline(
            args.claude_api_key,
            args.indices_file,
            args.output_csv,
            args.batch_size
        )
    elif args.command == 'evaluate':
        evaluate_pipeline(
            args.gemini_api_key,
            args.model_name,
            args.output_eval_file,
            args.claude_api_key,
            args.output_interp_file
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()