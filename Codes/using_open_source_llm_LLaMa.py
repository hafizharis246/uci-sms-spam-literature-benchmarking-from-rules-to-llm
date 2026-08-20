import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class OpenSourceLLMDetector:
    """
    SMS Spam Detection using Open-Source LLMs (LLaMA, Mixtral)
    """
    
    def __init__(self, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", use_quantization=True):
        """
        Initialize the LLM-based detector with an open-source model
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        
        print(f"Loading model: {model_name}")
        self.load_model()
        
    def load_model(self):
        """
        Load the LLM model and tokenizer
        """
        # Setup quantization for memory efficiency
        if self.use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
        
    def load_data(self, filepath='SMSSpamCollection'):
        """Load the UCI SMS Spam Collection dataset"""
        data = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
        data['label'] = data['label'].map({'ham': 0, 'spam': 1})
        return data
    
    def create_prompt(self, message, examples=None, use_cot=False):
        """
        Create a prompt for the LLM
        """
        if examples:
            # Few-shot prompt
            prompt = "Classify the following SMS messages as either 'spam' or 'ham'.\n\n"
            for ex in examples:
                label = "spam" if ex['label'] == 1 else "ham"
                prompt += f"Message: {ex['message']}\nClassification: {label}\n\n"
            prompt += f"Message: {message}\nClassification:"
        elif use_cot:
            # Chain-of-thought prompt
            prompt = f"""
            Analyze the following SMS message step by step to determine if it's spam or legitimate (ham):
            
            Message: {message}
            
            Steps:
            1. Does the message contain any suspicious links or phone numbers?
            2. Is there a sense of urgency or pressure to act quickly?
            3. Does it ask for personal information or money?
            4. Does it contain promotional language or offers that seem too good to be true?
            5. Is the sender identified as a known and trusted entity?
            
            Based on the analysis above, the message is:
            """
        else:
            # Zero-shot prompt
            prompt = f"Classify the following SMS message as either 'spam' or 'ham'. Respond with only one word.\n\nMessage: {message}\nClassification:"
            
        return prompt
    
    def classify_message(self, message, examples=None, use_cot=False):
        """
        Classify a single message using the LLM
        """
        # Create prompt
        prompt = self.create_prompt(message, examples, use_cot)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the classification
        # The response contains the prompt + generated text
        response_text = response[len(prompt):].strip().lower()
        
        # Parse result
        if 'spam' in response_text:
            return 1
        elif 'ham' in response_text or 'legitimate' in response_text:
            return 0
        else:
            # Try to find in the full response
            full_response = response.lower()
            if 'spam' in full_response and 'ham' not in full_response:
                return 1
            elif 'ham' in full_response:
                return 0
            else:
                return 0  # Default to ham
    
    def evaluate(self, messages, labels, examples=None, use_cot=False, verbose=True):
        """
        Evaluate the model on a set of messages
        """
        predictions = []
        
        for i, (message, true_label) in enumerate(zip(messages, labels)):
            if verbose and i % 10 == 0:
                print(f"Processing message {i+1}/{len(messages)}")
            
            pred_label = self.classify_message(message, examples, use_cot)
            predictions.append(pred_label)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        return {
            'predictions': predictions,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def fine_tune(self, train_data, val_data):
        """
        Fine-tune the model on SMS data (simplified version)
        For full fine-tuning, use QLoRA or LoRA
        """
        print("Preparing data for fine-tuning...")
        
        # Prepare training examples
        train_examples = []
        for _, row in train_data.iterrows():
            label = "spam" if row['label'] == 1 else "ham"
            text = f"Classify: {row['message']}\nAnswer: {label}"
            train_examples.append(text)
        
        print(f"Prepared {len(train_examples)} training examples")
        print("Note: Full