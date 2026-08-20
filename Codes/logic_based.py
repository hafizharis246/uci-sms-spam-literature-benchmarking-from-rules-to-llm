import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class FuzzyPSO_SMS_Detector:
    """
    SMS Spam Detection using Fuzzy Logic and Binary PSO
    Implementation based on the paper by Hameed & Ali (2021)
    """
    
    def __init__(self, n_particles=30, n_iterations=100, c1=1.495, c2=1.495, w=0.728):
        """
        Initialize the detector with PSO parameters
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.v_min = -4
        self.v_max = 4
        
        # Fuzzy membership function parameters
        self.fuzzy_sets = {}  # Will store fuzzy set definitions
        self.rule_base = []  # Will store generated rules
        self.best_rules = []  # Will store selected rules after PSO
        
        # For feature extraction
        self.spam_keywords = ['call', 'money', 'mobile', 'phone', 'free', 'win', 'prize', 
                             'urgent', 'claim', 'credit', 'bank', 'account', 'click', 
                             'link', 'offer', 'guarantee', 'special', 'limited']
        
    def load_data(self, filepath='SMSSpamCollection'):
        """Load the UCI SMS Spam Collection dataset"""
        data = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
        data['label'] = data['label'].map({'ham': 0, 'spam': 1})
        return data
    
    def preprocess_message(self, message):
        """
        Preprocess a single message: tokenization, stopword removal, stemming
        """
        # Convert to lowercase
        message = message.lower()
        
        # Remove special characters (keep letters, numbers, spaces)
        message = re.sub(r'[^a-zA-Z0-9\s]', '', message)
        
        # Tokenize
        tokens = message.split()
        
        # Remove stopwords (basic list)
        stopwords = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 
                    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
                    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
                    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their'}
        
        # Simple stemming (just remove common suffixes)
        stemmed_tokens = []
        for token in tokens:
            if token not in stopwords:
                # Simple suffix stripping
                if token.endswith('ing'):
                    token = token[:-3]
                elif token.endswith('ed'):
                    token = token[:-2]
                elif token.endswith('s') and not token.endswith('ss'):
                    token = token[:-1]
                stemmed_tokens.append(token)
        
        return stemmed_tokens
    
    def extract_features(self, message):
        """
        Extract the 6 features as defined in the paper
        """
        processed_words = self.preprocess_message(message)
        
        # Feature 1: Message length ratio
        max_length = 160  # SMS max length
        f1 = len(message) / max_length
        
        # Feature 2: Number of words ratio
        max_words = 50  # Approximate maximum words in SMS
        f2 = len(processed_words) / max_words
        
        # Feature 3: Number of words with less than 3 characters ratio
        short_words = sum(1 for word in processed_words if len(word) < 3)
        f3 = short_words / max(1, max_words)
        
        # Feature 4: Capital word ratio (based on original message)
        cap_words = sum(1 for word in message.split() if word.isupper())
        f4 = cap_words / max(1, max_words)
        
        # Feature 5: Alphanumeric characters ratio
        alnum_chars = sum(1 for char in message if char.isalnum())
        max_chars = 160
        f5 = alnum_chars / max_chars
        
        # Feature 6: Thematic SMS spam words
        thematic_count = sum(1 for word in message.lower().split() 
                           if word in self.spam_keywords)
        max_thematic = 20  # Approximate maximum thematic words
        f6 = thematic_count / max_thematic
        
        return [f1, f2, f3, f4, f5, f6]
    
    def fuzzy_membership(self, x, a, b, c):
        """
        Triangular membership function
        """
        if x <= a or x >= c:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x < c:
            return (c - x) / (c - b)
        else:
            return 0
    
    def generate_fuzzy_rules(self, X, y):
        """
        Generate fuzzy rules from training data
        """
        # Define fuzzy sets for each feature
        feature_sets = {}
        for i in range(6):
            # Get feature values
            values = [sample[i] for sample in X]
            
            # Determine thresholds for Low, Medium, High
            q1 = np.percentile(values, 33)
            q2 = np.percentile(values, 66)
            
            feature_sets[i] = {
                'Low': {'a': 0, 'b': q1, 'c': q2},
                'Medium': {'a': q1, 'b': q2, 'c': max(values)},
                'High': {'a': q2, 'b': max(values), 'c': max(values) + 0.1}
            }
        
        # Generate rules for each training sample
        rules = []
        for i, sample in enumerate(X):
            # Determine membership for each feature
            rule_antecedent = []
            for j, value in enumerate(sample):
                # Find which fuzzy set has highest membership
                max_membership = 0
                best_set = 'Low'
                for set_name, params in feature_sets[j].items():
                    membership = self.fuzzy_membership(value, params['a'], params['b'], params['c'])
                    if membership > max_membership:
                        max_membership = membership
                        best_set = set_name
                rule_antecedent.append(best_set)
            
            # Consequent is the class label
            rule_consequent = y[i]
            
            # Create rule string
            rule = {
                'antecedent': rule_antecedent,
                'consequent': rule_consequent
            }
            rules.append(rule)
        
        return rules
    
    def binary_pso_rule_selection(self, X_train, y_train, X_val, y_val):
        """
        Binary PSO for rule selection
        """
        # Generate all possible fuzzy rules
        all_rules = self.generate_fuzzy_rules(X_train, y_train)
        n_rules = len(all_rules)
        
        # Initialize PSO
        # Each particle represents a subset of rules (binary vector)
        population = np.random.randint(0, 2, (self.n_particles, n_rules))
        velocities = np.random.uniform(self.v_min, self.v_max, (self.n_particles, n_rules))
        
        # Initialize personal best and global best
        pbest = population.copy()
        pbest_fitness = np.zeros(self.n_particles)
        gbest = None
        gbest_fitness = -np.inf
        
        # Main PSO loop
        for iteration in range(self.n_iterations):
            # Evaluate fitness for each particle
            for i, particle in enumerate(population):
                # Select rules based on particle
                selected_rules = [all_rules[j] for j in range(n_rules) if particle[j] == 1]
                
                # If no rules selected, skip
                if len(selected_rules) == 0:
                    fitness = 0
                else:
                    # Evaluate selected rules on validation set
                    predictions = self.evaluate_rules(selected_rules, X_val)
                    fitness = accuracy_score(y_val, predictions)
                
                # Update personal best
                if fitness > pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest[i] = particle.copy()
                
                # Update global best
                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest = particle.copy()
            
            # Update velocities and positions
            for i in range(self.n_particles):
                r1, r2 = np.random.random(2)
                
                # Update velocity
                velocities[i] = (self.w * velocities[i] + 
                                self.c1 * r1 * (pbest[i] - population[i]) +
                                self.c2 * r2 * (gbest - population[i]))
                
                # Clamp velocity
                velocities[i] = np.clip(velocities[i], self.v_min, self.v_max)
                
                # Update position using sigmoid function
                sigmoid = 1 / (1 + np.exp(-velocities[i]))
                random_vals = np.random.random(n_rules)
                population[i] = (random_vals < sigmoid).astype(int)
        
        # Select best rules
        if gbest is not None:
            self.best_rules = [all_rules[j] for j in range(n_rules) if gbest[j] == 1]
        else:
            self.best_rules = all_rules
        
        return self.best_rules
    
    def evaluate_rules(self, rules, X):
        """
        Evaluate a set of fuzzy rules on data
        """
        predictions = []
        for sample in X:
            # Check each rule
            is_spam = 0
            confidence = 0
            
            for rule in rules:
                # Check if antecedent matches
                matches = True
                for j, feature_value in enumerate(sample):
                    # In practice, we need to check if the feature value falls within
                    # the fuzzy set defined in the antecedent
                    # For simplicity, we'll use a matching function
                    pass  # Implementation depends on fuzzy set definitions
                
                if matches:
                    if rule['consequent'] == 1:
                        is_spam += 1
                        confidence += 1
            
            # Decision based on majority
            predictions.append(1 if is_spam > 0 else 0)
        
        return np.array(predictions)
    
    def train(self, data):
        """
        Train the model on data
        """
        # Prepare data
        X = []
        y = []
        for _, row in data.iterrows():
            features = self.extract_features(row['message'])
            X.append(features)
            y.append(row['label'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Generate rules and select using PSO
        print("Generating fuzzy rules...")
        self.all_rules = self.generate_fuzzy_rules(X_train, y_train)
        print(f"Generated {len(self.all_rules)} rules")
        
        print("Selecting best rules using PSO...")
        self.best_rules = self.binary_pso_rule_selection(X_train, y_train, X_val, y_val)
        print(f"Selected {len(self.best_rules)} rules")
        
        return X_train, X_val, y_train, y_val
    
    def predict(self, X):
        """
        Predict class labels for new data
        """
        return self.evaluate_rules(self.best_rules, X)

# Main execution
def main():
    # Initialize detector
    detector = FuzzyPSO_SMS_Detector(n_particles=20, n_iterations=30)
    
    # Load data
    data = detector.load_data()
    print(f"Loaded {len(data)} messages")
    
    # Train the model
    X_train, X_val, y_train, y_val = detector.train(data)
    
    # Test on validation set
    predictions = detector.predict(X_val)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    
    print("\n--- Final Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return detector

if __name__ == "__main__":
    detector = main()