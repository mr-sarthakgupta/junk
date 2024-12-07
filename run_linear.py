import torch
import torch.nn as nn
import numpy as np
import string
import random
import re
import collections
import os
import json
import pickle

def generate_letter_cofrequency_matrices(words):
        """
        Generate co-frequency matrices for letters that succeed and precede each other.
        
        Parameters:
        words (list): List of words to analyze
        
        Returns:
        tuple: (succeeding_matrix, preceding_matrix)
        """
        # Create lowercase alphabet for matrix indexing
        alphabet = string.ascii_lowercase
        
        # Initialize matrices with zeros
        succeeding_matrix = np.zeros((26, 26), dtype=int)
        preceding_matrix = np.zeros((26, 26), dtype=int)
        
        # Process each word
        for word in words:
            # Convert to lowercase
            word = word.lower()
            
            # Analyze letter successions
            for i in range(len(word) - 1):
                # Current and next letter
                current_letter = word[i]
                next_letter = word[i + 1]
                
                # Skip if either letter is not in alphabet
                if current_letter not in alphabet or next_letter not in alphabet:
                    continue
                
                # Get matrix indices
                current_idx = alphabet.index(current_letter)
                next_idx = alphabet.index(next_letter)
                
                # Increment succeeding matrix
                succeeding_matrix[current_idx, next_idx] += 1
            
            # Analyze letter precedences
            for i in range(1, len(word)):
                # Current and previous letter
                current_letter = word[i]
                prev_letter = word[i - 1]
                
                # Skip if either letter is not in alphabet
                if current_letter not in alphabet or prev_letter not in alphabet:
                    continue
                
                # Get matrix indices
                current_idx = alphabet.index(current_letter)
                prev_idx = alphabet.index(prev_letter)
                
                # Increment preceding matrix
                preceding_matrix[current_idx, prev_idx] += 1
        
        return succeeding_matrix, preceding_matrix


with open("words_train_split.txt", "r") as file:
        words = file.read().splitlines()
        
succeeding_matrix, preceding_matrix = generate_letter_cofrequency_matrices(words)


from collections import Counter
# Join all the words in the training set into a single string
all_letters = ''.join(words)

# Count the frequency of each letter
letter_counts = Counter(all_letters)

# Sort the letters by frequency in decreasing order
sorted_letters_by_frequency = sorted(letter_counts.items(), key=lambda item: item[1], reverse=True)

# Extract just the letters in sorted order
sorted_letters = [letter for letter, count in sorted_letters_by_frequency]


with open("words_train_split.txt", "r") as file:
    words_split = file.read().splitlines()

# Generate bigrams from the words
bigrams = [word[i:i+2] for word in words_split for i in range(len(word) - 1)]

# Count the frequency of each bigram
bigram_counts = Counter(bigrams)

# Get the 100 most common bigrams
most_common_bigrams = bigram_counts.most_common(len(bigram_counts))

most_common_bigrams = [bigram for bigram, count in most_common_bigrams]



trigrams = [word[i:i+3] for word in words_split for i in range(len(word) - 3)]

# Count the frequency of each bigram
trigram_counts = Counter(trigrams)

# Get the 100 most common bigrams
most_common_trigrams = trigram_counts.most_common(len(trigram_counts))

most_common_trigrams = [trigram for trigram, count in most_common_trigrams]




qgrams = [word[i:i+4] for word in words_split for i in range(len(word) - 4)]

# Count the frequency of each bigram
qgram_counts = Counter(qgrams)

# Get the 100 most common bigrams
most_common_qgrams = qgram_counts.most_common(len(qgram_counts))

most_common_qgrams = [qgram for qgram, count in most_common_qgrams]

class MulticlassHammingClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim[0] * input_dim[1], num_classes)
        self.linear.requires_grad = True
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.softmax(self.linear(x))

class HangmanGame:
    def __init__(self, full_dictionary_location='words_train_split.txt'):

        self.model = MulticlassHammingClassifier(input_dim=(26, 6), num_classes=26)
        self.model.load_state_dict(torch.load('models/model_0.2915651202201843.pth'))
        self.model.to('cuda')

        self.guessed_letters = []

        full_dictionary_location = "words_train_split.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)

        val_dictionary_path = 'words_val_split.txt'
        self.val_dictionary = self.build_dictionary(val_dictionary_path)
            
        self.alphabet = string.ascii_lowercase

        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        
        self.current_dictionary = []        

        self.succeeding_matrix, self.preceding_matrix = succeeding_matrix, preceding_matrix

    # def guess(self, word, succeeding_matrix, preceding_matrix):
    #     # Predefined frequency lists
    #     letters_by_frequency = ['e',
    #                             'i',
    #                             'a',
    #                             'n',
    #                             'o',
    #                             'r',
    #                             's',
    #                             't',
    #                             'l',
    #                             'c',
    #                             'u',
    #                             'd',
    #                             'p',
    #                             'm',
    #                             'h',
    #                             'g',
    #                             'y',
    #                             'b',
    #                             'f',
    #                             'v',
    #                             'k',
    #                             'w',
    #                             'z',
    #                             'x',
    #                             'q',
    #                             'j']
    #     letters_by_frequency = letters_by_frequency[::-1]

    #     bigrams_by_frequency = most_common_bigrams
    #     bigrams_by_frequency = bigrams_by_frequency[::-1] 


    #     trigrams_by_frequency = most_common_trigrams
    #     trigrams_by_frequency = trigrams_by_frequency[::-1]


    #     quadgrams_by_frequency = most_common_qgrams        
    #     quadgrams_by_frequency = quadgrams_by_frequency[::-1]


    #     # Clean the word, stripping spaces and replacing "_" with placeholders
    #     clean_word = word[::2].replace("_", ".")
    
    #     # Score mechanism for letter selection
    #     letter_scores = {}
        
    #     # 1. Single Letter Frequency - Initial Base Score
    #     for letter in letters_by_frequency:
    #         if letter not in self.guessed_letters:
    #             letter_scores[letter] = [0]*6

    #     for letter in letters_by_frequency:
    #         if letter not in self.guessed_letters:
    #             letter_scores[letter][0] += letters_by_frequency.index(letter) + 1

    #     # 2. Bigram and  Scoring with Contextual Constraint
    #     for i in range(len(clean_word) - 1):
    #         # Extract 2-letter window
    #         window = clean_word[i:i+2]
            
    #         # Count known letters in the window
    #         known_letters_count = sum(1 for char in window if char != '.')
            
    #         # Only apply bigram scoring if 1 or more letters are known
    #         if known_letters_count == 1:
    #             for bigram in bigrams_by_frequency:
    #                 if self.is_bigram_window_match(window, bigram):
    #                     for letter in set(bigram):
    #                         if letter not in self.guessed_letters and letter not in window:
    #                             letter_scores[letter][1] += bigrams_by_frequency.index(bigram) + 1
                                        
        
    #     # 3. Trigram Scoring with Contextual Constraint
    #     for i in range(len(clean_word) - 2):
    #         # Extract 3-letter window
    #         window = clean_word[i:i+3]
            
    #         # Count known letters in the window
    #         known_letters_count = sum(1 for char in window if char != '.')
            
    #         # Only apply trigram scoring if 2 or more letters are known
    #         if known_letters_count >= 2:
    #             for trigram in trigrams_by_frequency:
    #                 if self.is_trigram_window_match(window, trigram):
    #                     for letter in set(trigram):
    #                         if letter not in self.guessed_letters and letter not in window:
    #                             letter_scores[letter][2] += trigrams_by_frequency.index(trigram) + 1
    
    #     # 4. Quadgram Scoring with Contextual Constraint
    #     for i in range(len(clean_word) - 3):
    #         # Extract 4-letter window
    #         window = clean_word[i:i+4]
            
    #         # Count known letters in the window
    #         known_letters_count = sum(1 for char in window if char != '.')
            
    #         # Only apply quadgram scoring if 2 or more letters are known
    #         if known_letters_count >= 3:
    #             for quadgram in quadgrams_by_frequency:
    #                 if self.is_quadgram_window_match(window, quadgram):
    #                     for letter in set(quadgram):
    #                         if letter not in self.guessed_letters and letter not in window:
    #                             letter_scores[letter][3] += quadgrams_by_frequency.index(quadgram) + 1

        
    #     # Fallback to most frequent unguessed letters
    #     if not letter_scores:
    #         for letter in letters_by_frequency:
    #             if letter not in self.guessed_letters:
    #                 return letter

    #     # input_vec = np.array([letter_scores[letter] if letter in letter_scores.keys() else [-1000]*6 for letter in string.ascii_lowercase])
    #     input_vec = np.array([sum(letter_scores[letter]) if letter in letter_scores.keys() else -1000*7 for letter in string.ascii_lowercase])

    #     # output = self.model(torch.tensor(input_vec).cuda().float().unsqueeze(0)).cpu().detach().numpy()

    #     guess_letter = string.ascii_lowercase[np.argmax(input)]

    #     # guess_letter = string.ascii_lowercase[np.argmax(output.squeeze())]
    #     # while string.ascii_lowercase[np.argmax(output.squeeze())] in self.guessed_letters:
    #     #     output[0, np.argmax(output)] = -100000
    #     # if clean_word == '.'*len(clean_word) and 'e' not in self.guessed_letters:
    #     #     guess_letter = 'e'
    #     # else:
    #     #     guess_letter = string.ascii_lowercase[np.argmax(output)]
    #     return guess_letter

    def guess(self, word, succeeding_matrix, preceding_matrix):
        # Predefined frequency lists
        letters_by_frequency = ['e',
                                'i',
                                'a',
                                'n',
                                'o',
                                'r',
                                's',
                                't',
                                'l',
                                'c',
                                'u',
                                'd',
                                'p',
                                'm',
                                'h',
                                'g',
                                'y',
                                'b',
                                'f',
                                'v',
                                'k',
                                'w',
                                'z',
                                'x',
                                'q',
                                'j']
        letters_by_frequency = letters_by_frequency[::-1]

        bigrams_by_frequency = most_common_bigrams
        bigrams_by_frequency = bigrams_by_frequency[::-1] 


        trigrams_by_frequency = most_common_trigrams
        trigrams_by_frequency = trigrams_by_frequency[::-1]


        quadgrams_by_frequency = most_common_qgrams        
        quadgrams_by_frequency = quadgrams_by_frequency[::-1]


        # Clean the word, stripping spaces and replacing "_" with placeholders
        clean_word = word[::2].replace("_", ".")
    
        # Score mechanism for letter selection
        letter_scores = {}
        
        # 1. Single Letter Frequency - Initial Base Score
        for letter in letters_by_frequency:
            if letter not in self.guessed_letters:
                letter_scores[letter] = [0]*6

        for letter in letters_by_frequency:
            if letter not in self.guessed_letters:
                letter_scores[letter][0] += letters_by_frequency.index(letter) + 1

        # 2. Bigram and  Scoring with Contextual Constraint
        for i in range(len(clean_word) - 1):
            # Extract 2-letter window
            window = clean_word[i:i+2]
            
            # Count known letters in the window
            known_letters_count = sum(1 for char in window if char != '.')
            
            # Only apply bigram scoring if 1 or more letters are known
            if known_letters_count == 1:
                for bigram in bigrams_by_frequency:
                    if self.is_bigram_window_match(window, bigram):
                        for letter in set(bigram):
                            if letter not in self.guessed_letters and letter not in window:
                                letter_scores[letter][1] += 676*bigrams_by_frequency.index(bigram) + 1
                                        
        
        # 3. Trigram Scoring with Contextual Constraint
        for i in range(len(clean_word) - 2):
            # Extract 3-letter window
            window = clean_word[i:i+3]
            
            # Count known letters in the window
            known_letters_count = sum(1 for char in window if char != '.')
            
            # Only apply trigram scoring if 2 or more letters are known
            if known_letters_count >= 2:
                for trigram in trigrams_by_frequency:
                    if self.is_trigram_window_match(window, trigram):
                        for letter in set(trigram):
                            if letter not in self.guessed_letters and letter not in window:
                                letter_scores[letter][2] += 26*trigrams_by_frequency.index(trigram) + 1
    
        # 4. Quadgram Scoring with Contextual Constraint
        for i in range(len(clean_word) - 3):
            # Extract 4-letter window
            window = clean_word[i:i+4]
            
            # Count known letters in the window
            known_letters_count = sum(1 for char in window if char != '.')
            
            # Only apply quadgram scoring if 2 or more letters are known
            if known_letters_count >= 2:
                for quadgram in quadgrams_by_frequency:
                    if self.is_quadgram_window_match(window, quadgram):
                        for letter in set(quadgram):
                            if letter not in self.guessed_letters and letter not in window:
                                letter_scores[letter][3] += quadgrams_by_frequency.index(quadgram) + 1

        
        # Fallback to most frequent unguessed letters
        if not letter_scores:
            for letter in letters_by_frequency:
                if letter not in self.guessed_letters:
                    return letter

        input = np.array([letter_scores[letter] if letter in letter_scores.keys() else [-1000]*6 for letter in string.ascii_lowercase])

        output_file_path = f'embeddings/{word.replace(".", "_")} ~ {self.secret_word}.txt'
        with open(output_file_path, 'w') as f:
            for letter, scores in letter_scores.items():
                f.write(f"{letter}: {scores}\n")       
        
        if clean_word == '.'*len(clean_word) and 'e' not in self.guessed_letters:
            guess_letter = 'e'
        else:
            output = self.model(torch.Tensor(input).cuda().float().unsqueeze(0)).cpu().detach().numpy()
            while string.ascii_lowercase[np.argmax(output.squeeze())] in self.guessed_letters:
                output[0, np.argmax(output)] = -100000
            guess_letter = string.ascii_lowercase[np.argmax(output.squeeze())]
        #     output = output[::-1]
        #     # guess_letter = string.ascii_lowercase[output[i]]
        #     for i in range(26):
        #         if string.ascii_lowercase[output[i]] not in self.guessed_letters:
        #             guess_letter = string.ascii_lowercase[output[i]]
        #             break

        # print(f"Guessed letter: {guess_letter}, rank: {i + 1}, score: {sum(letter_scores[guess_letter])}")
            
        return guess_letter
    
    def predict_multilabel(self, models, X_test, n_classes, threshold=0.5):
        # Predict probabilities for each label
        preds_proba = np.column_stack([
            model.predict(xgb.DMatrix(X_test)) for model in models
        ])
        
        # Convert probabilities to binary predictions
        preds_bin = (preds_proba >= threshold).astype(int)
        
        return preds_bin
    
    def is_bigram_window_match(self, window, bigram):
        """
        Check if a bigram is compatible with a 2-letter word window
        
        Example matches:
        '_ e' matches 'he'
        'h _' matches 'hi'
        """
        # Convert window to regex pattern, replacing dots with wildcards
        pattern = '^' + ''.join([c if c != '.' else '[a-z]' for c in window]) + '$'
        
        return re.match(pattern, bigram, re.IGNORECASE) is not None

    def is_trigram_window_match(self, window, trigram):
        """
        Check if a trigram is compatible with a 3-letter word window
        
        Example matches:
        '_ _ e' matches 'the'
        'h e _' matches 'her'
        """
        # Convert window to regex pattern, replacing dots with wildcards
        pattern = '^' + ''.join([c if c != '.' else '[a-z]' for c in window]) + '$'
        
        return re.match(pattern, trigram, re.IGNORECASE) is not None

    def is_quadgram_window_match(self, window, quadgram):
        """
        Check if a quadgram is compatible with a 4-letter word window
        
        Example matches:
        '_ _ e r' matches 'ther'
        'w i t _' matches 'with'
        """
        # Convert window to regex pattern, replacing dots with wildcards
        pattern = '^' + ''.join([c if c != '.' else '[a-z]' for c in window]) + '$'

        return re.match(pattern, quadgram, re.IGNORECASE) is not None

    def is_ngram_compatible(self, ngram, word_pattern):
        """
        Check if an ngram is compatible with the current word pattern
        """
        # Create a regex pattern from the ngram that respects the word pattern
        pattern = word_pattern.replace('.', '[a-z]')
        
        # Check if the ngram could exist within the pattern
        return re.search(f'(?=.{ngram}.)', pattern, re.IGNORECASE) is not None
    
    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
        
    def start_game(self, secret_word=None, verbose=True):
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
        if secret_word is None:
            secret_word = random.choice(self.val_dictionary)
        self.secret_word = secret_word
        word = ' '.join(['_' for _ in secret_word])
        tries_remains = 6
        if verbose:
            print("Successfully start a new game! # of tries remaining: {0}. Word: {1}.".format(tries_remains, word))
        while tries_remains > 0:
            # get guessed letter from user code
            guess_letter = self.guess(word, self.succeeding_matrix, self.preceding_matrix)
            
            # append guessed letter to guessed letters field in hangman object
            self.guessed_letters.append(guess_letter)
            if verbose:
                print("Guessing letter: {0}".format(guess_letter))

            if guess_letter in secret_word:
                # update word with guessed letter
                word = ' '.join([letter if letter == guess_letter else word[index*2] for index, letter in enumerate(secret_word)])
            if guess_letter not in secret_word:
                tries_remains -= 1

            if tries_remains > 0:
                status = 'ongoing'
            if '_' not in word:
                status = 'success'            
            if tries_remains == 0:
                status = 'failed'
            res = {'status': status, 'tries_remains': tries_remains, 'word': word, 'secret_word': secret_word, 'guessed_letters': self.guessed_letters}  
            if verbose:
                print(res)

            if status == 'success':
                print("Successfully finished game, the word was: {0}!".format(secret_word))
                return True

            if status == 'failed':
                print("Failed game because of: # of tries exceeded!")
                return False
            
        return status=="success"
    
game = HangmanGame()
game.start_game(secret_word='welder')

with open("words_train_split.txt", "r") as file:
        words = file.read().splitlines()

num_success = 0

for i, word in enumerate(words):
    print(i)
    if i == 1000:
        break
    game = HangmanGame()
    s = game.start_game(secret_word=word, verbose=True)
    if s:
        num_success += 1


print(f"Success rate: {num_success / 1000}")