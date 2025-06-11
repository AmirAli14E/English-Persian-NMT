"""
English-Persian Neural Machine Translation with LSTM
A seq2seq model implementation using TensorFlow/Keras
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import re
import pandas as pd

# ==================== Dataset Preparation ====================
def prepare_dataset():
    """Create and preprocess the translation dataset"""
    # Sample English-Persian parallel corpus
    data = {
        'english': [
            "hello", "how are you", "good morning", "what is your name",
            "thank you", "goodbye", "I love programming", "deep learning is amazing",
            "what time is it", "where are you going", "this is a book", "have a nice day",
            "how old are you", "I am hungry", "she is very kind", "we are happy",
            "the weather is nice", "I don't understand", "please help me", "what do you want"
        ],
        'persian': [
            "سلام", "حالت چطوره", "صبح بخیر", "اسم تو چیست",
            "متشکرم", "خداحافظ", "من عاشق برنامه‌نویسی هستم", "یادگیری عمیق شگفت‌انگیز است",
            "ساعت چنده", "کجا میری", "این یک کتاب است", "روز خوبی داشته باشی",
            "چند سالته", "من گرسنه هستم", "او بسیار مهربان است", "ما خوشحال هستیم",
            "هوا خوبه", "من متوجه نمی‌شوم", "لطفا کمکم کن", "چه می‌خواهی"
        ]
    }
    
    # Create DataFrame and expand dataset
    df = pd.DataFrame(data)
    df = pd.concat([df]*100, ignore_index=True)  # 2000 samples
    
    # Text cleaning function
    def clean_text(text):
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip().lower()
    
    # Apply cleaning and add special tokens
    english_sentences = [clean_text(sent) for sent in df['english']]
    persian_sentences = ['<start> ' + clean_text(sent) + ' <end>' for sent in df['persian']]
    
    return english_sentences, persian_sentences

# ==================== Model Configuration ====================
class Seq2SeqTranslator:
    """Sequence-to-sequence translation model"""
    
    def __init__(self, latent_dim=256, batch_size=32, epochs=30):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.eng_tokenizer = None
        self.per_tokenizer = None
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        
    def preprocess_data(self, english_sentences, persian_sentences):
        """Tokenize and prepare sequences for training"""
        # English tokenizer
        self.eng_tokenizer = Tokenizer(filters='')
        self.eng_tokenizer.fit_on_texts(english_sentences)
        eng_sequences = self.eng_tokenizer.texts_to_sequences(english_sentences)
        
        # Persian tokenizer
        self.per_tokenizer = Tokenizer(filters='')
        self.per_tokenizer.fit_on_texts(persian_sentences)
        per_sequences = self.per_tokenizer.texts_to_sequences(persian_sentences)
        
        # Calculate max lengths
        max_len_eng = max(len(seq) for seq in eng_sequences)
        max_len_per = max(len(seq) for seq in per_sequences)
        
        # Pad sequences
        eng_padded = pad_sequences(eng_sequences, maxlen=max_len_eng, padding='post')
        per_padded = pad_sequences(per_sequences, maxlen=max_len_per, padding='post')
        
        # Train-test split
        eng_train, eng_val, per_train, per_val = train_test_split(
            eng_padded, per_padded, test_size=0.2, random_state=42
        )
        
        return eng_train, eng_val, per_train, per_val, max_len_eng, max_len_per
    
    def build_model(self, vocab_size_eng, vocab_size_per):
        """Construct the seq2seq model architecture"""
        # Encoder
        encoder_inputs = Input(shape=(None,))
        enc_emb = Embedding(vocab_size_eng, self.latent_dim, mask_zero=True)(encoder_inputs)
        encoder_lstm = LSTM(self.latent_dim, return_state=True)
        _, state_h, state_c = encoder_lstm(enc_emb)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = Input(shape=(None,))
        dec_emb = Embedding(vocab_size_per, self.latent_dim, mask_zero=True)(decoder_inputs)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
        decoder_dense = Dense(vocab_size_per, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Full model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model, encoder_states
    
    def train(self, eng_train, per_train, eng_val, per_val):
        """Train the translation model"""
        self.history = self.model.fit(
            [eng_train, per_train[:, :-1]],
            per_train[:, 1:],
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=([eng_val, per_val[:, :-1]], per_val[:, 1:])
        )
    
    def build_inference_models(self):
        """Create models for prediction"""
        # Encoder inference model
        encoder_inputs = self.model.input[0]
        encoder_outputs = self.model.layers[4].output[1:]
        self.encoder_model = Model(encoder_inputs, encoder_outputs)
        
        # Decoder inference model
        decoder_inputs = self.model.input[1]
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        dec_emb = self.model.layers[5](decoder_inputs)
        decoder_lstm = self.model.layers[6]
        decoder_outputs, state_h, state_c = decoder_lstm(
            dec_emb, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_dense = self.model.layers[7]
        decoder_outputs = decoder_dense(decoder_outputs)
        
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )
    
    def translate(self, input_text, max_len=20):
        """Translate English text to Persian"""
        input_seq = self.eng_tokenizer.texts_to_sequences([input_text.lower()])
        input_padded = pad_sequences(input_seq, maxlen=self.max_len_eng, padding='post')
        
        states_value = self.encoder_model.predict(input_padded)
        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.per_tokenizer.word_index['<start>']
        
        translated = []
        for _ in range(max_len):
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)
            
            sampled_token = np.argmax(output_tokens[0, -1, :])
            word = self.per_tokenizer.index_word.get(sampled_token, '')
            
            if word == '<end>' or not word:
                break
                
            translated.append(word)
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token
            states_value = [h, c]
        
        return ' '.join(translated)

# ==================== Main Execution ====================
if __name__ == "__main__":
    print("Initializing translation model...")
    
    # Prepare data
    eng_sentences, per_sentences = prepare_dataset()
    
    # Initialize model
    translator = Seq2SeqTranslator()
    eng_train, eng_val, per_train, per_val, max_len_eng, max_len_per = translator.preprocess_data(
        eng_sentences, per_sentences)
    translator.max_len_eng = max_len_eng
    translator.max_len_per = max_len_per
    
    # Build and train model
    vocab_size_eng = len(translator.eng_tokenizer.word_index) + 1
    vocab_size_per = len(translator.per_tokenizer.word_index) + 1
    translator.model, _ = translator.build_model(vocab_size_eng, vocab_size_per)
    translator.train(eng_train, per_train, eng_val, per_val)
    
    # Prepare for inference
    translator.build_inference_models()
    
    # Test translations
    test_phrases = [
        "hello",
        "good morning",
        "what is your name",
        "I love programming"
    ]
    
    print("\nTranslation Examples:")
    for phrase in test_phrases:
        translation = translator.translate(phrase)
        print(f"{phrase} → {translation}")
