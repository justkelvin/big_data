#!/usr/bin/env python3

# Imports
try:
    import importlib
    import os
    import re

    import nltk
    import numpy
    import numpy.linalg as LA
    import pandas as pd
    from bs4 import BeautifulSoup as BS
    from config import Database as DB
    from dotenv import load_dotenv
    from nltk.stem import PorterStemmer
    from num2words import num2words
    from numpy import dot
    from numpy.linalg import norm
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from colorama import Fore, Back, Style
except ModuleNotFoundError:
    print(Fore.RED + "ModuleNotFoundError: Please run the following command to install the required modules: pip install -r requirements.txt" + Style.RESET_ALL)
    exit()

# Load the .env file
load_dotenv()

class Lab4: 
    database_instance = '' 
    PREPROCESSED_DOCUMENT = {}
    cosine_similarity_result = {}
    
    def __init__(self) -> None:
        self.database_instance = DB.Database(
            self.env('MYSQL_DB_HOST'),
            self.env('MYSQL_DB_USER'),
            self.env('MYSQL_DB_PASSWORD'),
            self.env('MYSQL_DB_DATABASE')
        )
    
    def env(self, variable):
        """ Get the value of an environment variable """
        return os.getenv(variable.upper())

    def read_lab3_extracted_data_table(self):
        """ Read the data from the lab3_extracted_data table """

        self.mydb, self.cursor = self.database_instance.db_connection()
        query =  f"SELECT * FROM `{self.env('LAB3_EXTRACTED_DATA_SQL_TABLE')}`"
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def part_1(self):
        """ Part 1 of the lab """
        
        # Get the raw document
        DOCUMENT_DICT = self.raw_document()
        
        # Preprocess the document
        print(Fore.GREEN + "[-] Document preprocessing." + Style.RESET_ALL)

        for key, value in DOCUMENT_DICT.items():
            self.PREPROCESSED_DOCUMENT[key] = self.preprocessor(value)

        print(Fore.GREEN + "[+] Document preprocessing completed." + Style.RESET_ALL)
        self.build_up_inverted_index() 
    
    def nltk_check(self):
        """ Check if the nltk packages are installed """
        stopwords = importlib.util.find_spec("stopwords")
        punkt = importlib.util.find_spec("punkt")
        wordnet = importlib.util.find_spec("wordnet")
        
        # Download the nltk packages if not installed already
        if stopwords is None:
            nltk.download('stopwords', quiet=True)
        
        if punkt is None:
            nltk.download('punkt', quiet=True)

        if wordnet is None:
            nltk.download("wordnet", quiet=True)

    def raw_document(self):
        """ Get the raw document """
        self.nltk_check()
        DOCUMENT_DICT = {}
        for row in self.read_lab3_extracted_data_table():
            DOCUMENT_DICT[row[0]] = row[1]
        return DOCUMENT_DICT
    
    def preprocessor(self, doc_content):
        """ Preprocess the document """
        from nltk.corpus import stopwords

        # Punctuate by removing Puntuation Marks
        document = []
        for word in doc_content:
            document.append(re.sub(r'[^\w\s]', '', word))
        
        punctuated_document = " " . join(document)

        # Tokenize the document
        tokenized_document = nltk.word_tokenize(punctuated_document)

        # change numbers to words
        number_to_words = []
        for word in tokenized_document:
            if word.isdigit():
                number_to_words.append(num2words(word))
            else:
                number_to_words.append(word)

        # suffix stripping
        suffix_stripped_document = []
        PorterStemmerInstance = PorterStemmer()

        for word in number_to_words:
            suffix_stripped_document.append(PorterStemmerInstance.stem(word))

        # Stopword removal and lowercasing
        stopword_removed_document = []
        for word in suffix_stripped_document:
            if word not in stopwords.words('english'):
                stopword_removed_document.append(word.lower())
            else:
                stopword_removed_document.append(word)
        
        # Convert words from uppercase to lower
        lowercase_document = []
        for word in stopword_removed_document:
            lowercase_document.append(word.lower())

        return lowercase_document
    
    def build_up_inverted_index(self):
        """ Build up the inverted index """

        print(Fore.GREEN + "[-] Building Inverted Index Tables." + Style.RESET_ALL)

        # Get the preprocessed document
        TF_DF = {}
        for key, value in self.PREPROCESSED_DOCUMENT.items():
            DF = {}
            for word in value:
                if word in DF:
                    DF[word] += 1
                else:
                    DF[word] = 1

            TF_DF[key] = DF
        
        pd.set_option('display.max_rows', None)
        df = pd.DataFrame.from_dict(TF_DF, orient='index')
        df = df.replace(numpy.nan, 0)
        df.to_csv("./output/Document_Term_Df.csv")
        print(Fore.GREEN + "[+] Document Term DF Table saved to ./output/Document_Term_Df.csv" + Style.RESET_ALL)

        DICTIONARY_FILE = {}
        POSTING_FILE = {} 
        Doc_Ids = []

        for file_name, TD in TF_DF.items():
            Doc_Ids.append(file_name)
        
        for file_name, TD in TF_DF.items():
            for word, freq in TD.items():
                if word in DICTIONARY_FILE.keys():
                    DICTIONARY_FILE[word] = {'DocFreq': file_name, 'CollectionFreq': DICTIONARY_FILE[word]['CollectionFreq'] + 1}
                else:
                    DICTIONARY_FILE[word] = {'DocFreq': file_name, 'CollectionFreq': freq}

                if word in POSTING_FILE.keys():
                    POSTING_FILE[word] = {'Doc Id' : [*set(Doc_Ids)].index(file_name), 'TermFreq': POSTING_FILE[word]['TermFreq'] + 1}
                else:
                    POSTING_FILE[word] = {'Doc Id' : [*set(Doc_Ids)].index(file_name), 'TermFreq': freq}

        # Save the dictionary file
        df = pd.DataFrame.from_dict(DICTIONARY_FILE, orient='index')
        df.index.name = 'Term'
        df = df.replace(numpy.nan, 0)
        df.transpose().to_csv("./output/DictionaryFile.csv")
        print(Fore.GREEN + "[+] Dictionary File saved to ./output/DictionaryFile.csv" + Style.RESET_ALL)

        # Save the posting file
        data = pd.read_csv(f"./output/DictionaryFile.csv", encoding='utf-8')
        df = pd.DataFrame(data, columns= ['Term', 'DocFreq', 'CollectionFreq'])

        for row in df.values.tolist():
            try:
                self.Insert_Dictionary_File(tuple(row))
            except:
                pass

        if True:
            # Save the posting file
            print(Fore.GREEN + "[+] Successfully Posted First Level Dictionary Table!" + Style.RESET_ALL)
        
        df = pd.DataFrame.from_dict(POSTING_FILE, orient='index')
        df.index.name = 'Term'
        df = df.replace(numpy.nan, 0)
        df.transpose().to_csv("./output/PostingFile.csv")
        print(Fore.GREEN + "[-] Posting File saved to ./output/PostingFile.csv" + Style.RESET_ALL)
        
        data = pd.read_csv(f"./output/PostingFile.csv", encoding='utf-8')
        df = pd.DataFrame(data, columns=['Term', 'Doc Id', 'TermFreq'])

        for row in df.values.tolist():
            try:
                self.Insert_Posting_File(tuple(row)) 
            except:
                pass
        
        if True:
            print(Fore.GREEN + "[+] Successfully Posted Second Level Posting Table!" + Style.RESET_ALL)
    
        self.cursor.close()
        self.mydb.close()

    def Insert_Dictionary_File(self, values):
        """ Insert the values into the dictionary file """
        sql = f"INSERT INTO '{self.env('FIRST_LEVEL_DICTIONARY_TABLE')}' (Term, DocFreq, CollectionFreq) VALUES (%s, %s, %s)"
        self.cursor.execute(sql, values)
        self.mydb.commit()

    def Insert_Posting_File(self, values):
        """ Insert the values into the posting file """
        sql = f"INSERT INTO '{self.env('SECOND_LEVEL_POSTING_TABLE')}' (Term, DocId, TermFreq) VALUES (%s, %s, %s)"
        self.cursor.execute(sql, values)
        self.mydb.commit()
        
    def part_2(self):
        """ Part 2 of the assignment """

        DOCUMENT_DICT = self.raw_document()
        print(Fore.GREEN + "[-] Preprocessing of the documents started. Please wait..." + Style.RESET_ALL)
        
        for key, value in DOCUMENT_DICT.items():
            processed_text = self.preprocessor(value)
            self.PREPROCESSED_DOCUMENT[key] = processed_text

        self.vectorize_document(self.PREPROCESSED_DOCUMENT)
        self.show_cosine_similarity(self.cosine_similarity_result)       

    def vectorize_document(self, processed_text_document):
        """ Vectorize the document """
        
        vectorizer = TfidfVectorizer()
        for doc_name, doc_content in processed_text_document.items():
            temp_cs_dict = {}
            train_corpus = [" " . join(doc_content)]
            try:
                X = vectorizer.fit_transform(train_corpus)
                X = X.toarray()
                for doc_name_2, doc_content_2 in processed_text_document.items():
                    temp_cs_dict[doc_name_2] = self.find_cosine_similarity(X, vectorizer.transform([" ".join(doc_content_2)]).toarray())
            except ValueError:
                continue
        self.cosine_similarity_result[doc_name] = temp_cs_dict

    def show_cosine_similarity(self, cosine_similarity_result):
        """ Show the cosine similarity """
        print(Fore.GREEN + "[-] Showing the cosine similarity. Please wait..." + Style.RESET_ALL)
        DFrame = pd.DataFrame.from_dict(cosine_similarity_result, orient='index')
        DFrame = DFrame.replace(numpy.nan, 0)
        DFrame.to_csv("./output/Cosine_Similarity.csv")
        print(Fore.GREEN + "[+] Cosine Similarity Table saved to ./output/Cosine_Similarity.csv" + Style.RESET_ALL)
    
    def find_cosine_similarity(self, a, b):
        """ Find the cosine similarity """
        a = numpy.array(a)
        b = numpy.array(b)
        dot_product = numpy.dot(a, b)
        magnitude = numpy.linalg.norm(a) * numpy.linalg.norm(b)
        if not magnitude:
            return 0
        return dot_product / magnitude