 # encoding=utf-8
# -*- coding: utf-8 -*-

from django.shortcuts import render
from django.http import HttpResponse
from .models import Documentdata
import os
import heapq
import math
import re
import nltk 
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from collections import defaultdict



lang=None
folder_path=None
wg_alg=None
sum_text=None
outtext=None





    
def index(request):
    global lang
    if request.method == 'POST':

            
        folder_path = request.POST.get('folder_path')
        
        lang = request.POST.get('language')
        wg_alg= request.POST.get('WAlg')
        word_files = []
        
       
        for file in os.listdir(folder_path):

            if file.endswith(".docx") or file.endswith(".doc") or file.endswith(".txt"):
                 

                 word_files.append(file)
                 print(word_files)
                 file_path = os.path.join(folder_path, file)  
                 file = open(file_path, "r")

                 content = file.read()
                 if (wg_alg=="TF-IDF"):
                     
                    
                    outtext=tf_idf(content)
                
                 if(wg_alg=="Bag of words"):
                    outtext=Bow(content)
 
                 file.close()          
                 data = Documentdata(
                 file_name=file_path,
                 content=outtext,
                 language=lang,
                 weighting_algorithm=wg_alg
                )
                             
                 data.save()
                 latest_row = Documentdata.objects.last()
                 #print(latest_row.content)
                 file1=open("dtabas.txt","w")
                 file1.write(file_path+'\n'+wg_alg+'\n'+str(latest_row.content))
                 count=len(word_files)
                 context = {'folder_path':folder_path ,'lang': lang  ,'In_alg':wg_alg , 'message':"the files indexed successfully"}
                
     
                # return HttpResponse(render_response,"<script>alert('Word file indexed successfully');</script>")
                 

            #else:
             #    
              #   return render(request, 'index.html')
                 
        return render(request,'search.html', context)
               
        #return HttpResponse("<h1>Indexing completed successfully!</h1> <a href='search'> Go to Search Page</a>")
    else:
        return render(request, 'index.html')


def search(request):
    search_results=""
    result_count=0
    if request.method == 'GET':
        query = request.GET.get('query')
        algorithm = request.GET.get('algorithm')

        # Logic for conducting searches using the specified algorithm
        # Example:
        if algorithm == 'boolean':
           num_words=20
           search_results = boolean_model_search(query)
           print(search_results)
          
          
           
        elif algorithm == 'extended_boolean':
            search_results = extended_boolean_model_search(query)
        elif algorithm == 'vector':
            search_results = vector_model_search(query)
            
        result_count = len(list(search_results))
        return render(request, 'search.html',  {'results': search_results,'result_count':search_results,'query':query,'result_count':result_count})
    else:
    
        return render(request, 'search.html')
    
    return render(request, 'search.html')
#{'results': search_results}





def boolean_model_search(query):
    print("Boolean used")
    # Tokenize the query
    query_terms = query.lower().split()

    # Construct the SQL query dynamically
    condition = " AND ".join(f"content LIKE '%%{term}%%'" for term in query_terms)
    sql_query = f"SELECT * FROM Appfiles_Documentdata  WHERE {condition};"

    # Perform the query
    results = Documentdata.objects.raw(sql_query)
  

       
  
      


    return results
    

def extended_boolean_model_search(query):
    # Tokenize the query
    print("extended Boolean used")
    # Tokenize the query
    query_terms = query.lower().split()

    # Construct the SQL query dynamically
    condition = " OR ".join(f"content LIKE '%%{term}%%'" for term in query_terms)
    sql_query = f"SELECT * FROM Appfiles_Documentdata  WHERE {condition};"

    # Perform the query
    results = Documentdata.objects.raw(sql_query)
    
  
    return results
    
#--------------------------------------------------------
def vector_model_search(query):
    query_terms = query.lower().split()
    condition = "".join(f"content LIKE '%%{term}%%'" for term in query_terms)
    sql_query = f"SELECT * FROM Appfiles_Documentdata  WHERE {condition};"
    results = Documentdata.objects.raw(sql_query)
    
    create_vector(results, query_terms)
    
   

    def calculate_tf(term, results):
        return results.split().count(term)

    def calculate_idf(term, results):
        if isinstance(results, set):
            document_count = 1 if term in results else 0
        else:
            document_count = sum(1 for doc in results.values() if term in doc)
        return math.log(len(documents) / (document_count + 1e-10))

    def create_vector(results, query_terms):
        vector = {}
        for term in query_terms:
            tf = calculate_tf(term, results)
            idf = calculate_idf(term, {results})  # Pass a set containing the current document
            vector[term] = tf * idf
        
        # Remove terms with zero vector values
        vector = {term: value for term, value in vector.items() if value != 0}
        
        return vector


    def cosine_similarity(vector1, vector2):
        dot_product = sum(vector1.get(term, 0) * vector2.get(term, 0) for term in set(vector1) & set(vector2))
        magnitude1 = math.sqrt(sum(value ** 2 for value in vector1.values()))
        magnitude2 = math.sqrt(sum(value ** 2 for value in vector2.values()))
        return dot_product / (magnitude1 * magnitude2 + 1e-10)

    def rank_documents(query_vector, document_vectors):
        rankings = [(doc, cosine_similarity(query_vector, doc_vector)) for doc, doc_vector in document_vectors.items()]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    def show_b(query):
        all_documents = results
        
        
        # Create vectors only for documents containing the query terms
        document_vectors = {doc: create_vector(content, query_terms) for doc, content in all_documents.items() if any(term in content for term in query_terms)}
        
        # If no documents contain the query terms, return "None"
        if not document_vectors:
            return "None"
        
        query_vector = create_vector(" ".join(all_documents.values()), query_terms)

        rankings = rank_documents(query_vector, document_vectors)

        result_text = ", ".join(doc for doc, _ in rankings) if rankings else "None"
        return result_text




























def tf_idf(text):
    global lang
  
    if(lang=="Arabic"):
        text = re.sub(r'\s*[A-Za-z]\s*', ' ' , text)
        #remove hashtags
        text = re.sub("#", " ", text)
        text = re.sub(r'\[0-9]*\]',' ',text)
        text = re.sub(r'\s+',' ',text)
        text = re.sub(r'\[^a-zA-Z]',' ',text)
        text = re.sub(r'\s+',' ',text)
        text = re.sub("-", " ", text)
        text = re.sub(r'•', '', text)
       
    elif(lang=="English") :
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\[0-9]*\]',' ',text)
        text = re.sub("-", " ", text)
        text = re.sub(r'•', '', text)
        


    sentences_tokens = text.split(".")
    documents = [];
   
    for sen in sentences_tokens:
        if len(sen)>2:
            documents.append(sen)
            
    # ============================     
    #2====== PreProcessing: Tokenization====== 
    dictOfWords = {}
    for index, sentence in enumerate(documents):
        tokenizedWords = sentence.split(' ')
        dictOfWords[index] = [(word,tokenizedWords.count(word)) for word in tokenizedWords]

    # ============================     


    #3 ====== Calculate term frequency (TF)====== 
    termFrequency = {}
    for i in range(0, len(documents)):
        listOfNoDuplicates = []
        for wordFreq in dictOfWords[i]:
            if wordFreq not in listOfNoDuplicates:
                listOfNoDuplicates.append(wordFreq)
            termFrequency[i] = listOfNoDuplicates

    normalizedTermFrequency = {}
    for i in range(0, len(documents)):
        sentence = dictOfWords[i]
        lenOfSentence = len(sentence)
    #    print(lenOfSentence)
        listOfNormalized = []
        for wordFreq in termFrequency[i]:
            normalizedFreq = wordFreq[1]/lenOfSentence
            listOfNormalized.append((wordFreq[0],normalizedFreq))
        normalizedTermFrequency[i] = listOfNormalized


    allDocuments = ''
    for sentence in documents:
        allDocuments += sentence + ' '
    allDocumentsTokenized = allDocuments.split(' ')
    # ============================   
    #---Calculate IDF
    allDocumentsNoDuplicates = []
    for word in allDocumentsTokenized:
        if word not in allDocumentsNoDuplicates:
            allDocumentsNoDuplicates.append(word)
            
            

    dictOfNumberOfDocumentsWithTermInside = {}
    # ovc = vocabilary OR word
    for index, voc in enumerate(allDocumentsNoDuplicates):
        count = 0
        for sentence in documents:
            if voc in sentence:
                count += 1
        dictOfNumberOfDocumentsWithTermInside[index] = (voc, count)

    dictOFIDFNoDuplicates = {} 

    for i in range(0, len(normalizedTermFrequency)):
        listOfIDFCalcs = []
        for word in normalizedTermFrequency[i]:
            for x in range(0, len(dictOfNumberOfDocumentsWithTermInside)):
                if word[0] == dictOfNumberOfDocumentsWithTermInside[x][0]:
                    listOfIDFCalcs.append((word[0],math.log(len(documents)/dictOfNumberOfDocumentsWithTermInside[x][1])))
        dictOFIDFNoDuplicates[i] = listOfIDFCalcs



    dictOFTF_IDF = {}
    for i in range(0,len(normalizedTermFrequency)):
        listOFTF_IDF = []
        TFsentence = normalizedTermFrequency[i]
        IDFsentence = dictOFIDFNoDuplicates[i]
        for x in range(0, len(TFsentence)):
           # print(TFsentence[x][0])
           # print(TFsentence[x][1])
            listOFTF_IDF.append((TFsentence[x][0],TFsentence[x][1]*IDFsentence[x][1]))
        dictOFTF_IDF[i] = listOFTF_IDF

 
    summary = heapq.nlargest(5, dictOFTF_IDF, key=dictOFTF_IDF.get)
    #print(" ========The Summary line via TF IDF approach=========\n")
    #for s in summary:
        #print(documents[s])
        
    sum_text=' '.join(documents)
    return sum_text
       
        
def Bow(text):
    global language
    
    # ============================     
    # ====== Clean input by removing all the HTML tags information======
    if(lang=="Arabic"):
        text = re.sub(r'\s*[A-Za-z]\s*', ' ' , text)
        #remove hashtags
        text = re.sub("#", " ", text)
        text = re.sub(r'\[0-9]*\]',' ',text)
        text = re.sub(r'\s+',' ',text)
        text = re.sub(r'\[^a-zA-Z]',' ',text)
        text = re.sub(r'\s+',' ',text)
        text = re.sub("-", " ", text)
        text = re.sub(r'•', '', text)
        
    elif(lang=="English") :
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\[0-9]*\]',' ',text)
        text = re.sub("-", " ", text)
        text = re.sub(r'•', '', text)  
        
        
  
    # ============================     
    # ====== PreProcessing: Sentence Splitting======  
    sentences_tokens = nltk.sent_tokenize(text)
    # ============================     
    # ====== PreProcessing: Tokenization======  
    words_tokens = nltk.word_tokenize(text)
    # ============================     
    # ====== PreProcessing: Arabic StopWords List======  
    stopwords_list = stopwords.words('arabic')
    # ============================     
    # ====== PreProcessing: Arabic Stemming======  
    st = ISRIStemmer()
    words_stemm = [st.stem(word) for word in words_tokens]
    # ============================     
    # ====== Calculate Each Term Frequency======  
    word_frequencies = {}
    for word in words_stemm:
        if word not in stopwords_list:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    #Get MAX
    maximum_frequency_word = max(word_frequencies.values())
    #Normalize
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency_word)
    # ============================     
    # ====== Calculate Each Sentence Frequency from the Term Frequencies======     
    #Calculate Sentence scores based on each word within sentence
    sentences_scores = {}
    wordsCounter = 0.0
    for sentence in sentences_tokens:
        for word in nltk.word_tokenize(sentence):
            word = st.stem(word)
            if word in word_frequencies.keys():
                wordsCounter += 1;
                if sentence not in sentences_scores.keys():
                    sentences_scores[sentence] = word_frequencies[word]
                else:
                    sentences_scores[sentence] += word_frequencies[word]
        #======Normalize======
        sentences_scores[sentence] = sentences_scores[sentence]/wordsCounter
        #Reset Counter
        wordsCounter = 0
    # ============================     
    # ====== Print Summary======  
    #Get summary with only highest top 3 
     
    summary = heapq.nlargest(10, sentences_scores, key=sentences_scores.get)
    #print("\nBag of Words based summary:\n")
    #print(summary)
    return summary
        

   