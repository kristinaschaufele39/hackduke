import torch
from flask import Flask, render_template,request
from sentence_transformers import SentenceTransformer, util
from annoy import AnnoyIndex

app = Flask(__name__)

# Using an instance of SBERT to create the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Single list of sentences - Possible tens of thousands of sentences --> input
sentences = ['The cat sits outside',
             'A man is playing guitar',
             'I love pasta',
             'The new movie is awesome',
             'The cat plays in the garden',
             'A woman watches TV',
             'The new movie is so great',
             'Do you like pizza?']

paraphrases = util.paraphrase_mining(model, sentences)

for paraphrase in paraphrases[0:10]:
    score, i, j = paraphrase
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], score))

@app.route("/")
def msg():
    return render_template('index.html')

#creates page to route to, using a function
@app.route("/summarize", methods=['POST','GET'])
def getTransformed(): #changed from summary
    body=request.form['data']
    result = model(body, num_sentences=5)
    return render_template('summary.html',result=result)

if __name__ =="__main__":
    app.run(debug=True,port=8000)

