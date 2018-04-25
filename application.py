import os
import gensim
import difflib
from gensim.models import word2vec
from flask import Flask, jsonify, request, render_template, make_response, redirect, url_for, flash, session
from flask_wtf import FlaskForm
from wtforms import StringField
import logging
import json
import string

with open('static/assets/perfumeslist.txt') as f:
    perfumeslist = f.readlines()
perfumeslist = [x.strip() for x in perfumeslist]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


model = gensim.models.Word2Vec.load('doc_tensor_perfume.doc2vec')
model.init_sims(replace=True)

app = Flask(__name__)
#app._static_folder = '/static'
app.config['SECRET_KEY']='\x9d|\xe4\x91\r<\xd8\x94\xf6\xa1\xde\xf7\xdc\x0b\x10\xb6\x92o9\xfe\x1e\xd8\xe9W'
class CompareForm(FlaskForm):
    name = StringField('name')

@app.route("/", methods=['GET', 'POST'])
def home():
    form = CompareForm()
    if form.validate_on_submit():
        name = request.form['name']
        number = request.form['number']
    return render_template("index.html")

@app.route('/similardoc',  methods=['GET', 'POST'])
def similardoc():
    name = request.form['name']
    if name:
        try:
            searchterm = difflib.get_close_matches(str(name), perfumeslist, 1, 0.8)
            sim = model.docvecs.most_similar(positive=searchterm, topn=10)
            return jsonify({"name": sim, "search": 'searched fragrances similar to: ' + str(searchterm[0])})
    
        except ValueError:
            sim = model.docvecs.most_similar(positive=[model.infer_vector(str(name))], topn=10)
            
            if sim:
                return jsonify({"name": sim, "search": 'searched fragrances using keywords: ' + str(name)})
    
    return jsonify({'error' : 'Could not find your perfume!'})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)