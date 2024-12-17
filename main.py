from flask import Flask,request,render_template
import os
import PyPdf2
import docx2txt

app=Flask(__name__)
app.config['UPLOAD_FOLDER']='uploads/'


# helper functions
def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""
@app.route("/")
def matchReume():
    return render_template('index.html')
@app.route("/matcher",methods=['GET','POST'])
def matcher():
    if request.method=='POST':
        job_description=request.form.get('job_description')
        resume_files=request.form.getList('resumes')

        resumes=[]
        for resume_file in resume_files:
            filename=os.path.join(app.config['UPLOAD_FOLDER'],resume_file.filename)
            resume_file.save(filename)
            resumes.append(filename)

def matchReume():
    return render_template('index.html')
if __name__=="__main__":
    app.run(debug=True)