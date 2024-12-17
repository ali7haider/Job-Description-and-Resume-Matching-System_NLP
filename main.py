from flask import Flask,request,render_template
import os
app=Flask(__name__)
app.config['UPLOAD_FOLDER']='uploads/'
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