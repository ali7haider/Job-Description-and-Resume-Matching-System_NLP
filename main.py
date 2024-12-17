from flask import Flask,request,render_template

app=Flask(__name__)

@app.route("/")
def matchReume():
    return render_template('index.html')
@app.route("/matcher",methods=['GET','POST'])
def matchReume():
    return render_template('index.html')
if __name__=="__main__":
    app.run(debug=True)