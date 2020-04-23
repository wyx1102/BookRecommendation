
from flask import Flask, request,render_template #import main Flask class and request object

import submit as sb
import get_prediction
import os
PROJECT = "bookrecommendation-267223"
PROJECT_NAME = "bookrecommendation"
BUCKET = "amazonbookrecommendation"
REGION = "us-east1"

os.environ["PROJECT"] = PROJECT
os.environ["PROJECT_NAME"] = PROJECT_NAME
os.environ["BUCKET"] = BUCKET
os.environ["REGION"] = REGION
os.environ["TFVERSION"] = "1.15"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/wuyuxuan/Desktop/test/bookrecommendation-267223-bf4a12419741.json'
user = 'AVC8ZAFPYOHZL'
app = Flask(__name__) #create the Flask app

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/recommendation', methods=['GET', 'POST']) #allow both GET and POST requests
def recommendation():
    if request.method == 'POST': #this block is only entered when the form is submitted        
        
        # with app.app_context():
        #     data=read(userid)
	    
        urls = get_prediction.main(user)
        print(urls)
        sb.main()
        return render_template('results.html', 
            urls=urls)
    else:
        list = [
        "http://ecx.images-amazon.com/images/I/51teQT7y83L.jpg" ,
        "http://ecx.images-amazon.com/images/I/51C0AJ2QAVL.jpg",
        "http://ecx.images-amazon.com/images/I/41A1dNkj%2BIL.jpg",
        "http://ecx.images-amazon.com/images/I/71V79Uw3%2BML.jpg",
        "http://ecx.images-amazon.com/images/I/41S16DCZ5JL.jpg",
        "http://ecx.images-amazon.com/images/I/51CD3VG60CL.jpg",
        "http://ecx.images-amazon.com/images/I/21HDCBWJBXL.jpg",
        "http://ecx.images-amazon.com/images/I/51RY565FH2L.jpg",
        "http://ecx.images-amazon.com/images/I/51DDHXDA7ML.jpg",
        "http://ecx.images-amazon.com/images/I/519R6MMGFEL.jpg"
        ]
        return render_template('recommendation.html', list = list)
    # return '''<form method="POST">
    #               userid: <input type="text" name="userid"><br>
    #               rating: <input type="text" name="rating"><br>
    #               <input type="submit" value="Submit"><br>
    #           </form>'''


if __name__ == '__main__':
    app.run(debug=True, port=5000) 