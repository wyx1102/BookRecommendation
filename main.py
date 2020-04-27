
from flask import Flask, request,render_template #import main Flask class and request object
import csv
import submit as sb
import get_prediction
import os
import random

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
# user = 'AVC8ZAFPYOHZL'
app = Flask(__name__) #create the Flask app

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/recommendation', methods=['GET', 'POST']) #allow both GET and POST requests
def recommendation():
    if request.method == 'POST': #this block is only entered when the form is submitted        
        
        # with app.app_context():
        #     data=read(userid)
	    
        offset = random.randint(1, 100)
        f = open('userid.csv')
        for _ in range(offset):  
            f.readline() 
            user = f.readline()  
        
        urls = get_prediction.main(user)
        print(urls)
        # sb.main()
        return render_template('results.html', urls=urls)
    else:

       
        
        data = []
        filesize = 11            
        samples = random.sample(range(1, filesize), 10)       
        f = open('books.csv')       
        for offset in samples:
            
            for _ in range(offset):  
                f.readline() 
           
            random_line = f.readline()     
            data.append(random_line)

        
        
        return render_template('recommendation.html', list = data)
    # return '''<form method="POST">
    #               userid: <input type="text" name="userid"><br>
    #               rating: <input type="text" name="rating"><br>
    #               <input type="submit" value="Submit"><br>
    #           </form>'''


if __name__ == '__main__':
    app.run(debug=True, port=5000) 