from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
                age = request.form.get('age'),
                job = request.form.get('job'),
                marital = request.form.get('marital'),
                education = request.form.get('education'),
                default = request.form.get('default'),
                housing = request.form.get('housing'),
                loan = request.form.get('loan'),
                contact = request.form.get('contact'),
                month = request.form.get('month'),
                day_of_week = request.form.get('day_of_week'),
                duration = request.form.get('duration'),
                campaign = request.form.get('campaign'),
                pdays = request.form.get('pdays'),
                previous = request.form.get('previous'),
                poutcome = request.form.get('poutcome'),
                emp_var_rate = float(request.form.get('emp_var_rate')),
                cons_price_idx = float(request.form.get('cons_price_idx')),
                cons_conf_idx = float(request.form.get('cons_conf_idx')),
                euribor3m = float(request.form.get('euribor3m')),
                nr_employed = float(request.form.get('nr_employed'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        if results[0] == 0:
            result = "No, The client will not subscribe a term deposit."
        else:
            result = "Yes, The client will subscribe a term deposit."
        return render_template('home.html',results=result)
    

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)