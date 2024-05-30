from flask import Flask, request, jsonify, render_template
from flask_apscheduler import APScheduler
import requests
import pdfplumber
import pandas as pd
import model

class Config:
    SCHEDULER_API_ENABLED = True

app = Flask(__name__)
app.config.from_object(Config())

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

def download_and_update_dataset():
    pdf_path = "Winning_Number_History.pdf"
    dataset_path = "draw-history-full.csv"
    pdf_url = "https://files.floridalottery.com/exptkt/ff.pdf"

    response = requests.get(pdf_url, timeout=10)
    if response.status_code == 200:
        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(response.content)
        print("PDF file downloaded successfully")

        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
            text = page.extract_text()
            lines = text.split('\n')
            start_index = next(i for i, line in enumerate(lines) if "Draw Date" in line) + 1
            data_lines = lines[start_index:start_index+2]

            data = []
            for line in data_lines:
                parts = line.split(maxsplit=2)
                draw_date = parts[0]
                draw_type = parts[1].lower()
                winning_numbers = parts[2].split()
                data.append([draw_date, draw_type] + winning_numbers)

            df_new = pd.DataFrame(data, columns=['Draw Date', 'Draw Type', 'First Number', 'Second Number', 'Third Number', 'Fourth Number', 'Fifth Number'])
            df_new['Draw Date'] = pd.to_datetime(df_new['Draw Date'], format='%m/%d/%y').dt.strftime('%A, %B %d, %Y')
            df_existing = pd.read_csv(dataset_path)
            df_updated = pd.concat([df_new, df_existing], ignore_index=True)
            df_updated.to_csv(dataset_path, index=False)
            print("Dataset has been updated with the latest draw data at the top.")
    else:
        print("Failed to download PDF, status code:", response.status_code)

# Schedule the task to run once every 24 hours
scheduler.add_job(id='Scheduled Task', func=download_and_update_dataset, trigger='interval', hours=24)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Reload the model and data each time a prediction is requested
    my_model, X_val = model.init_and_load_model('draw-history-full.csv')
    
    index = int(request.form.get('index', 0))
    index = min(index, len(X_val) - 1)
    single_sequence = X_val[index:index+1]
    single_prediction = my_model.predict(single_sequence)
    predicted_numbers = model.get_unique_numbers([single_prediction[j][0] for j in range(5)])
    predicted_numbers = [int(num) for num in predicted_numbers]

    return jsonify({'predicted_numbers': predicted_numbers})

if __name__ == '__main__':
    app.run(debug=True)
