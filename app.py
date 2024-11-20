from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from transformers import pipeline
from datetime import datetime
import torch
from sqlalchemy import text

# setting up the flask app
app = Flask(__name__)

# setting up the database
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# loading in our model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# setting up database models
class Summary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_text = db.Column(db.Text, nullable=False)
    summary_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    title = db.Column(db.String(200))
    compression_ratio = db.Column(db.Float)

    # turning data into json for formatting
    def to_dict(self):
        return {
            'id': self.id,
            'original_text': self.original_text,
            'summary_text': self.summary_text,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'title': self.title,
            'compression_ratio': self.compression_ratio
        }

# database ORM ---> maps python objects to databse tables:
# essentially does this:

# CREATE TABLE user_model (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     name VARCHAR(80) UNIQUE NOT NULL,
#     email VARCHAR(80) UNIQUE NOT NULL
# );

# registering summarizer endpoint 
@app.route("/api/summaries/", methods = ["POST"])
def create_summary():
    # retrieves the JSON data sent in the request body
    data = request.get_json()
    # checks if the data is present and if it contains the key 'text'
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    # checks if the text contains at least 10 words
    if len(data["text"].split()) < 10:
        return jsonify({"error":"this text is too short to summarize"})
    
    try:
        generated_summary = summarizer(
            data["text"],
            max_length=data.get("max_length", 130),
            min_length=data.get("min_length", 30)
        )[0]['summary_text']

        compression_ratio = round(len(generated_summary.split()) / len(data['text'].split()) * 100, 2)

        new_summary = Summary(
            original_text=data['text'],
            summary_text=generated_summary,
            title=data.get('title', 'Untitled'),
            compression_ratio=compression_ratio
        )
        
        db.session.add(new_summary)
        db.session.commit()

        return jsonify(new_summary.to_dict()), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# get all summaries
@app.route('/api/summaries', methods=['GET'])
def get_summaries():
    summaries = Summary.query.order_by(Summary.created_at.desc()).all()
    return jsonify([summary.to_dict() for summary in summaries])

# get a specific summary
@app.route('/api/summaries/<int:summary_id>', methods=['GET'])
def get_summary(summary_id):
    summary = Summary.query.get_or_404(summary_id)
    return jsonify(summary.to_dict())

# update a summary
@app.route('/api/summaries/<int:summary_id>', methods=['PUT'])
def update_summary(summary_id):
    summary = Summary.query.get_or_404(summary_id)
    data = request.get_json()
    
    if 'text' in data:
        try:
            generated_summary = summarizer(
                data['text'],
                max_length=data.get('max_length', 130),
                min_length=data.get('min_length', 30)
            )[0]['summary_text']
            
            summary.original_text = data['text']
            summary.summary_text = generated_summary
            summary.compression_ratio = round(len(generated_summary.split()) / len(data['text'].split()) * 100, 2)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    if 'title' in data:
        summary.title = data['title']
        
    summary.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify(summary.to_dict())

# delete a summary
@app.route('/api/summaries/<int:summary_id>', methods=['DELETE'])
def delete_summary(summary_id):
    summary = Summary.query.get_or_404(summary_id)
    db.session.delete(summary)
    db.session.commit()
    return jsonify({'message': 'Summary deleted successfully'})

# health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        # use text() to create a SQL expression
        with db.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            db_status = True
    except Exception:
        db_status = False

    return jsonify({
        'status': 'healthy',
        'model_loaded': summarizer is not None,
        'gpu_available': torch.cuda.is_available(),
        'database_connected': db_status
    })

# Create tables
with app.app_context():
    db.create_all()


if __name__ == '__main__':
    app.run(debug=True)