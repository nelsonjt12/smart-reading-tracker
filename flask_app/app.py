from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    # Sample data (replace with dynamic data as needed)
    summary = {
        "rating_diff": 0.52,
        "avg_length": 353,
        "shortest": 4,
        "longest": 980,
        "avg_days": 305.0,
        "avg_tbr": 2111.1,
        "authors": 321,
        "diversity": 45.7,
        "top_authors": [
            ("Scott Westerfeld", 15),
            ("Sarah J. Maas", 12),
            ("Maggie Stiefvater", 11),
            ("Victoria E. Schwab", 11),
            ("Leigh Bardugo", 11),
        ],
        "recommendations": [
            {"title": "Cinder House", "author": "Freya Marske", "rating": 5.00, "pages": 144},
            {"title": "The Last Contract of Isako", "author": "Fonda Lee", "rating": 5.00, "pages": None},
            {"title": "Untitled (The Singing Hills Cycle, #8)", "author": "Nghi Vo", "rating": 5.00, "pages": None},
            {"title": "Untitled (The Singing Hills Cycle, #7)", "author": "Nghi Vo", "rating": 5.00, "pages": None},
            {"title": "The Everlasting", "author": "Alix E. Harrow", "rating": 4.83, "pages": 400},
        ]
    }
    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
