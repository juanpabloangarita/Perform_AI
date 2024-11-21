## Getting Started

Create a virtual environment
```bash
python -m venv venv
```

Activate the virtual environment
```bash
source venv/bin/activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

Migrate the database
```bash
python manage.py migrate
```

Run the development server
```bash
python manage.py runserver
```