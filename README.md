# MAJOR PROJECT
Major Project: RAG-based MCQ Generation and Difficulty Estimation

How to run
Install the packages: pip install -r requirement.txt
Create a .env file next to headless.py with GROQ_API_KEY=your_key (never commit this file).
Generate questions from a PDF: python headless.py --pdf path/to/notes.pdf --topic "decision trees" --num 5 --save
Run the difficulty evaluation: python train_improved.py


