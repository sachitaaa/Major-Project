# MAJOR PROJECT
Major Project: RAG-based MCQ Generation and Difficulty Estimation

## How to run

1. Install the packages: `pip install -r requirement.txt`
2. Create a `.env` file in the project folder, next to `headless.py`, containing one line: `GROQ_API_KEY=your_key_here`. Never commit this file.
3. Generate questions from a PDF: `python headless.py --pdf path/to/notes.pdf --topic "decision trees" --num 5 --save`
   - `--pdf` is the PDF to read (required).
   - `--topic` should match a topic in the taxonomy file `Datasets/Jsons/datamining.json`. If nothing matches, the pipeline falls back to generic Remember and Understand questions.
   - `--num` is the number of questions (default 5), and `--save` writes the result as JSON into `mcq_output/`.
   - The first run on a PDF extracts and indexes it, which can take a while, and later runs reuse the stored index.
4. Run the difficulty evaluation: `python train_improved.py`. It reads `final2/master_dataset_final.csv` and prints the label agreement table and the model comparison.

