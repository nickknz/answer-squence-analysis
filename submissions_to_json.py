import psycopg2
import hashlib
import json

# config database
DB_CONFIG = {
    'dbname': 'answerbook-v2',
    'user': 'user',
    'password': 'pass',
    'host': 'localhost',
    'port': 5432
}

# connect to the database
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# select all records from the submission table
cur.execute("""
    SELECT username, question, part, section, task, timetz, answer
    FROM y2020_co141_exam.submission
""")
rows = cur.fetchall()

# initialize dictionaries 
# by_question: {question: [{username, timestamp, answer_length}]}
by_question = {}
by_student = {}

for username, question, part, section, task, timetz, answer in rows:
    # hash username
    # hashed_username = hashlib.sha256(username.encode()).hexdigest()[:12]
    timestamp = timetz.replace(microsecond=0).isoformat() + "Z"
    answer_length = len(answer) if answer else 0
    question_key = f"question{question}"
    part_key = f"part{part}"
    section_key = f"section{section}"
    task_key = f"task{task}"

    # by question
    by_question.setdefault(question_key, {}) \
        .setdefault(part_key, {}) \
        .setdefault(section_key, {}) \
        .setdefault(task_key, []) \
        .append({
            "username": username,
            "timestamp": timestamp,
            "answer_length": answer_length
        })

    # by student
    by_student.setdefault(username, []).append({
        "question": question,
        "part": part,
        "section": section,
        "task": task,
        "timestamp": timestamp,
        "answer_length": answer_length
    })

# write to JSON files
with open("example-question-view-v2.json", "w") as fq:
    json.dump(by_question, fq, indent=2)

with open("example-student-sequence-v2.json", "w") as fs:
    json.dump(by_student, fs, indent=2)

print("JSON file completed: by_question.json and by_student.json")

# close connection
cur.close()
conn.close()
