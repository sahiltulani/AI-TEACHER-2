from flask import Flask, request, jsonify, render_template
from crewai import Agent, Crew, Task, Process
import os

app = Flask(__name__)

# Set environment variables for API access
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_MODEL_NAME"] = "llama3-70b-8192"
os.environ["OPENAI_API_KEY"] = "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"

# Define the Explanation Generator Agent
explanation_generator = Agent(
    role='explanation generator',
    goal='to provide detailed explanations for student questions',
    backstory='you are an AI teacher helping to explain topics to students',
    verbose=True,
    allow_delegation=False
)

chat_history = []

def generate_explanation(question):
    generate_explanation_task = Task(
        description=f"generate an explanation for the following question: {question} with the context: {chat_history}",
        agent=explanation_generator,
        expected_output="detailed explanation of the topic"
    )

    crew = Crew(
        agents=[explanation_generator],
        tasks=[generate_explanation_task],
        verbose=2,
        process=Process.sequential
    )

    output = crew.kickoff()
    return output

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data['question']
    response = generate_explanation(question)
    chat_history.append({"user": question, "bot": response})
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
