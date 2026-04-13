import subprocess
scripts = ["graph_generator.py", "in_context_learning_examples.py", "Query_input_builder.py","LLM_Management.py"]
for script in scripts:
    subprocess.run(["python", script])
