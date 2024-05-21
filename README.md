# tiny-type-agent
Minimal example of a safe Autonomous DevTool for nopilot.dev blog

## Running

```
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python main.py <source_file> <mypy_command>
```

Example:

```
python main.py example true
```