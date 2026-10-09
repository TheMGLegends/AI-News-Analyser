# AI-News-Analyser
System that fetches news headlines from the provided RSS feed, has a local LLM analyse the headlines via ollama and shows results regarding areas like top global themes, affected sectors and affected companies.

## Requirements
After installing the repository, users should open their terminal inside the project folder and create a virtual terminal:

```python
python -m venv .venv
```

Users will then need to activate this virtual environment (Windows Command Prompt Example):

```python
.venv\Scripts\activate
```

Users may run into issues with this command if running from PowerShell, use the following command if this happens:

```python
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process; .\.venv\Scripts\Activate.ps1
```

Finally, users should install the required libraries which are stored in *requirements.txt*, like so:

```python
pip install -r requirements.txt
```
