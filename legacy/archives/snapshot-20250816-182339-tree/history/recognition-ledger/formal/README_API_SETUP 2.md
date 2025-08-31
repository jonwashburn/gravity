# API Key Setup

For security reasons, the Anthropic API key is not included in the code. 

To use the Recognition Science solvers, you need to set the `ANTHROPIC_API_KEY` environment variable:

## Option 1: Export in Terminal
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
python3 simple_solver.py
```

## Option 2: Create .env file
Create a `.env` file in the `formal/` directory:
```
ANTHROPIC_API_KEY=your-api-key-here
```

Then install python-dotenv:
```bash
pip install python-dotenv
```

## Option 3: Pass directly
```bash
ANTHROPIC_API_KEY="your-api-key-here" python3 simple_solver.py
```

## Getting an API Key

1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys
4. Create a new key
5. Copy and use as shown above

**Important**: Never commit API keys to version control! 