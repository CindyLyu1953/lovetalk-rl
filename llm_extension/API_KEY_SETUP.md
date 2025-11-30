# Gemini API Key Setup Guide

## How to Get an API Key

1. Visit Google AI Studio: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Select or create a Google Cloud project
4. Copy the generated API Key

---

## Configuration Methods

### Method 1: Environment Variable (Recommended)

#### macOS / Linux

Run in terminal:

```bash
export GEMINI_API_KEY="your-actual-api-key-here"
```

Or add to your shell configuration file (permanent):

```bash
# For zsh (macOS default)
echo 'export GEMINI_API_KEY="your-actual-api-key-here"' >> ~/.zshrc
source ~/.zshrc

# For bash
echo 'export GEMINI_API_KEY="your-actual-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

#### Windows

Run in PowerShell:

```powershell
$env:GEMINI_API_KEY="your-actual-api-key-here"
```

Or set permanent environment variable:
1. Search for "Environment Variables"
2. Click "Edit system environment variables"
3. Click "Environment Variables" button
4. In "User variables", create new:
   - Variable name: `GEMINI_API_KEY`
   - Variable value: your API key

---

### Method 2: Pass Directly in Code

```python
from llm_extension import DialogueRenderer

# Pass API key directly
renderer = DialogueRenderer(api_key="your-actual-api-key-here")
```

**Note:** Do NOT commit API keys to Git repository!

---

## Verify Configuration

Run the test script to check if configuration is correct:

```bash
python llm_extension/dialogue_renderer.py
```

If you see generated dialogue examples, the configuration is successful!

---

## FAQ

### Q: Where is my API key saved?

**A:**
- Method 1: In system environment variables (not saved to file)
- Method 2: In your shell configuration file (~/.zshrc or ~/.bashrc)

**NEVER save API keys in code files or Git repositories!**

### Q: How to check if API key is set?

**A:** Run in terminal:

```bash
echo $GEMINI_API_KEY
```

If it displays your API key, it's set correctly.

### Q: What if I forgot my API key?

**A:** Visit https://makersuite.google.com/app/apikey again to create a new key.

---

## Security Tips

**DO:**
- Use environment variables to save API keys
- Rotate API keys regularly
- Use different API keys for different projects

**DON'T:**
- Write API keys in code
- Commit API keys to Git
- Share API keys with others
- Display API keys publicly

---

## Pricing Information

Gemini Flash API has free quota:
- Free tier: 15 requests per minute
- Sufficient for development and small-scale testing

Detailed pricing: https://ai.google.dev/pricing

---

## Educational Purpose Note

This API key is used for educational simulations that teach conflict resolution and relationship management. All generated content is for learning purposes and does not represent real situations or cause real-world harm.
