# Sparkey Reflect — Plugin Page Brief

> Instructions for the website agent to generate the Sparkey Reflect plugin page on sparkey.ai

---

## Page Title

**Sparkey Reflect** — AI Coding Advisor for Claude Code

## Tagline

Analyze your AI coding sessions. Get coached. Level up.

## Hero Section

### Headline
Turn your AI coding sessions into skill growth

### Subheadline
Sparkey Reflect analyzes how you use Claude Code and generates personalized coaching insights grounded in DORA, SPACE, DevEx, GitClear, and METR research benchmarks.

### CTA Buttons
- **Install Plugin** (primary) → link to `#installation`
- **View on GitHub** (secondary) → `https://github.com/Sparkey-AI/sparkey-reflect`
- **View on PyPI** (tertiary) → `https://pypi.org/project/sparkey-reflect/`

---

## What It Does (Feature Grid)

### 7 Scoring Dimensions
Smooth, industry-benchmarked scoring across every aspect of AI-assisted development — not step-function grades, but continuous curves that reflect gradual improvement.

| Dimension | What It Measures | Key Benchmark |
|-----------|-----------------|---------------|
| Prompt Quality | Specificity, context richness, clarity, efficiency, chain of thought | GitClear: specific prompts → 40% less churn |
| Conversation Flow | Turns to resolution, correction rate, context retention, first acceptance, iteration velocity | DORA: fewer iterations = faster lead time |
| Context Management | File references, error context, code snippets, scope clarity, context window efficiency | METR: code context → better completions |
| Session Patterns | Duration, frequency, diversity, fatigue detection, deep work alignment | DORA 2024: uninterrupted blocks boost throughput |
| Tool Usage | Tool diversity, MCP utilization, slash commands, automation, tool appropriateness | Specialized tools = mastery signal |
| Rule File Quality | Completeness, specificity, actionability, currency, ecosystem coverage | DORA: stale docs = risk |
| Outcome Tracker | AI commit rate, productivity, rework rate, quality signals, quality trend | GitClear 2024: AI rework benchmarks |

### 3 Modes

#### Report Mode
Run `/sparkey:reflect` (or `daily`, `weekly`, `monthly`, `full`) to get a complete analysis with severity-ranked insights, real session evidence, and next-step recommendations.

#### Deep Dive Mode
Run `/sparkey:reflect deep-dive <skill>` to get an in-depth analysis of one skill area with before/after examples and practice exercises.

#### Update Rules Mode
Run `/sparkey:reflect update-rules` to automatically improve your CLAUDE.md based on real session data — close the gap between how you work and what your AI knows.

---

## Use Cases

### 1. Weekly Skill Check-In
A senior developer runs `/sparkey:reflect` every Monday to track AI coding effectiveness. They discover a 23% correction rate — nearly 1 in 4 AI responses need fixing. The deep-dive reveals they skip error context when debugging, forcing extra back-and-forth. After two weeks of including stack traces, their correction rate drops to 8%, saving ~30 minutes per day.

### 2. Onboarding Teams to AI-Assisted Development
An engineering manager installs the plugin for a team of 6 developers who recently adopted Claude Code. Junior developers run `/sparkey:reflect monthly` for a comprehensive baseline. Reports highlight specific patterns — one developer uses `Bash(sed)` instead of Edit, another has 3-hour marathon sessions with declining quality. Data-driven coaching sessions replace guesswork. After one quarter, the team's average score improves from 48 to 71.

### 3. Improving Project Rule Files
A tech lead runs `/sparkey:reflect update-rules` to optimize their CLAUDE.md. The analysis reveals only 1 instruction file with no code examples. The plugin proposes targeted additions — testing conventions, import path rules, project defaults. After the update, First Response Acceptance jumps from 52% to 74% because Claude follows project conventions on the first try.

---

## Installation Section

### Prerequisites
- Claude Code 1.0.33 or later
- Python 3.11+

### Install via Plugin Marketplace
```bash
# In Claude Code, run:
/plugin marketplace add Sparkey-AI/sparkey-reflect
/plugin install sparkey@sparkey-reflect
```

### Install CLI Standalone (optional)
```bash
pip install sparkey-reflect
```

### Quick Start
```bash
# Run your first analysis
/sparkey:reflect

# Deep dive into a specific skill
/sparkey:reflect deep-dive prompt_engineering

# Improve your CLAUDE.md
/sparkey:reflect update-rules
```

---

## How It Works (Visual Flow)

Create a visual diagram showing:

```
Your Claude Code Sessions
        ↓
   Session Reader (parses conversation logs)
        ↓
   7 Analyzers (smooth scoring curves)
        ↓
   Insight Generator (LLM-powered coaching)
        ↓
   Personalized Report with actionable insights
```

---

## Scoring Preview

Show an example score output (use this as mockup data):

```
  Prompt Quality            [################....] 78/100
  Conversation Flow         [##############......] 71/100
  Context Management        [############........] 62/100
  Session Patterns          [###########.........] 55/100
  Tool Usage                [################....] 80/100
  Rule File Quality         [########............] 41/100
  Outcome Tracker           [###############.....] 73/100

  Overall Score: 68/100
```

---

## Pricing Section

### Open Source (Free)
- Full scoring engine (7 dimensions, 35 sub-dimensions)
- CLI analysis (`sparkey-reflect analyze`)
- Claude Code plugin (`/sparkey:reflect`)
- Local SQLite trend storage
- MIT License

### Teams Edition (sparkey.ai)
Everything in Open Source, plus:
- Version Control + Project Management integrations
- Multi-developer trend tracking and comparison
- Team-level usage analytics
- Industry benchmark comparisons
- Manager dashboard
- **→ [Contact us](https://sparkey.ai)**

---

## Technical Details (Collapsible Section)

- **Language**: Python 3.11+
- **Framework**: Click CLI + Claude Code plugin system
- **Scoring**: Sigmoid, bell, diminishing return curves (no step functions)
- **Storage**: Local SQLite for trend history
- **AI**: Optional LLM-powered insights via Claude API
- **Privacy**: All analysis runs locally — no data leaves your machine
- **License**: MIT
- **PyPI**: `pip install sparkey-reflect`
- **GitHub**: `github.com/Sparkey-AI/sparkey-reflect`

---

## Footer CTA

> Ready to level up your AI coding skills?

**Install Sparkey Reflect** — takes 30 seconds, runs entirely on your machine.

```
/plugin marketplace add Sparkey-AI/sparkey-reflect
/plugin install sparkey@sparkey-reflect
/sparkey:reflect
```

---

## SEO Metadata

- **Title**: Sparkey Reflect — AI Coding Advisor Plugin for Claude Code
- **Description**: Analyze your AI coding sessions and get personalized coaching insights. Industry-benchmarked scoring across prompt quality, conversation flow, tool usage, and more. Free and open source.
- **Keywords**: AI coding advisor, Claude Code plugin, developer productivity, prompt engineering, DORA metrics, coding skills, AI coaching, sparkey reflect
- **OG Image**: Use the scoring preview mockup as the social card
