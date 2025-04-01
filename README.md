# SightLine

<div align="center">
  <img src="web/static/favicon.png" width=256 height=256 title="SightLine" alt="SightLine logo" />
</div>

## Intro

Sightline is an LLM-powered research assistant designed to help you quickly extract insights from research papers on [arXiv](https://arxiv.org/). 

It provides a structured summary of the paper, with more detail than just the abstract. It also answers questions based on the content of the paper, without making overconfident claims.

By simply pasting the link, you can save hours of sifting through multiple papers.

## Dirs

```plaintext
├── api                   # FastAPI app
│   └── app
│       ├── paper_reader  # Where the magic happens
│       ├── schemas       # Where the rules are imposed
│       └── tests         # Where the experiments take place
└── web                   # Svelte app
    ├── src
    │   ├── components    # All the pieces of the UI
    │   │   ├── buttons
    │   │   ├── cards
    │   │   └── fields
    │   ├── lib           # States and requests
    │   └── routes        # Routes
    └── static            # Images and stuff
```

## Tech

### Backend

- [FastAPI](https://fastapi.tiangolo.com/) for lightning-fast APIs
- [LangChain](https://www.langchain.com/) for unlocking the superpower of LLMs
- [Chroma](https://www.trychroma.com/) for a database filled with mysteries to humans

### Frontend

- [SvelteKit](https://svelte.dev/) for making the developer (me) happy
- Plain CSS for taking a break from Tailwind

## Dev

### Backend

Make sure you're at `./api`

Make sure the virtual environment is activated

Install required packages and run the dev server:
```bash
pip install -r requirements.txt
fastapi dev app/main.py
```

### Frontend

Make sure you're at `./web`

Install required packages and run the dev server:
```bash
yarn install
yarn run dev
```

## Next

- Enhance user interface
- Implement additional RAG techniques
- Extend coverage beyond arXiv papers

## Ethics

SightLine is not intended to, nor capable of, replacing human researchers. Despite constant improvements, AI can still make mistakes. It's important not to become overly reliant on AI.

The research is yours. While the literature review process can be challenging, it remains both essential and meaningful.
