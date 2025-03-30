from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from pydantic import BaseModel, Field


class PaperSummary(BaseModel):
    """Structure for the paper summary output."""

    title: str = Field(description="The title of the paper")
    authors: list[str] = Field(description="List of paper authors")
    abstract: str = Field(description="A concise summary of the paper's abstract")
    key_points: list[str] = Field(description="Key points and findings from the paper")
    methodology: str = Field(description="Description of the methodology used")
    results: str = Field(description="Main results and conclusions")
    implications: str = Field(
        description="Implications and potential impact of the research"
    )


class PaperSummarizer:
    def __init__(self, openai_api_key: str):
        """
        Initialize the PaperSummarizer with OpenAI API key.

        Args:
            openai_api_key (str): OpenAI API key for accessing the language model
        """
        self._llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.3,
            openai_api_key=openai_api_key,
        )
        self._output_parser = PydanticOutputParser(pydantic_object=PaperSummary)
        self._prompt_template = self._create_prompt_template()

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        Create the prompt template for paper summarization.

        Returns:
            ChatPromptTemplate: The formatted prompt template
        """
        system_template = """You are an expert at summarizing academic papers. Your task is to analyze academic papers and create comprehensive, well-structured summaries that capture the key aspects of the research."""

        human_template = """Please analyze the following paper and create a comprehensive summary.

Paper Title: {title}
Authors: {authors}
Abstract: {abstract}

Paper Content:
{content}

Please provide a detailed summary following this structure:
{format_instructions}

Focus on:
1. Capturing the main contributions and findings
2. Explaining the methodology clearly
3. Highlighting key results and their significance
4. Discussing the implications of the research

Make the summary clear, concise, and well-structured."""

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        return ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

    def _prepare_prompt_inputs(self, paper_data: dict) -> dict:
        """
        Prepare inputs for the prompt template.

        Args:
            paper_data (Dict): Paper data from ArXivPaper.get_paper_data()

        Returns:
            Dict: Formatted inputs for the prompt template
        """
        details = paper_data["details"]
        documents = paper_data["documents"]

        # Combine all document contents
        full_text = "\n\n".join(doc.page_content for doc in documents)

        return {
            "title": details["title"],
            "authors": ", ".join(details["authors"]),
            "abstract": details["abstract"],
            "content": full_text,
            "format_instructions": self._output_parser.get_format_instructions(),
        }

    def generate_summary(self, paper_data: dict) -> str:
        """
        Generate a markdown summary of the paper.

        Args:
            paper_data (Dict): Paper data from ArXivPaper.get_paper_data()

        Returns:
            str: Markdown-formatted summary of the paper
        """
        # Prepare prompt inputs and run chain with LCEL
        prompt_inputs = self._prepare_prompt_inputs(paper_data)

        chain = self._prompt_template | self._llm | self._output_parser
        summary = chain.invoke(prompt_inputs)

        # Format the markdown output
        markdown = f"""# Summary of {summary.title}

## Authors
{', '.join(summary.authors)}

## Abstract
{summary.abstract}

## Key Points
{chr(10).join(f"- {point}" for point in summary.key_points)}

## Methodology
{summary.methodology}

## Results
{summary.results}

## Implications
{summary.implications}
"""

        return markdown
