const API_URL = import.meta.env.VITE_API_URL;

export interface SummarizeRequest {
	arxiv_url: string;
}

export interface AskRequest {
	arxiv_url: string;
	question: string;
}

export interface SummarizeResponse {
	title: string;
	authors: string[];
	abstract: string;
	key_points: string[];
	methodology: string;
	results: string;
	implications: string;
}

export interface AskResponse {
	answer: string;
}

/**
 * Get a summary of the paper
 */
export const summarizePaper = async (arxivUrl: string): Promise<string> => {
	const response = await fetch(`${API_URL}/summarize`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({ paper_url: arxivUrl })
	});

	if (!response.ok) {
		throw new Error(`Error: ${response.status}`);
	}

	const data: SummarizeResponse = await response.json();

	const summary = `
	<h3>${data.title}</h3>
	<h4>Authors</h4>
	<p>${data.authors.join(', ')}</p>
	<h4>Abstract</h4>
	<p>${data.abstract}</p>
	<h4>Key Points</h4>
	<p>${data.key_points.join(', ')}</p>
	<h4>Methodology</h4>
	<p>${data.methodology}</p>
	<h4>Results</h4>
	<p>${data.results}</p>
	<h4>Implications</h4>
	<p>${data.implications}</p>
	`;

	return summary;
};

/**
 * Ask a question about the paper
 */
export const askQuestion = async (
	arxivUrl: string,
	question: string,
	strategy: string
): Promise<string> => {
	const response = await fetch(`${API_URL}/ask`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({
			paper_url: arxivUrl,
			question: question,
			strategy: strategy
		})
	});

	if (!response.ok) {
		throw new Error(`Error: ${response.status}`);
	}

	const data: AskResponse = await response.json();
	return data.answer;
};
