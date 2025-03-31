// Create state using runes
let arxivUrl = $state('');
let question = $state('');
let result = $state('');
let isLoading = $state(false);
let errorMessage = $state('');

// Create a sightline object that exposes the state
export const sightline = {
	// Getter functions for the state
	get arxivUrl() {
		return arxivUrl;
	},
	get question() {
		return question;
	},
	get result() {
		return result;
	},
	get isLoading() {
		return isLoading;
	},
	get errorMessage() {
		return errorMessage;
	},

	// Setter functions for the state
	set arxivUrl(value) {
		arxivUrl = value;
	},
	set question(value) {
		question = value;
	},
	set result(value) {
		result = value;
	},
	set isLoading(value) {
		isLoading = value;
	},
	set errorMessage(value) {
		errorMessage = value;
	},

	// Reset function
	reset() {
		arxivUrl = '';
		question = '';
		result = '';
		isLoading = false;
		errorMessage = '';
	}
};
