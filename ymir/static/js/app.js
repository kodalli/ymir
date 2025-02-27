// Ymir RLHF Dataset Builder - Client-side JavaScript

document.addEventListener('DOMContentLoaded', function () {
    // Initialize toasts
    initToasts();

    // Add event listeners for form submissions
    attachEventListeners();

    // Initialize RLHF dataset table handling
    initRLHFDatasetHandling();
});

/**
 * Initialize toast notification functionality
 */
function initToasts() {
    // Create toast container if it doesn't exist
    if (!document.querySelector('.toast-container')) {
        const toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }

    // Listen for custom toast events
    document.addEventListener('showToast', function (e) {
        const { message, type = 'info', duration = 3000 } = e.detail;
        showToast(message, type, duration);
    });
}

/**
 * Display a toast notification
 * @param {string} message - The message to display
 * @param {string} type - The type of toast (info, success, error)
 * @param {number} duration - How long to show the toast in milliseconds
 */
function showToast(message, type = 'info', duration = 3000) {
    const toastContainer = document.querySelector('.toast-container');

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    toastContainer.appendChild(toast);

    // Force reflow to enable transition
    void toast.offsetWidth;

    toast.classList.add('visible');

    setTimeout(() => {
        toast.classList.remove('visible');

        // Remove element after transition completes
        toast.addEventListener('transitionend', function () {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        });
    }, duration);
}

/**
 * Format the RLHF dataset data into a table
 * @param {Array} data - The RLHF dataset data
 * @returns {HTMLElement} - The table element
 */
function formatRLHFDataTable(data) {
    if (!data || data.length === 0) {
        const emptyMsg = document.createElement('p');
        emptyMsg.className = 'text-center text-gray-500 my-8';
        emptyMsg.textContent = 'No RLHF data available yet. Rate some responses to start building your dataset.';
        return emptyMsg;
    }

    const table = document.createElement('table');
    table.className = 'rlhf-table';

    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');

    const headers = [
        'User Prompt',
        'LLM1',
        'LLM2',
        'Response1',
        'Response2',
        'Rating',
        'Notes'
    ];

    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create body
    const tbody = document.createElement('tbody');

    data.forEach(entry => {
        const row = document.createElement('tr');

        // User Prompt
        let td = document.createElement('td');
        td.textContent = entry.user_prompt;
        td.style.maxWidth = '200px';
        td.style.overflow = 'hidden';
        td.style.textOverflow = 'ellipsis';
        row.appendChild(td);

        // LLM1
        td = document.createElement('td');
        td.textContent = entry.llm1;
        row.appendChild(td);

        // LLM2
        td = document.createElement('td');
        td.textContent = entry.llm2;
        row.appendChild(td);

        // Response1 (truncated with expand/collapse)
        td = document.createElement('td');
        const resp1Container = document.createElement('div');
        resp1Container.className = 'response-content';
        resp1Container.textContent = truncateText(entry.response1, 150);

        if (entry.response1.length > 150) {
            const expandBtn = document.createElement('button');
            expandBtn.className = 'text-blue-600 text-sm mt-2';
            expandBtn.textContent = 'Show more';
            expandBtn.addEventListener('click', function () {
                if (expandBtn.textContent === 'Show more') {
                    resp1Container.textContent = entry.response1;
                    expandBtn.textContent = 'Show less';
                } else {
                    resp1Container.textContent = truncateText(entry.response1, 150);
                    expandBtn.textContent = 'Show more';
                }
            });
            td.appendChild(resp1Container);
            td.appendChild(expandBtn);
        } else {
            td.appendChild(resp1Container);
        }
        row.appendChild(td);

        // Response2 (truncated with expand/collapse)
        td = document.createElement('td');
        const resp2Container = document.createElement('div');
        resp2Container.className = 'response-content';
        resp2Container.textContent = truncateText(entry.response2, 150);

        if (entry.response2.length > 150) {
            const expandBtn = document.createElement('button');
            expandBtn.className = 'text-blue-600 text-sm mt-2';
            expandBtn.textContent = 'Show more';
            expandBtn.addEventListener('click', function () {
                if (expandBtn.textContent === 'Show more') {
                    resp2Container.textContent = entry.response2;
                    expandBtn.textContent = 'Show less';
                } else {
                    resp2Container.textContent = truncateText(entry.response2, 150);
                    expandBtn.textContent = 'Show more';
                }
            });
            td.appendChild(resp2Container);
            td.appendChild(expandBtn);
        } else {
            td.appendChild(resp2Container);
        }
        row.appendChild(td);

        // Rating
        td = document.createElement('td');
        td.textContent = entry.rating;
        row.appendChild(td);

        // Notes
        td = document.createElement('td');
        td.textContent = entry.notes || '-';
        row.appendChild(td);

        tbody.appendChild(row);
    });

    table.appendChild(tbody);
    return table;
}

/**
 * Initialize RLHF dataset handling
 */
function initRLHFDatasetHandling() {
    // Add event listener for populating RLHF dataset
    htmx.on('#rlhf-dataset', 'htmx:afterSwap', function (evt) {
        try {
            const data = JSON.parse(evt.detail.xhr.response).data;
            const rlhfDataset = document.getElementById('rlhf-dataset');

            // Clear existing content
            rlhfDataset.innerHTML = '';

            // Create and add the table
            const table = formatRLHFDataTable(data);
            rlhfDataset.appendChild(table);
        } catch (error) {
            console.error('Error parsing RLHF dataset:', error);
            showToast('Error loading RLHF dataset', 'error');
        }
    });
}

/**
 * Attach event listeners for various UI interactions
 */
function attachEventListeners() {
    // Track chat message generation
    htmx.on('body', 'htmx:beforeSend', function (evt) {
        if (evt.detail.requestConfig.url === '/chat') {
            // Show loading indicator
            const targetId = JSON.parse(evt.detail.requestConfig.parameters).llm_key === 'llm_1'
                ? 'llm1-chat'
                : 'llm2-chat';

            const target = document.getElementById(targetId);
            const loadingIndicator = document.createElement('div');
            loadingIndicator.className = 'chat-message assistant-message';

            const spinner = document.createElement('div');
            spinner.className = 'spinner border-blue-600';
            spinner.style.borderTopColor = '#3b82f6';
            spinner.style.width = '1rem';
            spinner.style.height = '1rem';
            spinner.style.borderWidth = '0.2rem';

            loadingIndicator.appendChild(spinner);
            loadingIndicator.appendChild(document.createTextNode(' Generating response...'));

            target.appendChild(loadingIndicator);
            target.scrollTop = target.scrollHeight;
        }
    });

    // Handle provider changes and model selection
    htmx.on('body', 'htmx:afterSwap', function (evt) {
        // When provider is changed and model select is updated
        if (evt.detail.target.id === 'llm1-model-select' || evt.detail.target.id === 'llm2-model-select') {
            const select = evt.detail.target;

            // If there are options, select the first one by default
            if (select.options && select.options.length > 0) {
                // Trigger the model update
                const llmKey = select.id === 'llm1-model-select' ? 'llm_1' : 'llm_2';
                const selectedModel = select.options[0].value;

                // Show toast to indicate models were loaded
                showToast(`Models loaded for ${llmKey}`, 'success', 2000);

                // Make the POST request to update the model on the server
                fetch('/update_model', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `llm_key=${llmKey}&model=${selectedModel}`
                });
            } else {
                // If no options were found, show an error
                showToast('No models found for the selected provider', 'error');
            }
        }
    });

    // Handle model selection change
    document.addEventListener('change', function (evt) {
        if (evt.target.id === 'llm1-model-select' || evt.target.id === 'llm2-model-select') {
            const llmKey = evt.target.id === 'llm1-model-select' ? 'llm_1' : 'llm_2';
            const selectedModel = evt.target.value;

            // Make the POST request to update the model on the server
            fetch('/update_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `llm_key=${llmKey}&model=${selectedModel}`
            });
        }
    });

    // Handle successful rating
    htmx.on('#rating-result', 'htmx:afterSettle', function (evt) {
        if (evt.detail.requestConfig.url === '/rate' && evt.detail.successful) {
            // Clear rating notes
            document.getElementById('rating-notes').value = '';

            // Show success message
            showToast('Rating saved successfully!', 'success');
        }
    });

    // Handle download RLHF dataset
    htmx.on('#download-result', 'htmx:afterSettle', function (evt) {
        if (evt.detail.requestConfig.url === '/download_rlhf' && evt.detail.successful) {
            try {
                const response = JSON.parse(evt.detail.xhr.response);
                document.getElementById('download-result').textContent = `Dataset saved to: ${response.filename}`;
                showToast(`Dataset saved to: ${response.filename}`, 'success');
            } catch (error) {
                console.error('Error handling download result:', error);
            }
        }
    });
}

/**
 * Truncate text to specified length with ellipsis
 * @param {string} text - The text to truncate
 * @param {number} maxLength - Maximum length before truncation
 * @returns {string} - Truncated text
 */
function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}
