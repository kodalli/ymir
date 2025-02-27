// Ymir RLHF Dataset Builder - Minimal JavaScript with htmx

document.addEventListener('DOMContentLoaded', function () {
    // Initialize toast notifications (one JS feature we'll keep)
    initToasts();
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

    // Listen for custom toast events (can be triggered by htmx)
    document.addEventListener('showToast', function (e) {
        const { message, type = 'info', duration = 3000 } = e.detail;
        showToast(message, type, duration);
    });

    // Add event listener for htmx:afterSwap events that should trigger toasts
    document.body.addEventListener('htmx:afterSwap', function (evt) {
        // Check for toast triggers in the response headers
        const toastMessage = evt.detail.xhr.getResponseHeader('HX-Trigger-After-Swap');
        if (toastMessage) {
            try {
                const triggerData = JSON.parse(toastMessage);
                if (triggerData.showToast) {
                    showToast(
                        triggerData.showToast.message,
                        triggerData.showToast.type || 'info',
                        triggerData.showToast.duration || 3000
                    );
                }
            } catch (e) {
                console.error('Error parsing toast trigger data:', e);
            }
        }
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
 * Toggle content expansion for long text in tables
 * @param {HTMLElement} btn - The button that was clicked
 * @param {string} fullText - The full text content to show
 */
function toggleExpandText(btn, textId, isExpanded) {
    const container = document.getElementById(textId);

    if (!isExpanded) {
        // Load the full content via htmx
        // The expansion is now handled by the server returning the full content
    } else {
        // Collapse handled by htmx too
    }
}

// Export functions that need to be available globally
window.showToast = showToast;
window.toggleExpandText = toggleExpandText;
