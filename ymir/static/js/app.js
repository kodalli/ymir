// Ymir custom application logic
document.addEventListener('DOMContentLoaded', function () {
    initToasts();
});

function initToasts() {
    if (!document.querySelector('.toast-container')) {
        const toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }

    document.addEventListener('showToast', function (e) {
        const { message, type = 'info', duration = 3000 } = e.detail;
        showToast(message, type, duration);
    });

    document.body.addEventListener('htmx:afterSwap', function (evt) {
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
            } catch (e) {}
        }
    });
}

function showToast(message, type = 'info', duration = 3000) {
    const toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    let icon = 'info';
    if (type === 'success') icon = 'check-circle';
    if (type === 'error') icon = 'alert-circle';
    
    toast.innerHTML = `
        <i data-lucide="${icon}" class="w-5 h-5"></i>
        <span>${message}</span>
    `;

    toastContainer.appendChild(toast);
    
    if (window.lucide) {
        window.lucide.createIcons({
            root: toast
        });
    }

    void toast.offsetWidth;
    toast.classList.add('visible');

    setTimeout(() => {
        toast.classList.remove('visible');
        toast.addEventListener('transitionend', () => {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
        });
    }, duration);
}

// Global exposure
window.showToast = showToast;
